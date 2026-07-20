"""The TAMP control loop: interleaved symbolic planning + geometric refinement.

This is the core algorithm (Image 1's "constraint-aware planning") and the part
the reference repo never built. Pseudocode:

    problem = generate_problem(world, goal)        # bridge predicates optimistic
    repeat:
        plan = symbolic_planner.solve(problem)      # PDDL / Fast Downward
        if plan is None: return UNSOLVABLE
        for action in plan:
            verdict = refiner.refine(action)        # PyTAMP geometry
            if not feasible:
                problem := problem with falsified bridge predicate
                goto repeat                         # REPLAN
            apply action's symbolic effects to a shadow scene  # so later steps refine correctly
        return refined plan                          # geometrically feasible

The shadow scene mutates object poses as the (simulated) plan proceeds so that a
``stack`` refinement sees the bottom object where the previous ``place`` put it.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from unified_planning.model import Problem
from unified_planning.shortcuts import Not

from tamp.geometry.motion_planner import GeometricRefiner
from tamp.geometry.robot_model import RobotModel
from tamp.geometry.scene_interface import TampScene
from tamp.pddl.problem_generator import generate_problem
from tamp.planning.symbolic_planner import SymbolicPlanner
from tamp.types import Pose, TaskAction, WorldState

log = logging.getLogger("tamp.solver")


@dataclass
class TampResult:
    success: bool
    plan: List[TaskAction] = field(default_factory=list)
    replans: int = 0
    reason: str = ""
    falsified: List[tuple] = field(default_factory=list)


class TampSolver:
    def __init__(self, robot: Optional[RobotModel] = None, max_replans: int = 8,
                 engine: str = "fast-downward", default_surface: Optional[str] = None):
        self.robot = robot or RobotModel()
        self.planner = SymbolicPlanner(engine_name=engine)
        self.max_replans = max_replans
        self.default_surface = default_surface

    def solve(self, world: WorldState, goal_spec: Dict,
              pre_falsified: Optional[List[tuple]] = None) -> TampResult:
        # Falsified bridge predicates accumulated across replans, applied to the
        # PDDL problem's initial state each iteration. ``pre_falsified`` seeds the
        # loop with predicates learned OUTSIDE this call — e.g. an execution-time
        # failure reported by the robot (object not reached / dropped / collision)
        # that geometry alone judged feasible. This is what closes the loop across
        # plan→execute→replan, not just within geometric refinement.
        falsified: List[tuple] = list(pre_falsified or [])
        for attempt in range(self.max_replans + 1):
            problem = generate_problem(world, goal_spec, default_surface=self.default_surface)
            self._apply_falsified(problem, falsified)

            sym = self.planner.solve(problem)
            if not sym.success:
                return TampResult(False, replans=attempt,
                                  reason=f"symbolic planning failed ({sym.status})",
                                  falsified=falsified)

            refined, bad_pred = self._refine_plan(world, sym.actions)
            if bad_pred is None:
                log.info("TAMP solved in %d replan(s)", attempt)
                return TampResult(True, refined, attempt, "ok", falsified)

            if bad_pred in falsified:
                return TampResult(False, replans=attempt,
                                  reason=f"geometric failure not resolvable: {bad_pred}",
                                  falsified=falsified)
            log.info("replan: falsifying %s", bad_pred)
            falsified.append(bad_pred)

        return TampResult(False, replans=self.max_replans,
                          reason="max replans exceeded", falsified=falsified)

    # --- refine a symbolic plan against a shadow scene ---
    def _refine_plan(self, world: WorldState, actions: List[TaskAction]):
        scene = TampScene(); scene.update_scene(copy.deepcopy(world))
        refiner = GeometricRefiner(self.robot, scene)
        refined: List[TaskAction] = []
        for action in actions:
            res = refiner.refine(action)
            if not res.feasible:
                return refined, res.failed_predicate
            action.target_pose = res.target_pose
            action.params["approach"] = res.info
            refined.append(action)
            self._apply_effect_to_scene(scene, action)
        return refined, None

    @staticmethod
    def _apply_effect_to_scene(scene: TampScene, action: TaskAction) -> None:
        """Mutate the shadow scene so later refinements see the new geometry."""
        oid = action.object_id
        if action.action_type in ("PICK", "UNSTACK"):
            scene.set_held(oid)
        elif action.action_type in ("PLACE", "STACK") and action.target_pose is not None:
            scene.set_held(None)
            # place the object at the target footprint (drop EEF z to object center)
            tp = action.target_pose
            obj = scene.objects[oid]
            half = obj.dimensions[2] / 2
            scene.set_object_pose(oid, Pose(position=(tp.position[0], tp.position[1],
                                                      tp.position[2] - half),
                                            orientation=obj.pose.orientation))

    # --- inject falsified bridge predicates into the PDDL problem ---
    def _apply_falsified(self, problem: Problem, falsified: List[tuple]) -> None:
        objs = {o.name: o for o in problem.all_objects}
        for pred in falsified:
            name, args = pred[0], pred[1:]
            # A falsified predicate OUTLIVES the solve() that produced it: the ROS
            # loop carries `exec_falsified` across re-solves while perception keeps
            # republishing the world. If an object leaves the camera's view (or YOLO
            # re-IDs it), a stored predicate can reference an object that no longer
            # exists. Previously the `if a in objs` filter silently DROPPED such args,
            # handing the fluent the wrong arity -> UPExpressionDefinitionError, i.e.
            # a hard crash of the live closed loop. Skip the whole predicate instead.
            if not all(a in objs for a in args):
                log.warning("dropping stale falsified predicate %s "
                            "(references an object no longer in the world)", pred)
                continue
            fluent = problem.fluent(name)
            problem.set_initial_value(fluent(*[objs[a] for a in args]), False)
