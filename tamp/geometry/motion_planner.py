"""Geometric refiner: turns a symbolic action into a feasibility verdict.

For a ``pick`` / ``unstack``: find a grasp whose grasp **and** pre-grasp poses
have an IK solution and are collision-free → object is graspable+reachable.
For a ``place`` / ``stack``: find a target EEF pose above the destination with
IK + collision-free placement → object is placeable / stackable.

On failure it reports which **geometric bridge predicate** to falsify, which the
TAMP loop injects into the PDDL state to drive replanning. This is the concrete
realization of the symbolic↔geometric bridge.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger("tamp.refiner")

from tamp.geometry.collision import (arm_links_free, gripper_collision_free,
                                     swept_gripper_free)
from tamp.geometry.grasp_planner import PRE_GRASP_OFFSET, sample_grasps
from tamp.geometry.robot_model import RobotModel
from tamp.geometry.scene_interface import TampScene
from tamp.types import Pose, TaskAction


@dataclass
class RefineResult:
    feasible: bool
    target_pose: Optional[Pose] = None       # grasp/place EEF pose -> TaskAction.target_pose
    joint_waypoints: List[np.ndarray] = field(default_factory=list)
    failed_predicate: Optional[tuple] = None  # e.g. ("graspable", "A") or ("stackable","A","B")
    info: str = ""


def _pose_from_mat(T: np.ndarray) -> Pose:
    from tamp.geometry.robot_model import _rotation_error  # noqa: F401 (keep module cohesive)
    # convert rotation matrix -> quaternion (w,x,y,z)
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return Pose(position=tuple(T[:3, 3]), orientation=(float(w), float(x), float(y), float(z)))


class GeometricRefiner:
    #: Release the object this far ABOVE its resting height (m). It must exceed the
    #: arm's real positioning error, or a low-landing release drives the object INTO
    #: the support and the physics engine resolves that interpenetration explosively —
    #: bullet flung a placed cup several metres across the world. Measured arm arrival
    #: error on this setup is 13–19 mm, so 25 mm clears it with margin while keeping the
    #: drop small enough that the object settles instead of bouncing off a small tray.
    #: (Placing at exactly the resting height, i.e. 0.0, causes the fling; the old code
    #: instead baked a ~36 mm offset into the EEF pose, which dropped it too far.)
    #: 18 mm: 25 mm dropped the (now 40 mm wide, aspect ~2:1) cup far enough to TIP it
    #: onto its side on landing — inside the harness tolerance but ugly. Still above
    #: the arm's measured 13-19 mm vertical arrival error so a low landing does not
    #: press the object into the support.
    DEFAULT_PLACE_CLEARANCE = 0.018

    def __init__(self, robot: RobotModel, scene: TampScene,
                 place_clearance: float = DEFAULT_PLACE_CLEARANCE):
        self.robot = robot
        self.scene = scene
        self.place_clearance = place_clearance

    def refine(self, action: TaskAction) -> RefineResult:
        kind = action.action_type
        if kind in ("PICK", "UNSTACK"):
            return self._refine_grasp(action)
        if kind in ("PLACE", "STACK"):
            return self._refine_place(action)
        return RefineResult(False, info=f"unknown action {kind}")

    # --- grasp feasibility ---
    def _refine_grasp(self, action: TaskAction) -> RefineResult:
        oid = action.object_id
        obj = self.scene.objects[oid]
        cm = self.scene.collision_manager(exclude=oid)
        any_ik = False
        fails = []   # per-candidate failure reasons, logged before any falsification
        for g in sample_grasps(obj):
            # Solve the (harder, lower) grasp pose first, then seed the adjacent
            # pre-grasp IK from it so it converges reliably and within limits.
            grasp_ik = self.robot.solve_ik(g.grasp_pose)
            if not grasp_ik.success:
                # RETRY before giving up: for near-envelope objects the DLS solve is
                # marginal and can fail on solver noise alone — observed live as the
                # top grasp of a reachable cup failing in one process while identical
                # inputs solved in another. The falsified `graspable` that resulted
                # made the whole gather goal "UNSOLVABLE". The pre-grasp sits 0.10 m
                # higher and converges much more easily, so solve it unseeded and
                # re-seed the grasp from its solution (plus fresh random restarts).
                pre_ik = self.robot.solve_ik(g.pre_grasp_pose)
                if not pre_ik.success:
                    # Escalate the pre-grasp too before abandoning the candidate.
                    # PROVEN necessary offline: at the cup's exact (full-precision)
                    # settled pose both default solves fail — the target is
                    # chaotically sensitive to sub-mm pose changes — while fresh
                    # restarts converge. Without this rung the whole top grasp was
                    # discarded and `graspable` wrongly falsified.
                    pre_ik = self.robot.solve_ik(g.pre_grasp_pose, restarts=24, seed=3)
                if not pre_ik.success:
                    fails.append(f"{g.approach}:grasp_ik+pre_ik")
                    continue
                grasp_ik = self.robot.solve_ik(g.grasp_pose, q0=pre_ik.q,
                                               restarts=20, seed=1)
                if not grasp_ik.success:
                    grasp_ik = self.robot.solve_ik(g.grasp_pose, q0=pre_ik.q,
                                                   restarts=40, iters=400, seed=7)
                if not grasp_ik.success:
                    fails.append(f"{g.approach}:grasp_ik(retried,err={grasp_ik.error:.1e})")
                    continue
            else:
                pre_ik = self.robot.solve_ik(g.pre_grasp_pose, q0=grasp_ik.q)
                if not pre_ik.success:
                    # Same marginality applies to THIS leg: near the reach envelope
                    # the seeded pre-grasp solve sits right at the solver tolerance
                    # (observed err ~1e-3 == pos_tol) and can flip per-process. Retry
                    # with fresh random restarts before discarding the whole grasp —
                    # discarding it cascades into a false `graspable` falsification.
                    pre_ik = self.robot.solve_ik(g.pre_grasp_pose, restarts=24, seed=3)
                    if not pre_ik.success:
                        fails.append(f"{g.approach}:pre_ik(retried,err={pre_ik.error:.1e})")
                        continue
            any_ik = True
            pre_free = gripper_collision_free(cm, g.pre_grasp_pose)
            grasp_free = gripper_collision_free(cm, g.grasp_pose)
            if pre_free and grasp_free:
                # §4-C3 fix: endpoints alone are not a motion. Sweep the gripper
                # proxy along the joint path pre-grasp -> grasp (what the
                # controller actually executes) and coarsely check the arm links
                # at both configurations — a side grasp whose endpoints are free
                # can still drag the hand/forearm through a neighbouring object.
                if not swept_gripper_free(cm, self.robot, pre_ik.q, grasp_ik.q):
                    fails.append(f"{g.approach}:swept_path_collision")
                    continue
                if not (arm_links_free(cm, self.robot.link_positions(pre_ik.q))
                        and arm_links_free(cm, self.robot.link_positions(grasp_ik.q))):
                    fails.append(f"{g.approach}:arm_link_collision")
                    continue
                return RefineResult(
                    True, target_pose=_pose_from_mat(g.grasp_pose),
                    joint_waypoints=[pre_ik.q, grasp_ik.q],
                    info=f"grasp[{g.approach}]")
            fails.append(f"{g.approach}:collision({'pre' if not pre_free else ''}"
                         f"{'+' if not pre_free and not grasp_free else ''}"
                         f"{'grasp' if not grasp_free else ''})")
        # decide which predicate to falsify — and SAY WHY, per candidate. A silent
        # falsification here previously surfaced only as "UNSOLVABLE" at the
        # symbolic layer and was misread as a PDDL bug.
        log.info("grasp refinement failed for %s: %s", oid, "; ".join(fails) or "no candidates")
        pred = ("reachable", oid) if not any_ik else ("graspable", oid)
        return RefineResult(False, failed_predicate=pred,
                            info="no IK" if not any_ik else "grasp in collision")

    # --- placement feasibility ---
    def _refine_place(self, action: TaskAction) -> RefineResult:
        oid = action.object_id
        obj = self.scene.objects[oid]
        is_stack = action.action_type == "STACK"
        bottom = action.params.get("bottom")
        sid = action.params.get("s")
        fail_pred = ("stackable", oid, bottom) if is_stack else ("placeable", oid, sid)

        candidates = self._destination_candidates(action)
        if not candidates:
            return RefineResult(False, failed_predicate=fail_pred, info="no destination")

        cm = self.scene.collision_manager(exclude=oid)
        import trimesh
        from tamp.geometry.collision import in_collision
        ik_ok = False
        for target_xyz in candidates:
            # EEF pose: top-down over the destination, at the object's CENTRE.
            # This MUST use the same convention as _refine_grasp, which returns
            # g.grasp_pose — an EEF pose whose origin sits at the object centre. The
            # consumer (moveit_executor._pick_place_sequence) adds one fixed fingertip
            # offset (tcp) to target_pose for BOTH pick and place, so any extra offset
            # here is applied on top of that: this used to add half-height+1mm, which
            # released the cup ~36 mm ABOVE its resting height. It then dropped onto the
            # 0.34 m tray and could bounce off — cups ended up off the tray while every
            # step still reported ok.
            T = np.eye(4)
            T[:3, :3] = np.diag([1, -1, -1])  # z down (top-down place)
            T[:3, 3] = np.array(target_xyz)

            place_ik = self.robot.solve_ik(T)
            if not place_ik.success:
                continue
            ik_ok = True
            # The GRIPPER must also fit at the place pose. _refine_grasp checks this;
            # _refine_place previously did not, so it certified placements that drive
            # the hand straight through neighbouring geometry (only the placed object's
            # own footprint was checked).
            if not gripper_collision_free(cm, T):
                continue
            # footprint collision-free? (shrink + lift 1mm so resting contact isn't a hit)
            placed = trimesh.creation.box(extents=np.asarray(obj.dimensions, float) * 0.95)
            pt = np.eye(4); pt[:3, 3] = np.array(target_xyz) + np.array([0, 0, 0.002])
            placed.apply_transform(pt)
            if in_collision(cm, placed, "placed"):
                continue
            # §4-C3 fix: sweep the DESCENT. Execution approaches the place pose from
            # ~0.10 m above; certify that path (and the arm links at its endpoints),
            # not just the final pose — endpoint-only checks approved placements
            # whose descent swept the hand through a neighbouring stack.
            T_above = T.copy()
            T_above[2, 3] += 0.10
            above_ik = self.robot.solve_ik(T_above, q0=place_ik.q)
            if above_ik.success:
                if not swept_gripper_free(cm, self.robot, above_ik.q, place_ik.q):
                    continue
                if not (arm_links_free(cm, self.robot.link_positions(above_ik.q))
                        and arm_links_free(cm, self.robot.link_positions(place_ik.q))):
                    continue
            return RefineResult(True, target_pose=_pose_from_mat(T),
                                joint_waypoints=[place_ik.q], info="place")

        return RefineResult(False, failed_predicate=fail_pred,
                            info="place IK failed" if not ik_ok else "all placements in collision")

    def _destination_candidates(self, action: TaskAction):
        """Candidate object-centre XYZ spots for the placement, best first.

        STACK -> the single spot atop the bottom object. PLACE on a surface ->
        the surface centre then a small grid around it, so multiple objects can be
        placed on the same surface without overlapping."""
        oid = action.object_id
        obj = self.scene.objects[oid]
        if action.action_type == "STACK":
            b = self.scene.objects[action.params.get("bottom")]
            bc = np.asarray(b.pose.position, float)
            top_z = bc[2] + b.dimensions[2] / 2 + obj.dimensions[2] / 2 + self.place_clearance
            return [(float(bc[0]), float(bc[1]), float(top_z))]

        s = self.scene.surfaces[action.params.get("s")]
        sc = np.asarray(s.pose.position, float)
        top_z = float(sc[2] + s.dimensions[2] / 2 + obj.dimensions[2] / 2 + self.place_clearance)
        # Candidate spots within the surface footprint, ordered for reachability:
        # vary Y first (cheap to reach), and only offset X toward the robot (-x),
        # never outward (+x pushes past the arm's comfortable workspace).
        sx, sy = s.dimensions[0], s.dimensions[1]
        step = 0.10
        dxs = [0.0, -step]            # centre or toward the robot
        dys = [0.0, step, -step, 2 * step, -2 * step]
        cands = []
        for dx in dxs:
            for dy in dys:
                if abs(dx) <= sx / 2 - obj.dimensions[0] / 2 and abs(dy) <= sy / 2 - obj.dimensions[1] / 2:
                    cands.append((float(sc[0] + dx), float(sc[1] + dy), top_z))
        return cands or [(float(sc[0]), float(sc[1]), top_z)]
