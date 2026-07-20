"""Regressions for two core bugs an audit reproduced by execution.

H2 — `_refine_place` never collision-checked the GRIPPER (only the placed object's
     own footprint), so it certified placements that drive the hand through
     neighbouring geometry. `_refine_grasp` did check it; place did not.

M3 — `_apply_falsified` silently dropped args referencing objects that no longer
     exist, handing the fluent the wrong arity -> UPExpressionDefinitionError. This
     is reachable in the LIVE ROS loop: falsified predicates persist across re-solves
     while perception republishes the world, so an object leaving the camera's view
     crashed the closed loop.
"""
import pytest

from benchmarks import scene_pnp
from tamp.geometry.motion_planner import GeometricRefiner
from tamp.geometry.scene_interface import TampScene
from tamp.tamp_solver import TampSolver
from tamp.types import ObjectState, Pose, Surface, TaskAction, WorldState


# --- H2: the gripper must fit at the place pose ----------------------------
def _slot_world():
    """A narrow slot: two walls either side of the drop point. The 4cm cube fits
    between them, but the ~8x12cm gripper body does not."""
    table = Surface("table", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.6, 0.6, 0.02),
                    class_name="table")
    cube = ObjectState("cube", "block", Pose(position=(0.35, 0.25, 0.43)),
                       dimensions=(0.04, 0.04, 0.04))
    wall_l = ObjectState("wall_l", "wall", Pose(position=(0.5, 0.055, 0.47)),
                         dimensions=(0.10, 0.02, 0.12))
    wall_r = ObjectState("wall_r", "wall", Pose(position=(0.5, -0.055, 0.47)),
                         dimensions=(0.10, 0.02, 0.12))
    return WorldState(objects=[cube, wall_l, wall_r], surfaces=[table])


def test_place_between_walls_is_rejected_because_the_gripper_does_not_fit():
    world = _slot_world()
    scene = TampScene()
    scene.update_scene(world)
    refiner = GeometricRefiner(TampSolver().robot, scene)
    action = TaskAction(action_type="PLACE", object_id="cube", params={"s": "table"})
    res = refiner.refine(action)
    # The cube alone would fit in the 11cm gap between the walls; the gripper body
    # does not. The refiner must not certify this placement at the slot centre.
    if res.feasible:
        from tamp.geometry.collision import gripper_collision_free
        T = res.target_pose.homogeneous_matrix()
        cm = scene.collision_manager(exclude="cube")
        assert gripper_collision_free(cm, T), (
            "refiner returned a place pose where the GRIPPER is in collision")


# --- M3: a stale falsified predicate must not crash the solver --------------
def test_stale_falsified_predicate_is_dropped_not_crashed():
    """The object named by the falsified predicate is gone from the world (it left the
    camera's FOV). The solver must skip the predicate and keep planning, not crash."""
    world = scene_pnp.make_world()
    world.objects = [o for o in world.objects if o.object_id != "cup_1"]  # cup_1 vanished
    res = TampSolver().solve(
        world, {"type": "gather", "surface": "tray"},
        pre_falsified=[("reachable", "cup_1")],   # references the now-absent object
    )
    assert res.success, f"stale predicate should be dropped, not crash: {res.reason}"


def test_live_falsified_predicate_still_applies():
    """Guard the fix: a predicate whose object DOES exist must still be applied."""
    world = scene_pnp.make_world()
    res = TampSolver().solve(
        world, {"type": "gather", "surface": "tray"},
        pre_falsified=[("reachable", "cup_1")],
    )
    # cup_1 can never be picked -> the gather goal (ALL objects on the tray) fails.
    assert not res.success
    assert ("reachable", "cup_1") in res.falsified
