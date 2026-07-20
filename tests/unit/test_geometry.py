"""M2 unit tests: IK, grasp sampling, geometric refinement."""
import numpy as np
import pytest

from benchmarks.scene_stack3 import make_world
from tamp.geometry.grasp_planner import sample_grasps
from tamp.geometry.motion_planner import GeometricRefiner
from tamp.geometry.robot_model import RobotModel
from tamp.geometry.scene_interface import TampScene
from tamp.types import TaskAction


@pytest.fixture(scope="module")
def robot():
    return RobotModel()


@pytest.fixture(scope="module")
def scene():
    s = TampScene()
    s.update_scene(make_world())
    return s


def test_ik_roundtrip(robot):
    # FK then IK back to the same pose must succeed within limits.
    q = robot.home.copy()
    T = robot.fk(q)
    T2 = T.copy(); T2[:3, 3] += np.array([0.05, 0.0, -0.1])
    res = robot.solve_ik(T2)
    assert res.success
    assert np.linalg.norm(robot.fk(res.q)[:3, 3] - T2[:3, 3]) < 1e-2


def test_grasp_sampling_top_down(robot):
    w = make_world()
    grasps = sample_grasps(w.objects[0])
    assert any(g.approach == "top" for g in grasps)
    top = next(g for g in grasps if g.approach == "top")
    # pre-grasp must sit above the grasp (approach is -z world for top-down)
    assert top.pre_grasp_pose[2, 3] > top.grasp_pose[2, 3]


def test_all_blocks_graspable(robot, scene):
    ref = GeometricRefiner(robot, scene)
    for oid in ["A", "B", "C"]:
        r = ref.refine(TaskAction("PICK", oid, params={"o": oid, "s": "table", "g": "g"}))
        assert r.feasible, f"{oid} should be graspable: {r.info}"
        assert r.target_pose is not None


def test_unreachable_object_falsifies_predicate(robot):
    # An object 3 m away is unreachable -> refiner must report a failed predicate.
    s = TampScene(); s.update_scene(make_world())
    s.objects["A"].pose.position = (3.0, 0.0, 0.435)
    ref = GeometricRefiner(robot, s)
    r = ref.refine(TaskAction("PICK", "A", params={"o": "A", "s": "table", "g": "g"}))
    assert not r.feasible
    assert r.failed_predicate is not None and r.failed_predicate[0] in ("reachable", "graspable")
