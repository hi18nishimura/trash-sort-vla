"""§4-C3 regression tests: the refiner must certify MOTIONS, not just endpoints.

Before this fix, `motion_planner` checked only the two endpoint poses with a
fingertip-sized box: a path whose endpoints were free could still sweep the hand
(or the forearm) straight through an obstacle between them.
"""
import numpy as np
import trimesh

from tamp.geometry.collision import (ARM_CHECK_LINKS, arm_links_free,
                                     gripper_collision_free, swept_gripper_free)
from tamp.geometry.robot_model import RobotModel

_ROBOT = RobotModel()


def _manager_with_box(center, extents):
    m = trimesh.collision.CollisionManager()
    box = trimesh.creation.box(extents=extents)
    T = np.eye(4)
    T[:3, 3] = center
    box.apply_transform(T)
    m.add_object("obstacle", box)
    return m


def test_swept_path_rejects_midpoint_obstacle():
    """Two configurations whose EEF poses are free, with an obstacle parked at
    the midpoint of the joint path between them."""
    q_a = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    q_b = q_a.copy()
    q_b[0] += 0.9                       # swing the base joint ~52 degrees
    q_mid = (q_a + q_b) / 2
    mid_eef = _ROBOT.fk(q_mid)[:3, 3]
    m = _manager_with_box(mid_eef, (0.05, 0.05, 0.05))
    # endpoints are clear of the midpoint obstacle...
    assert gripper_collision_free(m, _ROBOT.fk(q_a))
    assert gripper_collision_free(m, _ROBOT.fk(q_b))
    # ...but the swept check must veto the motion.
    assert not swept_gripper_free(m, _ROBOT, q_a, q_b)


def test_swept_path_accepts_clear_scene():
    q_a = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    q_b = q_a + 0.1
    assert swept_gripper_free(trimesh.collision.CollisionManager(), _ROBOT, q_a, q_b)


def test_arm_links_flag_engulfing_obstacle():
    """A box swallowing the elbow region must fail the per-link check even though
    the fingertip-box endpoint check knows nothing about the arm."""
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    lp = _ROBOT.link_positions(q)
    assert set(ARM_CHECK_LINKS) <= set(lp), "pykin link names drifted"
    elbow = lp["panda_link_4"]
    m = _manager_with_box(elbow, (0.3, 0.3, 0.3))
    assert not arm_links_free(m, lp)
    assert arm_links_free(trimesh.collision.CollisionManager(), lp)
