"""Collision queries for the geometric refiner.

Thin helpers over trimesh ``CollisionManager``. Used to check whether a grasp
approach or a placement footprint is collision-free against the rest of the
scene. The robot arm is approximated by a small box at the end-effector (a
gripper proxy) — sufficient for the kinematic feasibility checks in the TAMP
loop; MoveIt's full-mesh planning scene is authoritative at execution time.
"""
from __future__ import annotations

import numpy as np
import trimesh

GRIPPER_PROXY = (0.08, 0.12, 0.06)  # rough Panda hand bounding box (m)


def gripper_proxy_mesh(eef_pose: np.ndarray) -> trimesh.Trimesh:
    """A box approximating the gripper, placed behind the fingertip along -z."""
    m = trimesh.creation.box(extents=np.asarray(GRIPPER_PROXY, float))
    offset = np.eye(4)
    offset[:3, 3] = (0, 0, -GRIPPER_PROXY[2] / 2)  # body sits behind fingertip
    m.apply_transform(eef_pose @ offset)
    return m


def in_collision(manager: trimesh.collision.CollisionManager,
                 mesh: trimesh.Trimesh, name: str = "probe") -> bool:
    """True if *mesh* collides with anything in *manager*."""
    probe = trimesh.collision.CollisionManager()
    probe.add_object(name, mesh)
    return bool(manager.in_collision_other(probe))


def gripper_collision_free(manager: trimesh.collision.CollisionManager,
                           eef_pose: np.ndarray) -> bool:
    return not in_collision(manager, gripper_proxy_mesh(eef_pose), "gripper")


def swept_gripper_free(manager: trimesh.collision.CollisionManager,
                       robot, q_from: np.ndarray, q_to: np.ndarray,
                       steps: int = 6) -> bool:
    """Check the gripper proxy along the JOINT-SPACE path from ``q_from`` to
    ``q_to`` (endpoints included). The refiner previously certified only the two
    endpoint poses — audit §4-C3: a motion whose endpoints are free could still
    sweep the hand straight through an obstacle between them (side grasps and
    place approaches especially). Joint interpolation mirrors what the
    trajectory controller actually executes."""
    q_from = np.asarray(q_from, float)
    q_to = np.asarray(q_to, float)
    for t in np.linspace(0.0, 1.0, max(2, steps)):
        q = (1.0 - t) * q_from + t * q_to
        if not gripper_collision_free(manager, robot.fk(q)):
            return False
    return True


#: Links checked by the coarse arm test and the sphere radius used for each.
#: HONESTY: this is a coarse capsule-less approximation — a sphere at each link
#: ORIGIN, not the true link volume. It catches gross pass-through (forearm
#: sweeping a tall obstacle, wrist dipping into the table) that the endpoint
#: gripper box could never see; MoveIt's full-mesh planning scene remains
#: authoritative at execution time.
ARM_CHECK_LINKS = {          # pykin frame names (underscored in its URDF)
    "panda_link_3": 0.07,
    "panda_link_4": 0.07,
    "panda_link_5": 0.07,
    "panda_link_6": 0.06,
    "panda_link_7": 0.06,
}

_UNIT_SPHERE = trimesh.creation.icosphere(subdivisions=1, radius=1.0)


def arm_links_free(manager: trimesh.collision.CollisionManager,
                   link_positions: dict) -> bool:
    """True if every checked arm-link sphere is clear of the scene.
    ``link_positions``: link name -> xyz origin (from RobotModel.link_positions)."""
    for link, radius in ARM_CHECK_LINKS.items():
        p = link_positions.get(link)
        if p is None:
            continue
        sphere = _UNIT_SPHERE.copy()
        sphere.apply_scale(radius)
        sphere.apply_translation(p)
        if in_collision(manager, sphere, f"arm_{link}"):
            return False
    return True
