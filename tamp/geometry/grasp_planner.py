"""Grasp sampling from an object's oriented bounding box.

Generates candidate end-effector poses (4x4, world frame) for a top-down and
side approaches, plus the pre-grasp pose offset along the approach axis (reusing
the reference's ``pre_grasp_offset = 0.1`` convention). Grasps are scored later
by the motion planner (IK existence + collision-free approach).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from tamp.types import ObjectState

PRE_GRASP_OFFSET = 0.10  # metres, matches reference action_executor.py


@dataclass
class Grasp:
    grasp_pose: np.ndarray      # 4x4 world transform for the EEF at grasp
    pre_grasp_pose: np.ndarray  # 4x4, offset back along approach (+approach = -z of EEF)
    approach: str               # "top" | "side_x" | "side_y"
    width: float                # required gripper opening (object extent across fingers)


def _eef_frame(position, z_axis, x_hint=(1.0, 0.0, 0.0)) -> np.ndarray:
    """Build an EEF transform whose approach (+z) points along ``z_axis``."""
    z = np.asarray(z_axis, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x = x - z * (x @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z * (np.array([0.0, 1.0, 0.0]) @ z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, position
    return T


def sample_grasps(obj: ObjectState, max_grasp_width: float = 0.12) -> List[Grasp]:
    """Return candidate grasps for *obj*, ranked top-down first."""
    c = np.asarray(obj.pose.position, float)
    dx, dy, dz = obj.dimensions
    grasps: List[Grasp] = []

    # Top-down: approach from +z, gripper closes across the smaller XY extent.
    if min(dx, dy) <= max_grasp_width:
        width = min(dx, dy)
        x_hint = (1, 0, 0) if dx <= dy else (0, 1, 0)
        T = _eef_frame(c, z_axis=(0, 0, -1), x_hint=x_hint)
        grasps.append(_with_pregrasp(T, "top", width))

    # Side approaches (close across the orthogonal horizontal extent).
    if dz <= max_grasp_width:
        for axis, name in (((1, 0, 0), "side_x"), ((0, 1, 0), "side_y")):
            T = _eef_frame(c, z_axis=tuple(-np.asarray(axis)), x_hint=(0, 0, 1))
            grasps.append(_with_pregrasp(T, name, dz))

    return grasps


def _with_pregrasp(T: np.ndarray, approach: str, width: float) -> Grasp:
    pre = T.copy()
    pre[:3, 3] = T[:3, 3] - T[:3, 2] * PRE_GRASP_OFFSET  # back off along approach
    return Grasp(grasp_pose=T, pre_grasp_pose=pre, approach=approach, width=float(width))
