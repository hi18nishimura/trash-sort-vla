"""Pure-Python data types for the TAMP core.

These mirror the ROS 2 messages defined in
``ros2_ws/src/pytamp_moveit_bridge/msg`` (``ObjectState``, ``WorldState``,
``TaskAction``) so the symbolic + geometric core can run and be unit-tested with
no ROS dependency. The ROS bridge (M4) converts between these and the real
messages 1:1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
QuatWXYZ = Tuple[float, float, float, float]  # (w, x, y, z) — matches reference order


@dataclass
class Pose:
    """A 6-DoF pose. Quaternion stored as (w, x, y, z) to match the reference
    ``scene_interface.py`` convention."""

    position: Vec3 = (0.0, 0.0, 0.0)
    orientation: QuatWXYZ = (1.0, 0.0, 0.0, 0.0)

    def homogeneous_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform for this pose."""
        w, x, y, z = self.orientation
        n = (w * w + x * x + y * y + z * z) ** 0.5
        if n == 0:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        else:
            w, x, y, z = w / n, x / n, y / n, z / n
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])
        h = np.eye(4)
        h[:3, :3] = rot
        h[:3, 3] = self.position
        return h


@dataclass
class ObjectState:
    """A detected/known object — mirrors ``ObjectState.msg``."""

    object_id: str
    class_name: str = ""
    pose: Pose = field(default_factory=Pose)
    dimensions: Vec3 = (0.05, 0.05, 0.05)  # bounding-box extents (x, y, z)
    confidence: float = 1.0


@dataclass
class Surface:
    """A support surface (table, tray, bin, region). Not in the reference msgs;
    surfaces are configured per-scene and seeded into the PDDL problem."""

    surface_id: str
    pose: Pose = field(default_factory=Pose)
    dimensions: Vec3 = (1.0, 1.0, 0.02)
    class_name: str = "table"


@dataclass
class WorldState:
    """Mirrors ``WorldState.msg`` (``ObjectState[] objects``), extended with the
    scene's surfaces (which the reference assumes implicitly)."""

    objects: List[ObjectState] = field(default_factory=list)
    surfaces: List[Surface] = field(default_factory=list)

    def object_ids(self) -> List[str]:
        return [o.object_id for o in self.objects]

    def get(self, object_id: str) -> ObjectState:
        for o in self.objects:
            if o.object_id == object_id:
                return o
        raise KeyError(object_id)


@dataclass
class TaskAction:
    """A single refined plan step — mirrors ``TaskAction.msg``. ``target_pose``
    is filled in by the geometric refiner (None until refined)."""

    action_type: str            # PICK / PLACE / STACK / UNSTACK
    object_id: str
    target_pose: Pose | None = None
    preconditions: Sequence[str] = field(default_factory=list)
    effects: Sequence[str] = field(default_factory=list)
    # extra context not in the ROS msg but useful for execution / logging:
    params: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # compact, plan-friendly
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.action_type}({self.object_id}{', ' + p if p else ''})"
