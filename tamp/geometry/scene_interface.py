"""Geometric scene model — the real implementation of what the reference repo's
``pytamp_moveit_bridge/src/scene_interface.py`` left stubbed.

Builds a trimesh-based world from a :class:`tamp.types.WorldState`: each object
and surface becomes an axis-aligned box at its pose, registered in a
``trimesh.collision.CollisionManager``. Keeps a **persistent** registry (objects
that leave the camera FOV are retained — the symbolic planner needs full scene
memory, exactly as the reference comments note).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import trimesh

from tamp.types import ObjectState, Surface, WorldState


class TampScene:
    def __init__(self):
        self.objects: Dict[str, ObjectState] = {}
        self.surfaces: Dict[str, Surface] = {}
        self._held: Optional[str] = None  # object currently grasped (excluded from collisions)

    # --- scene maintenance ---
    def update_scene(self, world: WorldState) -> None:
        """Merge a new observation into the persistent registry."""
        for o in world.objects:
            self.objects[o.object_id] = o
        for s in world.surfaces:
            self.surfaces[s.surface_id] = s

    def set_object_pose(self, object_id: str, pose) -> None:
        self.objects[object_id].pose = pose

    def set_held(self, object_id: Optional[str]) -> None:
        self._held = object_id

    # --- geometry export ---
    def _box(self, dims, h_mat) -> trimesh.Trimesh:
        m = trimesh.creation.box(extents=np.asarray(dims, float))
        m.apply_transform(h_mat)
        return m

    def collision_manager(self, exclude: Optional[str] = None) -> trimesh.collision.CollisionManager:
        """A CollisionManager containing all surfaces + objects, optionally
        excluding one object (e.g. the one being grasped/placed)."""
        cm = trimesh.collision.CollisionManager()
        skip = exclude or self._held
        for sid, s in self.surfaces.items():
            cm.add_object(f"surface::{sid}", self._box(s.dimensions, s.pose.homogeneous_matrix()))
        for oid, o in self.objects.items():
            if oid == skip:
                continue
            cm.add_object(f"object::{oid}", self._box(o.dimensions, o.pose.homogeneous_matrix()))
        return cm

    def object_box(self, object_id: str) -> trimesh.Trimesh:
        o = self.objects[object_id]
        return self._box(o.dimensions, o.pose.homogeneous_matrix())

    def as_world_state(self) -> WorldState:
        return WorldState(objects=list(self.objects.values()),
                          surfaces=list(self.surfaces.values()))
