"""Offline trimesh visualization of a TAMP scene + refined plan.

Renders surfaces (grey), objects (colored), and the end-effector target frames
of each refined plan step as RGB axis triads. Used for V3 offline validation.
Saves a PNG (headless-friendly) and can also open an interactive window.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import trimesh

from tamp.types import TaskAction, WorldState

_OBJ_COLORS = [
    [200, 60, 60, 255], [60, 160, 60, 255], [60, 90, 200, 255],
    [200, 160, 40, 255], [160, 60, 200, 255], [40, 180, 180, 255],
]


def build_scene(world: WorldState, plan: Optional[List[TaskAction]] = None) -> trimesh.Scene:
    scene = trimesh.Scene()
    for s in world.surfaces:
        box = trimesh.creation.box(extents=np.asarray(s.dimensions, float))
        box.visual.face_colors = [120, 120, 120, 180]
        box.apply_transform(s.pose.homogeneous_matrix())
        scene.add_geometry(box, geom_name=f"surface_{s.surface_id}")
    for i, o in enumerate(world.objects):
        box = trimesh.creation.box(extents=np.asarray(o.dimensions, float))
        box.visual.face_colors = _OBJ_COLORS[i % len(_OBJ_COLORS)]
        box.apply_transform(o.pose.homogeneous_matrix())
        scene.add_geometry(box, geom_name=f"object_{o.object_id}")
    if plan:
        for i, step in enumerate(plan):
            if step.target_pose is not None:
                scene.add_geometry(_eef_marker(step.target_pose.homogeneous_matrix()),
                                   geom_name=f"eef_{i}_{step.action_type}")
    return scene


def _eef_marker(T: np.ndarray) -> trimesh.Trimesh:
    """A small yellow sphere marking an end-effector target pose."""
    s = trimesh.creation.uv_sphere(radius=0.015)
    s.visual.face_colors = [255, 230, 0, 255]
    s.apply_transform(T)
    return s


def save_png(scene: trimesh.Scene, path: str, resolution=(1024, 768)) -> bool:
    try:
        png = scene.save_image(resolution=resolution, visible=True)
        with open(path, "wb") as fh:
            fh.write(png)
        return True
    except Exception as exc:  # headless without a GL backend
        print(f"[viz] could not render PNG ({exc}); export GLB instead")
        scene.export(path.replace(".png", ".glb"))
        return False
