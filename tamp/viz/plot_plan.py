"""Matplotlib renderer for a scene + refined plan.

Headless-safe alternative to the trimesh viewer (which needs a GL context):
draws the surfaces and objects of a :class:`~tamp.types.WorldState` in 3D and
overlays the plan's end-effector waypoints as a numbered path. Used by
``tamp-offline --viz out.png`` and for generating the repository figures.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from tamp.types import TaskAction, WorldState

_SURFACE_COLOR = "#8d6e63"
_TRAY_COLOR = "#5c6bc0"
_OBJECT_COLORS = ["#ef5350", "#66bb6a", "#ffca28", "#26c6da", "#ab47bc"]
_PATH_COLOR = "#212121"


def _box_faces(center, dims):
    cx, cy, cz = center
    dx, dy, dz = (d / 2 for d in dims)
    v = np.array([[cx - dx, cy - dy, cz - dz], [cx + dx, cy - dy, cz - dz],
                  [cx + dx, cy + dy, cz - dz], [cx - dx, cy + dy, cz - dz],
                  [cx - dx, cy - dy, cz + dz], [cx + dx, cy - dy, cz + dz],
                  [cx + dx, cy + dy, cz + dz], [cx - dx, cy + dy, cz + dz]])
    return [[v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
            [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
            [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]]


def render_plan_png(world: WorldState, plan: Optional[List[TaskAction]],
                    path: str, title: str = "") -> str:
    """Render the scene (and the plan's EEF waypoints, if given) to a PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(projection="3d", computed_zorder=False)

    for s in world.surfaces:
        color = _TRAY_COLOR if s.class_name == "tray" else _SURFACE_COLOR
        faces = _box_faces(s.pose.position, s.dimensions)
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=color, edgecolors="#00000033", alpha=0.55))
        ax.text(*s.pose.position[:2], s.pose.position[2] + s.dimensions[2],
                s.surface_id, fontsize=8, ha="center", color="#333333")

    for i, o in enumerate(world.objects):
        faces = _box_faces(o.pose.position, o.dimensions)
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=_OBJECT_COLORS[i % len(_OBJECT_COLORS)],
            edgecolors="#00000055", alpha=0.95))
        ax.text(o.pose.position[0], o.pose.position[1],
                o.pose.position[2] + o.dimensions[2], o.object_id,
                fontsize=8, ha="center", weight="bold")

    if plan:
        pts = np.array([a.target_pose.position for a in plan if a.target_pose])
        if len(pts):
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o",
                    color=_PATH_COLOR, linewidth=1.5, markersize=5, zorder=10)
            for i, (p, a) in enumerate(
                    zip(pts, [a for a in plan if a.target_pose]), 1):
                ax.text(p[0], p[1], p[2] + 0.02, f"{i}. {a.action_type}",
                        fontsize=7, color=_PATH_COLOR, zorder=11)

    pts_all = [s.pose.position for s in world.surfaces] + \
              [o.pose.position for o in world.objects]
    xs, ys, zs = zip(*pts_all)
    ax.set_xlim(min(xs) - 0.25, max(xs) + 0.25)
    ax.set_ylim(min(ys) - 0.25, max(ys) + 0.25)
    ax.set_zlim(min(zs) - 0.05, max(zs) + 0.35)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=28, azim=-50)
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
