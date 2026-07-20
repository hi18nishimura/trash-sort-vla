"""Derive symbolic relations from scene geometry.

Single source of truth for the *symbolic* facts that depend on geometry:
``on`` / ``stacked_on`` (support) and ``clear``. Keeping this logic in one place
prevents PDDL init facts from drifting away from the real scene (risk R7).

Heuristics (axis-aligned, gravity = +z):
- A rests on B if A's base z ≈ top of B and their XY footprints overlap.
- A surface supports A if A sits within the surface's XY extent at the surface
  top height.
- A is *clear* if nothing rests on top of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from tamp.types import ObjectState, Surface, WorldState

Z_TOL = 0.02   # metres — tolerance for "resting on" contact
XY_PAD = 0.0   # extra footprint padding


@dataclass
class Relations:
    support_of: Dict[str, str] = field(default_factory=dict)  # object_id -> (surface_id | object_id)
    clear: Set[str] = field(default_factory=set)
    on_top: Dict[str, List[str]] = field(default_factory=dict)  # id -> ids resting on it


def _xy_overlap(a_c, a_dim, b_c, b_dim) -> bool:
    return (abs(a_c[0] - b_c[0]) <= (a_dim[0] + b_dim[0]) / 2 + XY_PAD and
            abs(a_c[1] - b_c[1]) <= (a_dim[1] + b_dim[1]) / 2 + XY_PAD)


def _top_z(c, dim) -> float:
    return c[2] + dim[2] / 2


def _base_z(c, dim) -> float:
    return c[2] - dim[2] / 2


def infer_relations(world: WorldState) -> Relations:
    rel = Relations()
    objs: List[ObjectState] = world.objects
    surfs: List[Surface] = world.surfaces

    for a in objs:
        ac, adim = a.pose.position, a.dimensions
        best_support = None
        best_gap = Z_TOL

        # object-on-object support
        for b in objs:
            if a.object_id == b.object_id:
                continue
            bc, bdim = b.pose.position, b.dimensions
            gap = abs(_base_z(ac, adim) - _top_z(bc, bdim))
            if gap <= best_gap and _xy_overlap(ac, adim, bc, bdim) and _base_z(ac, adim) >= _base_z(bc, bdim):
                best_support, best_gap = b.object_id, gap

        # surface support (only if no object found below, or surface is closer).
        # An object rests ON a surface only if its CENTRE lies within the surface
        # footprint — using the half-sum overlap (as for objects) wrongly flags an
        # object that merely overlaps the edge of a large surface (e.g. a big tray).
        if best_support is None:
            for s in surfs:
                sc, sdim = s.pose.position, s.dimensions
                gap = abs(_base_z(ac, adim) - _top_z(sc, sdim))
                center_in = (abs(ac[0] - sc[0]) <= sdim[0] / 2 and
                             abs(ac[1] - sc[1]) <= sdim[1] / 2)
                if gap <= best_gap and center_in:
                    best_support, best_gap = s.surface_id, gap

        if best_support is not None:
            rel.support_of[a.object_id] = best_support
            rel.on_top.setdefault(best_support, []).append(a.object_id)

    # clear: an object is clear if nothing rests on it
    supported_targets = set(rel.on_top.keys())
    for a in objs:
        if a.object_id not in supported_targets:
            rel.clear.add(a.object_id)

    return rel
