"""Benchmark scene #1: three blocks on a table, goal = stack A on B on C.

Used by the offline validation (V1) and as a reusable fixture.
"""
from tamp.types import ObjectState, Pose, Surface, WorldState


def make_world() -> WorldState:
    table = Surface("table", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.8, 0.8, 0.02))
    z = 0.40 + 0.01 + 0.025  # table top + half block height
    a = ObjectState("A", "block", Pose(position=(0.40, -0.15, z)), dimensions=(0.05, 0.05, 0.05))
    b = ObjectState("B", "block", Pose(position=(0.50, 0.00, z)), dimensions=(0.05, 0.05, 0.05))
    c = ObjectState("C", "block", Pose(position=(0.60, 0.15, z)), dimensions=(0.05, 0.05, 0.05))
    return WorldState(objects=[a, b, c], surfaces=[table])


GOAL = {"type": "stack", "order": ["A", "B", "C"], "base_surface": "table"}
