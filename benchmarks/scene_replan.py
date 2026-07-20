"""Benchmark scene #2 (replanning recovery): a block must go onto one of two
candidate trays (disjunctive goal). ``tray_near`` is reachable; ``tray_far`` is
2.5 m away — far outside the Panda's workspace — so any plan that chooses it is
geometrically infeasible and the TAMP loop must recover onto ``tray_near``.

HONESTY NOTE (this scene was previously rigged): the unreachable tray used to be
named ``tray_z`` specifically so Fast Downward's arbitrary tie-break would explore
it first and the demo would always show ``replans == 1``. That "recovery" rode on
an accident of alphabetical ordering — renaming the tray made the showcase
silently demonstrate nothing. The scene now makes no assumption about which
disjunct the satisficing planner tries first. What is guaranteed — and what the
tests assert — is the property that actually matters:

* the final plan always places the block on the REACHABLE tray, and
* if the planner happens to pick ``tray_far`` first, the loop falsifies
  ``placeable(block, tray_far)`` and replans (organic in-loop recovery), and
* when NO feasible tray remains the solver reports failure instead of faking
  success (see ``tamp-offline --scene replan``, which demonstrates both paths).
"""
from tamp.types import ObjectState, Pose, Surface, WorldState


def make_world() -> WorldState:
    start = Surface("start", Pose(position=(0.45, 0.0, 0.40)), dimensions=(0.4, 0.4, 0.02))
    tray_near = Surface("tray_near", Pose(position=(0.55, 0.20, 0.40)),
                        dimensions=(0.2, 0.2, 0.02), class_name="tray")   # reachable
    tray_far = Surface("tray_far", Pose(position=(2.5, 0.0, 0.40)),
                       dimensions=(0.2, 0.2, 0.02), class_name="tray")    # out of reach
    z = 0.40 + 0.01 + 0.025
    block = ObjectState("block", "cube", Pose(position=(0.45, 0.0, z)),
                        dimensions=(0.05, 0.05, 0.05))
    return WorldState(objects=[block], surfaces=[start, tray_near, tray_far])


GOAL = {"type": "on_any", "object": "block", "surfaces": ["tray_near", "tray_far"]}
