"""unified-planning model of the ``tabletop`` domain.

This is the runtime twin of ``pddl/domains/tabletop.pddl``: it builds the same
types / fluents / actions as a ``unified_planning`` model so we can (a) generate
problems programmatically, (b) swap planning engines, and (c) emit a ``.pddl``
file that must match the canonical one (a test asserts this).

Keep this file and ``pddl/domains/tabletop.pddl`` in sync — the geometric bridge
predicates ``reachable / graspable / stackable / placeable`` are the contract
with the PyTAMP layer.
"""
from __future__ import annotations

from unified_planning.model import Problem
from unified_planning.shortcuts import BoolType, Not, UserType


# Geometric bridge fluents — seeded by the problem generator, falsified by the
# refiner during the TAMP loop. Names mirror tabletop.pddl exactly.
GEOMETRIC_FLUENTS = ("reachable", "graspable", "stackable", "placeable")


def build_domain(problem: Problem) -> dict:
    """Add the ``tabletop`` types, fluents and actions to *problem*.

    Returns a dict of the created UP objects (types, fluents, actions) so the
    problem generator can reference them when asserting initial state / goals.
    """
    # --- types ---
    Object = UserType("object")
    SurfaceT = UserType("surface")
    Gripper = UserType("gripper")

    # --- fluents ---
    on = problem.add_fluent("on", BoolType(), o=Object, s=SurfaceT, default_initial_value=False)
    stacked_on = problem.add_fluent("stacked_on", BoolType(), top=Object, bottom=Object, default_initial_value=False)
    clear = problem.add_fluent("clear", BoolType(), o=Object, default_initial_value=False)
    holding = problem.add_fluent("holding", BoolType(), g=Gripper, o=Object, default_initial_value=False)
    handempty = problem.add_fluent("handempty", BoolType(), g=Gripper, default_initial_value=False)
    reachable = problem.add_fluent("reachable", BoolType(), o=Object, default_initial_value=False)
    graspable = problem.add_fluent("graspable", BoolType(), o=Object, default_initial_value=False)
    stackable = problem.add_fluent("stackable", BoolType(), top=Object, bottom=Object, default_initial_value=False)
    placeable = problem.add_fluent("placeable", BoolType(), o=Object, s=SurfaceT, default_initial_value=False)

    from unified_planning.model import InstantaneousAction

    # --- pick ---
    pick = InstantaneousAction("pick", o=Object, s=SurfaceT, g=Gripper)
    o, s, g = pick.parameter("o"), pick.parameter("s"), pick.parameter("g")
    pick.add_precondition(on(o, s))
    pick.add_precondition(clear(o))
    pick.add_precondition(handempty(g))
    pick.add_precondition(reachable(o))
    pick.add_precondition(graspable(o))
    pick.add_effect(holding(g, o), True)
    pick.add_effect(on(o, s), False)
    pick.add_effect(handempty(g), False)
    pick.add_effect(clear(o), False)
    problem.add_action(pick)

    # --- place ---
    place = InstantaneousAction("place", o=Object, s=SurfaceT, g=Gripper)
    o, s, g = place.parameter("o"), place.parameter("s"), place.parameter("g")
    place.add_precondition(holding(g, o))
    place.add_precondition(placeable(o, s))
    place.add_effect(on(o, s), True)
    place.add_effect(clear(o), True)
    place.add_effect(handempty(g), True)
    place.add_effect(holding(g, o), False)
    problem.add_action(place)

    # --- stack ---
    stack = InstantaneousAction("stack", top=Object, bottom=Object, g=Gripper)
    top, bottom, g = stack.parameter("top"), stack.parameter("bottom"), stack.parameter("g")
    stack.add_precondition(holding(g, top))
    stack.add_precondition(clear(bottom))
    stack.add_precondition(stackable(top, bottom))
    stack.add_precondition(Not(up_equals(top, bottom)))
    stack.add_effect(stacked_on(top, bottom), True)
    stack.add_effect(clear(top), True)
    stack.add_effect(handempty(g), True)
    stack.add_effect(holding(g, top), False)
    stack.add_effect(clear(bottom), False)
    problem.add_action(stack)

    # --- unstack ---
    unstack = InstantaneousAction("unstack", top=Object, bottom=Object, g=Gripper)
    top, bottom, g = unstack.parameter("top"), unstack.parameter("bottom"), unstack.parameter("g")
    unstack.add_precondition(stacked_on(top, bottom))
    unstack.add_precondition(clear(top))
    unstack.add_precondition(handempty(g))
    unstack.add_precondition(reachable(top))
    unstack.add_precondition(graspable(top))
    unstack.add_precondition(Not(up_equals(top, bottom)))
    unstack.add_effect(holding(g, top), True)
    unstack.add_effect(clear(bottom), True)
    unstack.add_effect(stacked_on(top, bottom), False)
    unstack.add_effect(handempty(g), False)
    unstack.add_effect(clear(top), False)
    problem.add_action(unstack)

    return {
        "types": {"object": Object, "surface": SurfaceT, "gripper": Gripper},
        "fluents": {
            "on": on, "stacked_on": stacked_on, "clear": clear,
            "holding": holding, "handempty": handempty,
            "reachable": reachable, "graspable": graspable,
            "stackable": stackable, "placeable": placeable,
        },
        "actions": {"pick": pick, "place": place, "stack": stack, "unstack": unstack},
    }


def up_equals(a, b):
    """``Equals`` over user-type objects (for the ``(not (= ?x ?y))`` guards)."""
    from unified_planning.shortcuts import Equals

    return Equals(a, b)
