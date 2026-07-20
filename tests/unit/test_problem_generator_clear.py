"""Regression: `clear` must mean "nothing rests on me", regardless of default_surface.

The audit found that `default_surface` (set to "workspace" in the live ROS launches)
widened the `clear` assertion to every object whose *correctly inferred* support was
the default surface — so an object with another object stacked on top of it was
asserted clear(), and the planner would happily emit PICK on a buried object.

No existing fixture had a pre-existing stack, which is exactly why this hid. This
test builds one.
"""
import pytest

from tamp.pddl.problem_generator import generate_problem
from tamp.types import ObjectState, Pose, Surface, WorldState


def _stacked_world():
    """A on B, B on the table. So: clear(A) is true, clear(B) is FALSE."""
    table = Surface("table", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.8, 0.8, 0.02))
    z_b = 0.40 + 0.01 + 0.025          # table top + half block
    z_a = z_b + 0.05                   # A sits directly on B
    b = ObjectState("B", "block", Pose(position=(0.5, 0.0, z_b)), dimensions=(0.05, 0.05, 0.05))
    a = ObjectState("A", "block", Pose(position=(0.5, 0.0, z_a)), dimensions=(0.05, 0.05, 0.05))
    return WorldState(objects=[a, b], surfaces=[table])


def _clear_facts(problem):
    return {str(fnode.args[0]) for fnode, val in problem.initial_values.items()
            if fnode.fluent().name == "clear" and val.is_true()}


@pytest.mark.parametrize("default_surface", [None, "table"])
def test_buried_object_is_never_clear(default_surface):
    """B has A on top of it -> B must NOT be clear, with or without default_surface."""
    world = _stacked_world()
    problem = generate_problem(world, {"type": "clear", "object": "A"},
                               default_surface=default_surface)
    clear = _clear_facts(problem)
    assert "A" in clear, "A has nothing on it -> must be clear"
    assert "B" not in clear, (
        f"B has A stacked on it -> must NOT be clear (default_surface={default_surface})")


def test_default_surface_still_supports_objects_with_no_inferred_support():
    """The fallback itself must survive: a floating object (no inferable support, as in
    the YOLO path) still gets on(o, default_surface) and, having nothing on it, clear."""
    table = Surface("table", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.8, 0.8, 0.02))
    # z far above the table -> object_reasoner infers no support
    floating = ObjectState("cup_1", "cup", Pose(position=(0.5, 0.0, 0.9)),
                           dimensions=(0.05, 0.05, 0.07))
    world = WorldState(objects=[floating], surfaces=[table])
    problem = generate_problem(world, {"type": "on", "object": "cup_1", "surface": "table"},
                               default_surface="table")
    on = {(str(f.args[0]), str(f.args[1])) for f, v in problem.initial_values.items()
          if f.fluent().name == "on" and v.is_true()}
    assert ("cup_1", "table") in on          # fallback still applies
    assert "cup_1" in _clear_facts(problem)  # nothing on it -> clear
