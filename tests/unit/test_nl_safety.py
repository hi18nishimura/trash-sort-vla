"""SAFETY regressions for the always-available offline grammar parser.

An audit proved the earlier parser turned prohibitions, questions and removals into
robot motion — "do not put cup_1 on the tray" put cup_1 on the tray. Because this
layer is the final fallback (it runs when there is no API key), it is the layer that
MUST fail closed. A refusal is always better than a plausible-but-wrong goal: the
TAMP layer will happily execute a wrong-but-feasible plan.
"""
import pytest

from benchmarks import scene_pnp, scene_stack3
from tamp.nl import GoalGrounder, grammar
from tamp.types import ObjectState, Pose, Surface, WorldState


@pytest.fixture
def pnp():
    return scene_pnp.make_world()


@pytest.fixture
def stack3():
    return scene_stack3.make_world()


@pytest.fixture
def mixed():
    """A cup, a NON-cup, and three surfaces — the fixture whose absence hid several
    bugs (scene_pnp has only cups and exactly two surfaces)."""
    table = Surface("table", Pose(position=(0.5, 0.0, 0.40)), dimensions=(0.8, 0.8, 0.02),
                    class_name="table")
    tray = Surface("tray", Pose(position=(0.55, 0.2, 0.415)), dimensions=(0.2, 0.2, 0.01),
                   class_name="tray")
    bin_ = Surface("bin", Pose(position=(0.55, -0.2, 0.415)), dimensions=(0.2, 0.2, 0.01),
                   class_name="bin")
    cup = ObjectState("cup_1", "cup", Pose(position=(0.45, 0.0, 0.455)), dimensions=(0.05, 0.05, 0.07))
    bolt = ObjectState("bolt_1", "bolt", Pose(position=(0.50, 0.1, 0.44)), dimensions=(0.03, 0.03, 0.04))
    return WorldState(objects=[cup, bolt], surfaces=[table, tray, bin_])


# --- 1. Never invert an instruction -----------------------------------------
@pytest.mark.parametrize("cmd", [
    "take cup_1 off the tray",
    "remove cup_1 from the tray",
    "move cup_1 away from the tray",
])
def test_removal_is_refused_not_inverted(pnp, cmd):
    assert grammar.parse(cmd, pnp) is None, f"{cmd!r} must refuse, not become a placement"


# --- 2. Never act on a prohibition or a question -----------------------------
@pytest.mark.parametrize("cmd", [
    "do not put cup_1 on the tray",
    "don't put cup_1 on the tray",
    "never put cup_1 on the tray",
    "is cup_1 on the tray?",
    "where is cup_1",
])
def test_prohibitions_and_questions_are_refused(pnp, cmd):
    assert grammar.parse(cmd, pnp) is None, f"{cmd!r} must never become a motion"


def test_negated_stack_is_refused(stack3):
    assert grammar.parse("do not stack A on B", stack3) is None


# --- 3. Never guess when ambiguous ------------------------------------------
def test_destination_comes_from_sentence_position_not_world_order(pnp):
    """The object must go to the surface the SENTENCE names last, not whichever
    surface happens to come first in world.surfaces."""
    g = grammar.parse("take the cup off the table and put it on the tray", pnp)
    assert g is not None and g["surface"] == "tray"


def test_class_scoped_all_does_not_sweep_up_other_objects(mixed):
    """'all the cups' must not become an all-objects `gather` (which would also move
    the bolt). It must stay scoped to the named class."""
    g = grammar.parse("put all the cups on the tray", mixed)
    assert g is not None
    assert g["type"] != "gather", "class-scoped command must not sweep every object"
    assert g["type"] == "sort_by_class"
    assert g["assignments"] == [{"class": "cup", "surface": "tray"}]


def test_unscoped_everything_still_gathers(pnp):
    g = grammar.parse("put everything on the tray", pnp)
    assert g == {"type": "gather", "surface": "tray"}


def test_clear_refuses_when_destination_is_ambiguous(mixed):
    """3 surfaces -> 'clear the table' has two possible destinations -> refuse."""
    assert grammar.parse("clear the table", mixed) is None


def test_clear_resolves_when_unambiguous(pnp):
    """2 surfaces -> exactly one possible destination -> safe to resolve."""
    assert grammar.parse("clear the table", pnp) == {"type": "gather", "surface": "tray"}


# --- the refusal must propagate through the grounder, not fall through -------
def test_grounder_surfaces_a_refusal_rather_than_a_wrong_goal(pnp):
    r = GoalGrounder(use_llm=False).ground("do not put cup_1 on the tray", pnp)
    assert not r.success and r.goal is None


def test_llm_refusal_short_circuits_and_is_not_overridden_by_grammar(pnp):
    """A backend that explicitly refuses (type:'none') must END the chain. Previously
    it fell through to the grammar parser, which mis-parsed it into an action."""
    g = GoalGrounder(use_llm=False)
    # inject a fake backend that refuses, exactly as an LLM would
    g._backends = [("fake_llm", lambda cmd, world: {"type": "none", "reason": "prohibition"})]
    r = g.ground("do not put cup_1 on the tray", pnp)
    assert not r.success
    assert r.source == "fake_llm"        # NOT "rule" — the grammar must not have run
    assert "prohibition" in r.error


def test_malformed_llm_output_does_not_crash_the_chain(pnp):
    """Malformed LLM output is untrusted input: it must degrade to the next backend,
    never escape as an exception (which previously killed the whole fallback)."""
    g = GoalGrounder(use_llm=False)
    def bad(cmd, world):
        return {"type": "sort_by_class", "assignments": ["cup"]}   # str, not dict
    g._backends = [("bad_llm", bad)]
    r = g.ground("put everything on the tray", pnp)   # grammar can handle this one
    assert r.success and r.source == "rule"           # fell through cleanly
