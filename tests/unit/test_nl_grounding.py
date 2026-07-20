"""Flagship (Phase B) — natural-language goal grounding, offline path.

All deterministic and network-free (use_llm=False), so they exercise the grammar
parser + DSL validation + the same normalized goals the planner consumes. The
Claude-API path is a thin, separately-guarded layer on top of this."""
import pytest

from benchmarks import scene_pnp, scene_stack3
from tamp.nl import GoalGrounder
from tamp.nl.dsl import normalize_goal, validate_goal
from tamp.tamp_solver import TampSolver


@pytest.fixture
def pnp():
    return scene_pnp.make_world()


@pytest.fixture
def stack3():
    return scene_stack3.make_world()


# --- DSL validation --------------------------------------------------------
def test_validate_rejects_unknown_ids(pnp):
    ok, err = validate_goal({"type": "on", "object": "nope", "surface": "tray"}, pnp)
    assert not ok and "unknown object" in err
    ok, _ = validate_goal({"type": "gather", "surface": "tray"}, pnp)
    assert ok


def test_validate_place_class_needs_present_class(pnp):
    assert validate_goal({"type": "place_class", "class": "cup", "surface": "tray"}, pnp)[0]
    assert not validate_goal({"type": "place_class", "class": "banana", "surface": "tray"}, pnp)[0]


def test_normalize_sort_by_class_expands_to_object_map(pnp):
    spec = {"type": "sort_by_class", "assignments": [{"class": "cup", "surface": "tray"}]}
    out = normalize_goal(spec, pnp)
    assert out["type"] == "sort_by_class"
    assert out["assignment"] == {"cup_1": "tray", "cup_2": "tray"}


# --- grammar / grounder (offline) -----------------------------------------
@pytest.mark.parametrize("cmd,expect_type", [
    ("gather everything onto the tray", "gather"),
    ("put a cup on the tray", "place_class"),
    ("put cup_1 on the tray", "on"),
    ("clear the table", "gather"),
])
def test_grounds_pnp_commands(pnp, cmd, expect_type):
    r = GoalGrounder(use_llm=False).ground(cmd, pnp)
    assert r.success and r.source == "rule"
    assert r.goal["type"] == expect_type


def test_stack_order_preserved(stack3):
    r = GoalGrounder(use_llm=False).ground("stack A on B on C", stack3)
    assert r.success and r.goal["order"] == ["A", "B", "C"]   # not substring-matched


def test_unparseable_command_fails_gracefully(pnp):
    r = GoalGrounder(use_llm=False).ground("tell me a joke", pnp)
    assert not r.success and r.goal is None


def test_empty_command(pnp):
    assert not GoalGrounder(use_llm=False).ground("   ", pnp).success


# --- end-to-end: grounded goal actually solves -----------------------------
def test_grounded_goal_solves_on_real_solver(stack3):
    r = GoalGrounder(use_llm=False).ground("stack A on B on C", stack3)
    res = TampSolver().solve(stack3, r.goal)
    assert res.success
    assert [a.action_type for a in res.plan] == ["PICK", "STACK", "PICK", "STACK"]
