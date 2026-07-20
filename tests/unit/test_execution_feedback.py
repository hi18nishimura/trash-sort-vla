"""A2 — closed execution loop: execution-failure → bridge-predicate mapping and
the solver's ``pre_falsified`` seeding (ROS-free; this is the testable core of the
plan→execute→replan loop)."""
import pytest

from benchmarks import scene_stack3
from tamp.execution import feedback as fb
from tamp.tamp_solver import TampSolver
from tamp.types import TaskAction


# --- wire format -----------------------------------------------------------
def test_status_roundtrip():
    s = fb.format_status(False, "PICK", "cup_1", fb.NOT_REACHED)
    st = fb.parse_status(s)
    assert st is not None
    assert (st.ok, st.action_type, st.object_id, st.reason) == (False, "PICK", "cup_1", fb.NOT_REACHED)


def test_status_ok_and_malformed():
    assert fb.parse_status("ok:PLACE:cup_2:done").ok is True
    assert fb.parse_status("garbage") is None
    assert fb.parse_status("maybe:PICK:x:done") is None   # bad status token


# --- failure → predicate mapping ------------------------------------------
@pytest.mark.parametrize("at,reason,expected", [
    ("PICK",    fb.NOT_REACHED, ("reachable", "o")),
    ("PICK",    fb.PLAN_FAILED, ("reachable", "o")),
    ("UNSTACK", fb.COLLISION,   ("graspable", "o")),
    ("PICK",    fb.DROPPED,     ("graspable", "o")),
])
def test_predicate_for_pick(at, reason, expected):
    a = TaskAction(action_type=at, object_id="o")
    assert fb.predicate_for_action(a, reason) == expected


def test_predicate_for_place_and_stack():
    place = TaskAction(action_type="PLACE", object_id="o", params={"s": "tray"})
    assert fb.predicate_for_action(place, fb.NOT_REACHED) == ("placeable", "o", "tray")
    stack = TaskAction(action_type="STACK", object_id="o", params={"bottom": "b"})
    assert fb.predicate_for_action(stack, fb.COLLISION) == ("stackable", "o", "b")


# --- solver honours externally-supplied (execution-time) falsifications ----
def test_solver_pre_falsified_blocks_infeasible_plan():
    world, goal = scene_stack3.make_world(), scene_stack3.GOAL
    # baseline: the stack is solvable
    assert TampSolver().solve(world, goal).success
    # if the robot reports it can never grasp A (execution feedback), the
    # A-on-B-on-C stack becomes infeasible and the seed is carried into the loop.
    res = TampSolver().solve(world, goal, pre_falsified=[("graspable", "A")])
    assert not res.success
    assert ("graspable", "A") in res.falsified
