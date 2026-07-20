"""Regression: the MCP tools must honour the DSL they advertise, and never raise.

`tamp/mcp_server.py` previously had ZERO tests (0% coverage). An audit found that
`plan_goal` — the tool the MCP design tells Claude to call — sent the raw goal
straight to the solver, so the two wire forms the server hands Claude in its own
`tamp://goal-dsl` resource (`place_class`, `sort_by_class`) raised a Python
traceback, and a hallucinated object id was indistinguishable from a real
infeasibility. A single one of these assertions would have caught it.
"""
import pytest

from tamp import mcp_server as M


def test_tools_and_resource_are_registered():
    # what Claude actually sees
    import asyncio
    tools = {t.name for t in asyncio.run(M.mcp.list_tools())}
    assert {"list_scenes", "get_world_state", "plan_goal",
            "ground_command", "plan_command"} <= tools


@pytest.mark.parametrize("goal,scene", [
    # exactly as documented in the tamp://goal-dsl resource the server serves
    ({"type": "place_class", "class": "cup", "surface": "tray"}, "pnp"),
    ({"type": "sort_by_class",
      "assignments": [{"class": "cup", "surface": "tray"}]}, "pnp"),
    ({"type": "gather", "surface": "tray"}, "pnp"),
    ({"type": "stack", "order": ["A", "B", "C"]}, "stack3"),
])
def test_plan_goal_accepts_the_documented_wire_forms(goal, scene):
    res = M.plan_goal(goal, scene)
    assert res["success"], f"{goal} should plan, got: {res['reason']}"
    assert res["plan"]


@pytest.mark.parametrize("goal", [
    {"type": "stack", "order": ["A", "Z"]},        # hallucinated object id
    {"type": "on_any", "object": "A"},             # missing surfaces -> solver IndexError
    {"type": "stack", "order": "AB"},              # a string, not a list
    {"type": "stack", "order": ["A", "A"]},        # duplicate
    {"type": "sort_by_class", "assignments": ["cup"]},   # not a dict
    {"type": "nonsense"},
    {},
])
def test_bad_goals_are_rejected_not_raised(goal):
    """The whole point of 'LLM proposes, planner verifies' is that a bad proposal is
    REJECTED with a reason — never a traceback."""
    res = M.plan_goal(goal, "stack3")
    assert res["success"] is False
    assert res["reason"]                  # a human-readable reason, not an exception
    assert res["plan"] == []


def test_unknown_scene_is_reported_not_raised():
    assert M.plan_goal({"type": "gather", "surface": "tray"}, "nope")["success"] is False
    assert "unknown scene" in M.get_world_state("nope")["error"]
    assert M.ground_command("stack the blocks", "nope")["success"] is False


def test_ground_command_refuses_a_prohibition():
    r = M.ground_command("do not put cup_1 on the tray", "pnp")
    assert r["success"] is False and r["goal"] is None


def test_plan_command_end_to_end():
    res = M.plan_command("stack A on B on C", "stack3")
    assert res["success"] and len(res["plan"]) == 4
