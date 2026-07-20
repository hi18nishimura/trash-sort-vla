"""MCP server exposing the PDDL+TAMP robot brain to a Claude client.

This is the keyless flagship path: instead of the robot calling the Claude API
(needs an API key + billing), an MCP **client you already have** — Claude Code or
Claude Desktop, running on your Claude subscription — drives the robot through the
tools below. Claude reads the world, maps your English command to a goal using the
DSL resource, and calls ``plan_goal`` — so *Claude is the grounder*, the PDDL+TAMP
layer verifies feasibility, and no Anthropic API key is involved.

Run (stdio transport):  python -m tamp.mcp_server
Wire into Claude Code:  see .mcp.json (added to the repo) or `claude mcp add`.
"""
from __future__ import annotations

import importlib
from typing import Dict, List

from mcp.server.fastmcp import FastMCP

from tamp.nl import GoalGrounder
from tamp.nl.dsl import normalize_goal, validate_goal
from tamp.nl.llm_grounder import _SYSTEM
from tamp.tamp_solver import TampSolver

SCENES = {
    "stack3": "benchmarks.scene_stack3",   # 3 blocks A,B,C on a table
    "replan": "benchmarks.scene_replan",   # engineered closed-loop replan recovery
    "pnp": "benchmarks.scene_pnp",         # two cups + a tray (pick-and-place)
}

mcp = FastMCP("tamp-robot")
_solver = TampSolver()
_grounder = GoalGrounder(use_llm=False)   # deterministic helper; Claude is the LLM here


def _load(scene: str):
    if scene not in SCENES:
        raise ValueError(f"unknown scene '{scene}'; choose from {list(SCENES)}")
    return importlib.import_module(SCENES[scene]).make_world()


def _xyz(p):
    return [round(float(v), 3) for v in p]


@mcp.resource("tamp://goal-dsl")
def goal_dsl() -> str:
    """The goal-spec DSL: how to phrase a goal for `plan_goal`."""
    return _SYSTEM


@mcp.tool()
def list_scenes() -> List[str]:
    """List the available tabletop scenes."""
    return list(SCENES)


@mcp.tool()
def get_world_state(scene: str = "stack3") -> Dict:
    """Perceive a scene: the movable objects (id, class, position) and the surfaces
    available to place them on. Read this before constructing a goal."""
    try:
        w = _load(scene)
    except ValueError as exc:
        return {"error": str(exc), "objects": [], "surfaces": []}
    return {
        "scene": scene,
        "objects": [{"id": o.object_id, "class": o.class_name,
                     "position": _xyz(o.pose.position)} for o in w.objects],
        "surfaces": [{"id": s.surface_id, "class": s.class_name,
                      "position": _xyz(s.pose.position)} for s in w.surfaces],
    }


def _fail(reason: str) -> Dict:
    return {"success": False, "replans": 0, "reason": reason,
            "falsified_bridge_predicates": [], "plan": []}


@mcp.tool()
def plan_goal(goal: Dict, scene: str = "stack3") -> Dict:
    """Run the PDDL+TAMP solver on a goal spec (see the tamp://goal-dsl resource)
    and return the geometrically-feasible plan, or why it failed. This is where the
    symbolic↔geometric verification happens — an infeasible goal, or one naming an
    object that doesn't exist, comes back as success=false with a reason.

    The goal is VALIDATED against the perceived world and NORMALIZED to the solver's
    internal form before solving. Previously it went straight to the solver, so the
    two wire forms this server advertises in its own DSL resource (`place_class`,
    `sort_by_class`) raised a Python traceback, and a hallucinated object id was
    indistinguishable from a real infeasibility."""
    try:
        w = _load(scene)
    except ValueError as exc:
        return _fail(str(exc))
    try:
        ok, err = validate_goal(goal, w)
        if not ok:
            return _fail(f"invalid goal: {err}")
        res = _solver.solve(w, normalize_goal(goal, w))
    except Exception as exc:  # noqa: BLE001 — never hand Claude a raw traceback
        return _fail(f"planning error: {type(exc).__name__}: {exc}")
    return {
        "success": res.success,
        "replans": res.replans,
        "reason": res.reason,
        "falsified_bridge_predicates": [list(p) for p in res.falsified],
        "plan": [{"step": i, "action": a.action_type, "object": a.object_id,
                  "eef_target": _xyz(a.target_pose.position) if a.target_pose else None}
                 for i, a in enumerate(res.plan, 1)],
    }


@mcp.tool()
def ground_command(command: str, scene: str = "stack3") -> Dict:
    """Deterministic offline grounding of a common English command to a goal spec
    (a convenience for simple commands like "stack A on B on C" or "put the cups on
    the tray"). Refuses prohibitions, questions and removals rather than guessing.
    For anything nuanced, build the goal yourself from get_world_state + the
    tamp://goal-dsl resource and call plan_goal."""
    try:
        w = _load(scene)
    except ValueError as exc:
        return {"success": False, "goal": None, "source": "", "error": str(exc)}
    r = _grounder.ground(command, w)
    return {"success": r.success, "goal": r.goal, "source": r.source, "error": r.error}


@mcp.tool()
def plan_command(command: str, scene: str = "stack3") -> Dict:
    """One-shot: ground a simple English command and immediately plan it."""
    g = ground_command(command, scene)
    if not g["success"]:
        return {"success": False, "reason": g["error"], "plan": []}
    return plan_goal(g["goal"], scene)


def main():
    mcp.run()   # stdio transport


if __name__ == "__main__":
    main()
