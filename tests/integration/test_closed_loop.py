"""A4 — integration tests for the perception→planning pipeline and the closed
execution loop (A2), exercised at the Python level (ROS-free, deterministic).

These stand in for a live Gazebo run (which needs a display): they drive the real
``TampSolver`` + ``tamp.execution.feedback`` exactly as the ROS planner node does,
including the replan-on-execution-failure handshake.
"""
import numpy as np

from benchmarks import scene_pnp, scene_stack3
from tamp.execution.feedback import NOT_REACHED, predicate_for_action
from tamp.tamp_solver import TampSolver

WORKSPACE_XY = 1.0  # generous bound: refined EEF targets must be near the table


# --- perception -> plan ----------------------------------------------------
def test_perception_world_to_refined_plan():
    """A mock /world_state (as YOLO would publish) yields a valid, fully-refined
    plan: right action types, every step has a reachable target pose."""
    world = scene_pnp.make_world()
    goal = {"type": "gather", "surface": "tray"}
    res = TampSolver().solve(world, goal)
    assert res.success and res.plan
    # gather two cups -> 4 steps (PICK/PLACE each)
    assert [a.action_type for a in res.plan] == ["PICK", "PLACE", "PICK", "PLACE"]
    for a in res.plan:
        assert a.target_pose is not None
        xyz = np.asarray(a.target_pose.position)
        assert np.all(np.abs(xyz) < WORKSPACE_XY)          # sane, near-table targets


def _drive_closed_loop(solver, world, goal, fail_on, max_exec_replans=5):
    """Mimic the ROS planner's dispatch loop: solve, "execute" each step, and on a
    simulated execution failure falsify the offending predicate and re-solve —
    the same logic as tamp_planner_node._on_status. Returns (status, replans,
    falsified)."""
    falsified, replans = [], 0
    while True:
        res = solver.solve(world, goal, pre_falsified=falsified)
        if not res.success:
            return "failed", replans, falsified
        for step in res.plan:                              # execute steps in order
            if (step.action_type, step.object_id) in fail_on:
                pred = predicate_for_action(step, NOT_REACHED)
                if pred and pred not in falsified:
                    falsified.append(pred)
                replans += 1
                break                                      # -> replan
        else:
            return "success", replans, falsified           # all steps "executed" ok
        if replans > max_exec_replans:
            return "gave_up", replans, falsified


def test_closed_loop_runs_clean_when_nothing_fails():
    status, replans, _ = _drive_closed_loop(
        TampSolver(), scene_stack3.make_world(), scene_stack3.GOAL, fail_on=set())
    assert status == "success" and replans == 0


def test_closed_loop_falsifies_and_terminates_on_persistent_failure():
    """If a pick keeps failing at execution, the loop falsifies `reachable`,
    re-solves, and terminates gracefully (no infinite loop) — the gather goal
    becomes unsatisfiable because that object can never be placed."""
    status, replans, falsified = _drive_closed_loop(
        TampSolver(), scene_pnp.make_world(), {"type": "gather", "surface": "tray"},
        fail_on={("PICK", "cup_1")})
    assert ("reachable", "cup_1") in falsified
    assert status in ("failed", "gave_up")
    assert replans <= 6                                    # bounded by max_exec_replans
