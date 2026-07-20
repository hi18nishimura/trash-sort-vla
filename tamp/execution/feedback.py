"""Map execution-time failures to bridge predicates for closed-loop replanning.

The robot executor reports the outcome of each plan step on ``/execution_status``
as a compact, dependency-free string (the executor runs under system Python and
must not import ``tamp``):

    "<ok|fail>:<action_type>:<object_id>:<reason>"
        e.g. "fail:PICK:cup_1:not_reached", "ok:PLACE:cup_1:done"

It is keyed by ``action_type``+``object_id`` (both present in the TaskAction msg)
rather than a numeric index, so the planner can match the report to the step it is
waiting on without a shared counter. The planner parses this with
:func:`parse_status`, finds the matching :class:`TaskAction`, and calls
:func:`predicate_for_action` to get the bridge predicate to falsify. Feeding that
into ``TampSolver.solve(..., pre_falsified=[pred])`` produces a different,
hopefully-feasible plan — the same recovery the geometric loop does, but driven by
what actually happened on the robot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from tamp.types import TaskAction

# Failure reasons the executor may report (kept small and explicit).
PLAN_FAILED = "plan_failed"    # MoveIt found no motion plan to the target
NOT_REACHED = "not_reached"    # motion planned/executed but the pose wasn't achieved
COLLISION = "collision"        # a collision was detected at/near the target
DROPPED = "dropped"            # object was grasped then lost before placement
DONE = "done"                  # success
REASONS = {PLAN_FAILED, NOT_REACHED, COLLISION, DROPPED, DONE}

_PICKS = ("PICK", "UNSTACK")
_PLACES = ("PLACE", "STACK")


@dataclass
class ExecutionStatus:
    ok: bool
    action_type: str   # PICK / UNSTACK / PLACE / STACK
    object_id: str
    reason: str        # one of REASONS (best-effort; unknown reasons kept verbatim)


def format_status(ok: bool, action_type: str, object_id: str, reason: str) -> str:
    """Build the wire string (the executor builds the same inline, dep-free)."""
    return f"{'ok' if ok else 'fail'}:{action_type}:{object_id}:{reason}"


def parse_status(s: str) -> Optional[ExecutionStatus]:
    """Parse "<ok|fail>:<action_type>:<object_id>:<reason>"; None if malformed."""
    parts = s.strip().split(":")
    if len(parts) < 4 or parts[0] not in ("ok", "fail"):
        return None
    return ExecutionStatus(ok=(parts[0] == "ok"), action_type=parts[1],
                           object_id=parts[2], reason=":".join(parts[3:]))


def predicate_for_action(action: TaskAction, reason: str) -> Optional[Tuple]:
    """Bridge predicate to falsify when *action* fails at execution with *reason*.

    Mirrors what the geometric refiner falsifies so a re-solve avoids the same
    step: a pick that never reached the object falsifies ``reachable``; one that
    reached but collided or dropped falsifies ``graspable``; a failed place/stack
    falsifies ``placeable`` / ``stackable``.
    """
    at = action.action_type
    oid = action.object_id
    if at in _PICKS:
        if reason in (COLLISION, DROPPED):
            return ("graspable", oid)
        return ("reachable", oid)          # PLAN_FAILED / NOT_REACHED / unknown
    if at == "PLACE":
        return ("placeable", oid, action.params.get("s"))
    if at == "STACK":
        return ("stackable", oid, action.params.get("bottom"))
    return None
