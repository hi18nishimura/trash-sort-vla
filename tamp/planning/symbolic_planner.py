"""Symbolic planning over the ``tabletop`` domain via unified-planning.

Wraps a UP engine (Fast Downward by default) and returns an ordered list of
grounded :class:`tamp.types.TaskAction` steps. Supports *replanning from a
modified state*, which is what the TAMP loop calls after the geometric refiner
falsifies a bridge fluent (e.g. ``reachable``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import Problem
from unified_planning.shortcuts import OneshotPlanner, get_environment

from tamp.types import TaskAction

# Silence the per-solve engine "credits" banner.
get_environment().credits_stream = None

log = logging.getLogger("tamp.symbolic")

# Map UP action name -> TaskAction.action_type and the parameter that is the
# primary manipulated object (index into the grounded action's parameters).
_ACTION_MAP = {
    "pick":    ("PICK", 0),
    "place":   ("PLACE", 0),
    "stack":   ("STACK", 0),
    "unstack": ("UNSTACK", 0),
}


@dataclass
class PlanResult:
    success: bool
    actions: List[TaskAction]
    status: str
    engine: str = ""


class SymbolicPlanner:
    def __init__(self, engine_name: str = "fast-downward", timeout: float = 30.0):
        self.engine_name = engine_name
        self.timeout = timeout

    def solve(self, problem: Problem) -> PlanResult:
        log.info("solving (engine=%s, timeout=%.0fs)", self.engine_name, self.timeout)
        with OneshotPlanner(name=self.engine_name) as planner:
            # Pass the configured wall-clock timeout so a pathological problem
            # can't hang the TAMP loop (the param was previously declared but
            # never applied). Engines that ignore it just run to completion.
            try:
                result = planner.solve(problem, timeout=self.timeout)
            except TypeError:
                result = planner.solve(problem)   # engine without a timeout kwarg
        status = result.status
        ok = status in (
            PlanGenerationResultStatus.SOLVED_SATISFICING,
            PlanGenerationResultStatus.SOLVED_OPTIMALLY,
        )
        if not ok or result.plan is None:
            log.warning("no plan (status=%s)", status)
            return PlanResult(False, [], str(status), self.engine_name)

        actions = [self._to_task_action(ai) for ai in result.plan.actions]
        return PlanResult(True, actions, str(status), self.engine_name)

    @staticmethod
    def _to_task_action(action_instance) -> TaskAction:
        name = action_instance.action.name
        params = [str(p) for p in action_instance.actual_parameters]
        action_type, obj_idx = _ACTION_MAP[name]
        pnames = [p.name for p in action_instance.action.parameters]
        param_dict = dict(zip(pnames, params))
        return TaskAction(
            action_type=action_type,
            object_id=params[obj_idx],
            preconditions=[],   # filled by refiner / bridge if needed
            effects=[],
            params=param_dict,
        )


def solve(problem: Problem, engine_name: str = "fast-downward") -> PlanResult:
    """Convenience one-shot solve."""
    return SymbolicPlanner(engine_name).solve(problem)
