"""GoalGrounder — English command -> validated PDDL goal spec.

Resilient by a three-tier backend chain, tried in order and **all validated**
against the perceived world before use:

1. **Claude API** (``tamp.nl.llm_grounder``) — if the `anthropic` SDK + credentials
   are present;
2. **local LLM via Ollama** (``tamp.nl.local_grounder``) — keyless, if an Ollama
   server is running locally;
3. **offline grammar parser** (``tamp.nl.grammar``) — always available.

So it works with a cloud key, with a local model, or with neither. The returned
goal dict is exactly what ``TampSolver.solve`` / the planner node consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from tamp.types import WorldState
from tamp.nl import grammar, llm_grounder, local_grounder, openai_grounder
from tamp.nl.dsl import normalize_goal, validate_goal
from tamp.nl.llm_grounder import GroundingUnavailable


@dataclass
class GroundingResult:
    success: bool
    goal: Optional[dict] = None        # normalized, ready for the planner
    source: str = ""                   # "claude" | "local" | "rule" | ""
    raw: Optional[dict] = None         # pre-normalization spec (for explanations)
    error: str = ""


class GoalGrounder:
    def __init__(self, use_llm: Optional[bool] = None, model: str = llm_grounder.MODEL):
        # use_llm=False -> offline only (deterministic, network-free; used by tests).
        # None/True -> auto-detect available LLM backends, in priority order.
        self.model = model
        self._backends: List[Tuple[str, Callable]] = []
        if use_llm is not False:
            if llm_grounder.is_importable():
                self._backends.append(("claude", llm_grounder.ground))
            if openai_grounder.is_available():           # NVIDIA NIM / OpenAI-compatible
                self._backends.append(("nvidia", openai_grounder.ground))
            if local_grounder.is_available():            # local Ollama
                self._backends.append(("local", local_grounder.ground))
        self.use_llm = bool(self._backends)

    def ground(self, command: str, world: WorldState) -> GroundingResult:
        if not command or not command.strip():
            return GroundingResult(False, error="empty command")

        last_err = "llm disabled"
        # 1) LLM backends (each proposes a goal; we verify it against the world).
        for name, fn in list(self._backends):
            try:
                spec = fn(command, world)
                # An explicit refusal ("this command maps to no goal") is a DECISION,
                # not a failure. It must short-circuit the whole chain — otherwise it
                # falls through to the grammar parser, which is the dumbest layer in
                # the stack and would happily turn a prohibition into a robot motion.
                if isinstance(spec, dict) and spec.get("type") == "none":
                    return GroundingResult(
                        False, source=name, raw=spec,
                        error=spec.get("reason") or "command does not map to a goal")
                ok, err = validate_goal(spec, world)
                if ok:
                    return GroundingResult(True, normalize_goal(spec, world), name, spec)
                last_err = f"{name} goal rejected: {err}"   # invalid -> try next backend
            except GroundingUnavailable as exc:
                self._backends.remove((name, fn))           # don't retry this session
                last_err = str(exc)
            except Exception as exc:  # noqa: BLE001
                # LLM output is UNTRUSTED input. A malformed spec must never escape as
                # an exception and kill the fallback chain — degrade to the next backend.
                last_err = f"{name} raised on malformed output: {exc}"

        # 2) Offline grammar fallback.
        spec = grammar.parse(command, world)
        if spec is not None:
            ok, err = validate_goal(spec, world)
            if ok:
                return GroundingResult(True, normalize_goal(spec, world), "rule", spec)
            return GroundingResult(False, raw=spec, source="rule",
                                   error=f"parsed but invalid: {err}")

        reason = grammar.refusal_reason(command)
        if reason:
            return GroundingResult(False, source="rule", error=f"refused: {reason}")
        return GroundingResult(False, error=f"could not ground command ({last_err})")
