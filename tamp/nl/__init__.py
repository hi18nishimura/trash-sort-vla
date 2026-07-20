"""Natural-language goal grounding (the flagship): English command -> validated
PDDL goal spec, LLM-first with a fully-offline grammar fallback.

    from tamp.nl import GoalGrounder
    GoalGrounder(use_llm=False).ground("put the cups on the tray", world)
"""
from tamp.nl.grounder import GoalGrounder, GroundingResult

__all__ = ["GoalGrounder", "GroundingResult"]
