"""Claude-API natural-language goal grounder + plan narrator.

LLM proposes, the planner verifies: this turns an English command + the perceived
world into a *candidate* goal spec; ``tamp.nl.dsl.validate_goal`` and the PDDL+TAMP
layer are what actually accept or reject it. That neuro-symbolic split is the point
— the LLM never emits robot actions, only a symbolic goal that the geometry checks.

Best practices applied (see the claude-api skill):
* model ``claude-opus-4-8`` (latest);
* structured output via ``output_config.format`` with ``GOAL_JSON_SCHEMA`` so the
  response is guaranteed-parseable JSON;
* prompt caching of the stable system prompt + DSL schema (``cache_control``), with
  the volatile per-request world+command placed *after* in the user turn;
* ``anthropic`` imported lazily so this module loads with the dep absent — the
  grounder falls back to the offline grammar parser when there's no SDK/credentials.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from tamp.types import WorldState
from tamp.nl.dsl import GOAL_JSON_SCHEMA, GOAL_TYPES

MODEL = "claude-opus-4-8"

_SYSTEM = """You translate a natural-language manipulation command into ONE goal-spec \
JSON object for a tabletop robot. You do NOT plan or emit robot actions — you only \
choose the symbolic goal; a downstream task-and-motion planner verifies feasibility.

Output exactly one JSON object matching the provided schema. Allowed "type" values \
and their fields:
- stack: {order:[top..bottom ids], base_surface?} — build a tower in that order.
- on: {object:id, surface:id} — put one object on one surface.
- on_any: {object:id, surfaces:[ids]} — object on ANY of these surfaces.
- clear: {object:id} — nothing on top of this object.
- gather: {surface:id} — put EVERY movable object onto this surface.
- place_class: {class, surface:id} — put one object OF THIS CLASS on the surface.
- sort_by_class: {assignments:[{class, surface:id}]} — each class to its surface.
- none: {reason} — the command does not map to any goal, or references things not present.

Rules:
- Use ONLY the object ids, class names, and surface ids given in the world state.
- Never invent ids. If the command names something absent, return type "none".
- Prefer place_class over on when the user names a class ("a cup") not a specific id.
- Prefer gather for "everything/all/clear the table"; pick a destination that exists.
- Keep it to a single goal object; choose the closest-matching type."""

_NARRATE_SYSTEM = """You narrate a tabletop robot's plan for a human in ONE or TWO short \
sentences, plain and concrete. No preamble, no markdown, no lists. If asked about a \
replan, say briefly what failed and what it's trying instead."""


class GroundingUnavailable(RuntimeError):
    """Raised when the LLM path can't run (no SDK, no credentials, API error)."""


def is_importable() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except Exception:
        return False


def _client():
    try:
        import anthropic
    except Exception as exc:  # SDK not installed
        raise GroundingUnavailable(f"anthropic SDK unavailable: {exc}")
    return anthropic


def _world_brief(world: WorldState) -> str:
    objs = ", ".join(f"{o.object_id}(class={o.class_name})" for o in world.objects)
    surfs = ", ".join(f"{s.surface_id}(class={s.class_name})" for s in world.surfaces)
    return f"OBJECTS: {objs or '(none)'}\nSURFACES: {surfs or '(none)'}"


def ground(command: str, world: WorldState, model: str = MODEL) -> Dict:
    """Call Claude and return the raw goal-spec dict (validate it before use)."""
    anthropic = _client()
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            # Stable, cacheable prefix: instructions + the DSL/enum. The volatile
            # world+command go in the user turn, after this cache breakpoint.
            system=[{"type": "text",
                     "text": _SYSTEM + "\n\nValid types: " + ", ".join(GOAL_TYPES),
                     "cache_control": {"type": "ephemeral"}}],
            output_config={"format": {"type": "json_schema", "schema": GOAL_JSON_SCHEMA}},
            messages=[{"role": "user",
                       "content": f"WORLD STATE:\n{_world_brief(world)}\n\nCOMMAND: {command}"}],
        )
    except anthropic.AuthenticationError as exc:
        raise GroundingUnavailable(f"no/invalid credentials: {exc}")
    except anthropic.APIConnectionError as exc:
        raise GroundingUnavailable(f"connection error: {exc}")
    except Exception as exc:  # noqa: BLE001  any other API failure -> fall back
        raise GroundingUnavailable(f"grounding call failed: {exc}")

    text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise GroundingUnavailable("empty grounding response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GroundingUnavailable(f"non-JSON grounding response: {exc}")


def narrate(prompt: str, model: str = MODEL) -> Optional[str]:
    """Short natural-language narration of a plan/replan. None if LLM unavailable."""
    try:
        anthropic = _client()
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model, max_tokens=256,
            system=[{"type": "text", "text": _NARRATE_SYSTEM,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text.strip() for b in resp.content
                     if getattr(b, "type", None) == "text"), None)
    except Exception:
        return None
