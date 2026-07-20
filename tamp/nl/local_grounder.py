"""Keyless local-LLM goal grounding via Ollama (no Anthropic API key needed).

Same job as :mod:`tamp.nl.llm_grounder` — English command + world -> candidate
goal-spec JSON — but served by a model running locally through Ollama's HTTP API
(``http://localhost:11434`` by default). Uses Ollama's structured-output ``format``
field with ``GOAL_JSON_SCHEMA`` so the reply is parseable JSON. Pure stdlib
(urllib), so it adds no dependency; if Ollama isn't running it raises
``GroundingUnavailable`` and the grounder falls back to the offline grammar parser.

Set ``OLLAMA_MODEL`` (default ``llama3.2``) and ``OLLAMA_HOST`` to configure.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict

from tamp.types import WorldState
from tamp.nl.dsl import GOAL_JSON_SCHEMA
from tamp.nl.llm_grounder import GroundingUnavailable, _SYSTEM, _world_brief

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def is_available(timeout: float = 0.3) -> bool:
    """True if an Ollama server answers at HOST (cheap reachability probe)."""
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def ground(command: str, world: WorldState, model: str = None) -> Dict:
    model = model or MODEL
    body = json.dumps({
        "model": model,
        "stream": False,
        "format": GOAL_JSON_SCHEMA,                 # Ollama structured outputs
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",
             "content": f"WORLD STATE:\n{_world_brief(world)}\n\nCOMMAND: {command}"},
        ],
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
    except urllib.error.URLError as exc:
        raise GroundingUnavailable(f"ollama unreachable at {HOST}: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise GroundingUnavailable(f"ollama call failed: {exc}")

    content = (payload.get("message") or {}).get("content", "")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise GroundingUnavailable(f"non-JSON ollama response: {exc}")
