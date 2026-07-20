"""Goal grounding via any OpenAI-compatible chat API — NVIDIA NIM by default.

Lets the autonomous grounder use NVIDIA's hosted models (or OpenRouter, a local
vLLM, Ollama's /v1 endpoint, etc.) instead of the Anthropic API. NVIDIA's
``integrate.api.nvidia.com`` is OpenAI-compatible with a free tier; set
``NVIDIA_API_KEY`` (or ``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``) and it activates.
Pure stdlib (urllib) — no extra dependency; raises ``GroundingUnavailable`` (so
the chain falls through) when no key is configured or the call fails.

Env:  NVIDIA_API_KEY / OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.
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

# NVIDIA NIM is the default OpenAI-compatible endpoint.
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
API_KEY = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")
MODEL = os.environ.get("OPENAI_MODEL", "meta/llama-3.1-70b-instruct")


def is_available() -> bool:
    """True if a key is configured (we don't probe the network)."""
    return bool(API_KEY)


def ground(command: str, world: WorldState, model: str = None) -> Dict:
    if not API_KEY:
        raise GroundingUnavailable("no NVIDIA_API_KEY / OPENAI_API_KEY set")
    model = model or MODEL
    # JSON-object response_format is the broadly-supported constraint; the schema
    # is described in the system prompt and enforced by validate_goal downstream.
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system",
             "content": _SYSTEM + "\n\nReturn a JSON object with these fields: "
             + json.dumps(GOAL_JSON_SCHEMA["properties"])},
            {"role": "user",
             "content": f"WORLD STATE:\n{_world_brief(world)}\n\nCOMMAND: {command}"},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise GroundingUnavailable(f"openai-compat HTTP {exc.code}: {exc.reason}")
    except Exception as exc:  # noqa: BLE001
        raise GroundingUnavailable(f"openai-compat call failed: {exc}")

    try:
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise GroundingUnavailable(f"bad openai-compat response: {exc}")
