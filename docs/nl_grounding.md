# Language-grounded, self-verifying TAMP

Tell the robot what to do in English; it derives the symbolic goal, plans, and
executes — **and rejects anything infeasible, with a reason**. The design is
*neuro-symbolic*: a language model only ever proposes a **goal**, never robot
actions, and the PDDL+TAMP layer **validates and grounds** that goal against the
perceived world before anything moves. This "model proposes, planner verifies"
split is what separates the approach from direct language-to-action pipelines,
which have no feasibility check between the text and the motors.

```
"stack the blocks"  ──►  goal grounder  ──►  {type: stack, order: [A,B,C]}
                                              │  validated vs the perceived world
                                              ▼
                                     TampSolver (PDDL + geometric refine + replan)
                                              ▼
                                     PICK A · STACK A · PICK B · STACK B ...
```

## The goal grounder ([tamp/nl/](../tamp/nl/))

`GoalGrounder.ground(command, world)` returns a **validated** goal spec (the same
dict `TampSolver.solve` consumes). It tries a chain of backends and **validates
every one** against the actually-perceived objects and surfaces
([dsl.py](../tamp/nl/dsl.py)), so a hallucinated object id is caught, never executed:

| Backend | When it's used | Needs |
|---------|----------------|-------|
| **Anthropic API** ([llm_grounder.py](../tamp/nl/llm_grounder.py)) | `anthropic` SDK + credentials present | API key |
| **OpenAI-compatible / NVIDIA NIM** ([openai_grounder.py](../tamp/nl/openai_grounder.py)) | `NVIDIA_API_KEY`/`OPENAI_API_KEY` set | API key |
| **Local LLM (Ollama)** ([local_grounder.py](../tamp/nl/local_grounder.py)) | an Ollama server is running | a local model |
| **Deterministic grammar** ([grammar.py](../tamp/nl/grammar.py)) | always (final fallback) | nothing |

The chain degrades gracefully: with a cloud key, a local model, or **neither**,
common commands (stack, gather, "clear the table", "put a cup on the tray") parse
fully offline through the grammar. The grammar is written to **fail closed**:
questions, negations ("do *not* put the cup on the tray"), and commands without a
placement destination are refused rather than guessed — safety property tests
live in `tests/unit/test_nl_safety.py`.

### Try it
```bash
tamp-command "stack A on B on C"                 # one-shot
tamp-command --scene pnp "put the cups on the tray"
tamp-command                                     # interactive prompt
```

## MCP server: drive the planner from any MCP client

The planning stack is also exposed as a **Model Context Protocol server**
([tamp/mcp_server.py](../tamp/mcp_server.py)), so any MCP-capable client or agent
(e.g. Claude Code, Claude Desktop) can read the robot's world state and submit
goals conversationally — with the same rule as above: the client proposes a goal,
the symbolic + geometric layers verify it.

Tools: `list_scenes`, `get_world_state`, `plan_goal`, `ground_command`,
`plan_command`; resource: `tamp://goal-dsl` (the goal DSL, self-describing).

The repo ships a project-scoped [.mcp.json](../.mcp.json), so MCP clients that
read project configuration pick the server up automatically. To register it
manually in any other client:

```json
{ "mcpServers": { "tamp-robot": {
  "command": ".venv/bin/python",
  "args": ["-m", "tamp.mcp_server"] } } }
```

Run standalone (stdio transport): `python -m tamp.mcp_server`.

## Installing the optional extras
```bash
pip install -e ".[nl]"     # Anthropic-API grounding backend
pip install -e ".[mcp]"    # MCP server
# NVIDIA NIM / OpenAI-compatible: export NVIDIA_API_KEY=...  (no extra install)
# Ollama:                          run `ollama serve` + `ollama pull llama3.2`
```

## Status
- Grammar grounding, DSL validation, safety refusals, and end-to-end solves are
  covered by unit tests (`test_nl_grounding.py`, `test_nl_safety.py`,
  `test_mcp_server.py`).
- The MCP server registers its tools/resource and serves over stdio; every tool
  returns structured errors instead of raising.
- The hosted-API backends (Anthropic / NVIDIA / Ollama) activate only when their
  key or server is present; without one, the grounder falls through to the
  deterministic grammar and the planner verifies the result regardless.
