# A2A Completion Audit

Current audit status: not yet marked complete. The implementation has broad
coverage, but the goal should remain active until the verification set below has
been rerun after the final A2A changes and any remaining intentional gaps have
been accepted.

## Requirement Evidence

| Requirement | Current evidence | Status |
|---|---|---|
| HTTP A2A client support, no gRPC | `src/fast_agent/a2a/remote_agent.py` restricts default bindings to `JSONRPC` and `HTTP+JSON`; `src/fast_agent/a2a/connect.py` rejects gRPC aliases; `tests/integration/a2a/test_remote_agent_runtime.py` covers both HTTP bindings. | Implemented and tested. |
| HTTP/JSON-RPC A2A server support | `src/fast_agent/a2a/server.py` exposes SDK JSON-RPC and REST routes; `tests/integration/a2a/test_fast_agent_a2a_server.py` covers JSON-RPC and HTTP+JSON clients against fast-agent-as-server. | Implemented and tested. |
| Deployable like ACP/MCP | `fast-agent serve a2a` and `fast-agent serve --transport a2a` are covered by `tests/unit/fast_agent/cli/test_a2a_serve_options.py`; server docs show `fast-agent serve a2a`. | Implemented and tested at CLI request-construction level. |
| PromptMessageExtended API behaves like normal agents | Client and server bridges convert A2A parts to/from `PromptMessageExtended`; integration tests cover text, stream listeners, raw/blob/image/audio/data content, and history behavior. | Implemented and tested. |
| `INPUT_REQUIRED` turn management | Client preserves task/context only for `TASK_STATE_INPUT_REQUIRED`; server maps `LlmStopReason.PAUSE` to `TASK_STATE_INPUT_REQUIRED`; integration tests cover follow-up completion in both fake-server and fast-agent-server paths. | Implemented and tested. |
| Session/context correlation | A2A `contextId` is optional inbound; SDK resolves it. Server `connection` scope maps resolved context id to an instance; `shared` and `request` scopes are explicit alternatives. Tests cover context reuse, no-history fresh contexts, and all instance scopes. | Implemented and tested in-memory. |
| AgentCard and A2A AgentSkill docs | `docs/docs/a2a/server.md` explains AgentCard interfaces and one A2A `AgentSkill` per loaded fast-agent agent; integration tests assert skills, modes, descriptions, tags, and metadata routing. | Implemented, documented, and tested. |
| API documentation | `docs/docs/a2a/api.md` covers direct `A2ARemoteAgent`, embedded `AgentA2AServer`, raw JSON-RPC, raw HTTP+JSON, content mapping, and structured JSON data parts. | Documented. |
| Client/server docs pages and recordings | `docs/docs/a2a/client.md` and `server.md` embed deterministic recordings; client docs embed the real-LLM Hugging Face MCP streaming recording; `scripts/a2a_docs_pipeline.py check` verifies required assets/pages. | Documented and pipeline-checked. |
| Deterministic API/CLI/TUI tests | API/server integration tests cover runtime protocol behavior; CLI tests cover `--a2a`, `--a2a-transport`, and `serve a2a`; TUI unit tests cover `/a2a` parsing and command dispatch. | Covered, but full prompt-toolkit E2E remains a possible future hardening target. |
| Real LLM streaming demo with HF MCP | `docs/docs/assets/a2a/a2a-real-llm-hf-streaming.cast` is checked in; `scripts/a2a_docs_pipeline.py record-real-llm` regenerates it with `codexresponses.gpt-5.4-mini` and `https://hf.co/mcp`; client docs embed it. | Implemented as provider smoke artifact. |
| Structured JSON protocol answer | A2A structured JSON is represented as `Part.data`; fast-agent maps explicit JSON resources to data parts and leaves ordinary model text as text. Documented in API/server/compliance pages and addendum. | Implemented and documented. |
| Multimodal support | Tests cover raw image input and audio-as-blob preservation; docs list partial typed audio support as a known gap. | Partially implemented and documented. |
| Hooks/tools/skills bundle deployment | Server docs state A2A serving uses the normal fast-agent runtime, so AgentCards, MCP servers, tools, skills, hooks, model settings, and workflows load before serving. | Documented; mostly proven indirectly through shared serve/bootstrap path. |
| Review fixes: clone config | `A2ARemoteAgent._clone_constructor_kwargs()` preserves `a2a_config`; integration test covers detached clone contacting remote server. | Fixed and tested. |
| Review fixes: artifact append semantics | Client assembles per-artifact output and honors append/replacement; integration test covers replacement plus repeated appended chunks. | Fixed and tested. |
| Review fixes: default transport probing | Client defaults supported protocol bindings to `JSONRPC` and `HTTP+JSON` when no transport is requested. | Fixed and unit/integration covered. |
| Review fixes: routable AgentCard URLs | Served wildcard-host AgentCards are rewritten from the incoming request base URL; integration test covers wildcard bind. | Fixed and tested. |
| Review fixes: terminal task ids | Terminal full-task and status events clear task id except for `INPUT_REQUIRED`; unit tests cover full-task terminal behavior. | Fixed and tested. |
| Review fixes: raw file preservation | Inbound raw non-image file bytes become `BlobResourceContents`; outbound blobs become A2A raw parts. | Fixed and tested. |
| Review fixes: no-history A2A context reset | `use_history=False` gets a fresh context between completed turns but preserves context/task while continuing `INPUT_REQUIRED`. | Fixed and tested. |
| Review fixes: A2A instance scope | A2A serve path passes `instance_scope`; server implements `shared`, `connection`, and `request`; tests cover all scopes. | Fixed and tested. |

## Known Gaps

These are documented as current protocol-compliance gaps rather than hidden
unfinished work:

- gRPC transport is intentionally unsupported.
- A2A push notifications are not implemented; streaming and polling are the
  supported client update paths.
- Extended AgentCard, card signing, extension negotiation, transport-level
  security scheme advertisement/enforcement, idempotent message replay handling,
  and persistent task/session storage are not implemented.
- Audio is preserved as raw/blob content on the server rather than mapped to a
  dedicated fast-agent audio content object.
- Ordinary model text that contains JSON is not guessed into protocol data.

## Required Final Verification

Before marking the goal complete, rerun:

```bash
uv run pytest tests/integration/a2a \
  tests/unit/fast_agent/test_a2a_remote_agent_events.py \
  tests/unit/fast_agent/test_a2a_remote_agent_config.py \
  tests/unit/fast_agent/a2a_connect_test.py \
  tests/unit/fast_agent/cli/test_a2a_go_options.py \
  tests/unit/fast_agent/cli/test_a2a_serve_options.py \
  tests/unit/fast_agent/ui/test_parse_a2a_commands.py \
  tests/unit/fast_agent/ui/test_a2a_command_dispatch.py \
  tests/unit/fast_agent/core/test_a2a_error_formatting.py \
  tests/unit/test_a2a_docs_pipeline.py \
  -q
uv run scripts/a2a_docs_pipeline.py check
uv run scripts/lint.py
uv run scripts/typecheck.py
```

Also verify the checked-in real-LLM cast does not contain provider secrets:

```bash
rg -q "hf_|sk-|OPENAI|ANTHROPIC|Authorization|Bearer|HF_TOKEN|OPENAI_API_KEY|ANTHROPIC_API_KEY" \
  docs/docs/assets/a2a/a2a-real-llm-hf-streaming.cast
```

The secret scan should exit with status 1 because it finds no matches.
