# CC Refactor Implementation Prep

Date: 2026-03-14

This is a concrete implementation companion to `plan/03-14-cc-refactor.md`.
It does not change behavior by itself. It turns the refactor plan into an
ordered worklist with the current code seams, likely duplication hotspots, and
test targets.

## 1. Current hotspots and seams

### Primary hotspot

- `src/fast_agent/ui/prompt/parser.py`
  - `parse_special_input`
  - currently a large command multiplexer
  - currently owns:
    - command-family routing
    - family-specific token parsing
    - default/error behavior
    - some normalization logic

### Stable semantic boundary already present

- `src/fast_agent/ui/command_payloads.py`
  - semantic payload types for TUI parsing

### Shared execution path already present

- `src/fast_agent/ui/interactive/command_dispatch.py`
  - converts `CommandPayload` into handler calls and `DispatchResult`

### ACP surface to compare against

- `src/fast_agent/acp/slash_commands.py`
- `src/fast_agent/acp/slash/handlers/history.py`
- `src/fast_agent/acp/slash/handlers/mcp.py`
- `src/fast_agent/acp/slash/handlers/session.py`
- `src/fast_agent/acp/slash/handlers/model.py`

### Shared command handlers under the middle layer

- `src/fast_agent/commands/handlers/history.py`
- `src/fast_agent/commands/handlers/mcp_runtime.py`
- `src/fast_agent/commands/handlers/prompts.py`
- `src/fast_agent/commands/handlers/sessions.py`
- `src/fast_agent/commands/handlers/model.py`

## 2. Existing tests to preserve and later collapse

Current parser-focused tests:

- `tests/unit/fast_agent/ui/test_parse_history_commands.py`
- `tests/unit/fast_agent/ui/test_parse_mcp_commands.py`
- `tests/unit/fast_agent/ui/test_parse_model_commands.py`
- `tests/unit/fast_agent/ui/test_parse_models_commands.py`
- `tests/unit/fast_agent/ui/test_parse_cards_commands.py`
- `tests/unit/fast_agent/ui/test_hash_agent_command.py`

Current ACP tests already covering neighboring behavior:

- `tests/unit/fast_agent/acp/test_slash_commands_mcp.py`
- `tests/unit/fast_agent/acp/test_slash_commands_models.py`
- `tests/integration/acp/test_acp_slash_commands.py`

Current TUI-loop tests that may offer fixtures or stub patterns:

- `tests/unit/fast_agent/ui/test_interactive_prompt_agent_commands.py`

These should stay in place until the broader contract/integration/parity suites
exist. After that, narrow parser-only tests can be merged or deleted.

## 3. Highest-value duplication hotspots

### MCP connect parsing duplication

TUI parser:

- `src/fast_agent/ui/prompt/parser.py`
  - `_infer_mcp_connect_mode`
  - `_rebuild_mcp_target_text`
  - `/mcp connect` parsing branch
  - `/connect` alias branch

Shared/runtime parser already exists:

- `src/fast_agent/commands/handlers/mcp_runtime.py`
  - `infer_connect_mode`
  - `parse_connect_input`

ACP also parses connect arguments independently:

- `src/fast_agent/acp/slash/handlers/mcp.py`

Immediate implication:

- connect semantics are one of the clearest early extraction targets
- we should prefer reusing shared parsing utilities instead of maintaining
  separate TUI-only and ACP-only flag logic

### MCP session parsing duplication

Duplicated between:

- `src/fast_agent/ui/prompt/parser.py`
- `src/fast_agent/acp/slash/handlers/mcp.py`

Shared semantics today:

- `list`
- `jar`
- `new/create`
- `use/resume`
- `clear`

### History action parsing duplication

Duplicated between:

- `src/fast_agent/ui/prompt/parser.py`
- `src/fast_agent/acp/slash/handlers/history.py`

Not identical surfaces:

- TUI supports target-agent forms like `/history analyst`
- ACP is current-agent-oriented

But the overlap is still strong enough for parity tests on:

- `show`
- `detail/review`
- `save`
- `load`

### Session command parsing duplication

Duplicated between:

- `src/fast_agent/ui/prompt/parser.py`
- `src/fast_agent/acp/slash/handlers/session.py`

### Model command intent drift risk

Split across:

- `src/fast_agent/ui/prompt/parser.py`
- `src/fast_agent/acp/slash/handlers/model.py`
- shared handlers in `src/fast_agent/commands/handlers/model.py`

## 4. Recommended first test additions

### A. Intent contract suite

Create a single compact table-driven file, for example:

- `tests/unit/fast_agent/ui/test_command_intent_contract.py`

Target cases:

- `/history`
  - bare target
  - `show`
  - `load` missing filename
  - quoted reserved-name collision
- `/mcp`
  - `list`
  - `connect` URL mode
  - `session use`
  - invalid session usage
- `/connect`
  - URL
  - `npx`
  - `uvx`
  - plain stdio
- `#agent`
  - with message
  - without message
  - quiet `##`
- unknown command fallback

Assertions:

- concrete payload type
- important normalized fields
- error/default fields

This suite should be small enough to be read top-to-bottom as the canonical
intent contract.

### B. Simulator-backed TUI dispatch suite

Create:

- `tests/integration/ui/test_command_dispatch_flows.py`

Drive:

- `parse_special_input(...)`
- `dispatch_command_payload(...)`

Use real codepaths and stubbed in-memory collaborators, not mocks or
monkeypatching.

Recommended initial flows:

- session create/list/pin
- MCP list/connect/disconnect/session use
- history show/clear/rewind/fix
- prompt select/load
- `#agent` handoff and `DispatchResult` fields

Assertions:

- `DispatchResult`
- resulting simulator state
- coarse message content only

### C. ACP/TUI parity suite

Create:

- `tests/integration/acp/test_command_parity.py`

Initial parity targets:

- MCP session `list` / `use`
- MCP connect mode inference for URL / `npx` / `uvx` / stdio
- history `detail`
- model fast / reasoning / switch where the semantic outcome is shared

Assertions:

- equivalent normalized action
- equivalent core arguments fed into shared handlers
- equivalent outcome category or state effect

## 5. Proposed simulator/test-support module

Create a small shared helper module, for example:

- `tests/support/command_simulators.py`

Candidate classes:

- `InMemoryAgent`
- `InMemoryAgentProvider`
- `InMemorySessionState`
- `InMemoryMcpRegistry`
- `InMemoryPromptCatalog`
- `FakeInteractiveOwner`

Keep the simulator surface narrow. Only implement what the handlers exercised by
the new tests actually need.

### Minimum likely capabilities

`InMemoryAgent`

- `message_history`
- `usage_accumulator`
- `clear(...)`
- `pop_last_message()`
- `load_message_history(...)`

`InMemoryAgentProvider`

- `_agents`
- `_agent(name)`
- `agent_names()`
- `agent_types()`
- `list_prompts(...)`
- MCP attach/detach/list methods as needed by dispatch
- optional `_noenv_mode`

`FakeInteractiveOwner`

- `_get_agent_or_warn(...)`
- mutable `agent_types`

This module should favor plain dataclasses and tiny methods over clever test
abstractions.

## 6. Refactor extraction map for `parse_special_input`

Recommended extraction helpers:

- `_parse_history_command(remainder: str) -> CommandPayload`
- `_parse_session_command(remainder: str) -> CommandPayload`
- `_parse_card_command(remainder: str) -> CommandPayload`
- `_parse_agent_command(remainder: str) -> CommandPayload`
- `_parse_mcp_command(remainder: str) -> CommandPayload`
- `_parse_prompt_command(remainder: str) -> CommandPayload`
- `_parse_model_command(cmd_line: str, remainder: str) -> CommandPayload`

Keep these helpers pure and payload-focused.

Recommended small shared parsing utilities:

- `safe_shlex_split(...)`
- integer argument parser for history turn commands
- optional-flag extraction helper for small command families
- shared MCP session parser if feasible

## 7. Concrete implementation order

### Phase 1

1. Add `test_command_intent_contract.py`
2. Add simulator support module
3. Add first TUI dispatch flow file
4. Add first ACP/TUI parity file

### Phase 2

5. Extract command-family helpers from `parse_special_input`
6. Keep public behavior unchanged
7. Run tests after each helper extraction

### Phase 3

8. Move duplicated MCP parsing to shared helpers where practical
9. Reduce duplicated TUI/ACP session and history argument parsing
10. Keep help/rendering surface-specific

### Phase 4

11. Collapse redundant parser micro-tests
12. Keep only the canonical contract file plus broader integration/parity suites

## 8. Guardrails for implementation

- Do not add new parser-only micro-tests beyond the canonical contract table.
- Do not add mocks or monkeypatching in the new suites.
- Prefer stable assertions on payload fields, handler outcomes, and state.
- Avoid rich/markdown snapshot assertions.
- Keep diffs small: land test seam first, parser split second, duplication
  removal third.

## 9. Suggested first commit split

1. `tests: add command intent contract table`
2. `tests: add simulator-backed tui dispatch coverage`
3. `tests: add initial acp/tui command parity coverage`
4. `refactor: extract history/session/mcp/model parser helpers`
5. `refactor: share mcp parsing utilities across tui and acp`
6. `test: collapse redundant parser micro-tests`
