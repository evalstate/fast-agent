# 03-10 ty cleanup plan

## Status

- Branch: `fix/ty-upgrade`
- `ty` upgraded to `0.0.21`
- `src/` is now clean under `uv run ty check ./src`
- Full `uv run scripts/typecheck.py` still reports test-side diagnostics
- Current working strategy: keep `src/` green first, then sweep tests

## Completed

1. Upgraded repo dev dependency and lockfile to `ty>=0.0.21`
2. Removed deterministic stale suppressions and redundant casts across `src/` and `tests/`
3. Fixed `src/` typing regressions introduced by the newer `ty`
4. Ran:
   - `uv run scripts/lint.py --fix`
   - `uv run ty check ./src`
   - targeted smoke tests for touched runtime areas

## Remaining work

### Phase 1: test cleanup

Primary goal: make `uv run scripts/typecheck.py` pass without weakening rules.

Main remaining hotspots:

- `tests/unit/fast_agent/llm/providers/test_llm_anthropic_caching.py`
- `tests/unit/acp/test_content_conversion.py`
- `tests/unit/fast_agent/llm/providers/test_llm_openai_history.py`
- `tests/unit/fast_agent/commands/test_models_manager_handler.py`
- `tests/unit/fast_agent/mcp/test_agent_server_tool_description.py`
- `tests/unit/fast_agent/mcp/test_cimd.py`
- `tests/unit/fast_agent/mcp/test_mcp_connection_manager.py`

### Phase 2: test fix patterns

#### A. TypedDict-union narrowing in tests

Use explicit narrowing helpers or `assert isinstance(..., dict)` before key access.

Typical cases:

- `cache_control`
- `tool_use_id`
- `reasoning_content`
- `content`

Preferred approach:

- narrow with `assert isinstance(payload, dict)`
- materialize TypedDict unions to plain `dict(...)` when the test is intentionally probing ad-hoc keys
- avoid broad `cast(...)` unless we are crossing an external SDK boundary

#### B. Optional callback / optional attribute tests

Use explicit assertions before calling or dereferencing values:

- `assert callback_handler is not None`
- `assert oauth_filter is not None`
- `assert isinstance(content_block, TextContent)`

#### C. Test scaffolds that mimic framework/context objects

Where tests pass lightweight local stubs into typed runtime methods, either:

- make the stub satisfy the runtime protocol more explicitly, or
- add local helper factories that construct objects with the exact typed shape

#### D. Content block list variance

Where tests build narrower `list[...]` values and pass them to APIs typed as wider unions:

- prefer `Sequence[...]` for read-only inputs in test helpers
- or annotate local values with the wider content union up front

## Guardrails

- Do not globally ignore `invalid-key` or `invalid-argument-type`
- Prefer type narrowing over suppression
- Keep runtime behavior unchanged while cleaning tests
- Preserve the repo rule: no mocking / no monkeypatching in new work

## Validation plan

After each test-cleanup slice:

1. `uv run scripts/lint.py --fix`
2. `uv run scripts/typecheck.py`
3. targeted `pytest` for the touched test modules

Once full typecheck is green:

4. broader `pytest tests/unit`

## Commit plan

- Commit 1: deterministic cleanup and ty upgrade
- Commit 2: `src/` clean under `ty 0.0.21`
- Next commits: grouped test cleanup by hotspot area, not by individual file

