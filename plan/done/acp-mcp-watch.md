# ACP/MCP `--watch` + `--reload` support (spec)

## Goal
Enable AgentCard auto-reload and manual reload in server mode for:
- ACP transport (`fast-agent serve --transport acp`)
- MCP transports (`http`, `sse`, `stdio`)

`fast-agent go --watch/--reload` already works for interactive mode. This spec covers
equivalent behavior in `serve`.

## Current behavior (baseline)
- `fast-agent go --card <dir> --watch` starts the AgentCard watcher.
- `serve` does **not** expose `--watch` or `--reload`.
- `serve` uses `allow_extra_args` + `ignore_unknown_options`, so `--watch` is silently ignored.
- `FastAgent` only starts `_watch_agent_cards()` when `args.watch` is True.

## Desired behavior
1) [x] `fast-agent serve --card <dir> --watch` automatically reloads AgentCards on change.
2) [x] `fast-agent serve --card <dir> --reload` enables `/reload` (manual) from ACP slash command or MCP command hook.
3) [x] Behavior mirrors `go` where possible:
   - [x] Watcher only runs when AgentCards were loaded and `--watch` is set.
   - [x] Reload is safe in both shared and per-session scopes.

## Implementation plan

### 1) CLI: add flags to `serve`
File: `src/fast_agent/cli/commands/serve.py`
- [x] Add options:
  - [x] `reload: bool = typer.Option(False, "--reload", help="Enable manual AgentCard reloads (/reload)")`
  - [x] `watch: bool = typer.Option(False, "--watch", help="Watch AgentCard paths and reload")`
- [x] Pass through to `run_async_agent(..., reload=reload, watch=watch)`.

### 2) CLI: pass flags through server runner
File: `src/fast_agent/cli/commands/go.py`
- [x] `run_async_agent()` already accepts `reload` and `watch`.
- [x] Ensure `serve` path forwards them (no other changes needed).

### 3) FastAgent: start watcher in server mode
Already implemented in `FastAgent.run()`:
- [x] Watcher starts when `args.watch` is True and `_agent_card_roots` is non-empty.
- [x] `args.watch` is only set by CLI or programmatic calls.
No code change required beyond setting `args.watch` via CLI.

### 4) Manual reload in ACP
Behavior:
- [x] `/reload` is provided by ACP slash command handler.
- [x] It calls the `load_card_callback`/`reload` hooks supplied by `FastAgent`.
Ensure the reload callback is set for server mode:
- [x] In `FastAgent.run()`, `wrapper.set_reload_callback(...)` runs when `args.reload` or `args.watch` is True.
No change required if `args.reload` is passed through.

### 5) Manual reload in MCP
There is no built-in MCP slash command. Options:
1) [x] Add an MCP tool, e.g., `reload_agent_cards`, that calls `AgentApp.reload_agents()`.
2) [x] Expose reload via an MCP resource or prompt (lower priority).

Spec choice: **Add MCP tool**.
- File: `src/fast_agent/mcp/server/agent_server.py`
- [x] Register a tool `reload_agent_cards` when `args.reload` or `args.watch` is True.
- [x] Implementation: call `instance.app.reload_agents()` (or via `AgentApp`).
- [x] Return a boolean (changed or not).

### 6) Documentation
Update:
- [x] `docs/ACP_TESTING.md`: include `--watch` in ACP server examples.
- [x] `docs/ACP_IMPLEMENTATION_OVERVIEW.md`: mention watch/reload availability in server mode.
- [x] CLI docs: `src/fast_agent/cli/commands/README.md` to list `serve --watch/--reload`.

## Edge cases
- [x] **No AgentCards loaded**: `--watch` should be a no-op (same as now).
- [x] **Shared instance scope**: reloading replaces the primary instance; session maps are refreshed (already handled).
- [x] **Connection/request scope**: each session instance should refresh safely; reload should update the instance assigned to that session.
- [x] **Concurrent prompts**: ACP already blocks per-session overlap; MCP tool should respect request concurrency (use existing locks).

## Validation checklist
- [x] `fast-agent serve --transport acp --card ./agents --watch` reloads on file change.
- [x] `fast-agent serve --transport acp --card ./agents --reload` responds to `/reload`.
- [x] `fast-agent serve --transport http --card ./agents --watch` reloads on file change.
- [x] MCP tool `reload_agent_cards` is exposed when `--reload`/`--watch` is set.
- [x] No regressions in `go` behavior.
