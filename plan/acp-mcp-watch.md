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
1) `fast-agent serve --card <dir> --watch` automatically reloads AgentCards on change.
2) `fast-agent serve --card <dir> --reload` enables `/reload` (manual) from ACP slash command or MCP command hook.
3) Behavior mirrors `go` where possible:
   - Watcher only runs when AgentCards were loaded and `--watch` is set.
   - Reload is safe in both shared and per-session scopes.

## Implementation plan

### 1) CLI: add flags to `serve`
File: `src/fast_agent/cli/commands/serve.py`
- Add options:
  - `reload: bool = typer.Option(False, "--reload", help="Enable manual AgentCard reloads (/reload)")`
  - `watch: bool = typer.Option(False, "--watch", help="Watch AgentCard paths and reload")`
- Pass through to `run_async_agent(..., reload=reload, watch=watch)`.

### 2) CLI: pass flags through server runner
File: `src/fast_agent/cli/commands/go.py`
- `run_async_agent()` already accepts `reload` and `watch`.
- Ensure `serve` path forwards them (no other changes needed).

### 3) FastAgent: start watcher in server mode
Already implemented in `FastAgent.run()`:
- Watcher starts when `args.watch` is True and `_agent_card_roots` is non-empty.
- `args.watch` is only set by CLI or programmatic calls.
No code change required beyond setting `args.watch` via CLI.

### 4) Manual reload in ACP
Behavior:
- `/reload` is provided by ACP slash command handler.
- It calls the `load_card_callback`/`reload` hooks supplied by `FastAgent`.
Ensure the reload callback is set for server mode:
- In `FastAgent.run()`, `wrapper.set_reload_callback(...)` runs when `args.reload` or `args.watch` is True.
No change required if `args.reload` is passed through.

### 5) Manual reload in MCP
There is no built-in MCP slash command. Options:
1) Add an MCP tool, e.g., `reload_agent_cards`, that calls `AgentApp.reload_agents()`.
2) Expose reload via an MCP resource or prompt (lower priority).

Spec choice: **Add MCP tool**.
- File: `src/fast_agent/mcp/server/agent_server.py`
- Register a tool `reload_agent_cards` when `args.reload` or `args.watch` is True.
- Implementation: call `instance.app.reload_agents()` (or via `AgentApp`).
- Return a boolean (changed or not).

### 6) Documentation
Update:
- `docs/ACP_TESTING.md`: include `--watch` in ACP server examples.
- `docs/ACP_IMPLEMENTATION_OVERVIEW.md`: mention watch/reload availability in server mode.
- CLI docs: `src/fast_agent/cli/commands/README.md` to list `serve --watch/--reload`.

## Edge cases
- **No AgentCards loaded**: `--watch` should be a no-op (same as now).
- **Shared instance scope**: reloading replaces the primary instance; session maps are refreshed (already handled).
- **Connection/request scope**: each session instance should refresh safely; reload should update the instance assigned to that session.
- **Concurrent prompts**: ACP already blocks per-session overlap; MCP tool should respect request concurrency (use existing locks).

## Validation checklist
- `fast-agent serve --transport acp --card ./agents --watch` reloads on file change.
- `fast-agent serve --transport acp --card ./agents --reload` responds to `/reload`.
- `fast-agent serve --transport http --card ./agents --watch` reloads on file change.
- MCP tool `reload_agent_cards` is exposed when `--reload`/`--watch` is set.
- No regressions in `go` behavior.
