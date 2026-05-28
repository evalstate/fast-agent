# ACP OAuth Handling Summary + Next Plan (`hot-mcp-acp-oauth`)

Date: 2026-02-10

## Scope

This note summarizes ACP-specific OAuth behavior for runtime MCP `/mcp connect`, what we changed, what failed, and what to do next.

---

## 1) Current ACP OAuth flow (as implemented)

Runtime path:

1. ACP client sends slash command `/mcp connect <target>`
2. `SlashCommandHandler._handle_mcp()` executes connect
3. Calls `mcp_runtime.handle_mcp_connect(...)`
4. Builds `MCPAttachOptions` and routes into:
   - `MCPAggregator.attach_server(...)`
   - `MCPConnectionManager.get_server()/reconnect_server()`
   - `build_oauth_provider(...)`
5. OAuth events (`authorization_url`, `wait_start`, `wait_end`, `oauth_error`) flow back to slash handler progress reporting.

ACP progress transport now uses **tool-call updates** (`ToolCallStart`/`ToolCallProgress`) for `/mcp connect` instead of plain `agent_message_chunk` only.

---

## 2) What we changed (chronological, ACP-relevant)

## A. OAuth timeout separation + event model

- Added OAuth event model in `oauth_client.py`
- Separated startup timeout from OAuth wait budget in connection manager
- Added better startup vs OAuth timeout messaging

## B. ACP visibility fixes

- `/mcp connect` now sends OAuth progress to ACP client
- Final slash response includes OAuth link
- Switched ACP connect progress to tool-call style notifications (matching hf-inference-acp style)
- Ensured waiting-status updates include OAuth URL (so URL is still visible when clients only render latest update)

## C. ACP safety fix for stdin fallback

- Disabled OAuth paste-from-stdin fallback for ACP/non-interactive progress-hook paths
- If local callback server cannot be used in ACP mode, connect fails fast with explicit error

## D. Cancellation hardening (cross-cutting)

- Cancellation wiring added from Ctrl+C to in-flight connect task
- Scoped SIGINT handler around TUI `/mcp connect` to cancel connect task (not whole process)
- Cleanup in connection manager on cancellation/removal of in-flight server entries
- OAuth callback waiting is abortable via `abort_event`
- Suppressed expected MCP OAuth cancellation traceback spam

---

## 3) Symptoms observed and what they mean

## Symptom 1: “ACP shows waiting but no URL”

- Cause: later progress updates replaced URL-only updates in some ACP clients
- Fix: include URL directly in waiting-status progress text

## Symptom 2: “OAuth complete but nothing happens”

Most likely cause now:

- Redirect URI is loopback (`127.0.0.1:<port>/callback`)
- Browser used for auth is not reaching the fast-agent host callback listener
- In ACP mode, stdin paste fallback is intentionally disabled, so there is no fallback completion path

This is a topology/connectivity issue, not usually slash-command routing.

## Symptom 3: “Generation cancelled by user then exits shell”

- This was separate from OAuth; likely Ctrl+C residue/signal propagation interaction
- mitigated with prompt-loop handling and cancellation guards

---

## 4) Comparison with `publish/hf-inference-acp` `/connect`

`hf-inference-acp` `/connect` behavior:

- Emits `ToolCallStart` + `ToolCallProgress` updates for all connection stages
- Clients generally render this better than generic message chunks

Runtime `/mcp connect` now follows that tool-call update pattern for ACP progress.

---

## 5) Current behavior matrix

## TUI local

- OAuth URL printed to console
- Waiting status printed
- Ctrl+C should cancel connect and return to prompt
- Paste fallback still available when callback server unavailable

## ACP

- OAuth URL shown in tool-call progress + final response
- Waiting status shown in tool-call progress
- Paste fallback disabled (by design)
- Completion depends on callback endpoint reachability from browser

---

## 6) Known danger areas introduced

1. Scoped SIGINT handler swap in TUI `/mcp connect`
   - narrow scope, restored in `finally`
2. Prompt STOP suppression logic after interruption/cancellation
   - tuned to reduce accidental exits
3. Outer interactive cancellation recovery using `uncancel()`
   - powerful; should remain narrowly scoped
4. OAuth cancellation log filter
   - suppresses expected cancel noise only; keep narrow

---

## 7) Recommended next strategy (ACP)

## Immediate operator diagnostics (no code)

1. Verify callback listener is up on fast-agent host during connect
2. Verify browser can reach the callback host/port used in redirect URI
3. If remote host, use SSH/local port forwarding for callback port

## Product-level fix (recommended)

Implement **ACP manual completion** path to avoid loopback dependence:

- `/mcp connect` returns URL + pending auth context
- Add `/mcp connect-complete <callback-url-or-code>`
- Server completes token exchange from ACP-provided callback payload
- Works even when browser and fast-agent host differ

This removes the largest ACP OAuth fragility.

---

## 8) Proposed implementation slices

## Phase 1 — Better ACP diagnostics

- Include callback URI in ACP progress text explicitly
- Add actionable failure text when callback server is unreachable and paste fallback disabled

## Phase 2 — Manual ACP completion command

- Add pending-oauth state object per session/server
- Add slash command `/mcp connect-complete ...`
- Validate/parse callback URL and finalize flow

## Phase 3 — Hardening

- Add unit tests for manual completion success/failure/cancel
- Add ACP integration tests for remote-browser topology simulation

---

## 9) Test checklist for ACP OAuth

1. `/mcp connect` emits `ToolCallStart`
2. OAuth URL appears in progress update
3. Waiting update includes URL
4. Final slash response includes URL
5. ACP cancel stops connect promptly
6. Callback unreachable path fails with explicit ACP-safe error
7. (Future) `/mcp connect-complete` succeeds without loopback callback

