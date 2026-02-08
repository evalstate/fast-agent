---
title: "ACP Registry + Auth Methods Support"
status: draft
---

# ACP Registry + Auth Methods Support (fast-agent)

This document is a fast-agent oriented implementation plan for:

1. ACP Registry compatibility (current implementation at `agentclientprotocol/registry`).
2. ACP Authentication Methods RFD: `docs/rfds/auth-methods.mdx` in `agentclientprotocol/agent-client-protocol`.

Local references (cloned for review):

- `/home/tools/acp/agent-client-protocol`
- `/home/tools/acp/registry`
- `/home/tools/acp/python-sdk`
- `/home/tools/acp/codex-acp`

## Why This Matters

The ACP Registry get-started docs state the registry is curated and currently includes only agents that "support authentication".

In the current registry implementation (`/home/tools/acp/registry/AUTHENTICATION.md`), to be listed an agent must support at least:

- Agent Auth (`type: "agent"`; default if missing)
- Terminal Auth (`type: "terminal"`)

The Auth Methods RFD (`/home/tools/acp/agent-client-protocol/docs/rfds/auth-methods.mdx`) explains why the base `AuthMethod`
shape is insufficient and proposes richer, typed auth methods.

fast-agent already runs as an ACP agent (`src/fast_agent/acp/`), but currently advertises no auth methods:

- `/home/tools/fast-agent/src/fast_agent/acp/server/agent_acp_server.py` sets `auth_methods=[]` in `initialize()`.
- No `authenticate()` handler is implemented in `AgentACPServer` (so clients cannot trigger a login flow).

## Current Spec Constraints (Important)

1. ACP schema `AuthMethod` currently only defines: `id`, `name`, optional `description`, optional `_meta`.
   - See `/home/tools/acp/agent-client-protocol/schema/schema.json` (`$defs.AuthMethod`).
2. The Auth Methods RFD proposes adding fields like `type`, `varName`, `args`, `env`.
   - These fields are not present in the generated Python SDK models (`/home/tools/acp/python-sdk/src/acp/schema.py`).
3. JSON Schema for `AuthMethod` does not declare `additionalProperties: false`, so adding new fields is schema-valid.
   - But the Python SDK model will drop unknown fields on serialization, so use `_meta` if we need to transmit typed data
     before the SDK is updated.
4. ACP error objects are schema-limited to `{code, message, data?}`.
   - The RFD suggests adding `authMethods` at the top-level of the error; that is not compatible with the current schema.
   - If we want to return auth method hints today, they must go in `error.data`.

## Goal

Make fast-agent:

1. Advertise at least one auth method in ACP mode (starting with the simplest: Agent Auth).
2. Provide a predictable `authenticate()` behavior so ACP clients can trigger the flow.
3. (Optional) Add Terminal Auth support in a way that matches current registry expectations.
4. (Optional) Prepare a clean path to add typed methods (`env_var`, `terminal`) via `_meta` until ACP SDKs formalize them.

## Phase 1: Minimal "Agent Auth" (Default Type)

This is the lowest-risk, spec-compatible starting point.

### 1.1 Advertise a single AuthMethod in initialize()

Implement in:

- `/home/tools/fast-agent/src/fast_agent/acp/server/agent_acp_server.py`

Change `InitializeResponse(... auth_methods=[])` to advertise one method, for example:

- `id`: `fast-agent-config`
- `name`: `Configure fast-agent`
- `description`: "Set provider keys in fastagent.secrets.yaml or env vars. See docs: Configuration Reference"

Notes:

- Do not include `type` yet; per the RFD, missing `type` implies `"agent"`.
- Keep the description short and action-oriented, because this is what most ACP clients surface.

### 1.2 Implement authenticate(method_id)

Implement `authenticate()` in `AgentACPServer` and keep behavior conservative:

- If `method_id` is not recognized: raise `RequestError.invalid_params(...)`.
- If recognized:
  - Option A (purely informative): raise `RequestError.auth_required({"hint": "..."})` explaining that configuration is
    file/env driven, and point to docs + suggested commands (e.g. `fast-agent check`, `fast-agent auth status`).
  - Option B (best effort): run a local interactive helper if present (future phase), otherwise return OK.

Recommendation: start with Option A so the client gets a clear message when the user clicks "Authenticate".

### 1.3 (Optional) Add a clean `/status auth` UX in ACP

fast-agent already has a CLI `fast-agent auth` command family (OAuth tokens for MCP servers and Codex OAuth).
If we want to make "agent auth" more useful for ACP clients, add/extend ACP slash commands to:

- Show current auth/config readiness (keys present, keyring available, etc.)
- Provide copy-pastable instructions for fixing the most common missing bits.

This stays session-scoped, which ACP clients are better at rendering than a global authenticate response.

## Phase 2: Terminal Auth (Registry-Targeted)

The registry currently supports `terminal` as a first-class option.

We should implement Terminal Auth in a way that:

- Does not require ACP schema changes.
- Is robust across clients.

### 2.1 Pick a single, stable setup entrypoint

Candidate entrypoints (fast-agent already has these building blocks):

- `fast-agent auth codexplan` (Codex OAuth)
- `fast-agent auth login <server-or-url>` (OAuth for MCP servers)
- A new dedicated `fast-agent setup` command (wizard that can:
  - validate config,
  - prompt for provider keys,
  - store in keyring or write `fastagent.secrets.yaml`).

For registry compatibility, a single `--setup` style argument on the ACP binary is the cleanest target.

### 2.2 Advertise Terminal Auth via authMethods

We cannot safely serialize extra fields today via the Python SDK model, so encode type payload via `_meta` until ACP SDKs
ship the typed fields.

Example (conceptually):

- `AuthMethod(id="terminal-setup", name="Run setup in terminal", description="Interactive terminal setup")`
- `_meta` includes:
  - `{"auth": {"type": "terminal", "args": ["--setup"], "env": {...}}}`

Clients that understand the RFD can consume `_meta.auth` immediately. Others can still display `name/description`.

### 2.3 Implement the setup binary behavior

If the client launches the agent with `--setup` (or equivalent), fast-agent should:

- Run a focused interactive flow.
- Persist results (keyring or secrets file).
- Exit with a non-zero status on failure (so clients can show failure).

## Phase 3: Env Var Auth (Nice-to-Have, Not Registry-Supported Yet)

The RFD proposes `env_var` but the registry explicitly does not list it as supported today.

Still useful for local workflows and for clients that can set env vars and restart:

- Provide method metadata via `_meta`:
  - `varName` for the env var to set
  - optional link to obtain the key
- In fast-agent, validate at runtime:
  - If key missing, return `RequestError.auth_required({"authMethods": [...]})` (in `data`) so clients can hint.

## Phase 4: Registry Listing (Optional, After Phases 1-2)

Once fast-agent has an auth story that fits the registry expectations, submit it to the registry:

- Add an entry in `agentclientprotocol/registry` following `agent.schema.json`.
- Distribution recommendation: `uvx` targeting `fast-agent-acp` (already published as a convenience entrypoint in fast-agent).

Open question:

- The current registry schema is minimal (no capabilities/auth in `agent.json`), while the older "ACP Agent Registry" RFD
  describes a richer manifest. We should follow the current `agent.schema.json` and `AUTHENTICATION.md` requirements.

## Testing Plan (fast-agent)

1. Unit: validate `initialize()` includes at least one `authMethods` entry when `--transport acp`.
2. Integration (ACP): extend `tests/integration/acp/` with:
   - Auth methods present in initialize response.
   - Calling `authenticate` returns the expected error/response for unknown vs known method ids.
3. (If Terminal Auth implemented) add an e2e-ish test that runs the setup entrypoint and verifies persistence target.

