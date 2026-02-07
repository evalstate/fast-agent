---
title: "ACP Auth Methods Survey (Spec, Registry, SDKs)"
status: draft
---

# ACP Auth Methods Survey (Spec, Registry, SDKs)

This is a compact survey of the current ACP auth surface area and how the referenced repos implement it.

Repos pulled for reference:

- `/home/tools/acp/agent-client-protocol` (spec, schema, RFDs)
- `/home/tools/acp/python-sdk` (Python ACP SDK)
- `/home/tools/acp/registry` (ACP registry implementation + requirements)
- `/home/tools/acp/codex-acp` (real-world client/agent that exercises authMethods + authenticate)

## 1. Protocol: What Exists Today (Spec)

Spec/schema source of truth:

- `/home/tools/acp/agent-client-protocol/schema/schema.json`
- `/home/tools/acp/agent-client-protocol/docs/protocol/schema.mdx`

Key points:

1. `initialize` response includes `authMethods: AuthMethod[]` (default `[]`).
2. `AuthMethod` is currently:
   - `id: string`
   - `name: string`
   - `description?: string`
   - `_meta?: object`
3. `authenticate` is a defined agent method:
   - request: `{ "methodId": "..." }`
   - response: `{}` (empty object)
4. ACP error objects are schema-defined as `{code, message, data?}`.
   - Any extra fields (like a top-level `authMethods` on the error) are not schema-defined today.

Implication for fast-agent:

- We can be fully spec-compliant by:
  - advertising `authMethods` during `initialize`, and
  - implementing `authenticate(methodId)`.

## 2. RFD: "Authentication Methods" (Draft Direction)

Draft RFD:

- `/home/tools/acp/agent-client-protocol/docs/rfds/auth-methods.mdx`

Why it exists:

- The base `AuthMethod` shape does not tell the client what UI/action to offer (env var key entry vs OAuth vs terminal).

Proposed extensions (not yet in the generated SDK models):

1. `type: "agent"` (default if `type` missing)
2. `type: "env_var"` with:
   - `varName`
   - `link?`
3. `type: "terminal"` with:
   - `args: string[]`
   - `env: Record<string,string>`
   - note: `command` cannot be specified (client launches the same binary)

RFD also discusses adding auth method ids to `AUTH_REQUIRED` errors; the schema currently only supports `data`, so any such
payload must be nested under `error.data` until the protocol changes.

## 3. Registry: What Is Enforced Today

Registry implementation:

- `/home/tools/acp/registry/agent.schema.json` (agent entry schema)
- `/home/tools/acp/registry/AUTHENTICATION.md` (inclusion requirements)
- Published aggregate registry JSON:
  - `curl https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json`

Current format:

- Top-level keys: `version`, `agents[]`, `extensions[]`.
- `agents[]` entries currently include:
  - `id`, `name`, `version`, `description`, `distribution`
  - optional `authors`, `license`, `repository`, `icon`

Auth requirement (important):

- To be listed, an agent must support at least one of:
  - Agent Auth
  - Terminal Auth
- The registry explicitly says these are the only supported auth methods for now.

Notes:

- Registry auth docs describe Agent Auth primarily as an OAuth flow (local HTTP server + browser).
- This is stricter than the RFD's "agent handles auth" phrasing; treat it as "registry policy" rather than "protocol law".

## 4. Python SDK: What It Can Represent Today

Python SDK:

- `/home/tools/acp/python-sdk/src/acp/schema.py`
- `/home/tools/acp/python-sdk/src/acp/exceptions.py`

Observations:

1. `AuthMethod` model only has `id`, `name`, `description`, `_meta`.
   - No typed fields (`type`, `varName`, `args`, `env`) yet.
2. `RequestError.auth_required()` produces an error object with:
   - `code=-32000`, `message="Authentication required"`, `data=<optional>`
   - This maps to the schema's error object shape.

Practical takeaway:

- If we want to ship typed auth methods *before* SDK support, `_meta` is the safe, forward-compatible channel for extra
  fields (e.g. `_meta.auth = {type: ..., ...}`).

## 5. Codex ACP: Real Usage Pattern

Codex agent implementation (Rust):

- `/home/tools/acp/codex-acp/src/codex_agent.rs`

What it does:

1. Advertises a non-empty list of auth methods in `initialize()`:
   - ChatGPT login
   - API key based methods (Codex API key, OpenAI API key)
2. Implements `authenticate()` and performs the actual flow:
   - Browser/device login for ChatGPT
   - API key login via environment variables for key-based methods
3. Uses environment variables as a gating mechanism (e.g. `NO_BROWSER`) to disable browser auth where it cannot work.

Notably:

- It does not implement the RFD's `env_var` typed metadata; instead it relies on `description` text to tell the user what
  env var to set.

This is a good baseline precedent for fast-agent:

- Start by advertising a small number of methods with clear descriptions.
- Implement `authenticate` conservatively and fail with clear errors when prerequisites are missing.

## 6. Recommended fast-agent Strategy (From Survey)

1. Phase 1 (safe): advertise one `AuthMethod` in ACP mode that points users at fast-agent config/secrets docs.
2. Implement `authenticate()` with explicit error messages for unsupported/unknown method ids.
3. Phase 2 (registry-aligned): add a `terminal` auth story with a single stable setup entrypoint.
4. Phase 3 (typed metadata): if we want to ship `env_var`/`terminal` metadata early, encode it in `_meta` until ACP SDKs
   add first-class fields.

