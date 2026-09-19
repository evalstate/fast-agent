# Copilot direct-inference probe

This is an experimental diagnostic, not a registered fast-agent provider. It uses
Copilot SDK 1.0.14 and an installed Copilot CLI to discover the endpoint for a
model on your account. The SDK API is experimental and requires
`COPILOT_ALLOW_GET_PROVIDER_ENDPOINT=true`; the script sets this for its child
process. CLI 1.0.86 was used when preparing the probe.

## Credentials: sign in locally, not in chat or YAML

Install [Copilot CLI](https://github.com/github/copilot-cli#installation), then:

```bash
copilot login --device-code
```

Follow the URL/code shown by the CLI. The CLI stores the credential in the system
credential store, with a plaintext file under `~/.copilot` as its documented
fallback when secure storage is unavailable. Keep that directory private.
There is no need to copy tokens into fast-agent configuration or send them to
anyone. Subscription entitlement, model availability, and organizational policy
still apply.

The probe reuses that stored login. By default it removes `COPILOT_GITHUB_TOKEN`,
`GH_TOKEN`, `GITHUB_TOKEN`, `GITHUB_COPILOT_API_TOKEN`, and `COPILOT_API_URL` from
its child environment so an unrelated token/proxy override does not silently
change the test. It leaves your actual shell environment untouched.

For headless use, `--use-env-token` explicitly selects the first nonempty variable
among `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, and `GITHUB_TOKEN`. Use an appropriately
authorized OAuth token or a fine-grained PAT with **Copilot Requests** permission;
classic `ghp_` PATs are unsupported. Never pass a token as a command-line argument.

## Run from the repository root

```bash
# Authentication check and available model IDs; no inference.
uv run scripts/probe_copilot_endpoint.py

# Replace MODEL_ID with an ID returned above. Still no inference.
uv run scripts/probe_copilot_endpoint.py --model MODEL_ID

# Explicitly send ONE small, direct HTTP inference request (uses plan quota).
uv run scripts/probe_copilot_endpoint.py --model MODEL_ID --infer
```

`uv run` installs the script's pinned dependencies into an isolated environment;
no fast-agent dependency/configuration changes are needed. Use `--cli /path/to/copilot`
if the CLI is not on PATH. Output tokens default to 128; `--max-tokens` adjusts the
limit, which may need increasing for reasoning models.

## What it does and does not prove

The probe starts a local CLI runtime in an empty temporary working directory,
without available tools, permission approvals, repository configuration discovery,
file hooks, skills, or prompts submitted to Copilot's agent loop. It queries
`session.provider.getEndpoint` for the explicit model.

Discovery prints the provider family, wire API, transport, hostname, header
**names**, and whether credentials exist. It never prints credential values,
raw endpoint objects, full URLs, response bodies, or raw exceptions.

With `--infer`, it sends a fixed “Reply with exactly the word OK” prompt using
Chat Completions, Responses, or Anthropic Messages according to that snapshot.
All returned headers and any additional model-bound session credential accompany
the request. It uses HTTPS, does not follow redirects, ignores environment HTTP
proxies, and makes no automatic retries. No tool call or repository content is
sent. A successful HTTP result with the expected output-list shape is reported
as `direct_inference: accepted`; this does not prove useful text was generated,
streaming works, or a tool round-trip succeeds.

The probe currently handles HTTP only and rejects credential/query-bearing base
URLs. WebSocket endpoints or Azure-specific routing may require additional work.
Raw errors are intentionally withheld, so the reported HTTP status may not fully
explain a rejected request. Do not enable wire logging or dump SDK endpoint
objects to diagnose authentication problems.

A production integration would additionally need credential/session-token refresh,
capability discovery, streaming/tool tests, provider-specific routing, and lifecycle
integration. A newly discovered session token is used immediately here; the probe
does not retry authentication failures or cache tokens to disk.

## Sources

- [SDK authentication](https://github.com/github/copilot-sdk/blob/main/docs/auth/authenticate.md)
- [Python endpoint-discovery tests](https://github.com/github/copilot-sdk/blob/main/python/e2e/test_provider_endpoint_e2e.py)
- [Generated RPC contract](https://github.com/github/copilot-sdk/blob/main/python/copilot/generated/rpc.py)
