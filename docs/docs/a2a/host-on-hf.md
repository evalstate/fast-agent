---
title: Host A2A on Hugging Face
description: Deploy a fast-agent A2A server on Hugging Face Spaces with OAuth credential pass-through.
---

# Host A2A on Hugging Face

This page describes the target deployment shape for hosting fast-agent as an A2A
server on Hugging Face Spaces.

The important behavior is credential pass-through: the caller authenticates to
the hosted A2A server with a Hugging Face OAuth/bearer credential, and
fast-agent makes that credential available to the running agent. That lets the
agent use Hugging Face Inference Provider models, the Hugging Face MCP server,
and Hugging Face tools as the caller rather than as a shared server account.

## Current Status

fast-agent A2A serving supports Hugging Face bearer authentication for HTTP
`JSONRPC` and `HTTP+JSON` routes when `FAST_AGENT_SERVE_OAUTH=huggingface` is
set. The public AgentCard stays discoverable, and action routes require a bearer
token.

The implemented first pass supports Hugging Face bearer credentials and Space
header normalization. Served fast-agent A2A cards currently advertise bearer
security rather than OAuth2/OIDC security metadata; the fast-agent A2A client can
still use browser OAuth when a remote AgentCard advertises OAuth2 or OpenID
Connect.

## Space Layout

A minimal Space should contain:

```text
.
├── app.py
├── fast-agent.yaml
├── agents/
│   └── researcher.yaml
└── requirements.txt
```

`requirements.txt`:

```text
fast-agent-mcp
```

`app.py`:

```python
import os

from fast_agent.cli.main import app


if __name__ == "__main__":
    os.environ.setdefault("FAST_AGENT_SERVE_OAUTH", "huggingface")
    os.environ.setdefault("FAST_AGENT_OAUTH_SCOPES", "access")
    app()
```

Start the Space with:

```bash
fast-agent serve a2a \
  --host 0.0.0.0 \
  --port 7860 \
  --agent-cards ./agents \
  --model hf.moonshotai/Kimi-K2-Thinking
```

Use the model/provider alias that matches your application. The key point is
that Hugging Face provider credentials should come from the request token, not
from a shared `HF_TOKEN`, when OAuth pass-through is enabled.

## Space Environment

Set these environment variables in the Space:

```text
FAST_AGENT_SERVE_OAUTH=huggingface
FAST_AGENT_OAUTH_RESOURCE_URL=https://<space-subdomain>.hf.space
FAST_AGENT_OAUTH_SCOPES=access
```

Do not set a shared `HF_TOKEN` unless the Space intentionally needs a server
credential fallback. For user-scoped inference, the inbound bearer credential is
the credential source.

## Request Flow

OAuth-enabled A2A flow:

1. The client fetches `/.well-known/agent-card.json`.
2. The AgentCard advertises bearer/OAuth security requirements.
3. The client sends A2A requests to `/a2a/jsonrpc` or `/a2a/rest` with
   `Authorization: Bearer <hf-token>`, or uses OAuth when the card advertises an
   OAuth/OIDC challenge.
4. The A2A server validates that a bearer credential is present.
5. fast-agent stores the token in request context while the agent runs.
6. Hugging Face Inference Provider model calls and Hugging Face MCP/tool calls
   can use the request token.

Do not confuse this with fast-agent's ambient Hugging Face client policy. The
normal CLI can add a discovered local `HF_TOKEN` to Hugging Face URLs without an
explicit `--auth`: it sends `Authorization` to `hf.co` and `huggingface.co`, and
`X-HF-Authorization` to ordinary `*.hf.space` app URLs. That protects local
tokens from being sent as app-level `Authorization` to arbitrary Space apps.
When the Space endpoint itself is the authenticated A2A or MCP server, use
explicit endpoint auth instead: `--auth`, checked-in `headers: Authorization:
...`, or OAuth.

## AgentCard Security

An OAuth-enabled card should advertise security metadata so A2A clients know
that credentials are required.

The current implementation advertises bearer security:

```json
{
  "securitySchemes": {
    "hf_bearer": {
      "httpAuthSecurityScheme": {
        "scheme": "bearer",
        "bearerFormat": "HF_TOKEN",
        "description": "Hugging Face bearer token"
      }
    }
  },
  "securityRequirements": [
    {
      "schemes": {
        "hf_bearer": {}
      }
    }
  ]
}
```

Skills include the same `securityRequirements` entry. fast-agent A2A clients can
also use the existing browser OAuth flow when a remote AgentCard advertises
OAuth2 or OpenID Connect security schemes.

## Client Configuration

For a checked-in fast-agent A2A client card, explicit bearer headers remain the
most direct option:

```yaml
type: a2a
name: hf_space_agent
url: https://<space-subdomain>.hf.space
transport: JSONRPC
headers:
  Authorization: "Bearer ${HF_TOKEN}"
```

This is endpoint authentication: the Space-hosted A2A server is the protected
resource, so the standard `Authorization` header is the right header. The
ambient `X-HF-Authorization` Space policy is for ordinary Space apps, not for
authenticating to an A2A action route that advertises bearer/OAuth security.
When a fast-agent A2A client connects to a `*.hf.space` URL and the public
AgentCard advertises HTTP bearer security, a discovered local Hugging Face token
is automatically promoted to endpoint `Authorization` unless explicit headers
were configured.

For AgentCards that advertise OAuth2 or OpenID Connect instead of a static
bearer scheme, enable browser OAuth explicitly or allow the card to activate it:

```yaml
type: a2a
name: hosted_agent
url: https://<space-subdomain>.hf.space
transport: JSONRPC
auth:
  oauth: true
  persist: keyring
```

The same `--oauth` switch is available from the TUI:

```text
/a2a connect https://<space-subdomain>.hf.space --oauth --name hosted_agent
```

## Inference Provider Use

With request token pass-through, hosted A2A agents can use Hugging Face models
without putting a shared user token in the Space:

```yaml
name: researcher
type: agent
model: hf.moonshotai/Kimi-K2-Thinking
instruction: |
  Answer with concise Markdown.
  Use Hugging Face tools when current Hub context is needed.
mcp_connect:
  - name: huggingface
    target: "https://huggingface.co/mcp?bouquet=hub_repo_details_readme"
    auth:
      forward: huggingface
```

When the A2A request arrives with a user bearer token, Hugging Face provider
calls use that request token before falling back to Space configuration.

For client-managed Hugging Face MCP URLs, set `auth.forward: huggingface` to
forward the same inbound request token to `hf.co`, `huggingface.co`, or
`*.hf.space` upstreams. For Space upstreams, forwarded requests use
`X-HF-Authorization`; for `hf.co` and `huggingface.co`, they use
`Authorization`. This mode is intended for hosted Spaces where the agent should
act as the caller rather than as a shared Space identity. For the Hugging Face
MCP server itself, use `https://huggingface.co/mcp?...`; the forwarded request
token is sent there as `Authorization: Bearer ...`. It preserves explicit
`Authorization`/`X-HF-Authorization` headers when they are configured and
disables OAuth escalation for that MCP connection.

Do not combine `auth.forward: huggingface` with a shared `HF_TOKEN` expectation
for that MCP server: forward mode deliberately avoids capturing the Space
process token during configuration and uses the per-request bearer token at
connection time.

## Operational Notes

- Keep the AgentCard public so clients can discover endpoint and auth metadata.
- Require bearer credentials only on A2A action routes.
- Prefer `--instance-scope connection` for multi-turn authenticated sessions
  where A2A `contextId` should correlate with a fast-agent instance.
- Prefer `--instance-scope request` for stateless public endpoints.
- Use `--host 0.0.0.0` inside the Space; the served AgentCard should advertise
  the external Space hostname when fetched by clients.

## Verification Targets

The OAuth-enabled A2A implementation should include tests that prove:

- unauthenticated A2A requests are rejected;
- authenticated A2A requests reach the agent;
- the verified bearer token is saved at the A2A HTTP boundary and propagated
  into fast-agent request context;
- the AgentCard advertises security schemes and requirements;
- client-side Hugging Face token auto-headers are added for HF URLs;
- explicit user-supplied auth headers are preserved.
