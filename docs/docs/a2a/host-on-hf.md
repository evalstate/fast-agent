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

The implemented first pass supports static bearer credentials and Hugging Face
Space header normalization. Browser-based OAuth login for A2A clients is a later
phase.

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
3. The client sends A2A requests to `/a2a/jsonrpc` or `/a2a/rest` with either:
   - `Authorization: Bearer <hf-token>`;
   - `X-HF-Authorization: Bearer <hf-token>` when running through Hugging Face
     Space infrastructure.
4. The A2A server normalizes the Hugging Face header and validates that a bearer
   credential is present.
5. fast-agent stores the token in request context while the agent runs.
6. Hugging Face Inference Provider model calls and Hugging Face MCP/tool calls
   can use the request token.

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

Skills include the same `securityRequirements` entry. Later implementations can
advertise OAuth2 or OpenID Connect metadata when the client can complete the
browser OAuth flow directly from the AgentCard.

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

For Hugging Face Space routing, clients may also need:

```yaml
headers:
  Authorization: "Bearer ${HF_TOKEN}"
  X-HF-Authorization: "Bearer ${HF_TOKEN}"
```

fast-agent A2A clients reuse the existing Hugging Face token discovery used by
MCP URL connections, so explicit headers are not needed when the target is
`hf.co`, `huggingface.co`, or `*.hf.space` and no auth header has already been
configured.

## Inference Provider Use

With request token pass-through, hosted A2A agents can use Hugging Face models
without putting a shared user token in the Space:

```yaml
name: researcher
type: basic
model: hf.moonshotai/Kimi-K2-Thinking
instruction: |
  Answer with concise Markdown.
  Use Hugging Face tools when current Hub context is needed.
mcp_servers:
  - name: huggingface
    target: "https://huggingface.co/mcp"
```

When the A2A request arrives with a user bearer token, both provider calls and
the Hugging Face MCP server should be able to use that token through the normal
fast-agent request auth context.

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
- `Authorization` and `X-HF-Authorization` both propagate to request context;
- the AgentCard advertises security schemes and requirements;
- client-side Hugging Face token auto-headers are added for HF URLs;
- explicit user-supplied auth headers are preserved.
