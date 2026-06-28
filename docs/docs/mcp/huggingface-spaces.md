---
title: Host MCP Servers on Hugging Face Spaces
social:
  title: Host MCP Servers on Hugging Face Spaces
  tagline: Deploy fast-agent MCP servers with Hugging Face OAuth.
  description: Deploy fast-agent MCP servers with Hugging Face OAuth.
  alt: fast-agent social card — Host MCP Servers on Hugging Face Spaces
---

# Host MCP Servers on Hugging Face Spaces

Hugging Face Spaces are a good home for fast-agent MCP servers: they provide
hosting, OAuth, user-scoped access tokens, and scopes for Hub capabilities such
as inference providers, jobs, and repositories.

## Recommended shape

For hosted MCP, prefer request-scoped serving:

```bash
fast-agent serve \
  --transport http \
  --host 0.0.0.0 \
  --port 7860 \
  --instance-scope request
```

Request scope means every MCP tool call opens a transient harness session.
Durable application state should live in storage you control: buckets, repos,
databases, or job records keyed by the authenticated user and your own handles.

Use connection scope only when an MCP client needs conversation continuity from
the ambient `Mcp-Session-Id`.

## Enable Sign in with Hugging Face

Add OAuth metadata to the Space `README.md`:

```yaml
---
title: My fast-agent MCP server
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - inference-api
  - jobs
  - contribute-repos
---
```

Spaces always include `openid` and `profile`. Add only the scopes your tools
need.

Common scopes:

| Scope | Use it when your MCP tools need to... |
| --- | --- |
| `inference-api` | call Hugging Face Inference Providers on behalf of the user |
| `jobs` | create or manage Hugging Face Jobs |
| `contribute-repos` | create repos and access repos created by this OAuth app |
| `write-repos` | read/write the user's personal repos |
| `manage-repos` | create, delete, and fully manage the user's personal repos |
| `read-billing` | check whether the user has billing configured |

## Server environment

Spaces provide these OAuth environment variables:

- `OAUTH_CLIENT_ID`
- `OAUTH_CLIENT_SECRET`
- `OAUTH_SCOPES`
- `OPENID_PROVIDER_URL`
- `SPACE_HOST`

Configure fast-agent's inbound MCP auth:

```bash
export FAST_AGENT_SERVE_OAUTH=huggingface
export FAST_AGENT_OAUTH_RESOURCE_URL="https://${SPACE_HOST}"
export FAST_AGENT_OAUTH_SCOPES="${OAUTH_SCOPES}"
```

When enabled, fast-agent accepts Hugging Face bearer tokens from:

- `Authorization: Bearer ...`
- `X-HF-Authorization: Bearer ...`

The second form is normalized at the protocol edge for Hugging Face clients and
Spaces infrastructure.

## Minimal Dockerfile

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.13-bookworm

WORKDIR /app
COPY . .

RUN uv sync --frozen

ENV FAST_AGENT_SERVE_OAUTH=huggingface

CMD uv run fast-agent serve \
  --transport http \
  --host 0.0.0.0 \
  --port 7860 \
  --instance-scope request
```

If you use an `agent.py` application instead of `fast-agent serve`, use the same
transport options:

```bash
uv run agent.py \
  --transport http \
  --host 0.0.0.0 \
  --port 7860 \
  --instance-scope request
```

## Auth in tools

The verified Hugging Face caller is translated into `AgentAuth` for the harness
request. Use verified identity fields for resource ownership. Do not use raw
bearer tokens as durable user ids.

For privileged tools, prefer per-tool FastMCP auth checks:

```python
@mcp.tool(auth=requires_jobs_scope)
async def launch_job(repo: str, ctx: MCPContext) -> dict[str, str]:
    response = await adapter.invoke_agent(
        ctx=ctx,
        agent="job_runner",
        arguments={"repo": repo},
    )
    return {"status": "started", "summary": response.text_content()}
```

## State and storage

Request-scoped serving deliberately avoids hidden retained agent state.

For multi-step workflows:

1. create an explicit handle, such as `job_id`, `review_id`, or `workspace_id`;
2. store durable state in your own storage;
3. include the handle in backend MCP tool schemas;
4. use the authenticated subject/client id to isolate user data.

Example:

```python
@mcp.tool()
async def job_status(job_id: str, ctx: MCPContext) -> dict[str, str]:
    return await load_status_for_user(ctx, job_id)
```

## Organization resources

Hugging Face OAuth lets users grant access to selected organizations for scopes
such as repo or billing access. Treat organization access as an OAuth grant, not
as a fast-agent session feature.

If your app requires a specific organization, direct users through the normal
Hugging Face organization grant flow.

## Security checklist

- Use request scope for public hosted MCP unless you need connection continuity.
- Request the smallest OAuth scope set that works.
- Keep `HF_TOKEN` service credentials out of user-scoped tool paths unless the
  Space is private/trusted.
- Prefer verified identity claims over raw tokens for resource keys.
- Use explicit handles for durable application state.
- If serving shell/filesystem tools, keep the Space private or tightly scoped.

