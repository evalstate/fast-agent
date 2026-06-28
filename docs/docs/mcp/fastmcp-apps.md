---
title: FastMCP Apps with fast-agent
social:
  title: FastMCP Apps with fast-agent
  tagline: Build MCP Apps backed by fast-agent harness sessions.
  description: Build MCP Apps backed by fast-agent harness sessions.
  alt: fast-agent social card — FastMCP Apps with fast-agent
---

# FastMCP Apps with fast-agent

FastMCP Apps let an MCP server deliver user interface elements to clients that
support the MCP Apps extension. fast-agent fits behind those apps through the
Harness API.

This page describes **MCP Apps adapter mode**: you own a `FastMCPApp` provider
and call fast-agent from UI entry-point tools or app backend tools. If you only
need normal MCP tools, use [custom tool adapter mode](harness-adapter.md).

```text
FastMCPApp.ui() / FastMCPApp.tool()
└─ HarnessMCPAdapter
   └─ HarnessApp.open(...).invoke(...)
      └─ AgentResponse
```

FastMCP owns the UI surface. fast-agent owns the agent invocation.

## Minimal app

```python
from fastmcp import Context as MCPContext
from fastmcp import FastMCP, FastMCPApp

from fast_agent.mcp.server import HarnessMCPAdapter

mcp = FastMCP("workspace server")
ui = FastMCPApp("Workspace")
adapter = HarnessMCPAdapter(harness_app, options)


@ui.ui()
async def open_workspace(repo: str, ctx: MCPContext):
    response = await adapter.invoke_agent(
        ctx=ctx,
        agent="planner",
        arguments={"repo": repo},
    )
    return build_workspace_component(
        repo=repo,
        summary=response.text_content(),
    )


@ui.tool()
async def apply_plan(repo: str, plan_id: str, ctx: MCPContext) -> str:
    response = await adapter.invoke_agent(
        ctx=ctx,
        agent="developer",
        arguments={"repo": repo, "plan_id": plan_id},
    )
    return response.text_content()


mcp.add_provider(ui)
```

The same adapter works from normal `@mcp.tool()` handlers, MCP App entry-point
tools, and backend app tools.

## What belongs where

FastMCP Apps handle:

- UI metadata;
- tool visibility;
- UI resources;
- CSP and iframe permissions;
- backend app tool routing;
- Prefab/generative UI machinery.

fast-agent handles:

- agent definitions;
- model and tool configuration;
- skills;
- harness session opening;
- `AgentRequest` / `AgentResponse`;
- auth/progress/session context from MCP.

Do not make your harness app depend on FastMCP UI classes. Let the MCP App
handler project a protocol-neutral `AgentResponse` into UI.

## Returning UI from an agent response

Your harness invocation returns `AgentResponse`:

```python
response = await adapter.invoke_agent(
    ctx=ctx,
    agent="reporter",
    arguments={"dataset": dataset},
)
```

The MCP App handler decides how to display it:

```python
return build_report_component(
    title="Dataset report",
    summary=response.text_content(),
    metadata=response.metadata,
    artifacts=response.artifacts,
)
```

For ordinary MCP clients, return text. For MCP Apps clients, return components.
For large outputs, register resources or return resource links.

## HTML, files, and artifacts

If an agent produces HTML or files, keep the harness response neutral:

```python
return AgentResponse.text(
    "I created the report.",
    metadata={"ui_hint": "html_report"},
    artifacts=(report_artifact,),
)
```

Then project it in the MCP App layer:

```python
response = await adapter.invoke_agent(...)
report = find_report_artifact(response.artifacts)
return html_report_component(report.html)
```

This keeps the same agent usable from:

- MCP Apps;
- plain MCP tools;
- A2A;
- ACP;
- local/headless Harness API code.

## Long-running jobs

Use explicit job handles for workflows that outlive a single tool call.

```python
@ui.tool()
async def start_index(repo: str, ctx: MCPContext) -> dict[str, str]:
    job_id = await create_job_for_user(ctx, repo)
    await enqueue_index_job(job_id, repo)
    return {"job_id": job_id, "status": "queued"}


@ui.tool()
async def index_status(job_id: str, ctx: MCPContext) -> dict[str, str]:
    return await load_job_status_for_user(ctx, job_id)


@ui.tool()
async def index_result(job_id: str, ctx: MCPContext):
    result = await load_job_result_for_user(ctx, job_id)
    return build_index_result_component(result)
```

Store durable job state in author-managed storage keyed by the authenticated
user and the job handle.

## Progress while a call is active

While a tool call is active, the adapter forwards progress from the harness to
FastMCP:

```python
await request.report("Cloning repository", progress=1, total=4)
await request.report("Running tests", progress=2, total=4)
await request.report("Building report", progress=3, total=4)
```

For jobs that should continue after the initial call returns, use FastMCP task
support or explicit job/status tools.

## Sessions

MCP App UI entry points and backend app tools are separate MCP tool calls.

Recommended defaults:

- use request-scoped serving for hosted apps;
- put durable state in explicit handles and storage;
- use connection scope only when a client requires MCP session continuity;
- never rely on hidden global agent state for user-specific app data.

If an app needs continuity across clicks, include the handle in the backend tool
schema:

```python
@ui.tool()
async def refine_plan(workspace_id: str, instruction: str, ctx: MCPContext) -> str:
    response = await adapter.invoke_agent(
        ctx=ctx,
        agent="planner",
        session_id=workspace_id,
        arguments={"workspace_id": workspace_id, "instruction": instruction},
    )
    return response.text_content()
```

## Auth and CSP

Backend app tools receive auth through the MCP request context. The adapter
translates verified auth into `AgentAuth` for the harness request.

UI resources should not embed bearer tokens. If the app UI calls external
origins, declare accurate MCP Apps CSP metadata through FastMCP.

Use per-tool auth checks for privileged UI actions.
