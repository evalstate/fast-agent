---
title: MCP Apps
social:
  title: MCP Apps
  tagline: Discover standard MCP Apps tools, resources, and visibility metadata.
  description: Connect fast-agent to MCP servers that expose standard MCP Apps integrations.
  alt: fast-agent social card — MCP Apps
---

# MCP Apps

Fast-agent detects MCP Apps metadata while discovering an MCP server's tools
and resources.

An MCP Apps tool declares its resource in nested metadata:

```json
{
  "_meta": {
    "ui": {
      "resourceUri": "ui://workspace/app",
      "visibility": ["model", "app"]
    }
  }
}
```

The referenced resource must use:

```text
text/html;profile=mcp-app
```

Fast-agent also accepts the earlier flat `_meta["ui/resourceUri"]` form for
interoperability. New servers should emit nested `ui` metadata.

Fast-agent's terminal client does not advertise the MCP Apps host extension,
because it does not execute app HTML or provide the browser bridge. Servers
that expose app metadata only to negotiated app hosts will therefore retain
their normal non-app tool surface. The discovery described here applies when a
server exposes the metadata and resources to ordinary MCP clients.

## Discovery and visibility

For each MCP server, fast-agent:

- validates the declared `ui://` resource URI;
- lists and reads the referenced resource;
- checks the exact MCP Apps MIME type;
- reports missing, invalid, or unreferenced app resources;
- preserves normalized `ui.resourceUri` and `ui.visibility` metadata;
- excludes tools whose visibility is exactly `["app"]` from the model's tool
  catalog.

Tools visible to both the model and app remain ordinary callable MCP tools.
The `/mcp` display highlights the `Ui` capability when a valid app integration
is discovered.

## Programmatic access

MCP Apps and OpenAI Apps SDK discovery share neutral catalog models:

```python
from fast_agent.mcp.app_integrations import AppIntegrationKind

configs = await agent.aggregator.get_app_integration_configs()
config = await agent.aggregator.get_app_integration_config("workspace")

mcp_apps = [tool for tool in config.tools if tool.kind is AppIntegrationKind.MCP_APPS]
```

Each `AppServerConfig` includes discovered `resources`, linked `tools`, and
validation `warnings`.

## Rendering boundary

Fast-agent validates app metadata and presents structured tool results in the
terminal. It does not execute an MCP App's HTML or JavaScript as an embedded
terminal application.

To expose fast-agent-backed agents through a FastMCP Apps server, use the
[FastMCP Apps Adapter](fastmcp-apps.md). FastMCP owns the browser UI, CSP,
permissions, and app bridge; fast-agent owns harness invocation and agent
runtime behavior.
