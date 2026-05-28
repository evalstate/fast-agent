---
title: Configuring Servers
social:
  title: Configure MCP Servers
  tagline: Connect local and remote MCP servers to fast-agent.
  description: Connect local and remote MCP servers to fast-agent.
  alt: fast-agent social card — Configure MCP Servers
---


MCP Servers are configured in the `fast-agent.yaml` file. Secrets can be kept in `fast-agent.secrets.yaml`, which follows the same format (**fast-agent** merges the contents of the two files).

`mcp.servers.<name>` supports canonical server blocks and shorthand `target` entries:

```yaml
mcp:
  servers:
    remote_api:
      target: "https://api.example.com/mcp"
      headers:
        Authorization: "Bearer ${EXAMPLE_TOKEN}"
      auth:
        oauth: true
```

You can also supply a list of target-first entries via `mcp.targets`:

```yaml
mcp:
  targets:
    - target: "https://demo.hf.space"
    - target: "@modelcontextprotocol/server-filesystem /workspace"
      name: "filesystem"
      load_on_start: false
```

`mcp.targets` entries are normalized into named `mcp.servers` aliases. If both
forms define the same alias, the explicit `mcp.servers.<name>` entry wins.

`target` must be a pure target string (URL/package/command only). Do not embed
fast-agent CLI flags like `--auth`/`--oauth` inside `target`; use `headers` and
`auth` fields instead.

## AgentCard runtime MCP connections (`mcp_connect`)

AgentCards can also declare runtime MCP targets directly with `mcp_connect`.
This is useful when a card depends on MCP servers that are not predeclared in
`fast-agent.yaml`.

```yaml
mcp_connect:
  - target: "https://demo.hf.space"
    headers:
      Authorization: "Bearer ${DEMO_TOKEN}"
    auth:
      oauth: true
  - target: "@modelcontextprotocol/server-everything"
    name: "everything"
```

`mcp.servers` remains the place for reusable, preconfigured aliases.
`mcp_connect` is card-scoped runtime declaration.

## Provider-managed MCP

For remote HTTP/SSE MCP servers, you can ask the model provider to manage the
connection natively instead of having fast-agent connect to the server locally.

```yaml title="fast-agent.yaml"
mcp:
  servers:
    huggingface:
      management: provider
      transport: "http"
      url: "https://huggingface.co/mcp"
      access_token: "${HF_TOKEN}"
      description: "Hugging Face MCP"
```

AgentCards can use the same mode in `mcp_connect`:

```yaml
mcp_connect:
  - target: "https://huggingface.co/mcp"
    name: "huggingface"
    management: provider
    access_token: "${HF_TOKEN}"
```

Use provider-managed MCP when you want the upstream model API to handle remote
tool discovery and execution itself.

Notes:

- Supported providers: `anthropic` and OpenAI `responses`.
- Not supported with `codexresponses` / Codex OAuth aliases such as
  `codexplan`, `codexplan54`, and `codexspark`.
- Not supported with `openresponses`, `openai`, `anthropic-vertex`, or other
  client-managed providers.
- Provider-managed remote MCP is URL-only: use remote `http`/`sse` servers, not
  stdio/package targets.
- Use `access_token` for bearer auth. Provider-managed remote MCP does not use
  arbitrary local `headers` / `auth` settings.
- Tool filters must use exact tool names. Wildcards, prompt filters, and
  resource filters are not supported for provider-managed attachments.

### OpenAI Responses connectors

The OpenAI `responses` provider can also manage OpenAI hosted connectors through
the same `management: provider` lane. Configure exactly one of `url` or
`connector_id`:

```yaml title="fast-agent.yaml"
mcp:
  servers:
    dropbox:
      management: provider
      connector_id: connector_dropbox
      access_token: "${DROPBOX_OAUTH_ACCESS_TOKEN}"
      description: "Dropbox connector"
      defer_loading: true
```

For connector-backed entries, set `name`, omit `transport` and `url`, and provide
`access_token`. `defer_loading: true` enables server-side lazy tool loading for
Responses provider-managed remote MCP and connectors.

## Adding a STDIO Server

The below shows an example of configuring an MCP Server named `server_one`.

```yaml title="fast-agent.yaml"
mcp:
  servers:
    # name used in agent servers array
    server_one:
      # command to run
      command: "npx"
      # list of arguments for the command
      args: ["@modelcontextprotocol/server-brave-search"]
      # key/value pairs of environment variables
      env:
        BRAVE_API_KEY: your_key
        KEY: value
    server_two:
      # and so on ...

```

This MCP Server can then be used with an agent as follows:
```python
@fast.agent(name="Search", servers=["server_one"])
```


## Adding an SSE or HTTP Server

To use remote MCP Servers, specify either `http` or `sse` transport and the endpoint URL and headers:

```yaml title="fast-agent.yaml"
mcp:
  servers:
    # name used in agent servers array
    server_two:
      transport: "http"
      # url to connect
      url: "http://localhost:8000/mcp"
      # timeout in seconds to use for HTTP/SSE sessions (optional)
      read_transport_sse_timeout_seconds: 300
      # request headers for connection
      headers:
        Authorization: "Bearer <secret>"

    # name used in agent servers array
    server_three:
      transport: "sse"
      # url to connect
      url: "http://localhost:8001/sse"

```

## MCP Filtering

Agents and Workflows supporting the `servers` parameter have the ability to filter the tools, resources and prompts available to the agent.  This can greatly reduce the amount of context generated for the agents - which can both increase the accuracy of the responses and reduce costs due to the lower token count of the context.

The default behavior is to include all tools, prompts and resources from the configured MCP servers, but this can be overridden by the `tools`, `prompts` and `resources` parameters. These parameters accept a dict where each key is the server name to filter and each value is a list of tool, resource, or prompt names.

For example:
```python
@fast.agent(
  name="Search",
  instruction="You are a search agent that helps users find files using the provided tools.",
  servers=["server_one", "server_two"],  # use two MCP servers

  # Filter some of the MCP tools available to the agent
  tools={
    "server_one": ["search_files", "search_directory"],
    "server_two": ["regex_search"],
  },
  prompts=None,  # don't filter prompts (default behavior)
  resources={
    "server_two": ["file://get_tree"], # only filter resources on server_two
  },
)

```

## Implementation Spoofing

**`fast-agent`** can specify the implementation details sent to the MCP Server, enabling testing of servers that adapt their configuration based on the client connection. By default **`fast-agent`** uses `fast-agent-mcp` and its current version number.

```yaml title="fast-agent.yaml"
mcp:
  servers:
    server_one:
      transport: "http"
      url: "http://localhost:8000/mcp"
      implementation:
        name: "spoof-server"
        version: "9.9.9"
```


## Elicitations

Elicitations are configured by specifying a strategy for the MCP Server. The handler can be overridden with a custom handler in the Agent definition.

```yaml title="fast-agent.yaml"
mcp:
  servers:
    server_four:
      transport: "http"
      url: "http://localhost:8000/mcp"
      elicitation:
        mode: "forms"
```

`mode` can be one of:

- **`forms`** (default). Displays a form to respond to elicitations.
- **`auto-cancel`** The elicitation capability is advertised to the Server, but all solicitations are automatically cancelled.
- **`none`** No elicitation capability is advertised to the Server.


## Roots

!!! warning

    Roots are [being deprecated](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577) in future versions of MCP. They will remain supported in fast-agent.


**fast-agent** supports MCP Roots. Roots are configured on a per-server basis:

```yaml title="fast-agent.yaml"
mcp:
  servers:
    server_three:
      transport: "http"
      url: "http://localhost:8000/mcp"
      roots:
        - uri: "file:///path/to/workspace"
          name: "Optional Name"
          server_uri_alias: "file:///mnt/data" # optional
```

As per the [MCP specification](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/630db617baa801ef8ec99e64aa4b00e99c7165ec/schema/2025-11-25/schema.ts#L2108-L2133) roots MUST be a valid URI starting with `file://`.

If a server_uri_alias is supplied, **fast-agent** presents this to the MCP Server. This allows you to present a consistent interface to the MCP Server. An example of this usage would be mounting a local directory to a docker volume, and presenting it as `/mnt/data` to the MCP Server for consistency.

The data analysis example (`fast-agent quickstart data-analysis`) has a working example of MCP Roots.

## Sampling

!!! warning

    Sampling is [being deprecated](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577) in future versions of MCP. 

Sampling is configured by specifying a sampling model for the MCP Server.

```yaml title="fast-agent.yaml"
mcp:
  servers:
    server_four:
      transport: "http"
      url: "http://localhost:8000/mcp"
      sampling:
        model: "provider.model.<reasoning_effort>"
```

Read more about the model string and settings [here](../models/). Sampling requests support vision - try [`@llmindset/mcp-webcam`](https://github.com/evalstate/mcp-webcam) for an example.

## Experimental Session Capability (client-first demos)

By default, **fast-agent** does not advertise experimental session capability in
the client initialize payload. It detects server support from the server's
initialize response.

To demonstrate a client-first negotiation style, enable per-server advertising:

```yaml title="fast-agent.yaml"
mcp:
  servers:
    server_five:
      transport: "http"
      url: "http://localhost:8765/mcp"
      experimental_session_advertise: true
      experimental_session_advertise_version: 2
```

When enabled, those values are included under
`client.capabilities.experimental.session` during `initialize`.
