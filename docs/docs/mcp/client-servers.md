---
title: Connect to MCP Servers
social:
  title: Connect to MCP Servers
  tagline: Connect local and remote MCP servers to fast-agent.
  description: Connect local and remote MCP servers to fast-agent.
  alt: fast-agent social card — Connect to MCP Servers
---


# Connect to MCP Servers

For convenience, **`fast-agent`** lets you connect to MCP Servers with command
line switches or runtime commands. Use `/connect <name>` or `/mcp attach <name>`
for a server in the configured registry. Use `/connect <target>` or
`/mcp connect <target>` for an ad-hoc URL, package, or stdio command.

From the command line:

```bash
fast-agent --url https://huggingface.co/mcp # Streamable HTTP
fast-agent --npx @modelcontextprotocol/server-everything # NPX Package
fast-agent --uvx server-fetch # UVX Package
fast-agent --stdio /path/to/stdio-server # STDIO Server
```

Repeat `--url` to preserve input order. Comma-separated values remain accepted
temporarily:

```bash
fast-agent \
  --url https://one.example/mcp \
  --url https://two.example/mcp \
  --mcp-protocol modern
```

`--mcp-protocol` applies to every startup `--url`, `--npx`, `--uvx`, and
`--stdio` target. One `--auth` value applies to every startup URL. Startup
targets are all validated before connection; malformed or failed targets stop
startup. During the compatibility window, a URL missing `/mcp` or `/sse` is
shown with its resolved `/mcp` URL and a deprecation notice.

Connection failures use the same structured report at startup, in the terminal,
and over ACP. The report names the server, failure stage, redacted target,
details, and next action. URL user information, query values, fragments, and
common credential fields are redacted. If explicit bearer credentials are
rejected, fast-agent tells you to replace them; it does not silently switch to
OAuth.

MCP Servers are configured in the `fast-agent.yaml` file. Secrets can be kept in `fast-agent.secrets.yaml`, which follows the same format (**fast-agent** merges the contents of the two files).

The canonical MCP configuration is nested under `mcp`:

```yaml
mcp:
  defaults:
    protocol_mode: auto
    reconnect_on_disconnect: true
    include_instructions: true
  client:
    auto_sampling: true
  diagnostics:
    enabled: true
    timeline:
      steps: 20
      step_seconds: 30
  servers:
    remote_api:
      target: "https://api.example.com/mcp"
      headers:
        Authorization: "Bearer ${EXAMPLE_TOKEN}"
      auth:
        oauth: true
      protocol_mode: auto
```

The key under `mcp.servers` is the server's canonical name. Use that key in
agent `servers` lists; do not add a separate `name` field.

Inspect redacted declarations without expanding shorthand:

```bash
fast-agent config show-mcp fast-agent.yaml
```

Use `--view effective` to show resolved transports, inherited defaults, and
per-field provenance.

Each server uses either a shorthand `target` or expanded source fields:

```yaml
mcp:
  servers:
    filesystem:
      target: "@modelcontextprotocol/server-filesystem /workspace"
      load_on_start: false
    remote_api:
      transport: http
      url: "https://api.example.com/mcp"
      headers:
        Authorization: "Bearer ${EXAMPLE_TOKEN}"
```

`target` is mutually exclusive with the expanded source fields `transport`,
`url`, `command`, `args`, and `connector_id`. Other settings such as `headers`,
`auth`, `protocol_mode`, and `load_on_start` can accompany a `target`.

`target` must be a pure target string (URL/package/command only). Do not embed
fast-agent CLI flags like `--auth`/`--oauth` inside `target`; use `headers` and
`auth` fields instead.

`mcp.defaults` supplies `protocol_mode`, `reconnect_on_disconnect`, and
`include_instructions` when a server omits them. `mcp.client` controls client
behavior such as automatic sampling, while `mcp.diagnostics` controls
diagnostics collection and timeline display.

For legacy configuration, see [Migrate MCP configuration](migration.md).

## Protocol selection

By default, fast-agent asks the MCP SDK to negotiate the best supported
protocol:

```yaml
protocol_mode: auto
```

You can force a protocol era per client-managed server:

```yaml
mcp:
  servers:
    modern_only:
      target: "https://api.example.com/mcp"
      protocol_mode: modern
    legacy_only:
      target: "uvx legacy-mcp-server"
      protocol_mode: legacy
```

- `auto` attempts modern discovery and falls back when the peer is identified
  as a handshake-era server.
- `modern` pins negotiation to the modern protocol era. It sends
  `server/discover` to load capabilities and metadata, but does not fall back to
  initialization and cannot be combined with the legacy SSE transport.
- `legacy` uses the initialization handshake and does not attempt modern
  discovery.

Forced modern mode is useful for interoperability testing and known
modern-only peers. Unlike `auto`, a failed `server/discover` request is treated
as an error rather than evidence that the client should try legacy
initialization. Prefer `auto` for normal connections.

For an ad-hoc connection:

```text
/mcp connect --protocol modern https://api.example.com/mcp
```

Configured servers attached with `/mcp attach <name>` take their mode from
`protocol_mode`. A bare `/connect docs` attaches the configured server named
`docs` when it exists; `/mcp attach docs` is the explicit equivalent.
Otherwise `/connect` materializes an ad-hoc target. `/mcp connect` is always
ad-hoc. Use `--name` to give an ad-hoc target a non-conflicting runtime name.
Provider-managed MCP does not use fast-agent's MCP client and therefore does
not accept this setting.

## Runtime MCP commands

- `/mcp list` lists attached servers and configured servers that remain detached.
- `/mcp status` shows detailed protocol, transport, health, activity, and capability status.
- `/mcp attach <name>` attaches the named configured registry entry.
- `/mcp connect <target>` connects an ad-hoc target.
- `/connect <name|target>` attaches a configured name or connects an ad-hoc target.
- `/mcp disconnect <name>` detaches a server.
- `/mcp reconnect <name>` reconnects the attached server using its existing target.

In the terminal and ACP, bare `/mcp` is a shortcut for detailed status. Both
surfaces expose `/mcp list` for inventory and top-level `/connect` for
configured names and ad-hoc targets.

## AgentCard runtime MCP connections (`mcp_connect`)

AgentCards can also declare runtime MCP targets directly with `mcp_connect`.
This is useful when a card depends on MCP servers that are not predeclared in
`fast-agent.yaml`.

```yaml
mcp_connect:
  demo:
    target: "https://demo.hf.space"
    protocol_mode: modern
    headers:
      Authorization: "Bearer ${DEMO_TOKEN}"
    auth:
      oauth: true
  everything:
    target: "@modelcontextprotocol/server-everything"
```

`mcp.servers` remains the place for reusable, preconfigured definitions.
`mcp_connect` is card-scoped runtime declaration.
The named mapping is canonical; list entries remain compatible and retain
their list form when dumped. `protocol_mode` accepts `auto`, `modern`, or
`legacy`. Cards cannot set process fields (`command`, `args`, `env`, or `cwd`);
those are materialized only from the allow-listed `target`.

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
  huggingface:
    target: "https://huggingface.co/mcp"
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
