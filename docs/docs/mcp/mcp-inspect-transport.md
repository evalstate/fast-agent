---
title: Inspect MCP Transports
social:
  title: Inspect MCP Transports
  tagline: Read protocol, transport, activity, and capability status for MCP servers.
  description: Compare modern and legacy remote MCP connections in the terminal.
  alt: fast-agent social card — Inspect MCP Transports
---

Use `/mcp` or `/mcp status` to inspect attached MCP servers in detail. Use
`/mcp list` for the shorter configured-and-attached inventory.

Protocol era and transport are separate concepts. A remote Streamable HTTP
endpoint can negotiate modern or legacy protocol behavior; a legacy protocol
connection is not necessarily the configured legacy SSE transport.

| `/mcp` label | Meaning | Display difference |
| --- | --- | --- |
| `(modern)` | Automatic negotiation selected the discovery-era protocol. | Session, legacy Health, and ping columns are omitted. |
| `(legacy)` | Automatic negotiation selected the initialization-era protocol. | Session and legacy Health are shown. |
| `(forced modern)` / `(forced legacy)` | `protocol_mode` was explicitly selected for interoperability testing or debugging. | Uses the corresponding modern or legacy display. |

`auto` is the default and should be preferred for normal connections. To test
a server's legacy compatibility path, connect with `--protocol legacy`; forcing
a mode changes client negotiation behavior, not the HTTP transport.

## Modern remote MCP: JSON, SSE, and progress

This recording connects to Hugging Face's MCP Server, calls `hf_whoami` for a
direct JSON response, then generates an image over SSE. The image call receives
progress notifications and renders the result before `/mcp` opens. The
diagnostic timeline distinguishes direct JSON-RPC messages from SSE streams
and shows request, response, and notification activity. The `LISTEN (SSE)`
channel is disabled because the server does not advertise list-change
notifications or resource subscriptions.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/tui/hf-image-generation.cast"
  data-fa-asciinema-cols="120"
  data-fa-asciinema-rows="34"
  data-fa-asciinema-poster="npt:0:61.39"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

The command leaves protocol selection at its default, `auto`. Fast-agent
discovers the server's supported modern protocol and `/mcp` reports the
negotiated result:

```text
protocol  2026-07-28 (modern)
```

The modern display intentionally omits the legacy Session and Health fields,
and its channel timeline has no legacy ping column. The identity lookup appears
on `POST (JSON)`. The image request and its progress notifications appear on
`POST (SSE)`, both during execution and as notification activity in the
subsequent transport display.

<!--
Cast asset:
- Source: docs/docs/assets/tui/hf-image-generation.cast
- Regenerate: uv run scripts/docs.py cast-build hf-image-generation
- Replay locally: asciinema play docs/docs/assets/tui/hf-image-generation.cast
-->

## Legacy remote MCP: session and health

This deterministic recording connects to a local legacy Streamable HTTP
fixture. The server returns a fixed `MCP-Session-Id`; one-second health pings
establish a visible healthy state before `/mcp` opens its 60-segment timeline.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/mcp/mcp-inspect-legacy.cast"
  data-fa-asciinema-cols="112"
  data-fa-asciinema-rows="30"
  data-fa-asciinema-poster="npt:0:15"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

For a forced legacy connection, `/mcp` shows the initialization-era protocol,
the observed HTTP session identifier, and legacy health state:

```text
protocol  2025-11-25 (forced legacy)
session   docs-legacy-session
health    ok
```

`forced legacy` means the client was configured with `protocol_mode: legacy`.
The Session value is an observed `MCP-Session-Id` response header. Health and
ping information belong to the legacy client path; their absence from the
modern display is intentional.

<!--
Cast asset:
- Source: docs/docs/assets/mcp/mcp-inspect-legacy.cast
- Regenerate: uv run scripts/docs.py cast-build mcp-inspect-legacy
- Replay locally: asciinema play docs/docs/assets/mcp/mcp-inspect-legacy.cast
-->

## Reading the `/mcp` display

### Implementation, protocol, and session

The first lines identify the server implementation, version, client, and
negotiated protocol era. A forced mode is shown as `(forced modern)` or
`(forced legacy)`. The protocol label reports the negotiated result rather than
printing internal negotiation steps separately.

Successful negotiation requests are included in `mcp calls:` alongside normal
operations. Modern negotiation records `discover`; legacy negotiation records
`initialize`. Forced-modern connections still send `server/discover` so
capabilities and metadata are available, but they fail rather than falling back
to legacy initialization when discovery is rejected. Reconnecting repeats and
counts the corresponding negotiation request.

```text
mcp calls: discover:1, list_tools:1, tool:2
```

For modern-era connections, `/mcp` omits Session and legacy Health fields.
Non-modern displays include Session; it contains the observed
`MCP-Session-Id`, or `None` when no session header was captured. The `local`
value for stdio describes transport locality, not a protocol session.

### Transport channel history

Transport activity is shown only when fast-agent has a live diagnostics
source. Public `httpx` hooks classify outgoing Streamable HTTP requests,
responses, resumptions, and errors without consuming response bodies or
streamed SSE events.

`POST (JSON)` and `POST (SSE)` classify observed POST responses by their
`Content-Type`. `LISTEN (SSE)` is the separate server-event listening channel.
These labels describe live HTTP behavior; they do not mean the server was
configured with the legacy SSE transport.

Timeline symbols prioritize significant events in this order:

```text
error → disabled/request → response → notification/ping → none
```

When several events land in one segment, the higher-priority state remains
visible.

### Server capabilities

- `To`, `Pr`, `Re`: Tools, Prompts, and Resources. A highlighted token indicates
  list-change notifications.
- `Rs`: Resource subscriptions.
- `Lo`, `Co`: Logging and completions.
- `Ex`: Experimental capabilities.
- `In`: Server instructions. Warning and error colors indicate instructions
  that are available but not injected into the agent system prompt, or disabled.

### Extensions and client settings

- `Sk`: SEP-2640 Skills Extension Draft (`d7490ecd`) support. It does not
  indicate compatibility with legacy index/archive skill servers.
- `Ui`: Detected MCP Apps or OpenAI Apps SDK configuration.
- `Ro`: Roots offered to the MCP server.
- `El`: Elicitation mode.
- `Sa`: Sampling mode.
- `Sp`: Client-name spoofing.

Provider-managed MCP does not use fast-agent's MCP client, so these local
negotiation and transport diagnostics do not apply to provider-owned
connections.

## Diagnostics configuration

Configure timeline density in `fast-agent.yaml`:

```yaml
mcp:
  diagnostics:
    enabled: true
    timeline:
      steps: 60
      step_seconds: 1
```

`steps` controls the number of rendered segments and `step_seconds` controls
the duration represented by each segment. Set `mcp.diagnostics.enabled` to
`false` to disable MCP diagnostics collection.
