# MCP 2026-07-28 impact assessment

Date: 2026-07-24

## Scope and source revisions

This assessment compares:

- fast-agent `f848d91a` on `feat/mcp-2026-07-28`
- MCP specification `76346843`, whose draft schema declares
  `LATEST_PROTOCOL_VERSION = "2026-07-28"`
- MCP Python SDK v2 at `upstream/main` `00a70148`
  (`v2.0.0b2-7-g00a70148`)
- FastMCP v4 at `37fb0ad8` (`v4.0.0a2`)

The checked-out Python SDK `main` is still based on v1.24 and has local changes.
The v2 assessment therefore uses `upstream/main` and the `v2.0.0b2` tag without
altering the checkout.

## Implementation outcome

Fast-agent now owns configured server runtimes while the public
`mcp.client.Client` owns protocol behavior:

- `Client(mode="auto")` performs discovery and legacy fallback;
- high-level tool, prompt, and resource methods drive MRTR;
- attached runtimes use the SDK response cache;
- on-demand, request-authenticated clients disable caching;
- `Client.listen()` owns modern subscription demultiplexing and cache eviction;
- fast-agent callbacks are composed in `MCPClientCallbackRuntime` rather than
  attached to a `ClientSession` subclass;
- the custom `MCPAgentClientSession`, session factory, manual negotiation, and
  private input-required driver have been removed.

`MCPConnectionManager` remains as a product runtime owner for attach/detach,
stdio processes, OAuth UX, startup budgets, replacement, and status. It no
longer represents a pool or durable protocol-session identity.

## Executive assessment

Supporting MCP 2026-07-28 is a protocol-era migration, not a version constant
update.

The new revision removes protocol initialization and protocol-managed sessions.
Version, identity, and capabilities move to per-request metadata. The Streamable
HTTP transport becomes POST-only and loses `Mcp-Session-Id`, GET notification
streams, DELETE termination, SSE event replay, and `Last-Event-ID`. Server-
initiated requests are replaced by multi round-trip request (MRTR) results.

This directly conflicts with several central fast-agent assumptions:

- every connection is established with `ClientSession.initialize()`;
- `InitializeResult` is the source of protocol metadata;
- HTTP sessions and `Mcp-Session-Id` represent durable connection state;
- periodic `ping` determines connection health;
- Streamable HTTP has POST, GET, and resumption channels;
- roots, sampling, and elicitation arrive as server-initiated requests;
- tool-list changes arrive on the general connection receive loop;
- URL elicitation uses `elicitationId`, completion notifications, or `-32042`;
- fast-agent's served MCP agents can use an ambient MCP session as their
  application session key.

The Python SDK v2 and FastMCP v4 already implement dual-era discovery,
negotiation, result validation, MRTR, cache hints, extensions, and modern
transport behavior. Fast-agent should use those implementations and remove its
forks of SDK transport internals rather than porting those forks.

### Overall impact

| Area | Impact | Recommendation |
| --- | --- | --- |
| Dependency and protocol types | Very high, mechanical | Upgrade `fastmcp`, `mcp`, and `mcp-types` together; migrate all imports and model access |
| Client connection lifecycle | Very high, architectural | Replace initialize-centric state with SDK/FastMCP dual-era negotiation |
| Custom transports and metrics | Very high, architectural | Delete transport forks; retain a thin fast-agent diagnostics layer |
| Server-side session semantics | Very high, product behavior | Replace ambient MCP session identity with explicit application handles |
| MRTR and elicitation | High | Adopt the SDK/FastMCP driver and remove URL-elicitation compatibility hacks |
| Subscriptions and refresh | High | Use `subscriptions/listen`; retain explicit legacy behavior only when negotiated |
| `/mcp` diagnostics | Medium, high product value | Add negotiated protocol/era and truthful transport topology to `ServerStatus` |
| OAuth | Medium/high | Move to public v2/FastMCP APIs and issuer-keyed credential storage |
| Caching and JSON Schema | Medium | Prefer SDK/FastMCP behavior; validate fast-agent conversion boundaries |
| Deprecated roots/sampling/logging | Medium now, high later | Preserve only for legacy compatibility and publish a migration path |
| Tasks and Apps extensions | Optional | Add after core modern support; do not treat old core Tasks as compatible |

## Specification delta

The authoritative summary is
`modelcontextprotocol/docs/specification/draft/changelog.mdx`.

### Stateless protocol and discovery

MCP 2026-07-28 removes:

- `initialize`;
- `notifications/initialized`;
- protocol-level sessions;
- the assumption that capabilities or identity persist from an earlier
  exchange.

Every request now carries:

- `_meta.io.modelcontextprotocol/protocolVersion`;
- `_meta.io.modelcontextprotocol/clientCapabilities`;
- preferably `_meta.io.modelcontextprotocol/clientInfo`.

Results should carry
`_meta.io.modelcontextprotocol/serverInfo`. Servers must implement
`server/discover`, which returns supported versions, capabilities,
implementation information, instructions, and cache hints.

An unsupported version returns `UnsupportedProtocolVersionError` (`-32022`)
with the requested and supported versions. A dual-era client probes a server,
selects a mutually supported modern version, or falls back to the legacy
initialize handshake when the response identifies a legacy server.

Python SDK v2 implements this policy in:

- `mcp.client.Client(mode="auto")`;
- `mcp.client._probe.negotiate_auto`;
- `ClientSession.discover()`;
- `ClientSession.adopt()`.

The negotiated state is already exposed through:

- `ClientSession.protocol_version`;
- `ClientSession.server_info`;
- `ClientSession.server_capabilities`;
- `ClientSession.instructions`;
- `ClientSession.initialize_result`;
- `ClientSession.discover_result`.

FastMCP v4 exposes the first four as era-neutral `Client` properties.

### Required result discriminator and MRTR

All modern results have `resultType`:

- `complete` for an ordinary result;
- `input_required` when the server needs client input.

Clients must treat a missing discriminator from a legacy server as complete.

MRTR replaces independent server-to-client requests. A tool call, prompt get,
or resource read may return `InputRequiredResult` containing `inputRequests`
and opaque `requestState`. The client obtains the requested input and retries
the original operation with:

- a new JSON-RPC request ID;
- the exact request state;
- corresponding `inputResponses`.

This replaces the current push paths for roots, sampling, and elicitation.
Fast-agent should not implement its own retry state machine; SDK v2 and
FastMCP v4 already provide an input-required driver with a configurable round
limit.

### Streamable HTTP

Modern Streamable HTTP is:

- one POST per JSON-RPC request or notification;
- JSON or request-scoped SSE as the POST response;
- no independent GET stream;
- no protocol session ID;
- no DELETE session termination;
- no event IDs, replay, or resumption;
- a new request ID and complete reissue after an interrupted response stream.

Modern POSTs carry and validate:

- `MCP-Protocol-Version`;
- `Mcp-Method`;
- `Mcp-Name` for name-bearing operations.

Tool schemas can use `x-mcp-header` to map primitive tool parameters to
`Mcp-Param-*` headers. Header/body mismatches return `HeaderMismatchError`
(`-32020`).

This makes the current GET/resumption/session portions of the transport display
legacy-only information.

### Subscriptions

`resources/subscribe`, `resources/unsubscribe`, and the HTTP GET notification
stream are replaced by `subscriptions/listen`.

A client opens a filtered long-lived request for selected event classes:

- tool-list changes;
- prompt-list changes;
- resource-list changes;
- selected resource URI updates.

The server acknowledges the subscription and tags feed notifications with the
subscription ID. Request-scoped progress and log notifications remain on the
original request stream.

Fast-agent currently only reacts directly to
`ToolListChangedNotification`. Modern refresh support needs a subscription
owner that:

- starts after negotiation;
- requests only needed event classes;
- refreshes tools, prompts, and resources;
- invalidates response caches;
- restarts the subscription after loss without treating the entire server as
  disconnected.

### Caching

The following complete results now carry required `ttlMs` and `cacheScope`:

- `server/discover`;
- `tools/list`;
- `prompts/list`;
- `resources/list`;
- `resources/templates/list`;
- `resources/read`.

`cacheScope` distinguishes public results from results private to an
authorization context. Notifications invalidate related entries.

FastMCP v4 and SDK v2 provide client response caching. Fast-agent should use
that implementation rather than add another protocol cache to
`MCPAggregator`. Any higher-level aggregate cache must be keyed by server,
negotiated version, operation parameters, and authorization context.

### Feature changes

#### Tools, prompts, and resources

- List results must not vary by connection state.
- Deterministic tool ordering is recommended for prompt-cache stability.
- `tools/call`, `prompts/get`, and `resources/read` can return MRTR.
- Resource not found changes from `-32002` to JSON-RPC Invalid Params
  (`-32602`); clients should still accept `-32002` from legacy servers.
- Tool input/output schema accepts general JSON Schema 2020-12.
- `structuredContent` accepts any JSON value, not only an object.
- External `$ref` fetching must not happen implicitly; schema composition needs
  resource limits.

Fast-agent's provider converters and tool-result abstractions currently often
assume object-shaped structured content. Those boundaries need explicit tests
for arrays, scalars, booleans, and null.

#### Elicitation

Modern elicitation is MRTR. The revision removes:

- `elicitationId`;
- `notifications/elicitation/complete`;
- URL elicitation's `-32042` control flow.

Fast-agent can remove the result/exception payload attachment machinery in
`mcp_agent_client_session.py`, `url_elicitation_required.py`, and related
display tests after the MRTR UI path is complete. Legacy URL elicitation can
remain behind negotiated legacy behavior during the compatibility period.

#### Roots, sampling, and logging

Roots, Sampling, and Logging remain available but are deprecated as of
2026-07-28, with earliest removal in a revision released on or after
2027-07-28.

Recommended migrations are:

- roots: tool parameters, resource URIs, or server configuration;
- sampling: server-side provider integration;
- logging: stderr for stdio and OpenTelemetry for observability.

The revision also removes:

- `ping`;
- `logging/setLevel`;
- `notifications/roots/list_changed`.

Modern log level is request metadata, and servers cannot send log
notifications for a request that did not opt in.

Fast-agent should not add new roots/sampling/logging functionality. Existing
support should be explicitly marked legacy/deprecated in config and status
output.

#### Tasks

The experimental core Tasks API is removed. Tasks are redesigned as the
optional `io.modelcontextprotocol/tasks` extension:

- no core `ServerCapabilities.tasks`;
- no `tasks/list`;
- no blocking `tasks/result`;
- polling through `tasks/get`;
- input through `tasks/update`;
- cooperative cancellation;
- a task result type and extension-specific notifications.

`MCPAggregator.server_supports_feature(..., "tasks")` currently reads
`capabilities.tasks`; that code will fail against v2 types and must be replaced
by extension capability inspection.

#### Extensions and Apps

Core capabilities now contain a namespaced `extensions` map. Extensions are
disabled unless both peers advertise compatible support.

MCP Apps (`io.modelcontextprotocol/ui`) and Tasks should be implemented as
extension adapters, not added to core capability checks. Existing OpenAI Apps
SDK support should be assessed for convergence with the Apps extension rather
than maintained as an unrelated parallel protocol indefinitely.

### Authorization

The revision strengthens existing OAuth requirements and deprecates Dynamic
Client Registration in favor of Client ID Metadata Documents.

Required migration checks include:

- validate a returned RFC 9207 `iss` before redeeming an authorization code;
- key stored client credentials by authorization-server issuer;
- never reuse credentials after an issuer change;
- specify a suitable `application_type` during DCR fallback;
- preserve resource indicators and audience validation;
- prefer CIMD and retain DCR only as compatibility fallback.

Fast-agent already supports a client metadata URL, but its custom OAuth client
depends heavily on SDK internals. It should be rebuilt around FastMCP v4's
public `OAuth`/CIMD support or the SDK v2 public auth surface. The browser and
ACP progress UX can remain a fast-agent callback adapter.

## Python SDK v2 and FastMCP v4 migration

### Dependency migration

Current fast-agent pins:

```toml
fastmcp==3.4.4
mcp==1.28.1
```

FastMCP v4 currently requires the matching prerelease set:

```text
fastmcp 4.0.0a2
fastmcp-slim 4.0.0a2
mcp 2.0.0b2
mcp-types 2.0.0b2
```

These must be constrained together until stable releases provide normal
resolver guarantees. The lock update also needs explicit validation of:

- `opentelemetry-instrumentation-mcp==0.62.1` against SDK v2;
- `httpx2` exception handling and TLS trust-store behavior;
- the remaining `httpx` dependency used by model-provider SDKs;
- FastAPI/Starlette compatibility;
- supported Python versions and packaging extras.

### Protocol type split and naming

SDK v2 removes `mcp.types`. Protocol models move to `mcp_types`, and Python
attributes become snake_case while JSON aliases remain camelCase.

Examples:

| v1 attribute | v2 attribute |
| --- | --- |
| `inputSchema` | `input_schema` |
| `outputSchema` | `output_schema` |
| `isError` | `is_error` |
| `structuredContent` | `structured_content` |
| `mimeType` | `mime_type` |
| `nextCursor` | `next_cursor` |
| `serverInfo` | `server_info` |
| `protocolVersion` | `protocol_version` |
| `listChanged` | `list_changed` |
| `requestedSchema` | `requested_schema` |

FastMCP v4 installs a warning compatibility bridge for common reads, but it
does not restore `mcp.types`, old wrappers, private APIs, or all attributes.
Fast-agent should migrate fully and run CI with:

```text
FASTMCP_MCP_CAMELCASE_COMPAT=false
```

Wire-facing serialization must use:

```python
model.model_dump(by_alias=True, mode="json", exclude_none=True)
```

Other important model changes:

- JSON-RPC and request/notification unions are direct union values, not
  `RootModel` objects with `.root`;
- request metadata is a typed mapping rather than the old nested Pydantic
  model;
- URI fields are generally strings rather than `AnyUrl`;
- unknown Pydantic fields are not an application metadata channel;
- `McpError` becomes `MCPError`, with keyword/explicit construction;
- list cursors move into `PaginatedRequestParams`;
- session/request timeouts use float seconds.

This is an architecture-wide mechanical migration because MCP content and tool
types cross agents, providers, ACP, A2A, UI, history, trace export, and local
tool runtimes.

### FastMCP v4 breaking changes relevant to fast-agent

- Direct imports from `mcp.types` must move to `mcp_types`.
- Raw `client.session` and `ctx.request_context` expose SDK v2 shapes.
- FastMCP-owned HTTP uses `httpx2`, including its exception types.
- `StreamableHttpTransport(sse_read_timeout=...)` is removed.
- `Client` defaults to `mode="auto"`.
- Resource-not-found is `-32602`.
- Resource templates apply path-traversal screening by default.
- Deprecated proxy/mount/component import paths and several old tool
  parameters are removed.

Fast-agent's current server construction already passes transport settings to
`run_http_async` rather than the `FastMCP` constructor, so that part is aligned.

## Current fast-agent architecture and direct impact

### `MCPConnectionManager`

`src/fast_agent/mcp/mcp_connection_manager.py` owns:

- transport creation;
- OAuth preparation and escalation;
- long-lived task groups;
- initialization;
- reconnect policy;
- ping health;
- session ID capture;
- server metadata;
- stderr buffering;
- transport metrics.

Its `ServerConnection.initialize_session()` always calls
`session.initialize()`. It separately stores values that v2 exposes directly
and era-neutrally.

Required changes:

1. Replace initialize-only startup with `mode="auto"` negotiation.
2. Store `protocol_version`, era, discovery result, server info,
   capabilities, and instructions from public session/client properties.
3. Remove ping health for modern connections.
4. Treat the subscription stream and in-flight request streams independently
   from overall server reachability.
5. Remove protocol session termination and session-expiry assumptions on
   modern connections.
6. Stop importing SDK-private HTTP factories and transport internals.

### `MCPAgentClientSession`

`src/fast_agent/mcp/mcp_agent_client_session.py` subclasses the v1
`ClientSession` to add:

- sampling, roots, and elicitation callbacks;
- tool-list change handling;
- URL elicitation compatibility;
- transport response-channel attachment;
- offline/session-expiry translation;
- progress tracing.

Most of these can become composition:

- pass supported callbacks to the v2/FastMCP client;
- use the SDK MRTR driver for modern elicitation;
- use extension notification bindings and `subscriptions/listen`;
- consume OpenTelemetry spans or a public diagnostics callback;
- map connection errors at the fast-agent operation boundary.

Avoid recreating a large v2 `ClientSession` subclass. SDK v2's dispatcher and
callback concurrency differ materially from v1, making inherited internals an
unstable extension point.

### Transport forks

The following files copy or subclass SDK transport behavior:

- `mcp/streamable_http_tracking.py`;
- `mcp/sse_tracking.py`;
- `mcp/stdio_tracking_simple.py`.

The Streamable HTTP fork depends on exactly the features removed in 2026:

- GET stream;
- resumption stream;
- SSE event IDs;
- `Last-Event-ID`;
- `Mcp-Session-Id`;
- DELETE termination.

It also imports private SDK types and uses `httpx`/`httpx-sse`, while SDK v2
uses `httpx2`.

Recommendation:

- delete the Streamable HTTP implementation fork;
- use FastMCP v4 `ClientTransport` or the SDK v2 public transport;
- preserve `TransportChannelMetrics` as a fast-agent product model;
- feed it from public operation/notification callbacks, OpenTelemetry, and
  narrow transport event hooks;
- represent legacy-only channels only when the negotiated era actually has
  them.

If exact HTTP response mode remains important, add a small upstream extension
point to FastMCP/SDK rather than inheriting the transport implementation.

### `MCPAggregator`

`MCPAggregator` should continue to own fast-agent's namespacing and conversion
of MCP tools/prompts/resources into agent-visible components. It should stop
owning protocol-era behavior already provided upstream.

Specific changes:

- source metadata from an era-neutral connection descriptor;
- replace `capabilities.tasks` with extension inspection;
- add prompt/resource refresh handling, not only tool refresh;
- use SDK/FastMCP cache and listen behavior;
- update pagination and structured-content assumptions;
- keep authorization context out of shared cache entries;
- make attach results include negotiation and transport diagnostics.

### Server registry

`mcp_server_registry.py` caches `InitializeResult` and always initializes a
temporary session. This should become a cached era-neutral descriptor, such as:

```python
@dataclass(frozen=True, slots=True)
class MCPPeerInfo:
    protocol_version: str
    era: Literal["modern", "legacy"]
    server_info: Implementation | None
    capabilities: ServerCapabilities
    instructions: str | None
    supported_versions: tuple[str, ...]
```

The descriptor should be populated by the connection abstraction rather than
reconstructing negotiation in the registry.

### Served fast-agent MCP applications

`HarnessMCPAdapter` currently derives application continuity from
`Mcp-Session-Id`. Documentation explicitly describes that behavior.

That cannot work in modern MCP. A modern connection has no ambient protocol
session and every request may be independently routed.

Fast-agent needs separate product semantics:

- `request`: each call is isolated;
- `shared`: an explicitly configured shared agent instance;
- stateful: a server-minted opaque application handle passed as an ordinary
  tool argument.

FastMCP v4 includes an explicit server session store and `SessionId` tool
argument model for this purpose. The durable design is to expose an operation
that creates a handle and require the handle on subsequent calls. It must be
bound to the authenticated principal and treated as a bearer capability, not a
security boundary by itself.

Do not silently map modern no-session calls to connection scope. Either:

- change the modern default to request scope; or
- require the explicit application handle for stateful behavior.

Legacy clients may continue using ambient `Mcp-Session-Id` during the
compatibility period.

## `/mcp` and transport display

### Current state

There are currently two related command surfaces:

- `/mcp list` reports attached and configured-but-detached server names;
- `/mcpstatus` (also reached through status handling) renders the detailed
  implementation, capability, health, session, and channel display from
  `ServerStatus`.

`ServerStatus` already contains:

- server implementation name/version;
- server and client capabilities;
- configured transport;
- connection state;
- `Mcp-Session-Id`;
- ping health;
- POST JSON/SSE, GET, resumption, and stdio channel metrics.

It does not contain the negotiated protocol version or protocol era.

### Required status model

Add these fields to the connection status:

```python
protocol_version: str | None
protocol_era: Literal["modern", "legacy"] | None
supported_protocol_versions: tuple[str, ...]
negotiation: Literal["discover", "initialize", "pinned"] | None
server_implementation_source: Literal["discover", "initialize", "response_meta"] | None
extensions: Mapping[str, object]
transport_endpoint: str | None
transport_process: str | None
transport_security: Literal["none", "bearer", "oauth", "forwarded"] | None
subscription_state: Literal["open", "closed", "unsupported", "error"] | None
```

Do not display bearer tokens, OAuth client secrets, complete sensitive headers,
or URL userinfo. Query strings should be redacted by default.

### Recommended display

The top metadata section should show:

```text
Protocol    2026-07-28 (modern, discover)
Server      example-server 2.3.1
Transport   Streamable HTTP · https://example.test/mcp
Auth        OAuth
Session     none (sessionless protocol)
Extensions io.modelcontextprotocol/tasks, io.modelcontextprotocol/ui
```

For legacy:

```text
Protocol    2025-11-25 (legacy, initialize)
Transport   Streamable HTTP · session abc…789
```

Transport topology must be negotiated-era aware:

| Era/transport | Display |
| --- | --- |
| Modern HTTP | request POST JSON/SSE; subscription POST stream; no GET, resumption, protocol session, or ping |
| Legacy Streamable HTTP | POST JSON/SSE; optional GET; optional resumption; optional session ID; ping if enabled |
| Legacy SSE | clearly label deprecated HTTP+SSE |
| stdio modern | process/command, request activity, subscription-listen activity, no protocol ping |
| stdio legacy | process/command, request activity, legacy ping |

The existing channel panel should not render removed channels as merely idle;
that incorrectly suggests they are available. Mark them `n/a (modern)` or omit
them.

Modern health should be evidence-based:

- last successful operation;
- active subscription state;
- last transport error;
- process state for stdio;
- optional explicit reconnect/probe initiated by the user.

It should not send removed `ping` requests. An HTTP endpoint is not a durable
connection, so “connected” should mean “negotiated and currently usable,” not
“an HTTP socket or MCP session is open.”

### Command UX recommendation

Unify the simple and detailed views:

- `/mcp` or `/mcp list`: concise rows including server, state, protocol, and
  transport;
- `/mcp status [server]` or `/mcp list --verbose`: current detailed display;
- retain `/mcpstatus` as a compatibility alias.

This prevents negotiated protocol information from being hidden behind a
separate command.

## Refactor target

### Proposed boundaries

```mermaid
flowchart LR
    Config[MCPServerSettings] --> Factory[MCP client factory]
    Factory --> Client[FastMCP v4 Client / SDK v2 Client]
    Client --> Peer[MCPPeerInfo]
    Client --> Ops[MCP operations]
    Client --> Events[subscription + progress events]
    Client --> Diag[diagnostic events]
    Peer --> Aggregator[MCPAggregator]
    Ops --> Aggregator
    Events --> Aggregator
    Diag --> Status[ServerStatus]
    Aggregator --> Agents[Agent tool/prompt/resource surfaces]
    Status --> UI[/mcp display]
```

The central connection abstraction should expose product capabilities, not
transport streams:

```python
class MCPClientConnection(Protocol):
    @property
    def peer_info(self) -> MCPPeerInfo: ...

    @property
    def transport_info(self) -> MCPTransportInfo: ...

    async def list_tools(self, ...) -> ListToolsResult: ...
    async def call_tool(self, ...) -> CallToolResult: ...
    async def listen(self, ...) -> AsyncIterator[MCPServerEvent]: ...
    async def close(self) -> None: ...
```

Fast-agent can then change from FastMCP `Client` to the lower-level SDK client
without exposing that choice to the aggregator or UI.

### Code to delete or substantially reduce

Once equivalent behavior is covered:

- most of `streamable_http_tracking.py`;
- the SDK-copying portions of `sse_tracking.py`;
- stream adaptation helpers and session-ID callbacks in
  `mcp_connection_manager.py`;
- manual initialize metadata duplication;
- ping protocol and config for modern connections;
- GET/resumption metrics for modern connections;
- URL elicitation exception/result payload attachment;
- direct `ClientSession` inheritance used only for notification interception;
- old core Tasks capability logic;
- protocol-session-based server instance leasing for modern calls.

Keep:

- fast-agent's target parsing and configuration UX;
- OAuth browser/progress presentation;
- attach/detach/reconnect orchestration;
- aggregator namespacing and agent-visible conversion;
- stderr capture for stdio;
- the product-level diagnostics/status model;
- operation-level retry policy where it is not already protocol negotiation.

## Implementation plan

### Phase 0: dependency spike and migration gates

Impact: high, bounded.

1. Create a coordinated prerelease constraint set.
2. Verify OpenTelemetry MCP instrumentation against SDK v2.
3. Run import collection and type checking before behavior changes.
4. Add CI bans for:
   - `from mcp.types`;
   - `mcp.shared._httpx_utils`;
   - `GetSessionIdCallback`;
   - `streamablehttp_client`;
   - direct JSON-RPC `.root` access;
   - camelCase protocol attribute reads.
5. Run FastMCP with the camelCase bridge disabled.

Exit criterion: the dependency set resolves and failures are categorized.

### Phase 1: mechanical SDK v2 port

Impact: very high, mostly mechanical.

1. Move protocol imports to `mcp_types`.
2. Convert Python field access to snake_case.
3. Update union handling, errors, metadata, pagination, URI boundaries, and
   timeouts.
4. Make wire serialization alias-explicit.
5. Update provider, ACP, A2A, history, UI, and trace conversion tests.

Exit criterion: fast-agent imports, lints, and type-checks on v2 without relying
on FastMCP compatibility shims.

### Phase 2: replace transport and connection lifecycle

Impact: very high, architectural.

1. Introduce `MCPPeerInfo` and `MCPTransportInfo`.
2. Wrap FastMCP v4 `Client` or SDK v2 `Client` behind the connection protocol.
3. Use `mode="auto"` by default and expose an explicit legacy override only
   for compatibility.
4. Remove the Streamable HTTP fork.
5. Retain narrow diagnostics hooks and stdio stderr capture.
6. Replace ping health with era-aware health.

Exit criterion: stdio and Streamable HTTP work against modern, legacy, and
dual-era simulators with correct metadata.

### Phase 3: MRTR, subscriptions, and caching

Impact: high.

1. Route `InputRequiredResult` through existing human-input/elicitation UI.
2. Support repeated MRTR rounds and cancellation.
3. Remove modern URL-elicitation hacks.
4. Own one `subscriptions/listen` lifecycle per attached server as needed.
5. Refresh tools, prompts, resources, and templates.
6. Enable upstream cache handling and verify private cache isolation.

Exit criterion: modern elicitation and list/resource updates work without
server-initiated requests or GET streams.

### Phase 4: served fast-agent semantics

Impact: very high, product decision required.

1. Make modern request scope explicit.
2. Add server-minted application session handles for stateful agent use.
3. Bind handles to authenticated principals.
4. Preserve ambient MCP session behavior only for negotiated legacy clients.
5. Update `instance_scope` validation and documentation.

Exit criterion: no modern server behavior depends on connection affinity or
`Mcp-Session-Id`.

### Phase 5: `/mcp`, extensions, and deprecation

Impact: medium.

1. Add protocol/era/negotiation and transport endpoint details to status.
2. Render modern and legacy channel topology truthfully.
3. Replace Tasks core checks with extension checks.
4. Evaluate Tasks and Apps as separate optional deliverables.
5. Mark roots, sampling, logging, SSE, and DCR compatibility as deprecated.

Exit criterion: users can diagnose exactly what was negotiated and what
transport behavior is active.

## Test strategy

Prefer simulators and contract tests over mocks of SDK internals.

### Required matrix

| Dimension | Cases |
| --- | --- |
| Server era | modern-only, legacy-only, dual-era |
| Negotiation | discover success, unsupported-version retry, legacy fallback, no mutual version |
| Transport | stdio, modern HTTP JSON, modern HTTP SSE, legacy Streamable HTTP, deprecated SSE |
| Auth | none, bearer, OAuth, CIMD, DCR fallback, issuer change |
| Features | tools, prompts, resources, pagination, structured content, progress, MRTR, subscriptions |
| Failure | malformed result, interrupted stream, 401 escalation, 400 era fallback, subscription loss |

### Protocol invariants

- Every modern request has protocol version, client capabilities, and HTTP
  protocol headers where applicable.
- Every modern complete result has `resultType`, `ttlMs`, and `cacheScope`
  where required.
- No modern request uses initialization, ping, protocol sessions, GET
  notification streams, DELETE termination, or resumption.
- MRTR retries use a new JSON-RPC ID and preserve exact request state.
- Subscription notifications are filtered and correlated.
- Private cache entries do not cross authorization contexts.
- `serverInfo` is display-only and never drives a security decision.
- OAuth authorization response issuer is validated.
- JSON Schema external references are not fetched implicitly.

### fast-agent behavior contracts

- `/mcp` reports the negotiated version and era.
- Modern HTTP does not display GET/resumption/session/ping as active.
- Legacy HTTP still reports its actual session and channels.
- Redaction prevents credentials and URL userinfo from entering status output.
- Tool/prompt/resource refresh does not interrupt unrelated in-flight calls.
- Modern served agents preserve state only with explicit application handles.
- Legacy roots/sampling/elicitation remain isolated to negotiated legacy mode.

### Existing suites requiring early attention

- `tests/unit/fast_agent/mcp/test_mcp_aggregator_nonpersistent.py`;
- `tests/unit/fast_agent/mcp/test_harness_app_server.py`;
- `tests/unit/fast_agent/ui/test_mcp_display.py`;
- integration tests under sampling, roots, elicitation, resources, and OAuth;
- provider converter tests using MCP content/tool models;
- ACP/A2A content conversion and trace export tests.

Use SDK v2's in-process client/server path for focused contracts and real stdio
or ASGI transport simulators for transport behavior. Do not patch copied SDK
private methods into tests; those are the implementation being removed.

## Documentation and configuration impact

At minimum update:

- `docs/docs/mcp/mcp-server.md`;
- `docs/docs/mcp/harness-adapter.md`;
- `docs/docs/mcp/huggingface-spaces.md`;
- `docs/docs/mcp/mcp_display.md`;
- generated configuration references;
- `examples/setup/fast-agent.yaml`;
- MCP server/client examples.

Documentation must stop presenting `Mcp-Session-Id` as a universal application
session, distinguish protocol era from configured transport, and explain the
legacy-only status of roots, sampling, logging, SSE, DCR, and ping settings.

Configuration supports an explicit protocol mode for interoperability testing
and migration control:

```yaml
mcp:
  servers:
    example:
      protocol_mode: auto  # auto | modern | legacy
```

`auto` is the default. `modern` adopts the latest modern version supported by
the pinned SDK without fallback; `legacy` forces the initialization handshake.
A forced legacy setting is a migration escape hatch, not the long-term solution
for stateful servers.

Legacy-only settings such as ping interval and SSE read timeout should either
be grouped under a compatibility section or accepted with a warning when a
modern connection is negotiated.

## Risks and open decisions

### Product decisions

1. What should `instance_scope="connection"` mean for modern MCP?
   Recommendation: reject it or require an explicit application session
   handle; do not emulate connection affinity.
2. Should `/mcp list` become the concise status surface?
   Recommendation: yes, while retaining `/mcpstatus` as an alias.
3. Should fast-agent adopt Tasks in the first compatibility release?
   Recommendation: no; ship core modern support first.
4. Should OpenAI Apps SDK support converge on MCP Apps?
   Recommendation: assess after core extension negotiation is in place.
5. How long should forced legacy roots/sampling/elicitation remain supported?
   Recommendation: publish a removal plan aligned with MCP's earliest removal
   revision.

### Engineering risks

- private SDK transport dependencies make a line-by-line port fragile;
- `httpx` and `httpx2` exception classes can coexist and silently bypass the
  wrong handler;
- FastMCP's compatibility bridge can hide incomplete snake_case migration;
- authorization-scoped cache mistakes can leak data;
- direct MCP types are used far outside `fast_agent.mcp`;
- OpenTelemetry instrumentation may lag SDK v2;
- modern no-session behavior can silently lose conversation continuity if the
  served-agent design is not changed first;
- pre-existing local Python SDK changes must not be mistaken for v2 behavior.

## Recommendation

Proceed with the migration, but make refactoring the connection boundary part
of the first implementation milestone.

The least safe approach is to update imports and then port
`streamable_http_tracking.py` until tests pass. That would preserve an
obsolete transport model and continue depending on private SDK internals.

The preferred approach is:

1. mechanically adopt `mcp_types` and snake_case;
2. introduce era-neutral peer and transport descriptors;
3. delegate negotiation, transport, MRTR, subscriptions, caching, and
   extensions to SDK v2/FastMCP v4;
4. retain fast-agent ownership of aggregation, UI, auth presentation, and
   product lifecycle;
5. redesign served-agent continuity around explicit application handles;
6. make `/mcp` the authoritative display of negotiated protocol and actual
   transport topology.

This is a large migration, but it should leave fast-agent with materially less
protocol code and a more stable boundary for future MCP revisions.
