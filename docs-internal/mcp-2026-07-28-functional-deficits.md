# MCP 2026-07-28 functional-deficit ledger

This ledger records behavior intentionally not reimplemented during the MCP
SDK v2 / FastMCP v4 migration. Entries remain open until functionality is
restored, replaced, or explicitly removed as a product decision. `Owner`
identifies the party that can unblock the next action; `Blocker` records why
the entry cannot be closed today.

Current release baseline: `mcp==2.0.0`, `mcp-types==2.0.0`, and
`fastmcp-slim[server]==4.0.0b1`.

| ID | Area | Status | Owner | Blocker | Deficit | Recovery |
| --- | --- | --- | --- | --- | --- | --- |
| MCP2-004 | HTTP OAuth events | Upstream blocked | FastMCP | Public OAuth lifecycle and paste-fallback hooks | Detailed OAuth timing and paste fallback still depend on the migrated `_ProtectedResourceDiscoveryOAuthClientProvider` compatibility subclass. | Replace the subclass when upstream exposes equivalent lifecycle callbacks; retain focused compatibility coverage meanwhile. |
| MCP2-007 | MCP OpenTelemetry auto-instrumentation | Upstream blocked | OpenTelemetry instrumentation package | Current instrumentor patches a removed SDK v1 symbol | MCP-specific third-party auto-instrumentation is not installed or activated. `opentelemetry-instrumentation-mcp==0.62.1` crashes setup with SDK v2 because it patches `mcp.client.streamable_http.streamablehttp_client`. | Re-add and enable `McpInstrumentor` when an SDK-v2-compatible release is available. Fast-agent's own MCP progress and timing telemetry remains active. |

## Resolved or explicitly closed

| ID | Area | Status | Owner | Resolution | Verification |
| --- | --- | --- | --- | --- | --- |
| MCP2-001 | HTTP diagnostics | Resolved | fast-agent | Public HTTP response hooks now classify final POST JSON/SSE responses and record GET and `Last-Event-ID` resumption activity. | Unit coverage exercises mixed response modes and redirect-final response classification. |
| MCP2-002 | Legacy HTTP session diagnostics | Resolved | fast-agent | Legacy `Mcp-Session-Id` diagnostics are recovered from the public HTTP client's response event hook without removed SDK transport internals. | Connection-manager and HTTP simulator coverage. |
| MCP2-003 | Modern health | Removed by specification | MCP specification / fast-agent | Modern negotiation does not schedule removed `ping`; health uses negotiation, operation result, subscription, transport, and stdio process state. | Modern integration asserts zero ping activity and an open subscription. |
| MCP2-005 | Transport result channel | Explicitly retired | Product | Individual MCP `CallToolResult` objects will not receive synthetic channel metadata. `/mcp` retains server-level transport diagnostics and can observe modern `Mcp-Method`/`Mcp-Name` headers without coupling them to result objects. | Product decision; no MCP execution path currently writes per-result channel metadata. |
| MCP2-006 | Negotiation | Resolved | fast-agent | All negotiation runs through public `mcp.client.Client(mode="auto")`; fast-agent no longer imports `_probe.negotiate_auto`. | Modern auto and forced-mode integration coverage. |
| MCP3-001 | High-level MRTR client | Resolved | fast-agent | Tool, prompt, and resource MRTR use the SDK high-level client; custom `*_complete` methods and the private input-required driver were removed. | Modern MRTR integration coverage. |
| MCP3-002 | Attached response cache | Resolved | fast-agent | Attached runtimes use the SDK per-client response cache; request-scoped clients disable caching for authorization isolation. Modern listeners use the SDK eviction barrier. | Modern tool-list cache refresh integration coverage. |
| MCP3-003 | Subscription convergence | Resolved | fast-agent | Every acknowledged listener epoch force-refreshes and atomically recommits attachment-derived indexes before event consumption. Lost epochs retain bounded retry; partial acknowledgments retain the honored stream and add periodic authoritative refresh. | Simulator coverage drops an epoch, exercises partial acknowledgment, and verifies post-reconnect convergence. |
| MCP3-004 | Resource subscriptions | Resolved | fast-agent | Modern listeners maintain a canonical tuple of materialized UI resource URIs and rotate serially when authoritative discovery changes it. Initial attachment readiness prevents an empty-filter race; transient authoritative refresh failures retain the prior committed URI set and retry. | Unit and HTTP integration coverage verify initial rotation, resource-list rotation, updates, atomic refresh failure, and no overlapping listeners. |
| MCP3-005 | Shared cache | Explicitly unsupported | Product | Fast-agent will not provide persistent or cross-principal MCP protocol response caching. Attached caches remain client-local and request-scoped clients remain uncached. | Product decision; preserves principal isolation by construction. |
| MCP3-006 | Form elicitation examples | Resolved | fast-agent | Modern elicitation examples and integration simulators use SDK `MCPServer` resolver MRTR. Intentional standalone FastMCP cases are explicitly legacy. | Custom-handler elicitation integration coverage. |
| MCP3-008 | Modern stdio subscriptions | Resolved | MCP SDK / fast-agent | MCP SDK `2.0.0` supports `subscriptions/listen` over modern stdio; the supervisor shares the HTTP and stdio lifecycle. | Modern stdio negotiation integration coverage. |
| MCP3-009 | Sampling tool choice | Resolved | fast-agent | MCP `auto`, `required`, and `none`—including omitted choice as implicit `auto`—are converted to a provider-neutral policy and deliberately mapped at OpenAI, Responses, Anthropic, Google, and Bedrock boundaries. Provider-managed tools are excluded from MCP-scoped sampling. | Provider request-builder contract coverage. |
| MCP3-010 | Sampling and roots examples | Resolved | fast-agent | Modern sampling-with-tools and roots simulators use SDK `Resolve(Sample(...))` and `Resolve(ListRoots())`; retained standalone callback fixtures are explicitly legacy. | Sampling-with-tools and roots integration coverage. |
