# Diagnose an MCP server

```bash
fast-agent mcp diagnose SERVER
fast-agent mcp diagnose SERVER --json --timeout 15
fast-agent mcp diagnose SERVER --config ./fast-agent.yaml
```

`SERVER` is a configured `mcp.servers` name, not a URL. The command uses the
normal configuration/home selection, server registry, connection manager,
transport, headers and OAuth handling. It does not require a model or provider
API key. Only the selected server is connected.

The probe negotiates a connection, then lists advertised tools, prompts,
resources and resource templates, following pagination. Unsupported capabilities
are skipped. It never calls tools, reads resources or retrieves prompts. Server
sampling and elicitation callbacks are disabled, including on connection retries.
Configured server processes still run, and OAuth may open a browser; authenticate
first with `fast-agent auth mcp login SERVER` for unattended diagnostics.

Reports include negotiated protocol/negotiation, capability presence, per-phase
elapsed seconds, page/item counts, errors, and cleanup status. Discovery uses
`refresh`, not cached list results. `capability_cache` describes the fresh
registry's capability cache, not the server's negotiated capabilities or another
running agent's cache. No cache contents, item descriptions, server instructions,
endpoints or headers are printed verbatim. Failures include exception classes and
redacted cause details (up to eight exceptions, 2,000 characters per exception,
4,000 characters total), plus config/auth/reachability guidance. Recent stdio
stderr is included when the production connection error already contains it;
the command does not replay raw subprocess or SDK output. URLs, common credential
assignments, cookies and authorization values are redacted; terminal controls and
markup are neutralized. Avoid putting credentials in unlabeled error messages:
redaction cannot recognize arbitrary secret text. Underlying diagnostic output
remains suppressed in both text and JSON modes.

`--timeout` is a positive finite connection-and-discovery budget (default 30
seconds), including OAuth waits. Cleanup cancels connection lifecycle tasks and
has a separate five-second budget. As with other async connection handling,
resource release depends on transports cooperating with cancellation.

Exit status is **0** for successful supported probes and cleanup, **1** for a
configuration, connection, discovery or cleanup failure, and **2** for invalid
CLI arguments. `--json` emits a single report with `server`, `ok`, `protocol`,
`negotiation`, `capabilities`, `capability_cache`, `oauth_events` (event kinds only)
and `phases`. Failed discovery
retains completed phase results and partial counts; later phases may be absent
when the total budget expires.
