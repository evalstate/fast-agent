# MCP 2026-07-28 real-server validation

Validated on 24 July 2026 from branch `feat/mcp-2026-07-28`.

## Official server-everything

Source:

- repository: `modelcontextprotocol/servers`
- commit: `d31124c982401739917fd817c2a59db344529c16`
- package: `@modelcontextprotocol/server-everything@2.0.0`
- SDK dependency: `@modelcontextprotocol/sdk@^1.29.0`

The checkout was built outside the fast-agent tree:

```bash
git clone --depth 1 https://github.com/modelcontextprotocol/servers.git \
  /tmp/modelcontextprotocol-servers
cd /tmp/modelcontextprotocol-servers/src/everything
npm install
npm run build
```

The current fast-agent client passed the same operation probe over stdio and
Streamable HTTP:

- 19 tools discovered;
- 7 resources discovered;
- 4 prompts discovered;
- `echo` and `get-sum` calls;
- structured tool output;
- dynamic resource read;
- prompt retrieval;
- legacy server-initiated form elicitation through a deterministic callback;
- legacy sampling through the passthrough model callback.

The legacy sampling probe was repeated after replacing the custom
`ClientSession` boundary with public `mcp.client.Client`: the
`trigger-sampling-request` tool completed successfully through the SDK callback
table on negotiated `2025-11-25`.

Both transports correctly negotiated the real server as:

```text
protocol_version=2025-11-25
protocol_era=legacy
negotiation=initialize
```

## SDK v2 reference server

An SDK `MCPServer` fixture negotiated:

```text
protocol_version=2026-07-28
protocol_era=modern
supported_protocol_versions=(2026-07-28,)
negotiation=discover
```

Verified behaviors:

- tool, prompt, and resource operations;
- no legacy ping activity;
- SDK-native multi-round-trip elicitation using
  `Resolve(Elicit(...))`;
- automatic client callback dispatch and retry to a complete tool result;
- HTTP `subscriptions/listen`;
- SDK response-cache hits on repeated `tools/list` calls;
- tool-list change delivery with SDK cache eviction before fast-agent refresh.

The external `server-everything` and Hugging Face checks above are manual
network/process probes. Durable SDK-v2 integration coverage under
`tests/integration/mcp_2026/` covers modern negotiation, tool/prompt/resource
operations, no legacy ping activity, MRTR, response caching, and HTTP
tool-list subscriptions.

## Hugging Face MCP

Anonymous discovery against `https://huggingface.co/mcp` succeeded with MCP
Python SDK `2.0.0rc1` and an isolated `HF_HOME` so a developer credential could
not affect the probe:

```bash
HF_HOME=/tmp/empty-hf-home uv run --project . python /tmp/fa-hf-probe/probe.py
```

Observed on 27 July 2026:

```text
server=@huggingface/mcp-services 0.4.1
protocol_version=2026-07-28
protocol_era=modern
supported_protocol_versions=(2026-07-28,)
negotiation=discover
```

Anonymous `tools/list` returned the four-tool public Hugging Face surface and
`hf_whoami` completed successfully. Server identity is stamped in
`result._meta["io.modelcontextprotocol/serverInfo"]`, matching the final
post-specification-PR-3002 discover schema.

The earlier beta2 validation incorrectly reported this endpoint as legacy:
`mcp-types==2.0.0b2` still required the removed top-level
`DiscoverResult.serverInfo`, so `mode="auto"` silently fell back to
`initialize`. The final-schema contract is covered by
`tests/unit/fast_agent/mcp/test_discover_contract.py`.

A local cached Hugging Face token is automatically forwarded by fast-agent; an
invalid cached token can make this probe fail, which is why the validation
isolates `HF_HOME`.

## Existing examples

The elicitation integration suite currently passes 4 of 7 selected cases.
The three custom-handler failures are the FastMCP standalone-elicitation
deficit recorded as `MCP3-006`. A separate SDK-native MRTR probe passes.

The state-transfer example successfully exposes and calls `agent_one`, but
does not expose the documented `agent_one_history` prompt. This is recorded as
`MCP3-007`.
