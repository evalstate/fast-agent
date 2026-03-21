
Detailed information about the MCP Server connection can be displayed with the `/mcp` command.

![](./pics/mcp_transport_display.png)

### Section 1 - Implementation and Session

This section shows the MCP Server Implementation Details (`Name` and `Version`), along with any `Mcp-Session-Id` allocated by the MCP Server.

### Section 2 - Transport Channel History

Shows activity from the Streamable HTTP GET and POST handlers for the MCP Server. 

### Section 3 - Server Capabilities

- `To`, `Pr`, `Re`: Tools, Prompts and Resources. Green for available, Yellow for List Change notifications.
- `Rs`: Resource Subscriptions.
- `Lo`, `Co`: Logging and Completions.
- `Ex`: Experimental Capabilities
- `In`: Instructions. Green for available, and used - Yellow for available but not in Prompt, Red for available, but disabled.

### Section 4 - Client Capabilities.

- `Ro`: Roots offered to MCP Server.
- `El`: Elicitation offered to MCP Server. Red for `Cancel All` mode.
- `Sa`: Sampling offered to MCP Server. Green for auto, Yellow for manually configured.
- `Sp`: MCP Client Name has been spoofed.

### Configuration

The activity timeline shown in the transport section can be tailored in `fastagent.config.yaml` via the `mcp_timeline` block:

```yaml
mcp_timeline:
  steps: 20         # number of buckets rendered on the timeline
  step_seconds: 30  # duration of each bucket (supports values like "45s" or "2m")
```

These values flow through to both `fast-agent check` and the in-session `/mcp` display. When multiple events occur in the same bucket, higher priority states replace lower ones using this order: `error` → `disabled/request` → `response` → `notification/ping` → `none`. This keeps significant events (such as errors and requests) visible even if a subsequent ping lands in the same interval.
