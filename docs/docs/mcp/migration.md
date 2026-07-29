---
title: Migrate MCP Configuration
description: Migrate legacy MCP settings to the canonical nested schema.
---

# Migrate MCP Configuration

1. Preview without modifying the file:
   `fast-agent config migrate-mcp path/to/fast-agent.yaml`.
2. Resolve any reported conflicts.
3. Apply with
   `fast-agent config migrate-mcp path/to/fast-agent.yaml --write`. The command
   saves the exact original as `path/to/fast-agent.yaml.bak`.
4. Inspect the redacted result with
   `fast-agent config show-mcp path/to/fast-agent.yaml --view effective`.

The primary shape changes from a named list:

```yaml
mcp:
  targets:
    - name: docs
      target: https://example.com/mcp
```

to a server map:

```yaml
mcp:
  servers:
    docs:
      target: https://example.com/mcp
```

- Moved paths: `mcp.targets` → `mcp.servers`, top-level `auto_sampling` →
  `mcp.client.auto_sampling`, and top-level `mcp_timeline` →
  `mcp.diagnostics.timeline`.
- Resolve conflicts before rerunning. The command refuses files containing both
  an old and new path, duplicate migrated server names, or a `target` combined
  with source fields such as `transport`, `url`, `command`, `args`, or
  `connector_id`.
- Runtime commands are now explicit: `/mcp attach docs` attaches the configured
  `docs` definition; `/connect TARGET` and `/mcp connect TARGET` always create
  an ad-hoc session definition. `/mcp` and `/mcp status` show status.
- `/mcpstatus` remains available temporarily and emits a deprecation warning.
- Startup `--url` is repeatable. Prefer one flag per URL; comma-separated input
  is temporary compatibility behavior.
- Current schema: [Connect to MCP Servers](client-servers.md) and
  [Configuration Reference](../ref/config_file.md).
