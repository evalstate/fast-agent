---
title: Migrate MCP Configuration
description: Migrate legacy MCP settings to the canonical nested schema.
---

# Migrate MCP Configuration

- Preview changes without modifying the file:
  `fast-agent config migrate-mcp path/to/fast-agent.yaml`.
- Apply changes with
  `fast-agent config migrate-mcp path/to/fast-agent.yaml --write`. The command
  saves the exact original as `path/to/fast-agent.yaml.bak`.
- Moved paths: `mcp.targets` → `mcp.servers`, top-level `auto_sampling` →
  `mcp.client.auto_sampling`, and top-level `mcp_timeline` →
  `mcp.diagnostics.timeline`.
- Resolve conflicts before rerunning. The command refuses files containing both
  an old and new path, duplicate migrated server names, or a `target` combined
  with source fields such as `transport`, `url`, `command`, `args`, or
  `connector_id`.
- Current schema: [Connect to MCP Servers](client-servers.md) and
  [Configuration Reference](../ref/config_file.md).
