# Command Plugins

Command plugins package reusable slash commands such as `/find`, `/peek`, or
`/edit-last`. They install into the active fast-agent home under
`.fast-agent/plugins/` and are enabled by name from `fast-agent.yaml`.

```yaml
plugins:
  enabled:
    - agent-finder
    - edit-assistant
```

Install a plugin from the configured plugin registry:

```bash
fast-agent plugins add agent-finder
```

Manage installed plugins:

```bash
fast-agent plugins list
fast-agent plugins update
fast-agent plugins update all --yes
fast-agent plugins remove agent-finder
```

`fast-agent plugins list` shows each installed plugin, the slash commands it
adds, and any configured key bindings from the plugin manifest.

Use `--registry` to point at a local or remote marketplace:

```bash
fast-agent plugins --registry ./marketplace.json add agent-finder
```

## Global Plugins

Global plugin installs write to `FAST_AGENT_HOME` and enable the plugin in
`$FAST_AGENT_HOME/fast-agent.yaml`:

```bash
FAST_AGENT_HOME=~/.fast-agent fast-agent plugins add agent-finder --global
```

When `FAST_AGENT_HOME` is set, plugin names from
`$FAST_AGENT_HOME/fast-agent.yaml` are merged with the project configuration.
Only the global file's `plugins` block is merged; other settings still come
from the normal active config. Global plugins are loaded from
`$FAST_AGENT_HOME/plugins`, while project plugins are loaded from the active
environment's `plugins/` directory. This allows a central set of slash commands
to be available across projects while still letting each project enable
additional plugins.

If `FAST_AGENT_HOME` is not set, `--global` exits with a warning instead of
guessing a location.

## Plugin Manifests

A plugin is a directory containing `plugin.yaml`:

```yaml
schema_version: 1
name: agent-finder
version: 0.1.0
description: Discover skills and MCP servers.
commands:
  find:
    description: Discover skills and MCP servers with Agent Finder
    input_hint: "<query>"
    handler: ./agentfinder.py:find
```

Handlers use the same async command-action API as inline `commands:` entries.
Relative handler paths resolve from the plugin directory, so published plugins
can be moved between environments without editing command paths.

## Build a Plugin

The easiest development loop is to create a local plugin directory, point a
local marketplace at it, and install from that marketplace:

```text
my-plugin/
  plugin.yaml
  commands.py
```

```yaml
# my-plugin/plugin.yaml
schema_version: 1
name: my-plugin
version: 0.1.0
description: Developer tools for my workflow.
commands:
  draft-reply:
    description: Draft a reply from the current conversation
    input_hint: "[tone]"
    handler: ./commands.py:draft_reply
    key: "c-x r"
```

```python
# my-plugin/commands.py
from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionResult,
)


async def draft_reply(ctx: PluginCommandActionContext) -> PluginCommandActionResult:
    tone = ctx.arguments.strip() or "concise"
    last_message = ctx.message_history[-1] if ctx.message_history else None
    del last_message
    return PluginCommandActionResult(
        buffer_prefill=f"Draft a {tone} reply to the last user request."
    )
```

A local marketplace can live at the repository root:

```json
{
  "command_plugins": [
    {
      "name": "my-plugin",
      "description": "Developer tools for my workflow.",
      "repo_url": ".",
      "repo_path": "my-plugin"
    }
  ]
}
```

Install from the local marketplace:

```bash
fast-agent plugins --registry ./marketplace.json add my-plugin
```

For publication, add the plugin directory under the card-packs repository's
`plugins/` directory and add a `command_plugins` entry to its `marketplace.json`.

## Handler API

Handlers are async Python callables with this signature:

```python
async def handler(
    ctx: PluginCommandActionContext,
) -> PluginCommandActionResult | str | None:
    ...
```

Returning a plain string is shorthand for `PluginCommandActionResult(message=...)`.
Return `None` for no visible output.

The main result fields are:

| Field | Effect |
|-------|--------|
| `message` | Plain text shown in the command output. |
| `markdown` | Markdown output rendered by the UI. |
| `buffer_prefill` | Draft text inserted into the user's input buffer. |
| `switch_agent` | Switch the active TUI agent after the command. |
| `refresh_agents` | Refresh agent/card state after the command. |

The main context fields and helpers are:

| API | Description |
|-----|-------------|
| `ctx.command_name` | Slash command name being executed. |
| `ctx.arguments` | Raw text after the slash command. |
| `ctx.agent_name` | Active agent name. |
| `ctx.message_history` | Current agent message history. |
| `ctx.load_message_history(messages)` | Replace the active agent's message history. |
| `ctx.get_agent(name)` | Look up another registered agent. |
| `ctx.settings` | Resolved fast-agent settings, when available. |
| `ctx.session_cwd` | Working directory for the interactive session, when available. |
| `ctx.runtime` | Optional live-runtime capabilities. |
| `ctx.is_tui` / `ctx.is_acp` | Surface flags for UI-specific behavior. |
| `ctx.mark_user_adjusted(message, note=...)` | Mark a message as user-adjusted in the audit channel. |

Runtime capabilities are optional because not every surface can support live
changes:

```python
if ctx.runtime is not None:
    attached = await ctx.runtime.list_attached_mcp_servers()
```

The runtime API currently includes:

| API | Description |
|-----|-------------|
| `list_attached_mcp_servers(agent_name=None)` | List attached runtime MCP servers. |
| `list_configured_detached_mcp_servers(agent_name=None)` | List configured but detached MCP servers. |
| `attach_mcp_server(server_name=..., agent_name=None, server_config=None, options=None)` | Attach an MCP server and refresh instructions. |
| `detach_mcp_server(server_name=..., agent_name=None)` | Detach an MCP server and refresh instructions. |

## Card Packs

Card packs can reference command plugins by name in manifest schema v2:

```yaml
schema_version: 2
name: codex
kind: card
install:
  agent_cards:
    - agent-cards/dev.md
  files:
    - fast-agent.yaml
plugins:
  required:
    - edit-assistant
  recommended:
    - agent-finder
```

Required plugins are installed and enabled when the pack is installed.
Recommended plugins are discoverable metadata for users and future tooling.
