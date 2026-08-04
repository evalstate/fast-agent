---
social:
  title: Plugins
  tagline: Add slash commands and post-turn displays to fast-agent.
  description: Add slash commands and post-turn displays to fast-agent.
  alt: fast-agent social card — Plugins
---

# Plugins

Plugins package reusable slash commands such as `/find`, `/peek`, or
`/edit-last`, post-user-turn displays, or both. They install into the active
fast-agent home under `.fast-agent/plugins/` and are enabled by name from
`fast-agent.yaml`.

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

Plugin registries are used for direct plugin installs and updates. Card-pack
dependencies use the card-pack registry that supplied the selected pack; see
[Card Packs](#card-packs) for the coupling rule.

## Global Plugins

Global plugin installs write to `FAST_AGENT_HOME` when it is set; otherwise
they use `~/.fast-agent`. The plugin is enabled in that directory's
`fast-agent.yaml`:

```bash
FAST_AGENT_HOME=~/.fast-agent fast-agent plugins add agent-finder --global
```

When `FAST_AGENT_HOME` is set, plugin names from
`$FAST_AGENT_HOME/fast-agent.yaml` are merged with the active fast-agent home,
including when you run with `--home <dir>`. If `FAST_AGENT_HOME` is not set,
`~/.fast-agent/fast-agent.yaml` is used as the global plugin layer when it
exists.

Only the global file's `plugins` block is merged; other settings still come
from the normal active config. Global plugins are loaded from the global
`plugins/` directory, while project plugins are loaded from the active
environment's `plugins/` directory. This allows a central set of slash commands
to be available across projects while still letting each project enable
additional plugins.

Project plugin commands override global commands with the same name. Inline
`commands:` entries in the active config override both.

Plugin-specific configuration can be stored under `plugins.config`:

```yaml
plugins:
  enabled:
    - agent-finder
  config:
    agent-finder:
      urls:
        - https://evalstate-hf-agentfinder.hf.space/search
      page_size: 10
      prompt_when_multiple: true
```

Plugins can read their namespaced configuration from
`ctx.settings.plugins.config` and fall back to defaults when it is missing.

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

### Post-user-turn display hooks

Manifest schema v2 adds one optional display hook:

```yaml
schema_version: 2
name: price-calculator
version: 0.1.0
description: Display estimated cost for the last turn and current session.
hooks:
  post_user_turn: ./price_calculator.py:display_cost
```

`post_user_turn` runs once after each successful top-level interactive user
turn. It does not run for quiet nested sends, failed turns, programmatic
`agent.send(...)` calls, or once per subagent. This boundary ensures parallel
workflows, tool loops, retries, and subagents contribute to one consolidated
display.

The handler may be synchronous or asynchronous and returns trusted Rich markup
as `str`, or `None` to display nothing:

```python
from fast_agent.plugins import PluginPostUserTurnContext


def show_calls(ctx: PluginPostUserTurnContext) -> str | None:
    if not ctx.turn_usage:
        return None
    return (
        f"[dim]Provider calls:[/dim] {len(ctx.turn_usage)} last · {len(ctx.session_usage)} session"
    )
```

The context contains:

- `plugin_name` — manifest plugin name.
- `agent_name` — top-level agent selected by the user.
- `turn_usage` — canonical provider attempts for the completed user turn.
- `session_usage` — cumulative canonical provider attempts for the selected
  agent or parallel workflow.
- `config` — the plugin's mapping from `plugins.config.<plugin-name>`.

Handlers run in enabled-plugin order. A load or execution failure is logged and
does not fail the completed agent turn or prevent later display plugins from
running. Hooks require `schema_version: 2`; schema v1 remains supported for
command-only plugins.

### Price calculator

The
[card-packs registry](https://github.com/fast-agent-ai/card-packs/tree/main/plugins/price-calculator)
includes a price calculator using hardcoded GPT-5.6, Kimi K3, and DeepSeek V4
Flash rates. Install and enable it with:

```bash
fast-agent plugins add price-calculator
```

Use `/cost` for one rollup per top-level user turn, including explanatory
subagent or parallel-child ledgers. Use `/cost detail` for the provider-attempt
table with model, service tier, context band, token/cache partitions, and
estimated cost.

GPT-5.6 prompts above 272,000 tokens use the long-context table. Standard and
Flex prices are supported; Fast-tier calls are shown as unpriced until a Fast
price table is configured. Kimi K3 and DeepSeek V4 Flash cache writes are also
shown as unpriced because their supplied tables do not specify a cache-write
rate. Unknown calls never count as zero: the display labels partial totals with
the number of unpriced model calls.

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
  "entries": [
    {
      "name": "my-pack",
      "description": "My local card pack.",
      "kind": "card",
      "repo_url": ".",
      "repo_path": "packs/my-pack"
    }
  ],
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

--8<-- "_generated/plugin_api.md"

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

Required plugins are resolved from the same marketplace that supplied the
selected card pack. This keeps private/custom registries self-contained:

```bash
fast-agent cards --registry ./my-packs.json add codex
```

If `./my-packs.json` contains the `codex` entry above, it should also contain a
matching `command_plugins` entry for `edit-assistant`. `fast-agent` will use
that registry for the pack's required plugins during install and update, even if
your normal plugin registry points somewhere else.

You can still keep a separate plugin registry for ad hoc plugin installs:

```yaml
plugins:
  marketplace_urls:
    - ./my-plugins.json
```

That registry is used by `fast-agent plugins add ...`. It is not the dependency
source of truth for a card pack installed from another registry. The simple rule
is: if a pack declares `plugins.required`, publish matching `command_plugins`
entries alongside that pack in the same marketplace file.
