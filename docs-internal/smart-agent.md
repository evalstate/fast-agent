# Smart Agent Design Notes

Date: 2026-01-25

## Goals

- Introduce a new **Smart** agent type that ships with a single built-in `smart` tool.
- Provide card management capabilities (list/read/write/load/check) and a way to spawn agents with a message (`run`).
- Use a sensible default card root (environment `agent-cards`), but **do not restrict** user-supplied paths.
- Keep tool surface stable (one tool with an `action` field).

## Proposed Agent Type

- Add `AgentType.SMART` (new enum value).
- Add card loader and dump support for `type: smart`.
- Provide a `SmartAgent` implementation (likely deriving from `AgentsAsToolsAgent` so it can also expose child agents as tools if configured).
- Inject a single tool named `smart` (via `ToolAgent.add_tool` or by overriding `list_tools`).

### Card Schema

```yaml
---
type: smart
name: Smart
instruction: ...
# optional: smart_root: ./agent-cards  # default root override (optional)
---
```

Notes:
- `smart_root` (if added) is only a **default**; explicit `path` values must be accepted.
- We intentionally **avoid** dynamic tool attachment (`attach`); Smart uses `run(agent, message)` instead.

### Allowed Fields for `smart` Type

```python
_ALLOWED_FIELDS_BY_TYPE["smart"] = {
    *_COMMON_FIELDS,
    # LLM/model configuration
    "model",
    "use_history",
    "request_params",
    "api_key",
    # MCP integration (the smart agent can use MCP servers/tools alongside its built-in tool)
    "servers",
    "tools",
    "resources",
    "prompts",
    # Skills
    "skills",
    # Human input support
    "human_input",
    # History/messages
    "messages",
    "history_source",
    "history_merge_target",
    # Child agents (exposed as tools via AgentsAsToolsAgent base)
    "agents",
    # Smart-specific
    "smart_root",  # default directory for card operations when path omitted
}
```

Note: `agents` IS included — the smart agent can have statically declared child agents
(exposed as tools via AgentsAsToolsAgent base) alongside its dynamic `smart` tool capabilities.

Required fields:
```python
_REQUIRED_FIELDS_BY_TYPE["smart"] = set()  # No required fields beyond common ones
```

Default use_history:
```python
_DEFAULT_USE_HISTORY_BY_TYPE["smart"] = True
```

## `smart` Tool Interface

### Actions (single tool)

- `list`: enumerate card definitions in a directory
- `read`: return card file content
- `write`: write card content
- `check`: validate card directory
- `load`: load cards into registry from file or directory
- `run`: spawn a named agent and send a message

### Parameters

Required:
- `action`: one of `list | read | write | check | load | run`

Optional:
- `path`: directory for `list`/`check`, file (or directory) for `read`/`write`/`load`
- `content`: for `write`
- `agent`, `message`: for `run`

### Defaults

If `path` is omitted for `list`/`check`, use:

```
resolve_environment_paths(context.config).agent_cards
```

### List results

`list` should return **raw error strings** from validation and include `description` when possible.

Behavior:
- Use `scan_agent_card_directory()` to enumerate files and collect `errors`.
- For entries **without errors**, load the card to extract `config.description`.
- For entries **with errors**, do **not** attempt a full load; return:
  - `status: "error"`
  - `errors: [...]` (raw validation errors)

### Read/Write format

- Use **Markdown** as the default format (frontmatter + body).
- `read` returns the file content as-is.
- `write` expects full Markdown content.

If YAML output is needed later, add a dedicated action (e.g. `dump`) rather than adding a `format` parameter.

## Implementation Touchpoints

- `src/fast_agent/agents/agent_types.py`
  - add `AgentType.SMART`
- `src/fast_agent/core/agent_card_loader.py`
  - `_TYPE_MAP`: add `"smart": AgentType.SMART`
  - `_ALLOWED_FIELDS_BY_TYPE`: add `"smart": {...}` (see above)
  - `_REQUIRED_FIELDS_BY_TYPE`: add `"smart": set()`
  - `_AGENT_TYPE_TO_CARD_TYPE`: add `AgentType.SMART.value: "smart"`
  - `_DEFAULT_USE_HISTORY_BY_TYPE`: add `"smart": True`
- `src/fast_agent/core/agent_card_validation.py`
  - add `"smart"` to the allowed type set in `_normalize_card_type`
- `src/fast_agent/core/direct_factory.py`
  - add `elif agent_type == AgentType.SMART:` branch to instantiate `SmartAgent`

## Registry/Manager Access

The `smart` tool needs access to:
- load/dump cards (`FastAgent.load_agents`, `FastAgent.dump_agent_card_text`)
- validation (`scan_agent_card_directory`)
- spawning agents (`spawn_detached_instance` on target agent)

We should define a small manager interface (similar to `AgentCardManager`) to pass into `SmartAgent` so it can call into the registry without depending on UI handlers.

---

# Introspect Tool Design (Read-only)

The `introspect` tool is a **read-only** system inspection tool. It reports the
current Fast-Agent configuration, environment, MCP server settings, provider
configuration, and available model metadata. It also exposes curated documentation
resources (for example, tool card format guidance).

## Interface

Single tool with an `action` field (stable tool surface):

```json
{
  "action": "summary | config | environment | providers | models | mcp | resources | resource",
  "provider": "optional provider id (e.g. openai, anthropic)",
  "model": "optional model name",
  "resource": "optional resource id"
}
```

### Actions

- `summary`
  - High-level snapshot of config, providers, and model defaults.
  - Convenience action for quick checks.
- `config`
  - Current `Settings` summary (default model, logger config, session history, MCP UI mode).
- `environment`
  - Environment directory and resolved paths.
- `providers`
  - Provider configuration status (configured via config/env, masked keys).
- `models`
  - Known model inventory and alias mapping.
- `mcp`
  - MCP server configuration summary (server names, transport, command/url, load flags).
- `resources`
  - List of available documentation resources.
- `resource`
  - Return a specific resource content by id.

## Data Sources

- **Settings**
  - `fast_agent.config.Settings`
  - Default model, logger config, session history, MCP settings, environment_dir.
- **Environment paths**
  - `resolve_environment_paths(settings)` from `fast_agent.paths`.
- **Providers**
  - Enumerate `Provider` (`fast_agent.llm.provider_types`).
  - For each provider, determine config/env presence using `ProviderKeyManager` and
    the `check_api_keys` logic in `fast_agent/cli/commands/check_config.py`.
  - Mask API keys (e.g., last 5 chars) and report the **source** (`config`, `env`,
    `DefaultAzureCredential`, or `none`).
- **Models**
  - `ModelDatabase.list_models()` for known models.
  - `ModelFactory.MODEL_ALIASES` for aliases.
  - `ModelFactory.DEFAULT_PROVIDERS` to map known models to a default provider.
  - Return `default_model` from settings and optionally the resolved model source
    via `resolve_model_spec` (config/env/hardcoded).
  - When model identifiers overlap with aliases, **aliases take precedence** in
    any merged or de-duplicated list.
- **MCP servers**
  - `Settings.mcp.servers` / `ServerRegistry.registry` to list server configs.

## Provider Status Output (example)

```json
{
  "provider": "openai",
  "configured": true,
  "source": "env",
  "key_hint": "...abcd1"
}
```

## Model Output (example)

```json
{
  "default_model": "gpt-5-mini.low",
  "known_models": ["gpt-4.1", "claude-3-5-sonnet-20241022", "gemini-2.5-pro", "..."],
  "aliases": {"sonnet": "claude-sonnet-4-5", "haiku": "claude-haiku-4-5"},
  "default_providers": {"gpt-4.1": "openai", "claude-3-5-sonnet-20241022": "anthropic"}
}
```

Note: if an alias maps to a model identifier already present in the known model list,
prefer the alias entry in merged outputs.

## Documentation Resources

Curated, read-only resources returned by `introspect(action="resource")`.

### Resource Packaging

Resources are stored in `.fast-agent/shared/` at the repository root and copied to
`src/fast_agent/resources/shared/` during the build process (via `hatch_build.py`).

**Build hook addition** (`hatch_build.py`):
```python
# Add to example_mappings or create shared_mappings:
shared_mappings = {
    ".fast-agent/shared": "src/fast_agent/resources/shared",
}
```

**Runtime access** uses `importlib.resources`:
```python
from importlib.resources import files

def load_internal_resource(resource_id: str) -> str:
    """Load a packaged internal resource by ID."""
    resource_path = (
        files("fast_agent")
        .joinpath("resources")
        .joinpath("shared")
        .joinpath(f"{resource_id}.md")
    )
    if resource_path.is_file():
        return resource_path.read_text()
    raise ValueError(f"Unknown internal resource: {resource_id}")
```

**URI scheme**: Resources are referenced as `internal:<resource_id>`, e.g.:
- `internal:tool-card-format`
- `internal:agent-card-overview`
- `internal:agent-card-rfc`

### Source Files (`.fast-agent/shared/`)

| Resource ID | Source File | Description |
|-------------|-------------|-------------|
| `tool-card-format` | `.fast-agent/shared/tool-card-format.md` | Tool card design guidance |
| `agent-card-overview` | `.fast-agent/shared/agent-card-overview.md` | Agent card implementation overview |
| `agent-card-rfc` | `.fast-agent/shared/agent-card-rfc.md` | Agent card RFC specification |

**Initial population**: Copy content from current locations:
- `examples/hf-toad-cards/skills/tool-card-eval/references/tool-card-design.md` → `tool-card-format.md`
- `docs-internal/AGENT_CARD_IMPLEMENTATION_OVERVIEW.md` → `agent-card-overview.md`
- `plan/agent-card-rfc.md` → `agent-card-rfc.md`

Notes:
- The tool returns resource **content** and source **path** (the `internal:` URI).
- Resources are versioned with the package—no external file dependencies at runtime.

## Non-goals

- No mutation of configuration or registry state.
- No file writes.
- No provider credential disclosure (masked only).
