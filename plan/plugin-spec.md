# Plugin Runtime Capability Spec

## Goal

Expose a small, stable runtime facade to plugin command actions so plugins can
mutate live agent capabilities without depending on concrete agent classes,
slash-command handlers, or internal rebuild details.

Initial capabilities:

- Attach/detach MCP servers at runtime.
- Refresh skills after a plugin writes or installs skill files.

The API should make common plugin code straightforward while keeping the
implementation free to reuse the existing `AgentApp` runtime callbacks and
instruction refresh machinery.

## Current state

Plugin command actions receive a `PluginCommandActionContext`:

```python
@dataclass(frozen=True, slots=True)
class PluginCommandActionContext:
    command_name: str
    arguments: str
    agent: PluginCommandAgentProtocol
    settings: Settings | None = None
    session_cwd: Path | None = None
```

Today a plugin can technically reach live MCP and skill refresh behavior by
using concrete/internal APIs, for example:

- `McpAgentProtocol.attach_mcp_server(...)`
- `rebuild_agent_instruction(...)`
- skill registry reload helpers

That works, but it is not a supported plugin contract.

## Proposal

Add a plugin runtime facade to `PluginCommandActionContext`:

```python
@dataclass(frozen=True, slots=True)
class PluginCommandActionContext:
    command_name: str
    arguments: str
    agent: PluginCommandAgentProtocol
    settings: Settings | None = None
    session_cwd: Path | None = None
    runtime: PluginRuntime | None = None
```

`runtime` is optional for backward compatibility and for execution contexts
that cannot provide live runtime mutation.

Plugin authors should feature-test the facade:

```python
if ctx.runtime is None:
    return PluginCommandActionResult(message="Runtime capabilities are not available.")
```

## Runtime protocol

Minimal v1:

```python
from __future__ import annotations

from typing import Protocol

from fast_agent.config import MCPServerSettings
from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult


class PluginRuntime(Protocol):
    """Stable live-runtime capabilities exposed to plugin command actions."""

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        """Attach an MCP server to a running MCP-capable agent and refresh instructions."""

    async def detach_mcp_server(
        self,
        *,
        server_name: str,
        agent_name: str | None = None,
    ) -> MCPDetachResult:
        """Detach an MCP server from a running MCP-capable agent and refresh instructions."""

    async def refresh_skills(
        self,
        *,
        agent_name: str | None = None,
    ) -> PluginSkillRefreshResult:
        """Reload skill manifests and rebuild the target agent's instructions."""
```

### Agent targeting

If `agent_name` is omitted, methods operate on the plugin context's current
agent.

This keeps common plugin code terse:

```python
await ctx.runtime.attach_mcp_server(
    server_name="github",
    server_config=server_config,
)
```

Plugins can still target another registered agent explicitly:

```python
await ctx.runtime.refresh_skills(agent_name="planner")
```

## Result types

Skill refresh should return simple structured data rather than a UI-oriented
`CommandOutcome`.

```python
from dataclasses import dataclass

from fast_agent.skills import SkillManifest
from fast_agent.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class PluginSkillRefreshResult:
    agent_name: str
    manifests: tuple[SkillManifest, ...]
    registry: SkillRegistry
    skill_count: int
```

For MCP attach/detach, v1 can reuse existing runtime result types:

- `MCPAttachResult`
- `MCPDetachResult`

If those are too internal or unstable, add plugin-specific wrappers later.

## Optional future methods

These are intentionally not part of minimal v1, but the protocol should leave
room for them.

```python
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PluginSkillInstallResult:
    agent_name: str
    name: str
    path: Path
    refreshed: PluginSkillRefreshResult


@dataclass(frozen=True, slots=True)
class PluginSkillRemoveResult:
    agent_name: str
    name: str
    removed_paths: tuple[Path, ...]
    refreshed: PluginSkillRefreshResult


class PluginRuntime(Protocol):
    async def install_skill(
        self,
        skill: str,
        *,
        agent_name: str | None = None,
        registry_url: str | None = None,
    ) -> PluginSkillInstallResult:
        """Install a marketplace skill, refresh manifests, and rebuild instructions."""

    async def remove_skill(
        self,
        skill: str,
        *,
        agent_name: str | None = None,
    ) -> PluginSkillRemoveResult:
        """Remove a managed skill, refresh manifests, and rebuild instructions."""
```

A lower-level instruction refresh hook may also be useful, but should be added
carefully because it allows plugins to alter instruction context directly:

```python
async def refresh_agent_instruction(
    self,
    *,
    agent_name: str | None = None,
) -> None:
    """Rebuild the target agent's instruction from current runtime state."""
```

## Example plugin: connect an MCP server

```python
from fast_agent.command_actions.models import (
    PluginCommandActionContext,
    PluginCommandActionResult,
)
from fast_agent.config import MCPServerSettings


async def connect_github(ctx: PluginCommandActionContext) -> PluginCommandActionResult:
    if ctx.runtime is None:
        return PluginCommandActionResult(message="Runtime capabilities are not available.")

    server_config = MCPServerSettings(
        command="uvx",
        args=["mcp-server-github"],
    )

    await ctx.runtime.attach_mcp_server(
        server_name="github",
        server_config=server_config,
    )

    return PluginCommandActionResult(message="Connected MCP server: github")
```

## Example plugin: write a skill then refresh

```python
from pathlib import Path

from fast_agent.command_actions.models import (
    PluginCommandActionContext,
    PluginCommandActionResult,
)


async def add_local_skill(ctx: PluginCommandActionContext) -> PluginCommandActionResult:
    if ctx.runtime is None:
        return PluginCommandActionResult(message="Runtime capabilities are not available.")

    skill_dir = Path(".fast-agent/skills/example")
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_dir.joinpath("SKILL.md").write_text(
        "# Example\n\nUse this skill for example tasks.\n",
        encoding="utf-8",
    )

    refreshed = await ctx.runtime.refresh_skills()

    return PluginCommandActionResult(
        message=f"Skills refreshed. Loaded {refreshed.skill_count} skills."
    )
```

## Implementation notes

### Runtime facade construction

The plugin dispatch sites should construct a runtime facade next to the existing
`PluginCommandActionContext`.

Interactive dispatch and ACP slash dispatch should pass equivalent capability
objects so plugin behavior is consistent across transports.

### MCP implementation

The facade should reuse existing `AgentApp` callbacks where available:

- `attach_mcp_server(...)`
- `detach_mcp_server(...)`
- `list_attached_mcp_servers(...)`, if added later
- `list_configured_detached_mcp_servers(...)`, if added later

The existing callback path already performs instruction rebuild after attach or
detach.

### Skill refresh implementation

`refresh_skills(...)` should mirror the current `/skills` refresh behavior:

1. Resolve skill directories from settings.
2. Reload skill manifests.
3. Format skill instructions.
4. Rebuild the target agent instruction with the refreshed manifests, registry,
   and instruction context.

It should not require a full `CommandContext` or UI `CommandIO`.

## Error behavior

Runtime methods should raise normal Python exceptions for programmer-visible
failures, matching existing plugin command behavior where dispatcher catches
exceptions and reports command failure.

Expected examples:

- target agent not found
- target agent does not support MCP server management
- invalid MCP server configuration
- skill manifest reload failure

Plugins that want friendlier UX can catch and translate exceptions into
`PluginCommandActionResult`.

## Backward compatibility

- Existing plugin command handlers continue to work.
- `runtime` is optional on the context.
- New code should check `ctx.runtime is not None` before using live runtime
  capabilities.

## Open questions

1. Should v1 expose MCP list methods as well as attach/detach?
2. Should `runtime` be guaranteed in interactive and ACP modes, or optional in
   all modes?
3. Should `MCPAttachResult`/`MCPDetachResult` be considered stable enough for
   plugin authors, or should plugin-specific wrappers be introduced immediately?
4. Should marketplace `install_skill(...)` be part of v1, or should plugins
   initially write/copy skill files themselves and call `refresh_skills(...)`?
5. Should plugin runtime methods enforce any allowlist/safety policy before
   attaching external MCP servers or writing skills?
