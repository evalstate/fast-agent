"""
Instruction building and refresh utilities.

This module provides the central logic for building agent instructions from
templates. It consolidates instruction building that was previously spread
across McpAgent and other modules.

The InstructionBuilder handles template resolution (placeholders like {{currentDate}},
{{file:path}}, etc.), while this module provides higher-level functions that:
- Gather data from agent sources (MCP servers, skills, etc.)
- Build the complete instruction using InstructionBuilder
- Set the instruction on the agent via set_instruction()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, runtime_checkable
from weakref import WeakKeyDictionary

from fast_agent.core.instruction import InstructionBuilder
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.common import create_namespaced_name

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import MCPAggregator
    from fast_agent.skills import SkillManifest
    from fast_agent.skills.registry import SkillRegistry

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Protocols
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class InstructionCapable(Protocol):
    """Protocol for agents that support instruction get/set."""

    @property
    def instruction(self) -> str: ...

    def set_instruction(self, instruction: str) -> None: ...


@runtime_checkable
class McpInstructionCapable(InstructionCapable, Protocol):
    """Protocol for MCP agents that support full instruction refresh."""

    @property
    def instruction_template(self) -> str: ...

    @property
    def instruction_context(self) -> dict[str, str]: ...

    @property
    def aggregator(self) -> "MCPAggregator": ...

    @property
    def skill_manifests(self) -> Sequence["SkillManifest"]: ...

    @property
    def skill_registry(self) -> "SkillRegistry | None": ...

    @skill_registry.setter
    def skill_registry(self, value: "SkillRegistry | None") -> None: ...

    def set_skill_manifests(self, manifests: Sequence["SkillManifest"]) -> None: ...

    def set_instruction_context(self, context: dict[str, str]) -> None: ...

    @property
    def has_filesystem_runtime(self) -> bool: ...


# ─────────────────────────────────────────────────────────────────────────────
# Instruction Building
# ─────────────────────────────────────────────────────────────────────────────


def format_server_instructions(
    instructions_data: dict[str, tuple[str | None, list[str]]]
) -> str:
    """
    Format server instructions with XML tags and tool lists.

    Args:
        instructions_data: Dict mapping server name to (instructions, tool_names)

    Returns:
        Formatted string with server instructions
    """
    if not instructions_data:
        return ""

    formatted_parts = []
    for server_name, (instructions, tool_names) in instructions_data.items():
        if instructions is None:
            continue

        prefixed_tools = [create_namespaced_name(server_name, tool) for tool in tool_names]
        tools_list = ", ".join(prefixed_tools) if prefixed_tools else "No tools available"

        formatted_parts.append(
            f'<fastagent:mcp-server name="{server_name}">\n'
            f"<tools>{tools_list}</tools>\n"
            f"<instructions>\n{instructions}\n</instructions>\n"
            f"</fastagent:mcp-server>"
        )

    return "\n\n".join(formatted_parts) if formatted_parts else ""


def format_agent_skills(
    manifests: Sequence["SkillManifest"],
    has_filesystem_runtime: bool = False,
) -> str:
    """
    Format skill manifests for inclusion in the instruction.

    Args:
        manifests: List of skill manifests
        has_filesystem_runtime: Whether filesystem runtime is available (affects tool name)

    Returns:
        Formatted skills text
    """
    from fast_agent.skills.registry import format_skills_for_prompt

    read_tool_name = "read_text_file" if has_filesystem_runtime else "read_skill"
    return format_skills_for_prompt(manifests, read_tool_name=read_tool_name)


async def build_instruction(
    template: str,
    *,
    aggregator: "MCPAggregator | None" = None,
    skill_manifests: Sequence["SkillManifest"] | None = None,
    has_filesystem_runtime: bool = False,
    context: Mapping[str, str] | None = None,
) -> str:
    """
    Build an instruction string from a template with all placeholders resolved.

    This is the main entry point for building agent instructions. It:
    1. Creates an InstructionBuilder with the template
    2. Sets up resolvers for serverInstructions and agentSkills
    3. Applies any context values
    4. Builds and returns the final instruction

    Args:
        template: The instruction template with {{placeholder}} patterns
        aggregator: MCP aggregator for fetching server instructions
        skill_manifests: List of skill manifests for {{agentSkills}}
        has_filesystem_runtime: Whether filesystem runtime is available
        context: Additional context values (env, workspaceRoot, etc.)

    Returns:
        The fully resolved instruction string
    """
    builder = InstructionBuilder(template)

    # Set up server instructions resolver
    if aggregator is not None:

        async def resolve_server_instructions() -> str:
            try:
                instructions_data = await aggregator.get_server_instructions()
                return format_server_instructions(instructions_data)
            except Exception as e:
                logger.warning(f"Failed to get server instructions: {e}")
                return ""

        builder.set_resolver("serverInstructions", resolve_server_instructions)

    # Set up agent skills resolver
    if skill_manifests is not None:

        async def resolve_agent_skills() -> str:
            return format_agent_skills(skill_manifests, has_filesystem_runtime)

        builder.set_resolver("agentSkills", resolve_agent_skills)

    # Apply context values
    if context:
        builder.set_many(dict(context))

    return await builder.build()


# ─────────────────────────────────────────────────────────────────────────────
# Agent Instruction Refresh
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InstructionRefreshResult:
    updated_skill_manifests: bool = False
    updated_skill_registry: bool = False
    updated_context: bool = False
    rebuilt_instruction: bool = False


_instruction_locks: "WeakKeyDictionary[object, asyncio.Lock]" = WeakKeyDictionary()
_fallback_instruction_locks: dict[int, asyncio.Lock] = {}


def _get_instruction_lock(agent: object) -> asyncio.Lock:
    try:
        lock = _instruction_locks.get(agent)
        if lock is None:
            lock = asyncio.Lock()
            _instruction_locks[agent] = lock
        return lock
    except TypeError:
        key = id(agent)
        lock = _fallback_instruction_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _fallback_instruction_locks[key] = lock
        return lock


async def rebuild_agent_instruction(
    agent: object,
    *,
    skill_manifests: list[Any] | None = None,
    skill_registry: Any | None = None,
    context: Mapping[str, str] | None = None,
) -> InstructionRefreshResult:
    """
    Rebuild an agent's instruction from its template.

    This function:
    1. Optionally updates skill_manifests and skill_registry on the agent
    2. Optionally updates instruction_context if context is provided
    3. Builds the instruction using the agent's template and data sources
    4. Sets the new instruction on the agent

    Args:
        agent: The agent to refresh (must implement McpInstructionCapable)
        skill_manifests: Optional new skill manifests to set
        skill_registry: Optional new skill registry to set
        context: Optional new context values to set and use in the build.
                 If not provided, uses the agent's stored instruction_context.

    Returns:
        InstructionRefreshResult indicating what was updated
    """
    lock = _get_instruction_lock(agent)
    async with lock:
        updated_skill_manifests = False
        updated_skill_registry = False
        updated_context = False
        rebuilt_instruction = False

        if not isinstance(agent, McpInstructionCapable):
            return InstructionRefreshResult()

        # Update agent state if new values provided
        if skill_manifests is not None:
            agent.set_skill_manifests(skill_manifests)
            updated_skill_manifests = True

        if skill_registry is not None:
            agent.skill_registry = skill_registry
            updated_skill_registry = True

        if context is not None:
            agent.set_instruction_context(dict(context))
            updated_context = True

        # Build the instruction using the agent's current state
        template = agent.instruction_template
        if not template:
            return InstructionRefreshResult(
                updated_skill_manifests=updated_skill_manifests,
                updated_skill_registry=updated_skill_registry,
                updated_context=updated_context,
            )

        # Use agent's stored context (which may have just been updated)
        build_context = agent.instruction_context

        new_instruction = await build_instruction(
            template,
            aggregator=agent.aggregator,
            skill_manifests=agent.skill_manifests,
            has_filesystem_runtime=agent.has_filesystem_runtime,
            context=build_context,
        )

        agent.set_instruction(new_instruction)
        rebuilt_instruction = True

        return InstructionRefreshResult(
            updated_skill_manifests=updated_skill_manifests,
            updated_skill_registry=updated_skill_registry,
            updated_context=updated_context,
            rebuilt_instruction=rebuilt_instruction,
        )
