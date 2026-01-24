"""Shared AgentCard command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fast_agent.commands.results import CommandOutcome

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fast_agent.commands.context import CommandContext


class AgentCardManager(Protocol):
    def can_load_agent_cards(self) -> bool: ...

    def can_dump_agent_cards(self) -> bool: ...

    def can_attach_agent_tools(self) -> bool: ...

    def can_detach_agent_tools(self) -> bool: ...

    def can_reload_agents(self) -> bool: ...

    async def load_agent_card(
        self, source: str, parent_agent: str | None = None
    ) -> tuple[list[str], list[str]]: ...

    async def dump_agent_card(self, agent_name: str) -> str: ...

    async def attach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]: ...

    async def detach_agent_tools(self, parent_agent: str, child_agents: Sequence[str]) -> list[str]: ...

    async def reload_agents(self) -> bool: ...

    def agent_names(self) -> Iterable[str]: ...


async def handle_card_load(
    ctx: CommandContext,
    *,
    manager: AgentCardManager,
    filename: str | None,
    add_tool: bool,
    remove_tool: bool,
    current_agent: str | None,
) -> CommandOutcome:
    del ctx

    outcome = CommandOutcome()

    if not filename:
        outcome.add_message(
            "Filename required for /card command.",
            channel="error",
        )
        outcome.add_message(
            "Usage: /card <filename|url> [--tool]",
            channel="info",
        )
        return outcome

    if not manager.can_load_agent_cards():
        outcome.add_message(
            "AgentCard loading is not available in this session.",
            channel="warning",
        )
        return outcome

    try:
        if add_tool and not remove_tool:
            loaded_names, attached_names = await manager.load_agent_card(
                filename,
                current_agent,
            )
        else:
            loaded_names, attached_names = await manager.load_agent_card(filename)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"AgentCard load failed: {exc}", channel="error")
        return outcome

    if loaded_names:
        loaded_list = ", ".join(loaded_names)
        outcome.add_message(f"Loaded AgentCard(s): {loaded_list}", channel="info")
    else:
        outcome.add_message("AgentCard loaded.", channel="info")

    if add_tool and remove_tool:
        if not manager.can_detach_agent_tools():
            outcome.add_message(
                "Agent tool detachment is not available in this session.",
                channel="warning",
            )
            return outcome

        if not current_agent:
            outcome.add_message("No active agent available for tool detachment.", channel="error")
            return outcome

        try:
            removed = await manager.detach_agent_tools(current_agent, loaded_names)
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"Agent tool detach failed: {exc}", channel="error")
            return outcome

        if removed:
            removed_list = ", ".join(removed)
            outcome.add_message(f"Detached agent tool(s): {removed_list}", channel="info")
        else:
            outcome.add_message("No agent tools detached.", channel="warning")
        return outcome

    if add_tool and attached_names:
        attached_list = ", ".join(attached_names)
        outcome.add_message(f"Attached agent tool(s): {attached_list}", channel="info")

    outcome.requires_refresh = True
    return outcome


async def handle_agent_command(
    ctx: CommandContext,
    *,
    manager: AgentCardManager,
    current_agent: str,
    target_agent: str | None,
    add_tool: bool,
    remove_tool: bool,
    dump: bool,
    error: str | None = None,
) -> CommandOutcome:
    del ctx

    outcome = CommandOutcome()

    if error:
        outcome.add_message(error, channel="error")
        return outcome

    if add_tool and remove_tool and target_agent is None:
        outcome.add_message(
            "Agent name is required for /agent --tool remove.",
            channel="error",
        )
        return outcome

    if add_tool and not dump and target_agent is None:
        outcome.add_message("Agent name is required for /agent --tool.", channel="error")
        return outcome

    target = target_agent or current_agent

    if dump:
        if not manager.can_dump_agent_cards():
            outcome.add_message(
                "AgentCard dumping is not available in this session.",
                channel="warning",
            )
            return outcome
        try:
            card_text = await manager.dump_agent_card(target)
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"AgentCard dump failed: {exc}", channel="error")
            return outcome
        outcome.add_message(card_text)
        return outcome

    if add_tool and remove_tool:
        if not manager.can_detach_agent_tools():
            outcome.add_message(
                "Agent tool detachment is not available in this session.",
                channel="warning",
            )
            return outcome
        try:
            removed = await manager.detach_agent_tools(current_agent, [target])
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"Agent tool detach failed: {exc}", channel="error")
            return outcome
        if removed:
            removed_list = ", ".join(removed)
            outcome.add_message(f"Detached agent tool(s): {removed_list}", channel="info")
        else:
            outcome.add_message("No agent tools detached.", channel="warning")
        return outcome

    if add_tool:
        if target == current_agent:
            outcome.add_message("Can't attach agent to itself.", channel="warning")
            return outcome
        if target not in set(manager.agent_names()):
            outcome.add_message(f"Agent '{target}' not found", channel="error")
            return outcome
        if not manager.can_attach_agent_tools():
            outcome.add_message(
                "Agent tool attachment is not available in this session.",
                channel="warning",
            )
            return outcome
        try:
            attached = await manager.attach_agent_tools(current_agent, [target])
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"Agent tool attach failed: {exc}", channel="error")
            return outcome

        if attached:
            attached_list = ", ".join(attached)
            outcome.add_message(f"Attached agent tool(s): {attached_list}", channel="info")
        else:
            outcome.add_message("No agent tools attached.", channel="warning")
        return outcome

    outcome.add_message("Invalid /agent command.", channel="error")
    return outcome


async def handle_reload_agents(
    ctx: CommandContext,
    *,
    manager: AgentCardManager,
) -> CommandOutcome:
    del ctx

    outcome = CommandOutcome()

    if not manager.can_reload_agents():
        outcome.add_message(
            "Reload is not available in this session.",
            channel="warning",
        )
        return outcome

    try:
        changed = await manager.reload_agents()
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Reload failed: {exc}", channel="error")
        return outcome

    if not changed:
        outcome.add_message("No AgentCard changes detected.", channel="warning")
        return outcome

    outcome.add_message("AgentCards reloaded.", channel="info")
    outcome.requires_refresh = True
    return outcome
