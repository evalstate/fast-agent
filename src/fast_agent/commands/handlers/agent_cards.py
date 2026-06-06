"""Shared AgentCard command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from fast_agent.commands.results import CommandOutcome
from fast_agent.utils.count_display import plural_label

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.core.agent_app import AgentCardLoadResult


class AgentCardManager(Protocol):
    def can_load_agent_cards(self) -> bool: ...

    def can_dump_agent_cards(self) -> bool: ...

    def can_attach_agent_tools(self) -> bool: ...

    def can_detach_agent_tools(self) -> bool: ...

    def can_reload_agents(self) -> bool: ...

    async def load_agent_card(
        self, source: str, parent_agent: str | None = None
    ) -> AgentCardLoadResult: ...

    async def dump_agent_card(self, agent_name: str) -> str: ...

    async def attach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]: ...

    async def detach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]: ...

    async def reload_agents(self) -> bool: ...

    def registered_agent_names(self) -> Iterable[str]: ...


class CardLoadAction(StrEnum):
    LOAD = "load"
    LOAD_AND_ATTACH = "load_and_attach"
    LOAD_AND_DETACH = "load_and_detach"


class AgentCommandAction(StrEnum):
    DUMP = "dump"
    ATTACH_TOOL = "attach_tool"
    DETACH_TOOL = "detach_tool"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class CardLoadToolActionPolicy:
    can_perform: "Callable[[AgentCardManager], bool]"
    unavailable_message: str
    missing_agent_message: str


@dataclass(frozen=True, slots=True)
class AgentCommandRequest:
    action: AgentCommandAction
    target: str


@dataclass(frozen=True, slots=True)
class AgentCommandActionPolicy:
    requires_target: bool = False
    missing_target_message: str = ""


_CARD_LOAD_TOOL_ACTION_POLICIES: dict[CardLoadAction, CardLoadToolActionPolicy] = {
    CardLoadAction.LOAD_AND_ATTACH: CardLoadToolActionPolicy(
        can_perform=lambda manager: manager.can_attach_agent_tools(),
        unavailable_message="Agent tool attachment is not available in this session.",
        missing_agent_message="No active agent available for tool attachment.",
    ),
    CardLoadAction.LOAD_AND_DETACH: CardLoadToolActionPolicy(
        can_perform=lambda manager: manager.can_detach_agent_tools(),
        unavailable_message="Agent tool detachment is not available in this session.",
        missing_agent_message="No active agent available for tool detachment.",
    ),
}

_AGENT_COMMAND_ACTION_POLICIES: dict[AgentCommandAction, AgentCommandActionPolicy] = {
    AgentCommandAction.DUMP: AgentCommandActionPolicy(),
    AgentCommandAction.ATTACH_TOOL: AgentCommandActionPolicy(
        requires_target=True,
        missing_target_message="Agent name is required for /agent --tool.",
    ),
    AgentCommandAction.DETACH_TOOL: AgentCommandActionPolicy(
        requires_target=True,
        missing_target_message="Agent name is required for /agent --tool remove.",
    ),
}
_AGENT_TOOL_NOOP_ACTION_LABELS = {
    "Attached": "attached",
    "Detached": "detached",
}


def _resolve_card_load_action(*, add_tool: bool, remove_tool: bool) -> CardLoadAction:
    if add_tool and remove_tool:
        return CardLoadAction.LOAD_AND_DETACH
    if add_tool:
        return CardLoadAction.LOAD_AND_ATTACH
    return CardLoadAction.LOAD


def _resolve_agent_command_action(
    *,
    add_tool: bool,
    remove_tool: bool,
    dump: bool,
) -> AgentCommandAction:
    if dump:
        return AgentCommandAction.INVALID if add_tool or remove_tool else AgentCommandAction.DUMP
    if add_tool and remove_tool:
        return AgentCommandAction.DETACH_TOOL
    if add_tool:
        return AgentCommandAction.ATTACH_TOOL
    return AgentCommandAction.INVALID


def _agent_command_action_policy(
    action: AgentCommandAction,
) -> AgentCommandActionPolicy | None:
    return _AGENT_COMMAND_ACTION_POLICIES.get(action)


def _agent_command_request(
    outcome: CommandOutcome,
    *,
    action: AgentCommandAction,
    current_agent: str,
    target_agent: str | None,
    error: str | None,
) -> AgentCommandRequest | None:
    if error:
        outcome.add_message(error, channel="error")
        return None

    policy = _agent_command_action_policy(action)
    if policy is None:
        outcome.add_message("Invalid /agent command.", channel="error")
        return None

    if policy.requires_target and target_agent is None:
        outcome.add_message(policy.missing_target_message, channel="error")
        return None

    return AgentCommandRequest(action=action, target=target_agent or current_agent)


def _card_load_tool_action_policy(action: CardLoadAction) -> CardLoadToolActionPolicy | None:
    return _CARD_LOAD_TOOL_ACTION_POLICIES.get(action)


def _format_agent_card_names(names: Sequence[str]) -> str:
    return f"{plural_label(len(names), 'AgentCard')}: {', '.join(names)}"


def _format_agent_tool_names(action: str, names: Sequence[str]) -> str:
    return f"{action} {plural_label(len(names), 'agent tool')}: {', '.join(names)}"


def _add_agent_tool_change_message(
    outcome: CommandOutcome,
    *,
    action: str,
    names: Sequence[str],
) -> None:
    if names:
        outcome.add_message(_format_agent_tool_names(action, names), channel="info")
        return

    action_label = _AGENT_TOOL_NOOP_ACTION_LABELS.get(action, action)
    outcome.add_message(f"No agent tools {action_label}.", channel="warning")


def _add_card_load_success_message(outcome: CommandOutcome, loaded_names: Sequence[str]) -> None:
    if loaded_names:
        outcome.add_message(
            f"Loaded {_format_agent_card_names(loaded_names)}",
            channel="info",
        )
        return

    outcome.add_message("AgentCard loaded.", channel="info")


async def _load_agent_card_for_action(
    manager: AgentCardManager,
    *,
    filename: str,
    action: CardLoadAction,
    current_agent: str | None,
) -> AgentCardLoadResult:
    if action is CardLoadAction.LOAD_AND_ATTACH:
        return await manager.load_agent_card(filename, current_agent)
    return await manager.load_agent_card(filename)


async def _handle_loaded_card_tool_action(
    outcome: CommandOutcome,
    *,
    manager: AgentCardManager,
    action: CardLoadAction,
    current_agent: str | None,
    loaded_names: Sequence[str],
    attached_names: Sequence[str],
) -> CommandOutcome:
    if action is CardLoadAction.LOAD_AND_DETACH:
        if current_agent is None:
            outcome.add_message("No active agent available for tool detachment.", channel="error")
            return outcome
        try:
            removed = await manager.detach_agent_tools(current_agent, loaded_names)
        except Exception as exc:
            outcome.add_message(f"Agent tool detach failed: {exc}", channel="error")
            return outcome

        _add_agent_tool_change_message(outcome, action="Detached", names=removed)
        return outcome

    if action is CardLoadAction.LOAD_AND_ATTACH and attached_names:
        _add_agent_tool_change_message(outcome, action="Attached", names=attached_names)

    return outcome


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
    action = _resolve_card_load_action(add_tool=add_tool, remove_tool=remove_tool)

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

    tool_action_policy = _card_load_tool_action_policy(action)
    if tool_action_policy is not None:
        if not tool_action_policy.can_perform(manager):
            outcome.add_message(
                tool_action_policy.unavailable_message,
                channel="warning",
            )
            return outcome
        if not current_agent:
            outcome.add_message(tool_action_policy.missing_agent_message, channel="error")
            return outcome

    try:
        loaded_card = await _load_agent_card_for_action(
            manager,
            filename=filename,
            action=action,
            current_agent=current_agent,
        )
    except Exception as exc:
        outcome.add_message(f"AgentCard load failed: {exc}", channel="error")
        return outcome

    _add_card_load_success_message(outcome, loaded_card.loaded_names)
    outcome.requires_refresh = True
    return await _handle_loaded_card_tool_action(
        outcome,
        manager=manager,
        action=action,
        current_agent=current_agent,
        loaded_names=loaded_card.loaded_names,
        attached_names=loaded_card.attached_names,
    )


async def _handle_agent_dump(
    outcome: CommandOutcome,
    *,
    manager: AgentCardManager,
    target: str,
) -> CommandOutcome:
    if not manager.can_dump_agent_cards():
        outcome.add_message(
            "AgentCard dumping is not available in this session.",
            channel="warning",
        )
        return outcome
    try:
        card_text = await manager.dump_agent_card(target)
    except Exception as exc:
        outcome.add_message(f"AgentCard dump failed: {exc}", channel="error")
        return outcome

    outcome.add_message(card_text)
    return outcome


async def _handle_agent_detach_tool(
    outcome: CommandOutcome,
    *,
    manager: AgentCardManager,
    current_agent: str,
    target: str,
) -> CommandOutcome:
    if not manager.can_detach_agent_tools():
        outcome.add_message(
            "Agent tool detachment is not available in this session.",
            channel="warning",
        )
        return outcome
    try:
        removed = await manager.detach_agent_tools(current_agent, [target])
    except Exception as exc:
        outcome.add_message(f"Agent tool detach failed: {exc}", channel="error")
        return outcome

    _add_agent_tool_change_message(outcome, action="Detached", names=removed)
    return outcome


async def _handle_agent_attach_tool(
    outcome: CommandOutcome,
    *,
    manager: AgentCardManager,
    current_agent: str,
    target: str,
) -> CommandOutcome:
    if target == current_agent:
        outcome.add_message("Can't attach agent to itself.", channel="warning")
        return outcome
    if target not in set(manager.registered_agent_names()):
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
    except Exception as exc:
        outcome.add_message(f"Agent tool attach failed: {exc}", channel="error")
        return outcome

    _add_agent_tool_change_message(outcome, action="Attached", names=attached)
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
    action = _resolve_agent_command_action(add_tool=add_tool, remove_tool=remove_tool, dump=dump)
    request = _agent_command_request(
        outcome,
        action=action,
        current_agent=current_agent,
        target_agent=target_agent,
        error=error,
    )
    if request is None:
        return outcome

    if request.action is AgentCommandAction.DUMP:
        return await _handle_agent_dump(outcome, manager=manager, target=request.target)

    if request.action is AgentCommandAction.DETACH_TOOL:
        return await _handle_agent_detach_tool(
            outcome,
            manager=manager,
            current_agent=current_agent,
            target=request.target,
        )

    if request.action is AgentCommandAction.ATTACH_TOOL:
        return await _handle_agent_attach_tool(
            outcome,
            manager=manager,
            current_agent=current_agent,
            target=request.target,
        )

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
    except Exception as exc:
        outcome.add_message(f"Reload failed: {exc}", channel="error")
        return outcome

    if not changed:
        outcome.add_message("No AgentCard changes detected.", channel="warning")
        return outcome

    outcome.add_message("AgentCards reloaded.", channel="info")
    outcome.requires_refresh = True
    return outcome
