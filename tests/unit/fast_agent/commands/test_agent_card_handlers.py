from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.commands.handlers.agent_cards import handle_agent_command, handle_card_load
from fast_agent.core.agent_app import AgentCardLoadResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fast_agent.commands.context import CommandContext


class _CardManager:
    def __init__(
        self,
        *,
        can_attach: bool = True,
        can_detach: bool = True,
        attach_result: Sequence[str] | None = None,
        detach_result: Sequence[str] | None = None,
    ) -> None:
        self.can_attach = can_attach
        self.can_detach = can_detach
        self.attach_result = list(attach_result) if attach_result is not None else None
        self.detach_result = list(detach_result) if detach_result is not None else None
        self.loaded: list[tuple[str, str | None]] = []
        self.attached: list[tuple[str, list[str]]] = []
        self.detached: list[tuple[str, list[str]]] = []

    def can_load_agent_cards(self) -> bool:
        return True

    def can_dump_agent_cards(self) -> bool:
        return True

    def can_attach_agent_tools(self) -> bool:
        return self.can_attach

    def can_detach_agent_tools(self) -> bool:
        return self.can_detach

    def can_reload_agents(self) -> bool:
        return True

    async def load_agent_card(
        self, source: str, parent_agent: str | None = None
    ) -> AgentCardLoadResult:
        self.loaded.append((source, parent_agent))
        return AgentCardLoadResult(
            loaded_names=["helper"],
            attached_names=["helper"] if parent_agent else [],
        )

    async def dump_agent_card(self, agent_name: str) -> str:
        return f"card:{agent_name}"

    async def attach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        attached = list(child_agents)
        self.attached.append((parent_agent, attached))
        return attached if self.attach_result is None else self.attach_result

    async def detach_agent_tools(
        self, parent_agent: str, child_agents: Sequence[str]
    ) -> list[str]:
        removed = list(child_agents)
        self.detached.append((parent_agent, removed))
        return removed if self.detach_result is None else self.detach_result

    async def reload_agents(self) -> bool:
        return False

    def registered_agent_names(self) -> Iterable[str]:
        return ["root", "helper"]


def _ctx() -> "CommandContext":
    return cast("CommandContext", object())


@pytest.mark.asyncio
async def test_card_load_detach_preflights_before_loading() -> None:
    manager = _CardManager(can_detach=False)

    outcome = await handle_card_load(
        _ctx(),
        manager=manager,
        filename="helper.md",
        add_tool=True,
        remove_tool=True,
        current_agent="root",
    )

    assert manager.loaded == []
    assert manager.detached == []
    assert outcome.requires_refresh is False
    assert outcome.messages[-1].text == "Agent tool detachment is not available in this session."


@pytest.mark.asyncio
async def test_card_load_detach_refreshes_after_successful_load() -> None:
    manager = _CardManager()

    outcome = await handle_card_load(
        _ctx(),
        manager=manager,
        filename="helper.md",
        add_tool=True,
        remove_tool=True,
        current_agent="root",
    )

    assert manager.loaded == [("helper.md", None)]
    assert manager.detached == [("root", ["helper"])]
    assert outcome.requires_refresh is True


@pytest.mark.asyncio
async def test_card_load_attach_preflights_before_loading() -> None:
    manager = _CardManager(can_attach=False)

    outcome = await handle_card_load(
        _ctx(),
        manager=manager,
        filename="helper.md",
        add_tool=True,
        remove_tool=False,
        current_agent="root",
    )

    assert manager.loaded == []
    assert outcome.requires_refresh is False
    assert outcome.messages[-1].text == "Agent tool attachment is not available in this session."


@pytest.mark.asyncio
async def test_agent_command_rejects_conflicting_tool_dump_flags() -> None:
    manager = _CardManager()

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent="helper",
        add_tool=True,
        remove_tool=False,
        dump=True,
    )

    assert manager.attached == []
    assert outcome.messages[-1].text == "Invalid /agent command."


@pytest.mark.asyncio
async def test_agent_command_validates_missing_target_once_for_detach() -> None:
    manager = _CardManager()

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent=None,
        add_tool=True,
        remove_tool=True,
        dump=False,
    )

    assert manager.detached == []
    assert outcome.messages[-1].text == "Agent name is required for /agent --tool remove."


@pytest.mark.asyncio
async def test_agent_command_validates_missing_target_once_for_attach() -> None:
    manager = _CardManager()

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent=None,
        add_tool=True,
        remove_tool=False,
        dump=False,
    )

    assert manager.attached == []
    assert outcome.messages[-1].text == "Agent name is required for /agent --tool."


@pytest.mark.asyncio
async def test_agent_command_attach_uses_explicit_action() -> None:
    manager = _CardManager()

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent="helper",
        add_tool=True,
        remove_tool=False,
        dump=False,
    )

    assert manager.attached == [("root", ["helper"])]
    assert outcome.messages[-1].text == "Attached agent tool: helper"


@pytest.mark.asyncio
async def test_agent_command_attach_reports_empty_result_with_explicit_label() -> None:
    manager = _CardManager(attach_result=[])

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent="helper",
        add_tool=True,
        remove_tool=False,
        dump=False,
    )

    assert manager.attached == [("root", ["helper"])]
    assert outcome.messages[-1].text == "No agent tools attached."


@pytest.mark.asyncio
async def test_agent_command_detach_reports_empty_result_with_explicit_label() -> None:
    manager = _CardManager(detach_result=[])

    outcome = await handle_agent_command(
        _ctx(),
        manager=manager,
        current_agent="root",
        target_agent="helper",
        add_tool=True,
        remove_tool=True,
        dump=False,
    )

    assert manager.detached == [("root", ["helper"])]
    assert outcome.messages[-1].text == "No agent tools detached."
