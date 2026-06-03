from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.acp.slash.handlers import cards_manager as cards_slash_handler
from fast_agent.commands.results import CommandOutcome

if TYPE_CHECKING:
    from fast_agent.acp.slash_commands import SlashCommandHandler


class _CardsHandler:
    current_agent_name = "main"

    def _build_command_context(self) -> object:
        return SimpleNamespace(io=object())

    def _format_outcome_as_markdown(
        self,
        outcome: CommandOutcome,
        heading: str,
        *,
        io: object,
    ) -> str:
        del outcome, io
        return f"# {heading}"


def test_parse_cards_arguments_normalizes_aliases() -> None:
    assert cards_slash_handler._parse_cards_arguments("show alpha") == ("readme", "alpha")
    assert cards_slash_handler._parse_cards_arguments("install alpha") == ("add", "alpha")
    assert cards_slash_handler._parse_cards_arguments(None) == ("list", "")


@pytest.mark.asyncio
async def test_handle_cards_uses_canonical_heading_for_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    async def handle_cards_command(
        _ctx: object,
        *,
        agent_name: str,
        action: str | None,
        argument: str | None,
    ) -> CommandOutcome:
        captured.update(agent_name=agent_name, action=action, argument=argument)
        return CommandOutcome()

    monkeypatch.setattr(cards_slash_handler.cards_handlers, "handle_cards_command", handle_cards_command)

    rendered = await cards_slash_handler.handle_cards(
        cast("SlashCommandHandler", _CardsHandler()),
        "show alpha",
    )

    assert captured == {"agent_name": "main", "action": "readme", "argument": "alpha"}
    assert rendered == "# cards readme"
