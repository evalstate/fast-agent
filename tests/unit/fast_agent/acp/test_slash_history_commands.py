from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.acp.slash.handlers import history as history_slash_handlers
from fast_agent.commands.results import CommandOutcome

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.acp.slash_commands import SlashCommandHandler


class _Handler:
    current_agent_name = "main"
    instance = SimpleNamespace(agents={"main": object()})

    def _get_current_agent(self) -> object:
        return object()


class _HistoryLoadHandler(_Handler):
    history_exporter = object()

    def __init__(self, *, session_cwd: object) -> None:
        self.ctx = SimpleNamespace(io=object(), session_cwd=session_cwd)

    def _get_current_agent_or_error(self, heading: str, missing_template: str):
        del heading, missing_template
        return object(), None

    def _build_command_context(self) -> object:
        return self.ctx

    def _format_outcome_as_markdown(
        self, outcome: CommandOutcome, heading: str, *, io: object
    ) -> str:
        del heading, io
        return "\n".join(message.plain_text() for message in outcome.messages)


@pytest.mark.asyncio
async def test_handle_history_uses_shared_parser_for_empty_overview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    async def fake_overview(handler: object) -> str:
        calls.append(handler)
        return "# history"

    monkeypatch.setattr(history_slash_handlers, "render_history_overview", fake_overview)

    handler = cast("SlashCommandHandler", _Handler())
    output = await history_slash_handlers.handle_history(handler, "")

    assert output == "# history"
    assert calls == [handler]


@pytest.mark.asyncio
async def test_handle_history_uses_shared_parser_for_unknown_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        history_slash_handlers.history_handlers,
        "web_tools_enabled_for_agent",
        lambda _agent: False,
    )

    handler = cast("SlashCommandHandler", _Handler())
    output = await history_slash_handlers.handle_history(handler, "bogus action")

    assert "Unknown /history action: bogus" in output
    assert (
        "Usage: /history [show|detail <turn>|save|load|clear [last]|rewind <turn>|fix] [args]"
        in output
    )


@pytest.mark.asyncio
async def test_handle_history_unknown_action_includes_webclear_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        history_slash_handlers.history_handlers,
        "web_tools_enabled_for_agent",
        lambda _agent: True,
    )

    handler = cast("SlashCommandHandler", _Handler())
    output = await history_slash_handlers.handle_history(handler, "bogus action")

    assert "Unknown /history action: bogus" in output
    assert (
        "Usage: /history "
        "[show|detail <turn>|save|load|clear [last]|rewind <turn>|fix|webclear] [args]"
        in output
    )


@pytest.mark.asyncio
async def test_history_webclear_missing_agent_reports_missing_agent() -> None:
    handler = cast("SlashCommandHandler", _Handler())
    output = await history_slash_handlers.handle_history(handler, "webclear missing")

    assert "Unable to locate agent 'missing' for this session." in output
    assert "Unknown /history action: webclear" not in output


@pytest.mark.asyncio
async def test_history_load_resolves_relative_file_from_session_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str | None] = []
    history_file = tmp_path / "saved.jsonl"
    history_file.write_text("[]")

    async def fake_load(_ctx: object, **kwargs: object) -> CommandOutcome:
        captured.append(cast("str | None", kwargs["filename"]))
        outcome = CommandOutcome()
        outcome.add_message("loaded")
        return outcome

    monkeypatch.setattr(history_slash_handlers.history_handlers, "handle_history_load", fake_load)

    handler = cast("SlashCommandHandler", _HistoryLoadHandler(session_cwd=tmp_path))
    output = await history_slash_handlers.handle_load(handler, "saved.jsonl")

    assert output == "loaded"
    assert captured == [str(history_file)]


@pytest.mark.asyncio
async def test_history_save_treats_blank_filename_as_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str | None] = []

    async def fake_save(_ctx: object, **kwargs: object) -> CommandOutcome:
        captured.append(cast("str | None", kwargs["filename"]))
        outcome = CommandOutcome()
        outcome.add_message("saved")
        return outcome

    monkeypatch.setattr(history_slash_handlers.history_handlers, "handle_history_save", fake_save)

    handler = cast("SlashCommandHandler", _HistoryLoadHandler(session_cwd=tmp_path))
    output = await history_slash_handlers.handle_save(handler, "   ")

    assert output == "saved"
    assert captured == [None]


@pytest.mark.asyncio
async def test_history_load_treats_blank_filename_as_missing(tmp_path: Path) -> None:
    handler = cast("SlashCommandHandler", _HistoryLoadHandler(session_cwd=tmp_path))

    output = await history_slash_handlers.handle_load(handler, "   ")

    assert "Filename required for /history load." in output
