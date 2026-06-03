from typing import TYPE_CHECKING, Literal, cast

import pytest
from mcp.types import TextContent

from fast_agent.commands.handlers.display import (
    _decoded_process_output,
    _last_assistant_message,
    _last_assistant_text,
    handle_check,
    handle_commands,
    handle_show_markdown,
)
from fast_agent.types import PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


def _message(role: Literal["user", "assistant"], text: str) -> PromptMessageExtended:
    return PromptMessageExtended(role=role, content=[TextContent(type="text", text=text)])


def test_last_assistant_message_returns_most_recent_assistant_message() -> None:
    first = _message("assistant", "first")
    latest = _message("assistant", "latest")

    assert _last_assistant_message([first, _message("user", "question"), latest]) is latest


def test_last_assistant_message_returns_none_without_assistant_messages() -> None:
    assert _last_assistant_message([_message("user", "question")]) is None


def test_last_assistant_text_skips_empty_assistant_messages() -> None:
    assert (
        _last_assistant_text(
            [
                _message("assistant", "display me"),
                PromptMessageExtended(role="assistant", content=[]),
            ]
        )
        == "display me"
    )


def test_decoded_process_output_normalizes_blank_output() -> None:
    assert _decoded_process_output(b" ok \n") == "ok"
    assert _decoded_process_output(b" \n\t") is None


class _Provider:
    def __init__(self, agent) -> None:
        self._selected_agent = agent

    def _agent(self, agent_name: str):
        del agent_name
        return self._selected_agent


class _Context:
    def __init__(self, agent) -> None:
        self.agent_provider = _Provider(agent)


class _Agent:
    llm = object()

    def __init__(self, message_history: list[PromptMessageExtended]) -> None:
        self.message_history = message_history


@pytest.mark.asyncio
async def test_show_markdown_returns_renderable_command_message() -> None:
    agent = _Agent([_message("assistant", "**render me**")])
    ctx = cast("CommandContext", _Context(agent))
    outcome = await handle_show_markdown(ctx, agent_name="default")

    assert len(outcome.messages) == 1
    message = outcome.messages[0]
    assert message.text == "**render me**"
    assert message.title == "Last Assistant Response"
    assert message.right_info == "display"
    assert message.agent_name == "default"
    assert message.render_markdown is True


@pytest.mark.asyncio
async def test_show_markdown_uses_latest_assistant_text() -> None:
    agent = _Agent(
        [
            _message("assistant", "**render me**"),
            PromptMessageExtended(role="assistant", content=[]),
        ]
    )
    ctx = cast("CommandContext", _Context(agent))

    outcome = await handle_show_markdown(ctx, agent_name="default")

    assert len(outcome.messages) == 1
    assert outcome.messages[0].text == "**render me**"


@pytest.mark.asyncio
async def test_handle_check_reports_invalid_argument_syntax() -> None:
    ctx = cast("CommandContext", _Context(_Agent([])))

    outcome = await handle_check(ctx, agent_name="default", argument='"')

    assert len(outcome.messages) == 1
    assert outcome.messages[0].channel == "warning"
    assert str(outcome.messages[0].text).startswith("Invalid check arguments:")


@pytest.mark.asyncio
async def test_handle_check_captures_subprocess_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    class _Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"ok\n", b""

    async def fake_create_subprocess_exec(*command: str, **_kwargs: object) -> _Process:
        calls.append(command)
        return _Process()

    monkeypatch.setattr(
        "fast_agent.commands.handlers.display.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    ctx = cast("CommandContext", _Context(_Agent([])))

    outcome = await handle_check(ctx, agent_name="default", argument="models --for-model gpt-5")

    assert calls
    assert calls[0][-4:] == ("check", "models", "--for-model", "gpt-5")
    assert outcome.messages[0].text == "ok"
    assert outcome.messages[0].right_info == "check"


@pytest.mark.asyncio
async def test_handle_check_trims_argument_before_splitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    class _Process:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"ok\n", b""

    async def fake_create_subprocess_exec(*command: str, **_kwargs: object) -> _Process:
        calls.append(command)
        return _Process()

    monkeypatch.setattr(
        "fast_agent.commands.handlers.display.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    ctx = cast("CommandContext", _Context(_Agent([])))

    await handle_check(ctx, agent_name="default", argument="  models  ")

    assert calls[0][-2:] == ("check", "models")


@pytest.mark.asyncio
async def test_handle_commands_delegates_argument_normalization_to_parser() -> None:
    ctx = cast("CommandContext", _Context(_Agent([])))

    outcome = await handle_commands(ctx, agent_name="default", argument="  skills  ")

    assert len(outcome.messages) == 1
    assert outcome.messages[0].right_info == "commands"
    assert "commands skills" in str(outcome.messages[0].text)
