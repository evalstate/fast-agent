from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.ui.prompt import input as prompt_input


class _FakeBuffer:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _FakeSession:
    def __init__(self, text: str = "") -> None:
        self.default_buffer = _FakeBuffer(text)


def test_build_prompt_text_resolver_omits_default_agent_name() -> None:
    session = _FakeSession()
    resolver = prompt_input._build_prompt_text_resolver(
        session_factory=lambda: cast("PromptSession[Any]", session),
        agent_name="dev",
        default_agent_name="dev",
        show_default=False,
        default="",
        shell_enabled=False,
    )

    assert resolver().value == "❯ "


def test_build_prompt_text_resolver_shows_named_non_default_agent() -> None:
    session = _FakeSession()
    resolver = prompt_input._build_prompt_text_resolver(
        session_factory=lambda: cast("PromptSession[Any]", session),
        agent_name="review",
        default_agent_name="dev",
        show_default=False,
        default="",
        shell_enabled=False,
    )

    assert resolver().value == "<ansibrightblue>review</ansibrightblue> ❯ "


def test_cycle_agent_mode_reports_explicit_subagent_disable(capsys) -> None:
    prompt_input._cycle_agent_mode(ToolAgent(AgentConfig("dev", subagents=False)))

    assert "Subagents are disabled by configuration." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_get_selection_input_escapes_error_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []

    class _PromptSession:
        def __init__(self, **_kwargs: object) -> None:
            self.app = type("_App", (), {"is_running": False})()

        async def prompt_async(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("bad [selection]")

    monkeypatch.setattr(prompt_input, "rich_print", printed.append)
    monkeypatch.setattr(prompt_input, "PromptSession", _PromptSession)

    result = await prompt_input.get_selection_input("choose", ["one"])

    assert result is None
    assert printed == ["\n[red]Error getting selection: bad \\[selection][/red]"]


@pytest.mark.asyncio
async def test_get_argument_input_escapes_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    prompts: list[object] = []

    class _App:
        is_running = False

        def exit(self) -> None:
            raise AssertionError("exit should not be called when app is not running")

    class _PromptSession:
        app = _App()

        async def prompt_async(self, prompt_text: object, **_kwargs: object) -> str:
            prompts.append(prompt_text)
            return "value"

    monkeypatch.setattr(prompt_input, "rich_print", printed.append)
    monkeypatch.setattr(prompt_input, "PromptSession", _PromptSession)

    result = await prompt_input.get_argument_input(
        "name <draft> [local]",
        description="Use [literal] text",
    )

    assert result == "value"
    assert printed == [r"  [dim]name <draft> \[local]: Use \[literal] text[/dim]"]
    assert prompts
    assert getattr(prompts[0], "value", "") == (
        "Enter value for <ansibrightcyan>name &lt;draft&gt; [local]</ansibrightcyan> (required): "
    )
