from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from prompt_toolkit.formatted_text import to_formatted_text

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


@pytest.mark.parametrize(
    ("agent_names", "show_name"),
    [(None, False), ([], False), (["dev"], False), (["dev", "review"], True)],
)
def test_toolbar_shows_agent_name_only_when_agents_can_be_switched(
    monkeypatch: pytest.MonkeyPatch, agent_names: list[str] | None, show_name: bool
) -> None:
    # A previous multi-agent prompt must not leave a stale identity in a new one.
    monkeypatch.setattr(prompt_input, "available_agents", {"dev", "previous"})
    monkeypatch.setattr(prompt_input, "agent_histories", {})
    monkeypatch.setattr(prompt_input, "in_multiline_mode", False)
    prompt_input._initialize_prompt_input_state(
        agent_name="dev",
        multiline=False,
        available_agent_names=agent_names,
        agent_provider=None,
    )
    session = _FakeSession()
    toolbar = prompt_input._build_toolbar(
        agent_name="dev",
        toolbar_color="ansiblue",
        agent_provider=None,
        shell_context=prompt_input.ShellInputContext(),
        session_factory=lambda: cast("PromptSession[Any]", session),
    )

    text = "".join(fragment[1] for fragment in to_formatted_text(toolbar()))

    assert ("dev" in text) is show_name
    assert "NRM" in text
    assert "fast-agent" in text
