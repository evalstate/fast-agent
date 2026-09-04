from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.agent_app import AgentApp, _format_interactive_final_error

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session import SessionManager
    from fast_agent.types import RequestParams
    from fast_agent.ui.interactive_prompt import PromptLoopResult


class _Agent:
    def __init__(self, name: str, *, default: bool = False) -> None:
        self.name = name
        self.config = SimpleNamespace(default=default)
        self.agent_type = AgentType.BASIC


class _PromptAgentApp(AgentApp):
    async def interactive(
        self,
        agent_name: str | None = None,
        default_prompt: str = "",
        pretty_print_parallel: bool = False,
        request_params: "RequestParams | None" = None,
        session_manager: "SessionManager | None" = None,
        harness_session=None,
    ) -> "PromptLoopResult":
        del (
            agent_name,
            default_prompt,
            pretty_print_parallel,
            request_params,
            session_manager,
            harness_session,
        )
        return cast("PromptLoopResult", "done")


def test_format_interactive_final_error_uses_type_when_message_is_blank() -> None:
    text = _format_interactive_final_error(ValueError("   "))

    assert "Error details: ValueError" in text


def test_get_default_agent_name_prefers_explicit_non_tool_default() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool", default=True)),
            "main": cast("AgentProtocol", _Agent("main", default=True)),
            "other": cast("AgentProtocol", _Agent("other")),
        },
        tool_only_agents={"tool"},
    )

    assert app.get_default_agent_name() == "main"
    assert app._agent(None).name == "main"


def test_get_default_agent_name_falls_back_to_first_non_tool_agent() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.get_default_agent_name() == "main"
    assert app._agent(None).name == "main"


def test_visible_agent_names_can_include_targeted_tool_only_agent() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.visible_agent_names(force_include="tool") == ["tool", "main"]


def test_resolve_target_agent_name_prefers_explicit_name_over_default() -> None:
    app = AgentApp(
        agents={
            "main": cast("AgentProtocol", _Agent("main", default=True)),
            "other": cast("AgentProtocol", _Agent("other")),
        }
    )

    assert app.resolve_target_agent_name("other") == "other"
    assert app.resolve_target_agent_name() == "main"


def test_registered_agent_names_include_tool_only_agents() -> None:
    app = AgentApp(
        agents={
            "tool": cast("AgentProtocol", _Agent("tool")),
            "main": cast("AgentProtocol", _Agent("main")),
        },
        tool_only_agents={"tool"},
    )

    assert app.registered_agent_names() == ["tool", "main"]


def test_no_home_mode_defaults_false_and_can_be_updated() -> None:
    app = AgentApp(agents={"main": cast("AgentProtocol", _Agent("main"))})

    assert app.no_home_mode is False

    app.no_home_mode = True

    assert app.no_home_mode is True


@pytest.mark.asyncio
async def test_prompt_warns_and_delegates_to_interactive() -> None:
    app = _PromptAgentApp(agents={"main": cast("AgentProtocol", _Agent("main", default=True))})

    with pytest.warns(
        DeprecationWarning,
        match=r"AgentApp\.prompt\(\) is deprecated; use interactive\(\) instead",
    ):
        result = await app.prompt(agent_name="main", default_prompt="hello")

    assert result == "done"


@pytest.mark.asyncio
async def test_prompt_records_and_reports_user_text_before_sending(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    agent = cast("AgentProtocol", _Agent("main", default=True))
    app = AgentApp(agents={"main": agent})
    session = SimpleNamespace(
        set_last_user_prompt=lambda prompt: (
            events.append(("prompt", prompt)),
            "latest prompt",
        )[1]
    )
    manager = SimpleNamespace(current_session=session)

    class _InteractivePrompt:
        def __init__(self, **_kwargs) -> None:
            pass

        async def prompt_loop(self, *, send_func, **_kwargs):
            return await send_func(" latest\nprompt ", "main")

    async def send_message(*_args, **_kwargs) -> str:
        events.append(("send", "latest prompt"))
        return "done"

    monkeypatch.setattr(
        "fast_agent.ui.interactive_prompt.InteractivePrompt",
        _InteractivePrompt,
    )
    monkeypatch.setattr(app, "_send_interactive_message", AsyncMock(side_effect=send_message))
    monkeypatch.setattr(
        "fast_agent.integrations.herdr_session.report_session",
        lambda reported_session, *, agent: events.append(("report", (reported_session, agent))),
    )

    result = await app.interactive(session_manager=cast("SessionManager", manager))

    assert result == "done"
    assert events == [
        ("prompt", " latest\nprompt "),
        ("report", (session, agent)),
        ("send", "latest prompt"),
    ]
