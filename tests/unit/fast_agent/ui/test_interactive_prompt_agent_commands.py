from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import enhanced_prompt, interactive_prompt
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _FakeAgent:
    agent_type = AgentType.BASIC


class _FakeAgentApp:
    def __init__(self, agent_names: list[str]) -> None:
        self._agents = {name: _FakeAgent() for name in agent_names}
        self.attached: list[str] = []
        self.detached: list[str] = []
        self.loaded: list[str] = []

    async def refresh_if_needed(self) -> bool:
        return False

    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def can_attach_agent_tools(self) -> bool:
        return True

    async def attach_agent_tools(self, _parent: str, child_names: list[str]) -> list[str]:
        self.attached.extend(child_names)
        return child_names

    def can_detach_agent_tools(self) -> bool:
        return True

    async def detach_agent_tools(self, _parent: str, child_names: list[str]) -> list[str]:
        self.detached.extend(child_names)
        return child_names

    def can_dump_agent_cards(self) -> bool:
        return True

    async def dump_agent_card(self, name: str) -> str:
        return f"card:{name}"

    def can_load_agent_cards(self) -> bool:
        return True

    async def load_agent_card(
        self, filename: str, _parent: str | None = None
    ) -> tuple[list[str], list[str]]:
        self.loaded.append(filename)
        loaded = ["sizer"]
        attached = ["sizer"] if _parent else []
        return loaded, attached

    def can_reload_agents(self) -> bool:
        return False


class _ReloadAgentApp(_FakeAgentApp):
    def __init__(self, agent_names: list[str], changed: bool) -> None:
        super().__init__(agent_names)
        self._changed = changed

    def can_reload_agents(self) -> bool:
        return True

    async def reload_agents(self) -> bool:
        return self._changed



def _patch_input(monkeypatch, inputs: list[str]) -> None:
    iterator = iter(inputs)

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(iterator)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)


@pytest.mark.asyncio
async def test_agent_command_missing_agent(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/agent sizer --tool", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Agent 'sizer' not found" in output


@pytest.mark.asyncio
async def test_agent_command_attach_and_detach(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/agent sizer --tool", "/agent sizer --tool remove", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag", "sizer"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag", "sizer"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Attached agent tool(s): sizer" in output
    assert "Detached agent tool(s): sizer" in output


@pytest.mark.asyncio
async def test_agent_command_dump(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/agent sizer --dump", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag", "sizer"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag", "sizer"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "card:sizer" in output


@pytest.mark.asyncio
async def test_card_command_attach(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/card sizer.md --tool", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Loaded AgentCard(s): sizer" in output
    assert "Attached agent tool(s): sizer" in output


@pytest.mark.asyncio
async def test_history_fix_trims_pending_tool_call(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/history fix", "STOP"])

    class _HistoryAgent(_FakeAgent):
        def __init__(self) -> None:
            pending_tool_call = CallToolRequest(
                params=CallToolRequestParams(name="fake_tool", arguments={})
            )
            self._message_history = [
                Prompt.assistant(
                    "pending",
                    stop_reason=LlmStopReason.TOOL_USE,
                    tool_calls={"call-1": pending_tool_call},
                )
            ]

        @property
        def message_history(self):
            return self._message_history

        def load_message_history(self, history):
            self._message_history = list(history)

    class _HistoryAgentApp(_FakeAgentApp):
        def __init__(self):
            super().__init__(["test"])
            self._agents["test"] = _HistoryAgent()

        def _agent(self, agent_name: str | None):
            if agent_name is None:
                return self._agents["test"]
            return self._agents[agent_name]

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _HistoryAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="test",
        available_agents=["test"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Removed pending tool call" in output
    assert agent_app._agent("test").message_history == []


@pytest.mark.asyncio
async def test_history_fix_notice_on_cancelled_turn(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["STOP"])

    class _CancelledAgent(_FakeAgent):
        def __init__(self) -> None:
            self._last_turn_cancelled = True
            self._last_turn_cancel_reason = "cancelled"

    class _CancelledAgentApp(_FakeAgentApp):
        def __init__(self):
            super().__init__(["test"])
            self._agents["test"] = _CancelledAgent()

        def _agent(self, agent_name: str | None):
            if agent_name is None:
                return self._agents["test"]
            return self._agents[agent_name]

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _CancelledAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="test",
        available_agents=["test"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Previous turn was cancelled" in output


@pytest.mark.asyncio
async def test_reload_agents_no_changes(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/reload", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _ReloadAgentApp(["vertex-rag"], changed=False)

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "No AgentCard changes detected" in output


@pytest.mark.asyncio
async def test_reload_agents_with_changes(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/reload", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _ReloadAgentApp(["vertex-rag"], changed=True)

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "AgentCards reloaded" in output
