from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.constants import (
    ANTHROPIC_ASSISTANT_RAW_CONTENT,
    ANTHROPIC_CITATIONS_CHANNEL,
    ANTHROPIC_SERVER_TOOLS_CHANNEL,
)
from fast_agent.core.exceptions import PromptExitError
from fast_agent.core.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui import enhanced_prompt, interactive_prompt
from fast_agent.ui.command_payloads import InterruptCommand
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

    async def attach_mcp_server(
        self,
        _agent_name: str,
        server_name: str,
        server_config: object | None = None,
        options: object | None = None,
    ) -> object:
        del server_config, options
        self.attached.append(server_name)
        return object()

    async def detach_mcp_server(self, _agent_name: str, server_name: str) -> object:
        self.detached.append(server_name)
        return object()

    async def list_attached_mcp_servers(self, _agent_name: str) -> list[str]:
        return []

    async def list_configured_detached_mcp_servers(self, _agent_name: str) -> list[str]:
        return []


class _ReloadAgentApp(_FakeAgentApp):
    def __init__(self, agent_names: list[str], changed: bool) -> None:
        super().__init__(agent_names)
        self._changed = changed

    def can_reload_agents(self) -> bool:
        return True

    async def reload_agents(self) -> bool:
        return self._changed


class _DetachCancelledAgentApp(_FakeAgentApp):
    async def detach_mcp_server(self, _agent_name: str, server_name: str) -> object:
        self.detached.append(server_name)
        raise asyncio.CancelledError()


class _PreAttachedMcpAgentApp(_FakeAgentApp):
    async def list_attached_mcp_servers(self, _agent_name: str) -> list[str]:
        return ["demo"]


def _patch_input(monkeypatch, inputs: list[str]) -> None:
    iterator = iter(inputs)

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(iterator)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)


@pytest.mark.asyncio
async def test_prompt_loop_skips_shell_cwd_startup_prompt_when_policy_not_ask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_input(monkeypatch, ["STOP"])

    def _unexpected_collect(*_args: Any, **_kwargs: Any) -> list[Any]:
        raise AssertionError("Shell cwd startup collection should not run outside ask policy.")

    async def _unexpected_selection(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("Shell cwd startup prompt should not run outside ask policy.")

    monkeypatch.setattr(
        interactive_prompt,
        "collect_shell_cwd_issues_from_runtime_agents",
        _unexpected_collect,
    )
    monkeypatch.setattr(interactive_prompt, "get_selection_input", _unexpected_selection)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])
    setattr(agent_app, "_missing_shell_cwd_policy_override", "warn")

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )


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
async def test_history_webclear_removes_web_channels(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["/history webclear", "STOP"])

    class _HistoryAgent(_FakeAgent):
        def __init__(self) -> None:
            class _LlmStub:
                web_tools_enabled = (True, False)

            self.llm = _LlmStub()
            self._message_history = [
                PromptMessageExtended(
                    role="assistant",
                    content=[TextContent(type="text", text="done")],
                    channels={
                        ANTHROPIC_SERVER_TOOLS_CHANNEL: [
                            TextContent(type="text", text='{"type":"server_tool_use"}')
                        ],
                        ANTHROPIC_ASSISTANT_RAW_CONTENT: [
                            TextContent(
                                type="text",
                                text='{"type":"server_tool_use","name":"web_search","id":"srv_1"}',
                            )
                        ],
                        ANTHROPIC_CITATIONS_CHANNEL: [
                            TextContent(
                                type="text",
                                text='{"type":"web_search_result_location","url":"https://example.com"}',
                            )
                        ],
                    },
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
    assert "Removed 3 web metadata block(s)" in output
    channels = agent_app._agent("test").message_history[0].channels
    assert channels is None


@pytest.mark.asyncio
async def test_cancelled_turn_auto_fixes_pending_tool_call(monkeypatch, capsys: Any) -> None:
    _patch_input(monkeypatch, ["STOP"])

    class _CancelledHistoryAgent(_FakeAgent):
        def __init__(self) -> None:
            pending_tool_call = CallToolRequest(
                params=CallToolRequestParams(name="fake_tool", arguments={})
            )
            self._last_turn_cancelled = True
            self._last_turn_cancel_reason = "interrupted"
            self._message_history = [
                Prompt.user("hello"),
                Prompt.assistant(
                    "pending",
                    stop_reason=LlmStopReason.TOOL_USE,
                    tool_calls={"call-1": pending_tool_call},
                ),
            ]

        @property
        def message_history(self):
            return self._message_history

        def load_message_history(self, history):
            self._message_history = list(history)

    class _CancelledHistoryAgentApp(_FakeAgentApp):
        def __init__(self):
            super().__init__(["test"])
            self._agents["test"] = _CancelledHistoryAgent()

        def _agent(self, agent_name: str | None):
            if agent_name is None:
                return self._agents["test"]
            return self._agents[agent_name]

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _CancelledHistoryAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="test",
        available_agents=["test"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Removed pending tool call from history" in output
    history = agent_app._agent("test").message_history
    assert len(history) == 1
    assert history[0].role == "user"


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


@pytest.mark.asyncio
async def test_mcp_connect_ctrl_c_cancels_and_returns_to_prompt(
    monkeypatch, capsys: Any
) -> None:
    _patch_input(monkeypatch, ["/mcp connect npx demo-server --name demo", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    async def fake_handle_mcp_connect(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        interactive_prompt.mcp_runtime_handlers,
        "handle_mcp_connect",
        fake_handle_mcp_connect,
    )

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "MCP connect cancelled; returned to prompt" in output
    assert "demo" in agent_app.detached


@pytest.mark.asyncio
async def test_mcp_connect_cancel_survives_detach_cancelled_error(
    monkeypatch, capsys: Any
) -> None:
    _patch_input(monkeypatch, ["/mcp connect npx demo-server --name demo", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    async def fake_handle_mcp_connect(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        interactive_prompt.mcp_runtime_handlers,
        "handle_mcp_connect",
        fake_handle_mcp_connect,
    )

    prompt_ui = InteractivePrompt()
    agent_app = _DetachCancelledAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "MCP connect cancelled; returned to prompt" in output
    assert "demo" in agent_app.detached


@pytest.mark.asyncio
async def test_mcp_connect_cancel_does_not_detach_previously_attached_server(
    monkeypatch, capsys: Any
) -> None:
    _patch_input(monkeypatch, ["/mcp connect npx demo-server --name demo --reconnect", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    async def fake_handle_mcp_connect(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        interactive_prompt.mcp_runtime_handlers,
        "handle_mcp_connect",
        fake_handle_mcp_connect,
    )

    prompt_ui = InteractivePrompt()
    agent_app = _PreAttachedMcpAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "MCP connect cancelled; returned to prompt" in output
    assert "demo" not in agent_app.detached


@pytest.mark.asyncio
async def test_mcp_connect_cancel_allows_stop_immediately(monkeypatch, capsys: Any) -> None:
    inputs = iter([
        "/mcp connect npx demo-server --name demo",
        "STOP",
    ])
    input_calls = {"count": 0}

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        input_calls["count"] += 1
        return next(inputs)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    async def fake_handle_mcp_connect(*_args: Any, **_kwargs: Any):
        raise asyncio.CancelledError()

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)
    monkeypatch.setattr(
        interactive_prompt.mcp_runtime_handlers,
        "handle_mcp_connect",
        fake_handle_mcp_connect,
    )

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "MCP connect cancelled; returned to prompt" in output
    # STOP immediately exits after cancellation.
    assert input_calls["count"] == 2


@pytest.mark.asyncio
async def test_mcp_connect_oauth_link_is_not_repeated_in_final_outcome(
    monkeypatch,
    capsys: Any,
) -> None:
    _patch_input(monkeypatch, ["/connect https://example.com/api", "STOP"])

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    auth_url = "https://auth.example.com/authorize?state=abc"

    async def fake_handle_mcp_connect(*_args: Any, **kwargs: Any):
        progress_cb = kwargs.get("on_progress")
        if progress_cb is not None:
            await progress_cb(f"Open this link to authorize: {auth_url}")

        return CommandOutcome(
            messages=[
                CommandMessage("Connected MCP server 'example-com' (url)."),
                CommandMessage(f"OAuth authorization link: {auth_url}", channel="info"),
            ]
        )

    monkeypatch.setattr(
        interactive_prompt.mcp_runtime_handlers,
        "handle_mcp_connect",
        fake_handle_mcp_connect,
    )

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert output.count(auth_url) == 1
    assert "OAuth authorization link:" not in output


@pytest.mark.asyncio
async def test_prompt_loop_recovers_from_keyboard_interrupt_in_input(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])
    input_calls = {"count": 0}

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        input_calls["count"] += 1
        if input_calls["count"] == 1:
            raise KeyboardInterrupt()
        return next(inputs)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Press Ctrl+C again within 2 seconds" in output
    # KeyboardInterrupt + STOP(exit)
    assert input_calls["count"] == 2


@pytest.mark.asyncio
async def test_prompt_loop_recovers_from_keyboard_interrupt_during_send(
    monkeypatch,
    capsys: Any,
) -> None:
    inputs = iter(["hello", "STOP"])
    input_calls = {"count": 0}
    send_calls = {"count": 0}

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        input_calls["count"] += 1
        return next(inputs)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        send_calls["count"] += 1
        if send_calls["count"] == 1:
            raise KeyboardInterrupt()
        return ""

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Generation cancelled by user." in output
    assert input_calls["count"] == 2
    assert send_calls["count"] == 1


@pytest.mark.asyncio
async def test_prompt_loop_recovers_from_cancelled_error_during_send(
    monkeypatch,
    capsys: Any,
) -> None:
    inputs = iter(["hello", "STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        raise asyncio.CancelledError()

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    output = capsys.readouterr().out
    assert "Generation cancelled by user." in output


@pytest.mark.asyncio
async def test_prompt_loop_double_ctrl_c_exits(monkeypatch) -> None:
    inputs = iter([InterruptCommand(), InterruptCommand()])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> InterruptCommand:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp(["vertex-rag"])

    with pytest.raises(PromptExitError):
        await prompt_ui.prompt_loop(
            send_func=fake_send,
            default_agent="vertex-rag",
            available_agents=["vertex-rag"],
            prompt_provider=cast("AgentApp", agent_app),
        )


@pytest.mark.asyncio
async def test_prompt_loop_stop_exits_after_cancelled_generation(
    monkeypatch,
) -> None:
    class _CancelledGenerationAgent(_FakeAgent):
        def __init__(self) -> None:
            self.message_history: list[object] = []

    class _CancelledGenerationApp(_FakeAgentApp):
        def __init__(self) -> None:
            super().__init__(["test"])
            self._agents["test"] = _CancelledGenerationAgent()

        def _agent(self, agent_name: str | None):
            if agent_name is None:
                return self._agents["test"]
            return self._agents[agent_name]

    inputs = iter(["hello", "STOP"])
    input_calls = {"count": 0}
    send_calls = {"count": 0}

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        input_calls["count"] += 1
        return next(inputs)

    async def fake_send(_message: Any, _agent_name: str) -> str:
        send_calls["count"] += 1
        return ""

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    prompt_ui = InteractivePrompt()
    agent_app = _CancelledGenerationApp()

    async def send_wrapper(message: Any, agent_name: str) -> str:
        result = await fake_send(message, agent_name)
        if send_calls["count"] == 1:
            cancelled_message = Prompt.assistant(
                "",
                stop_reason=LlmStopReason.CANCELLED,
            )
            agent_app._agent(agent_name).message_history.append(cancelled_message)
        return result

    await prompt_ui.prompt_loop(
        send_func=send_wrapper,
        default_agent="test",
        available_agents=["test"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    # hello + STOP(exit)
    assert input_calls["count"] == 2
    assert send_calls["count"] == 1
