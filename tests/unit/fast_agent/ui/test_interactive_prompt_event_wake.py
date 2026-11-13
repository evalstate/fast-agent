from __future__ import annotations

from typing import Any, cast

import pytest

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.ui import interactive_prompt
from fast_agent.ui.command_payloads import EventWakeCommand
from fast_agent.ui.interactive_prompt import InteractivePrompt


class _WakeAgent:
    def __init__(self) -> None:
        self.message_history: list[Any] = []
        self.pending = ["<mcp-event>one</mcp-event>"]

    def take_event_messages(self) -> list[str]:
        messages, self.pending = self.pending, []
        return messages

    async def wait_for_event_wake(self) -> None:
        return None


class _WakeAgentApp:
    def __init__(self) -> None:
        self.agent = _WakeAgent()
        self.no_home_mode = False
        self.missing_shell_cwd_policy_override = None

    async def refresh_if_needed(self) -> bool:
        return False

    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=False)

    def _agent(self, _name: str) -> _WakeAgent:
        return self.agent

    def visible_agent_names(self, *, force_include: str | None = None) -> list[str]:
        del force_include
        return ["agent1"]

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        del force_include
        return {"agent1": AgentType.BASIC}

    def registered_agent_names(self) -> list[str]:
        return ["agent1"]

    def registered_agents(self) -> dict[str, _WakeAgent]:
        return {"agent1": self.agent}

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name or "agent1"


@pytest.mark.asyncio
async def test_event_wake_starts_a_turn_and_returns_the_typed_text(monkeypatch) -> None:
    app = _WakeAgentApp()
    inputs = iter([EventWakeCommand(buffer_text="draft"), "STOP"])
    prompts: list[dict[str, Any]] = []

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str | EventWakeCommand:
        prompts.append(kwargs)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    sent: list[Any] = []

    async def fake_send(payload, _agent_name: str) -> str:
        sent.append(payload)
        return "ok"

    result = await InteractivePrompt().prompt_loop(
        send_func=fake_send,
        default_agent="agent1",
        available_agents=["agent1"],
        prompt_provider=cast("Any", app),
    )

    assert sent == ["<mcp-event>one</mcp-event>"]
    assert result == "ok"
    assert prompts[0]["wait_for_wake"] == app.agent.wait_for_event_wake
    assert prompts[0]["pre_populate_buffer"] == ""
    assert prompts[1]["pre_populate_buffer"] == "draft"
