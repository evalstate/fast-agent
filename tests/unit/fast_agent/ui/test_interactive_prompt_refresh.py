from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentType
from fast_agent.ui import enhanced_prompt, interactive_prompt
from fast_agent.ui.interactive_prompt import InteractivePrompt

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


class _FakeAgent:
    agent_type = AgentType.BASIC


class _FakeAgentApp:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {"vertex-rag": _FakeAgent()}
        self._refreshed = False

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._agents["sizer"] = _FakeAgent()
        self._refreshed = True
        return True

    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


class _FakeAgentAppRemove:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {
            "vertex-rag": _FakeAgent(),
            "sizer": _FakeAgent(),
        }
        self._refreshed = False

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._agents.pop("sizer", None)
        self._refreshed = True
        return True

    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


class _FakeToolOnlyAgentApp:
    def __init__(self) -> None:
        self._agents: dict[str, _FakeAgent] = {
            "tool-only": _FakeAgent(),
            "vertex-rag": _FakeAgent(),
        }
        self._tool_only = {"tool-only"}
        self._refreshed = False

    async def refresh_if_needed(self) -> bool:
        if self._refreshed:
            return False
        self._refreshed = True
        return True

    def agent_names(self) -> list[str]:
        return [name for name in self._agents.keys() if name not in self._tool_only]

    def agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    def can_load_agent_cards(self) -> bool:
        return False

    def can_reload_agents(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_prompt_loop_refreshes_agent_list(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    capsys.readouterr()
    assert "sizer" in enhanced_prompt.available_agents


@pytest.mark.asyncio
async def test_prompt_loop_prunes_removed_agent(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeAgentAppRemove()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="vertex-rag",
        available_agents=["vertex-rag", "sizer"],
        prompt_provider=cast("AgentApp", agent_app),
    )

    capsys.readouterr()
    assert enhanced_prompt.available_agents == {"vertex-rag"}


@pytest.mark.asyncio
async def test_prompt_loop_preserves_pinned_tool_only_agent(monkeypatch, capsys: Any) -> None:
    inputs = iter(["STOP"])

    async def fake_get_enhanced_input(*_args: Any, **kwargs: Any) -> str:
        available_agent_names = kwargs.get("available_agent_names")
        if available_agent_names is not None:
            enhanced_prompt.available_agents = set(available_agent_names)
        return next(inputs)

    monkeypatch.setattr(interactive_prompt, "get_enhanced_input", fake_get_enhanced_input)

    async def fake_send(*_args: Any, **_kwargs: Any) -> str:
        return ""

    prompt_ui = InteractivePrompt()
    agent_app = _FakeToolOnlyAgentApp()

    await prompt_ui.prompt_loop(
        send_func=fake_send,
        default_agent="tool-only",
        available_agents=["tool-only", "vertex-rag"],
        prompt_provider=cast("AgentApp", agent_app),
        pinned_agent="tool-only",
    )

    capsys.readouterr()
    assert enhanced_prompt.available_agents == {"tool-only", "vertex-rag"}
