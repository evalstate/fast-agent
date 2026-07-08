from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from mcp.types import TextContent

from fast_agent.cli.runtime.session_resume import emit_resume_assistant_preview
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import AgentProtocol


class _Agent:
    def __init__(
        self,
        name: str,
        history: list[PromptMessageExtended] | None = None,
    ) -> None:
        self.name = name
        self.message_history = list(history or [])


class _AgentApp:
    def __init__(self, agents: list[_Agent]) -> None:
        self._agents = {agent.name: agent for agent in agents}

    def get_agent(self, name: str) -> _Agent | None:
        return self._agents.get(name)


def _message(role: Literal["user", "assistant"], text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role,
        content=[TextContent(type="text", text=text)],
    )


def test_resume_assistant_preview_uses_preferred_agent_history() -> None:
    queued: list[tuple[str, str | None]] = []
    preferred = _Agent("main", [_message("assistant", "main response")])
    other = _Agent("other", [_message("assistant", "other response")])

    emit_resume_assistant_preview(
        cast("AgentApp", _AgentApp([preferred, other])),
        cast("AgentProtocol", preferred),
        {"other": Path("other.json")},
        True,
        lambda text, **kwargs: queued.append((text, kwargs.get("agent_name"))),
    )

    assert queued == [("main response", "main")]


def test_resume_assistant_preview_falls_back_to_loaded_agent_history() -> None:
    queued: list[tuple[str, str | None]] = []
    preferred = _Agent("main", [_message("user", "last prompt")])
    loaded = _Agent("worker", [_message("assistant", "worker response")])

    emit_resume_assistant_preview(
        cast("AgentApp", _AgentApp([preferred, loaded])),
        cast("AgentProtocol", preferred),
        {"worker": Path("worker.json")},
        True,
        lambda text, **kwargs: queued.append((text, kwargs.get("agent_name"))),
    )

    assert queued == [("worker response", "worker")]
