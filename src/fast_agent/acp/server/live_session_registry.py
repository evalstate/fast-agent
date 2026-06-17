"""ACP live session registry state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.core.fastagent import AgentInstance


@dataclass(slots=True)
class ACPLiveSessionRegistry:
    """Own ACP live session maps while preserving transport-specific state."""

    sessions: dict[str, AgentInstance] = field(default_factory=dict)
    session_state: dict[str, ACPSessionState] = field(default_factory=dict)
    prompt_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    active_prompts: set[str] = field(default_factory=set)
    session_tasks: dict[str, asyncio.Task[Any]] = field(default_factory=dict)

    def clear_prompt_state(self) -> None:
        self.session_tasks.clear()
        self.active_prompts.clear()
        self.prompt_locks.clear()

    def clear_all(self) -> None:
        self.session_state.clear()
        self.clear_prompt_state()
        self.sessions.clear()


__all__ = ["ACPLiveSessionRegistry"]
