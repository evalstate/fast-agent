"""Data types for plugin command actions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Awaitable, Mapping, Protocol

from mcp.types import TextContent

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.command_actions.runtime import PluginRuntime
    from fast_agent.config import Settings
    from fast_agent.context import Context
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended


FAST_AGENT_AUDIT_CHANNEL = "fast-agent.audit"


class PluginCommandAgentProtocol(Protocol):
    """Agent surface required by plugin command actions."""

    @property
    def name(self) -> str: ...

    @property
    def message_history(self) -> list["PromptMessageExtended"]: ...

    def load_message_history(self, messages: list["PromptMessageExtended"] | None) -> None: ...

    @property
    def context(self) -> "Context | None": ...

    @property
    def config(self) -> "AgentConfig": ...

    @property
    def agent_registry(self) -> "Mapping[str, AgentProtocol] | None": ...

    def get_agent(self, name: str) -> "AgentProtocol | None": ...

    async def send(self, message: str) -> str: ...


@dataclass(frozen=True, slots=True)
class PluginCommandActionSpec:
    """Configured plugin slash-command metadata."""

    name: str
    description: str
    handler: str
    input_hint: str | None = None
    key: str | None = None


@dataclass(slots=True)
class PluginCommandActionResult:
    """Result returned by a plugin command action."""

    message: str | None = None
    markdown: str | None = None
    buffer_prefill: str | None = None
    switch_agent: str | None = None
    refresh_agents: bool = False


class PluginCommandActionFunction(Protocol):
    """Async plugin command action callable."""

    __name__: str

    def __call__(
        self,
        ctx: "PluginCommandActionContext",
    ) -> Awaitable[PluginCommandActionResult | str | None]: ...


@dataclass(frozen=True, slots=True)
class PluginCommandAction:
    """Loaded plugin command action."""

    spec: PluginCommandActionSpec
    handler: PluginCommandActionFunction


@dataclass(slots=True)
class PluginCommandActionContext:
    """Runtime context passed to plugin command actions."""

    command_name: str
    arguments: str
    agent: PluginCommandAgentProtocol
    settings: "Settings | None" = None
    session_cwd: Path | None = None
    runtime: "PluginRuntime | None" = None
    is_tui: bool = False
    is_acp: bool = False

    @property
    def agent_name(self) -> str:
        return self.agent.name

    @property
    def context(self) -> "Context | None":
        return self.agent.context

    @property
    def message_history(self) -> list["PromptMessageExtended"]:
        return self.agent.message_history

    @property
    def agent_registry(self) -> "Mapping[str, AgentProtocol] | None":
        return self.agent.agent_registry

    def load_message_history(self, messages: list["PromptMessageExtended"] | None) -> None:
        self.agent.load_message_history(messages)

    def get_agent(self, name: str) -> "AgentProtocol | None":
        return self.agent.get_agent(name)

    def mark_user_adjusted(
        self,
        message: "PromptMessageExtended",
        *,
        note: str | None = None,
    ) -> None:
        payload = {
            "event": "user_adjusted",
            "command": self.command_name,
            "at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
        if note:
            payload["note"] = note

        channels = dict(message.channels or {})
        existing = list(channels.get(FAST_AGENT_AUDIT_CHANNEL, ()))
        existing.append(TextContent(type="text", text=json.dumps(payload, separators=(",", ":"))))
        channels[FAST_AGENT_AUDIT_CHANNEL] = existing
        message.channels = channels
