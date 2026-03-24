from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.context import StaticAgentProvider
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.experimental_session_client import SessionJarEntry
from fast_agent.ui.command_payloads import is_command_payload
from fast_agent.ui.interactive.command_dispatch import DispatchResult, dispatch_command_payload
from fast_agent.ui.prompt import parse_special_input

if TYPE_CHECKING:
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


@dataclass
class DisplayRecorder:
    messages: list[str] = field(default_factory=list)

    def show_status_message(self, content: object) -> None:
        plain = getattr(content, "plain", None)
        self.messages.append(plain if isinstance(plain, str) else str(content))


@dataclass
class SessionClient:
    server_name: str = "demo"
    active_session_id: str = "sess-1"

    async def list_jar(self) -> list[SessionJarEntry]:
        return [
            SessionJarEntry(
                server_name=self.server_name,
                server_identity=self.server_name,
                target="cmd:python demo.py",
                cookie={"id": self.active_session_id},
                cookies=(
                    {
                        "id": self.active_session_id,
                        "title": "Demo",
                        "expiry": None,
                        "updatedAt": "2026-03-14T00:00:00Z",
                        "active": True,
                    },
                ),
                last_used_id=self.active_session_id,
                title="Demo",
                supported=True,
                features=("create", "list"),
                connected=True,
            )
        ]

    async def resolve_server_name(self, server_identifier: str | None) -> str:
        return server_identifier or self.server_name

    async def list_server_cookies(
        self,
        server_identifier: str | None,
    ) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
        server_name = server_identifier or self.server_name
        return (
            server_name,
            server_name,
            self.active_session_id,
            [{"id": self.active_session_id, "title": "Demo"}],
        )

    async def create_session(
        self,
        server_identifier: str | None,
        *,
        title: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        del title
        server_name = server_identifier or self.server_name
        self.active_session_id = "sess-created"
        return server_name, {"id": self.active_session_id}

    async def resume_session(
        self,
        server_identifier: str | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        server_name = server_identifier or self.server_name
        self.active_session_id = session_id
        return server_name, {"id": session_id}

    async def clear_cookie(self, server_identifier: str | None) -> str:
        return server_identifier or self.server_name

    async def clear_all_cookies(self) -> list[str]:
        return [self.server_name]


@dataclass
class Aggregator:
    experimental_sessions: SessionClient = field(default_factory=SessionClient)


@dataclass
class CommandSurfaceAgent:
    name: str
    display: DisplayRecorder = field(default_factory=DisplayRecorder)
    message_history: list[PromptMessageExtended] = field(default_factory=list)
    usage_accumulator: object | None = None
    template_messages: list[PromptMessageExtended] | None = None
    aggregator: Aggregator = field(default_factory=Aggregator)
    agent_type: AgentType = AgentType.BASIC

    def clear(self, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def pop_last_message(self) -> PromptMessageExtended | None:
        if not self.message_history:
            return None
        return self.message_history.pop()

    def load_message_history(self, history: list[PromptMessageExtended] | None) -> None:
        self.message_history = list(history or [])


class CommandSurfaceProvider(StaticAgentProvider):
    def __init__(
        self,
        agents: dict[str, CommandSurfaceAgent],
        *,
        attached_mcp_servers: list[str] | None = None,
        detached_mcp_servers: list[str] | None = None,
        noenv_mode: bool = False,
    ) -> None:
        super().__init__(agents)
        self._attached_mcp_servers = list(attached_mcp_servers or [])
        self._detached_mcp_servers = list(detached_mcp_servers or ["docs"])
        self._noenv_mode = noenv_mode

    def _agent(self, name: str) -> CommandSurfaceAgent:
        return cast("CommandSurfaceAgent", super()._agent(name))

    def agent_names(self) -> list[str]:
        return list(self.registered_agent_names())

    def agent_types(self) -> dict[str, AgentType]:
        return {
            name: cast("CommandSurfaceAgent", agent).agent_type
            for name, agent in self.registered_agents().items()
        }

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, AgentType]:
        visible = set(self.visible_agent_names(force_include=force_include))
        return {
            name: cast("CommandSurfaceAgent", agent).agent_type
            for name, agent in self.registered_agents().items()
            if name in visible
        }

    async def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> object:
        del namespace, agent_name
        return {}

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: object | None = None,
        options: object | None = None,
    ) -> object:
        del agent_name, server_config, options
        if server_name not in self._attached_mcp_servers:
            self._attached_mcp_servers.append(server_name)
        if server_name in self._detached_mcp_servers:
            self._detached_mcp_servers.remove(server_name)
        return object()

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> object:
        del agent_name
        if server_name in self._attached_mcp_servers:
            self._attached_mcp_servers.remove(server_name)
        if server_name not in self._detached_mcp_servers:
            self._detached_mcp_servers.append(server_name)
        return object()

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        del agent_name
        return list(self._attached_mcp_servers)

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        del agent_name
        return list(self._detached_mcp_servers)


@dataclass
class CommandSurfaceOwner:
    agent_types: dict[str, AgentType] = field(default_factory=dict)

    def _get_agent_or_warn(
        self,
        prompt_provider: CommandSurfaceProvider,
        agent_name: str,
    ) -> CommandSurfaceAgent | None:
        try:
            return prompt_provider._agent(agent_name)
        except KeyError:
            return None


def merge_pinned_agents(agent_names: list[str]) -> list[str]:
    return agent_names


async def dispatch_tui_command(
    raw_input: str,
    *,
    owner: CommandSurfaceOwner,
    prompt_provider: CommandSurfaceProvider,
    agent_name: str = "main",
    buffer_prefill: str = "",
) -> DispatchResult:
    parsed = parse_special_input(raw_input)
    assert is_command_payload(parsed)
    return await dispatch_command_payload(
        cast("Any", owner),
        parsed,
        prompt_provider=cast("Any", prompt_provider),
        agent=agent_name,
        available_agents=prompt_provider.agent_names(),
        available_agents_set=set(prompt_provider.agent_names()),
        merge_pinned_agents=merge_pinned_agents,
        buffer_prefill=buffer_prefill,
    )


def build_acp_handler(
    provider: CommandSurfaceProvider,
    *,
    agent_name: str = "main",
) -> SlashCommandHandler:
    instance = AgentInstance(
        app=cast("Any", provider),
        agents=cast("dict[str, Any]", provider._agents),
        registry_version=0,
    )
    return SlashCommandHandler(
        session_id="test-session",
        instance=instance,
        primary_agent_name=agent_name,
    )
