from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.agents.agent_types import AgentType
from fast_agent.config import get_settings, update_global_settings
from fast_agent.core.fastagent import AgentInstance
from fast_agent.mcp.experimental_session_client import SessionJarEntry
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.ui.command_payloads import is_command_payload
from fast_agent.ui.interactive.command_dispatch import dispatch_command_payload
from fast_agent.ui.prompt import parse_special_input

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class _DisplayRecorder:
    messages: list[str] = field(default_factory=list)

    def show_status_message(self, content: object) -> None:
        plain = getattr(content, "plain", None)
        self.messages.append(plain if isinstance(plain, str) else str(content))


@dataclass
class _SessionClient:
    server_name: str = "demo"
    session_id: str = "sess-initial"

    async def list_jar(self) -> list[SessionJarEntry]:
        return [
            SessionJarEntry(
                server_name=self.server_name,
                server_identity=self.server_name,
                target="cmd:python demo.py",
                cookie={"id": self.session_id},
                cookies=(
                    {
                        "id": self.session_id,
                        "title": "Demo",
                        "expiry": None,
                        "updatedAt": "2026-03-14T00:00:00Z",
                        "active": True,
                    },
                ),
                last_used_id=self.session_id,
                title="Demo",
                supported=True,
                features=("create", "list"),
                connected=True,
            )
        ]

    async def resolve_server_name(self, server_identifier: str | None) -> str:
        return server_identifier or self.server_name

    async def list_server_cookies(
        self, server_identifier: str | None
    ) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
        resolved = server_identifier or self.server_name
        return resolved, resolved, self.session_id, [{"id": self.session_id, "title": "Demo"}]

    async def create_session(
        self,
        server_identifier: str | None,
        *,
        title: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        del title
        resolved = server_identifier or self.server_name
        self.session_id = "sess-created"
        return resolved, {"id": self.session_id}

    async def resume_session(
        self,
        server_identifier: str | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        resolved = server_identifier or self.server_name
        self.session_id = session_id
        return resolved, {"id": session_id}

    async def clear_cookie(self, server_identifier: str | None) -> str:
        return server_identifier or self.server_name

    async def clear_all_cookies(self) -> list[str]:
        return [self.server_name]


@dataclass
class _Aggregator:
    experimental_sessions: _SessionClient = field(default_factory=_SessionClient)


@dataclass
class _Agent:
    name: str
    display: _DisplayRecorder = field(default_factory=_DisplayRecorder)
    aggregator: _Aggregator = field(default_factory=_Aggregator)
    message_history: list[PromptMessageExtended] = field(default_factory=list)
    usage_accumulator: object | None = None
    agent_type: AgentType = AgentType.BASIC


@dataclass
class _Provider:
    _agents: dict[str, _Agent]

    def _agent(self, name: str) -> _Agent:
        return self._agents[name]

    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def agent_types(self) -> dict[str, AgentType]:
        return {name: agent.agent_type for name, agent in self._agents.items()}

    async def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> object:
        del namespace, agent_name
        return {}


@dataclass
class _Owner:
    agent_types: dict[str, AgentType]

    def _get_agent_or_warn(self, prompt_provider: _Provider, agent_name: str) -> _Agent | None:
        try:
            return prompt_provider._agent(agent_name)
        except KeyError:
            return None


def _merge_pinned_agents(agent_names: list[str]) -> list[str]:
    return agent_names


async def _dispatch_tui(
    raw_input: str,
    *,
    owner: _Owner,
    provider: _Provider,
    agent_name: str = "main",
) -> None:
    parsed = parse_special_input(raw_input)
    assert is_command_payload(parsed)
    await dispatch_command_payload(
        cast("Any", owner),
        parsed,
        prompt_provider=cast("Any", provider),
        agent=agent_name,
        available_agents=provider.agent_names(),
        available_agents_set=set(provider.agent_names()),
        merge_pinned_agents=_merge_pinned_agents,
    )


def _build_acp_handler(provider: _Provider, *, agent_name: str = "main") -> SlashCommandHandler:
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


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_session_pin_state_effect(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    update_global_settings(old_settings.model_copy(update={"environment_dir": str(env_dir)}))
    reset_session_manager()

    try:
        provider = _Provider({"main": _Agent(name="main")})
        owner = _Owner(agent_types=provider.agent_types())

        manager = get_session_manager()
        session = manager.create_session("sprint")
        session.set_pinned(False)

        await _dispatch_tui("/session pin on", owner=owner, provider=provider)
        assert manager.current_session is not None
        assert manager.current_session.info.metadata.get("pinned") is True
        assert any("Pinned session:" in message for message in provider._agent("main").display.messages)

        manager.current_session.set_pinned(False)

        handler = _build_acp_handler(provider)
        response = await handler.execute_command("session", "pin on")

        assert manager.current_session.info.metadata.get("pinned") is True
        assert "Pinned session:" in response
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_mcp_session_use_state_effect() -> None:
    provider = _Provider(
        {
            "main": _Agent(
                name="main",
                message_history=[
                    PromptMessageExtended(
                        role="user",
                        content=[TextContent(type="text", text="hello")],
                    )
                ],
            )
        }
    )
    owner = _Owner(agent_types=provider.agent_types())
    session_client = provider._agent("main").aggregator.experimental_sessions

    await _dispatch_tui("/mcp session use demo sess-123", owner=owner, provider=provider)
    assert session_client.session_id == "sess-123"
    assert any(
        "Selected MCP session for demo." in message
        for message in provider._agent("main").display.messages
    )

    session_client.session_id = "sess-initial"

    handler = _build_acp_handler(provider)
    response = await handler.execute_command("mcp", "session use demo sess-123")

    assert session_client.session_id == "sess-123"
    assert "Selected MCP session for demo." in response


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tui_and_acp_share_history_detail_error_intent() -> None:
    provider = _Provider({"main": _Agent(name="main")})
    owner = _Owner(agent_types=provider.agent_types())

    await _dispatch_tui("/history detail", owner=owner, provider=provider)
    emitted = "\n".join(provider._agent("main").display.messages)
    assert "Turn number required for /history detail" in emitted

    handler = _build_acp_handler(provider)
    response = await handler.execute_command("history", "detail")

    assert "Turn number required for /history detail" in response
