from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentType
from fast_agent.config import get_settings, update_global_settings
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import get_session_manager, reset_session_manager
from fast_agent.ui.command_payloads import is_command_payload
from fast_agent.ui.interactive.command_dispatch import DispatchResult, dispatch_command_payload
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
    active_session_id: str = "sess-1"

    async def list_jar(self) -> list[object]:
        return []

    async def resolve_server_name(self, server_identifier: str | None) -> str:
        return server_identifier or "demo"

    async def list_server_cookies(
        self, server_identifier: str | None
    ) -> tuple[str, str | None, str | None, list[dict[str, Any]]]:
        server_name = server_identifier or "demo"
        return server_name, server_name, self.active_session_id, [{"id": self.active_session_id}]

    async def create_session(
        self,
        server_identifier: str | None,
        *,
        title: str | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        del title
        server_name = server_identifier or "demo"
        self.active_session_id = "sess-created"
        return server_name, {"id": self.active_session_id}

    async def resume_session(
        self,
        server_identifier: str | None,
        *,
        session_id: str,
    ) -> tuple[str, dict[str, Any]]:
        server_name = server_identifier or "demo"
        self.active_session_id = session_id
        return server_name, {"id": session_id}

    async def clear_cookie(self, server_identifier: str | None) -> str:
        return server_identifier or "demo"

    async def clear_all_cookies(self) -> list[str]:
        return ["demo"]


@dataclass
class _Aggregator:
    experimental_sessions: _SessionClient = field(default_factory=_SessionClient)


@dataclass
class _Agent:
    name: str
    display: _DisplayRecorder = field(default_factory=_DisplayRecorder)
    message_history: list[PromptMessageExtended] = field(default_factory=list)
    usage_accumulator: object | None = None
    template_messages: list[PromptMessageExtended] | None = None
    aggregator: _Aggregator = field(default_factory=_Aggregator)
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


@dataclass
class _PromptProvider:
    _agents: dict[str, _Agent]
    _attached_mcp_servers: list[str] = field(default_factory=list)
    _detached_mcp_servers: list[str] = field(default_factory=lambda: ["docs"])
    _noenv_mode: bool = False

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
class _Owner:
    agent_types: dict[str, AgentType] = field(default_factory=dict)

    def _get_agent_or_warn(self, prompt_provider: _PromptProvider, agent_name: str) -> _Agent | None:
        try:
            return prompt_provider._agent(agent_name)
        except KeyError:
            return None


def _merge_pinned_agents(agent_names: list[str]) -> list[str]:
    return agent_names


async def _dispatch_raw_command(
    raw_input: str,
    *,
    owner: _Owner,
    prompt_provider: _PromptProvider,
    agent_name: str = "main",
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
        merge_pinned_agents=_merge_pinned_agents,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_session_flow_updates_session_state(tmp_path: Path) -> None:
    old_settings = get_settings()
    env_dir = tmp_path / "env"
    update_global_settings(old_settings.model_copy(update={"environment_dir": str(env_dir)}))
    reset_session_manager()

    try:
        provider = _PromptProvider({"main": _Agent(name="main")})
        owner = _Owner(agent_types=provider.agent_types())

        create_result = await _dispatch_raw_command(
            "/session new sprint",
            owner=owner,
            prompt_provider=provider,
        )
        pin_result = await _dispatch_raw_command(
            "/session pin on",
            owner=owner,
            prompt_provider=provider,
        )
        await _dispatch_raw_command(
            "/session list",
            owner=owner,
            prompt_provider=provider,
        )

        assert create_result == DispatchResult(handled=True)
        assert pin_result == DispatchResult(handled=True)

        manager = get_session_manager()
        current_session = manager.current_session
        assert current_session is not None
        assert current_session.info.metadata.get("pinned") is True

        emitted = "\n".join(provider._agent("main").display.messages)
        assert "Created session:" in emitted
        assert current_session.info.name in emitted
        assert "Pinned session:" in emitted
        assert "Sessions:" in emitted
        assert "(pin)" in emitted
    finally:
        update_global_settings(old_settings)
        reset_session_manager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_history_rewind_updates_history_and_prefills_buffer() -> None:
    agent = _Agent(
        name="main",
        message_history=[
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="first question")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="first answer")],
            ),
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="second question")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="second answer")],
            ),
        ],
    )
    provider = _PromptProvider({"main": agent})
    owner = _Owner(agent_types=provider.agent_types())

    result = await _dispatch_raw_command(
        "/history rewind 2",
        owner=owner,
        prompt_provider=provider,
    )

    assert result.buffer_prefill == "second question"
    assert [message.role for message in agent.message_history] == ["user", "assistant"]
    assert agent.message_history[0].first_text() == "first question"
    assert agent.message_history[1].first_text() == "first answer"
    assert any("History rewound" in message for message in agent.display.messages)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dispatch_hash_agent_sets_message_handoff() -> None:
    provider = _PromptProvider(
        {
            "main": _Agent(name="main"),
            "review": _Agent(name="review"),
        }
    )
    owner = _Owner(agent_types=provider.agent_types())

    result = await _dispatch_raw_command(
        "##review please assess this change",
        owner=owner,
        prompt_provider=provider,
    )

    assert result.handled is True
    assert result.hash_send_target == "review"
    assert result.hash_send_message == "please assess this change"
    assert result.hash_send_quiet is True
