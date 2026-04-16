from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import (
    SessionAgentSnapshot,
    SessionAttachmentRef,
    SessionContinuationSnapshot,
    SessionHydrationPolicy,
    SessionHydrator,
    SessionRequestSettingsSnapshot,
    SessionSnapshot,
)
from fast_agent.session.session_manager import SessionManager

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol
    from fast_agent.session.session_manager import Session


class _ProviderStub:
    def __init__(self) -> None:
        self.config_name = None


class _TurnRecord(str):
    def model_copy(self, *, deep: bool = False) -> "_TurnRecord":
        del deep
        return _TurnRecord(self)


class _UsageAccumulator:
    def __init__(self, turns: list[str] | None = None) -> None:
        self.turns = [_TurnRecord(turn) for turn in (turns or [])]
        self.model = None
        self.last_cache_activity_time = None
        self.context_window_size = None

    def model_copy(self, *, deep: bool = False) -> "_UsageAccumulator":
        del deep
        copied = _UsageAccumulator(list(self.turns))
        copied.model = self.model
        copied.last_cache_activity_time = self.last_cache_activity_time
        copied.context_window_size = self.context_window_size
        return copied

    def reset(self) -> None:
        self.turns.clear()

    def set_context_window_size(self, value) -> None:
        self.context_window_size = value

    def get_summary(self) -> dict[str, int]:
        return {}


class _Agent:
    def __init__(
        self,
        *,
        name: str,
        instruction: str,
        history: list[PromptMessageExtended] | None = None,
        attached_servers: list[str] | None = None,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self.config = AgentConfig(name, instruction=instruction, model="passthrough")
        self.config.default_request_params = RequestParams(
            use_history=self.config.use_history,
            systemPrompt=instruction,
        )
        self.llm = SimpleNamespace(
            default_request_params=self.config.default_request_params.model_copy(deep=True),
            model_name=self.config.model,
            provider=_ProviderStub(),
            resolved_model=SimpleNamespace(overlay=None),
        )
        self.usage_accumulator = _UsageAccumulator()
        self.message_history = list(history or [])
        self.model_updates: list[str | None] = []
        self.attached_servers = list(attached_servers or [])
        self._agent_tools: dict[str, _Agent] = {}

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts
        self.message_history.clear()

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction
        self.config.instruction = instruction
        params = self.config.default_request_params
        assert params is not None
        params.systemPrompt = instruction
        self.llm.default_request_params.systemPrompt = instruction

    async def set_model(self, model: str | None) -> None:
        self.model_updates.append(model)
        self.config.model = model

    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: object | None = None,
        options: object | None = None,
    ) -> None:
        del server_config, options
        self.attached_servers.append(server_name)

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self.attached_servers)

    @property
    def agent_backed_tools(self) -> dict[str, "_Agent"]:
        return dict(self._agent_tools)

    def add_agent_tool(
        self,
        child: "_Agent",
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> str:
        del description
        tool_name = name or f"agent__{child.name}"
        self._agent_tools[tool_name] = child
        return tool_name


class _FailingPromptAgent(_Agent):
    def set_instruction(self, instruction: str) -> None:
        del instruction
        raise RuntimeError("boom")


def _message(role: Literal["user", "assistant"], text: str) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role,
        content=[TextContent(type="text", text=text)],
    )


def _message_texts(agent: _Agent) -> list[str]:
    return [
        content.text
        for message in agent.message_history
        for content in message.content
        if isinstance(content, TextContent)
    ]


def _write_snapshot(session: Session, snapshot: SessionSnapshot) -> None:
    (session.directory / "session.json").write_text(
        json.dumps(snapshot.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_hydrate_session_restores_transcript_prompt_and_active_agent(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()

    saved_foo = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "hello foo"), _message("assistant", "done foo")],
    )
    saved_bar = _Agent(
        name="bar",
        instruction="Stored bar prompt",
        history=[_message("user", "hello bar"), _message("assistant", "done bar")],
    )

    await session.save_history(
        cast("AgentProtocol", saved_foo),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )
    await session.save_history(
        cast("AgentProtocol", saved_bar),
        agent_registry={
            "foo": cast("AgentProtocol", saved_foo),
            "bar": cast("AgentProtocol", saved_bar),
        },
    )

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _Agent(name="foo", instruction="Changed foo prompt")
    runtime_bar = _Agent(name="bar", instruction="Changed bar prompt")

    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={
            "foo": cast("AgentProtocol", runtime_foo),
            "bar": cast("AgentProtocol", runtime_bar),
        },
        fallback_agent_name="foo",
    )

    assert set(result.loaded_agents) == {"foo", "bar"}
    assert result.restored_prompts == {
        "foo": "Stored foo prompt",
        "bar": "Stored bar prompt",
    }
    assert result.active_agent == "bar"
    assert result.warnings == []
    assert runtime_foo.instruction == "Stored foo prompt"
    assert runtime_bar.instruction == "Stored bar prompt"
    assert _message_texts(runtime_foo) == ["hello foo", "done foo"]
    assert _message_texts(runtime_bar) == ["hello bar", "done bar"]


@pytest.mark.asyncio
async def test_hydrate_session_warns_for_missing_agent_and_history_file(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="bar",
            agents={
                "foo": SessionAgentSnapshot(history_file="missing.json"),
                "bar": SessionAgentSnapshot(history_file="history_bar.json"),
            },
        ),
    )
    _write_snapshot(session, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _Agent(name="foo", instruction="runtime foo")
    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"foo": cast("AgentProtocol", runtime_foo)},
        fallback_agent_name="foo",
    )

    assert result.loaded_agents == {}
    assert result.skipped_agents == ["bar"]
    assert result.missing_history_files == ["missing.json"]
    assert result.active_agent == "foo"
    assert {warning.code for warning in result.warnings} == {
        "missing-active-agent",
        "missing-agent",
        "missing-history-file",
    }


@pytest.mark.asyncio
async def test_hydrate_session_does_not_report_prompt_when_restore_fails(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            agents={"foo": SessionAgentSnapshot(resolved_prompt="Stored foo prompt")},
        ),
    )
    _write_snapshot(session, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _FailingPromptAgent(name="foo", instruction="Runtime foo prompt")
    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"foo": cast("AgentProtocol", runtime_foo)},
        fallback_agent_name="foo",
    )

    assert result.restored_prompts == {}
    assert runtime_foo.instruction == "Runtime foo prompt"
    assert {warning.code for warning in result.warnings} == {"prompt-restore-failed"}


@pytest.mark.asyncio
async def test_hydrate_session_falls_back_to_latest_history_for_thin_snapshot(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    saved = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "thin hello"), _message("assistant", "thin done")],
    )
    await session.save_history(cast("AgentProtocol", saved))

    thin_snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(),
    )
    _write_snapshot(session, thin_snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _Agent(name="foo", instruction="Changed foo prompt")
    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"foo": cast("AgentProtocol", runtime_foo)},
        fallback_agent_name="foo",
    )

    assert set(result.loaded_agents) == {"foo"}
    assert result.active_agent == "foo"
    assert _message_texts(runtime_foo) == ["thin hello", "thin done"]


@pytest.mark.asyncio
async def test_hydrate_session_restores_runtime_state_and_replaces_usage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    saved = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "resume hello"), _message("assistant", "resume done")],
    )
    await session.save_history(cast("AgentProtocol", saved))
    history_path = session.latest_history_path("foo")
    assert history_path is not None

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="foo",
            agents={
                "foo": SessionAgentSnapshot(
                    history_file=history_path.name,
                    resolved_prompt="Stored foo prompt",
                    model="sonnet-4",
                    provider="anthropic",
                    request_settings=SessionRequestSettingsSnapshot(
                        max_tokens=2048,
                        temperature=0.3,
                        use_history=True,
                        max_iterations=7,
                    ),
                    attachment_refs=[SessionAttachmentRef(ref="mcp_server:filesystem")],
                )
            },
        ),
    )
    _write_snapshot(session, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _Agent(
        name="foo",
        instruction="Runtime foo prompt",
        attached_servers=["already-attached"],
    )
    runtime_foo.usage_accumulator.turns.extend([_TurnRecord("stale-usage")])

    def _fake_rehydrate_usage(agent: _Agent, path):
        assert path == history_path
        assert agent.usage_accumulator.turns == []
        agent.usage_accumulator.turns.append(_TurnRecord("restored-usage"))
        return "usage restored"

    monkeypatch.setattr(
        "fast_agent.session.hydrator.rehydrate_usage_from_history",
        _fake_rehydrate_usage,
    )

    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"foo": cast("AgentProtocol", runtime_foo)},
        fallback_agent_name="foo",
    )

    params = runtime_foo.config.default_request_params
    assert params is not None
    assert result.active_agent == "foo"
    assert result.usage_notices == ["usage restored"]
    assert runtime_foo.instruction == "Stored foo prompt"
    assert runtime_foo.config.model == "anthropic.sonnet-4"
    assert runtime_foo.model_updates == ["anthropic.sonnet-4"]
    assert params.maxTokens == 2048
    assert params.temperature == 0.3
    assert params.use_history is True
    assert params.max_iterations == 7
    assert params.systemPrompt == "Stored foo prompt"
    assert runtime_foo.llm.default_request_params.systemPrompt == "Stored foo prompt"
    assert runtime_foo.attached_servers == ["already-attached", "filesystem"]
    assert runtime_foo.usage_accumulator.turns == [_TurnRecord("restored-usage")]


@pytest.mark.asyncio
async def test_hydrate_session_refresh_policy_skips_prompt_and_runtime_state(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    saved = _Agent(
        name="foo",
        instruction="Stored foo prompt",
        history=[_message("user", "refresh hello"), _message("assistant", "refresh done")],
    )
    await session.save_history(cast("AgentProtocol", saved))
    history_path = session.latest_history_path("foo")
    assert history_path is not None

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="foo",
            agents={
                "foo": SessionAgentSnapshot(
                    history_file=history_path.name,
                    resolved_prompt="Stored foo prompt",
                    model="sonnet-4",
                    provider="anthropic",
                    request_settings=SessionRequestSettingsSnapshot(
                        max_tokens=1024,
                        temperature=0.1,
                        use_history=True,
                    ),
                    attachment_refs=[SessionAttachmentRef(ref="mcp_server:filesystem")],
                )
            },
        ),
    )
    _write_snapshot(session, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_foo = _Agent(
        name="foo",
        instruction="Updated card prompt",
        attached_servers=["existing-server"],
    )
    runtime_foo.config.model = "openai.gpt-5-mini"
    refresh_params = runtime_foo.config.default_request_params
    assert refresh_params is not None
    refresh_params.maxTokens = 512

    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={"foo": cast("AgentProtocol", runtime_foo)},
        fallback_agent_name="foo",
        policy=SessionHydrationPolicy.for_refresh(),
    )

    assert result.active_agent == "foo"
    assert result.restored_prompts == {}
    assert runtime_foo.instruction == "Updated card prompt"
    assert runtime_foo.config.model == "openai.gpt-5-mini"
    refreshed_params = runtime_foo.config.default_request_params
    assert refreshed_params is not None
    assert refreshed_params.maxTokens == 512
    assert runtime_foo.attached_servers == ["existing-server"]
    assert _message_texts(runtime_foo) == ["refresh hello", "refresh done"]


@pytest.mark.asyncio
async def test_hydrate_session_restores_persisted_agent_tools(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    saved_parent = _Agent(
        name="parent",
        instruction="Stored parent prompt",
        history=[_message("user", "delegate"), _message("assistant", "done")],
    )
    await session.save_history(cast("AgentProtocol", saved_parent))
    history_path = session.latest_history_path("parent")
    assert history_path is not None

    snapshot = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            active_agent="parent",
            agents={
                "parent": SessionAgentSnapshot(
                    history_file=history_path.name,
                    resolved_prompt="Stored parent prompt",
                    attachment_refs=[SessionAttachmentRef(ref="agent_tool:child")],
                ),
                "child": SessionAgentSnapshot(resolved_prompt="Stored child prompt"),
            },
        ),
    )
    _write_snapshot(session, snapshot)

    persisted_session = manager.load_session(session.info.name)
    assert persisted_session is not None

    runtime_parent = _Agent(name="parent", instruction="Runtime parent prompt")
    runtime_child = _Agent(name="child", instruction="Runtime child prompt")

    result = await SessionHydrator().hydrate_session(
        session=persisted_session,
        agents={
            "parent": cast("AgentProtocol", runtime_parent),
            "child": cast("AgentProtocol", runtime_child),
        },
        fallback_agent_name="parent",
    )

    assert result.warnings == []
    assert runtime_parent.agent_backed_tools == {"agent__child": runtime_child}
