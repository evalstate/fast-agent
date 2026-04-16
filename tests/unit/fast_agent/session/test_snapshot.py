from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.session import (
    SESSION_SNAPSHOT_SCHEMA_VERSION,
    SessionAgentSnapshot,
    SessionAnalysisSnapshot,
    SessionAttachmentRef,
    SessionCardProvenanceRef,
    SessionContinuationSnapshot,
    SessionDiagnosticSnapshot,
    SessionLineageSnapshot,
    SessionMetadataSnapshot,
    SessionModelOverlayRef,
    SessionRequestSettingsSnapshot,
    SessionSnapshot,
    SessionTimingSummarySnapshot,
    SessionUsageSummarySnapshot,
    capture_session_snapshot,
    load_session_snapshot,
)
from fast_agent.session.identity import SessionSaveIdentity
from fast_agent.session.session_manager import SessionManager

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


class _Overlay:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path


class _ResolvedModel:
    def __init__(self, overlay: _Overlay | None = None) -> None:
        self.overlay = overlay


class _Llm:
    def __init__(
        self,
        *,
        model_name: str | None,
        provider_name: str,
        request_params: RequestParams,
        overlay_manifest_path: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = SimpleNamespace(config_name=provider_name)
        self.default_request_params = request_params
        self.resolved_model = _ResolvedModel(
            _Overlay(overlay_manifest_path) if overlay_manifest_path is not None else None
        )


class _UsageAccumulator:
    def __init__(self, summary: dict[str, object]) -> None:
        self._summary = summary

    def get_summary(self) -> dict[str, object]:
        return dict(self._summary)


class _Agent:
    def __init__(
        self,
        *,
        name: str,
        instruction: str,
        config: AgentConfig,
        llm: _Llm | None = None,
        usage_summary: dict[str, object] | None = None,
        attached_mcp_servers: list[str] | None = None,
        child_agents: dict[str, object] | None = None,
        message_history: list[PromptMessageExtended] | None = None,
    ) -> None:
        self.name = name
        self.instruction = instruction
        self.config = config
        self.llm = llm
        self.usage_accumulator = (
            _UsageAccumulator(usage_summary) if usage_summary is not None else None
        )
        self._attached_mcp_servers = list(attached_mcp_servers or [])
        self._child_agents = dict(child_agents or {})
        self.message_history = list(message_history or [])

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self._attached_mcp_servers)

    @property
    def agent_backed_tools(self) -> dict[str, object]:
        return dict(self._child_agents)


def test_legacy_session_synthesizes_into_typed_snapshot() -> None:
    payload = {
        "name": "2604141705-AbCd12",
        "created_at": "2026-04-14T17:05:00",
        "last_activity": "2026-04-14T17:09:00",
        "history_files": ["history_dev_previous.json", "history_dev.json"],
        "metadata": {
            "agent_name": "dev",
            "cwd": "/tmp/workspace",
            "forked_from": "2604141600-ZzYyXx",
            "last_history_by_agent": {"dev": "history_dev.json"},
            "first_user_preview": "hello",
            "model": "passthrough",
        },
    }

    snapshot = load_session_snapshot(payload)

    assert snapshot.schema_version == SESSION_SNAPSHOT_SCHEMA_VERSION
    assert snapshot.session_id == "2604141705-AbCd12"
    assert snapshot.continuation.active_agent == "dev"
    assert snapshot.continuation.cwd == "/tmp/workspace"
    assert snapshot.continuation.lineage.forked_from == "2604141600-ZzYyXx"
    assert snapshot.continuation.agents["dev"].history_file == "history_dev.json"
    assert snapshot.metadata.first_user_preview == "hello"
    assert snapshot.metadata.extras == {"model": "passthrough"}

    agent_snapshot = snapshot.continuation.agents["dev"]
    assert agent_snapshot.resolved_prompt is None
    assert agent_snapshot.provider is None
    assert agent_snapshot.request_settings is None
    assert agent_snapshot.card_provenance == []
    assert agent_snapshot.attachment_refs == []
    assert agent_snapshot.model_overlay_refs == []


def test_session_snapshot_v2_round_trips_unchanged() -> None:
    snapshot = SessionSnapshot(
        session_id="2604141705-AbCd12",
        created_at=datetime(2026, 4, 14, 17, 5, 0),
        last_activity=datetime(2026, 4, 14, 17, 9, 0),
        metadata=SessionMetadataSnapshot(
            title="Demo",
            first_user_preview="hello",
            pinned=True,
            extras={"model": "passthrough"},
        ),
        continuation=SessionContinuationSnapshot(
            active_agent="dev",
            cwd="/tmp/workspace",
            lineage=SessionLineageSnapshot(
                forked_from="2604141600-ZzYyXx",
                acp_session_id="acp-123",
            ),
            agents={
                "dev": SessionAgentSnapshot(
                    history_file="history_dev.json",
                    resolved_prompt="You are dev.",
                    model="passthrough",
                    provider="test",
                    request_settings=SessionRequestSettingsSnapshot(
                        max_tokens=2048,
                        temperature=0.2,
                        use_history=True,
                    ),
                    card_provenance=[SessionCardProvenanceRef(ref="cards/dev.md")],
                    attachment_refs=[SessionAttachmentRef(ref="attachments/readme.md")],
                    model_overlay_refs=[SessionModelOverlayRef(ref="overlay://fast")],
                )
            },
        ),
        analysis=SessionAnalysisSnapshot(
            usage_summary=SessionUsageSummarySnapshot(total_tokens=42),
            timing_summary=SessionTimingSummarySnapshot(duration_seconds=1.5),
            provider_diagnostics=[
                SessionDiagnosticSnapshot(message="provider ok", details={"kind": "info"})
            ],
            transport_diagnostics=[
                SessionDiagnosticSnapshot(message="transport ok", details={"kind": "info"})
            ],
        ),
    )

    payload = snapshot.model_dump(mode="json")
    reloaded = load_session_snapshot(payload)

    assert reloaded == snapshot


def test_load_session_rewrites_legacy_file_as_v2_snapshot(tmp_path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session_id = "2604141705-AbCd12"
    session_dir = manager.base_dir / session_id
    session_dir.mkdir(parents=True)
    payload = {
        "name": session_id,
        "created_at": "2026-04-14T17:05:00",
        "last_activity": "2026-04-14T17:09:00",
        "history_files": ["history_dev.json"],
        "metadata": {
            "agent_name": "dev",
            "last_history_by_agent": {"dev": "history_dev.json"},
        },
    }
    metadata_path = session_dir / "session.json"
    original_text = json.dumps(payload, indent=2)
    metadata_path.write_text(original_text, encoding="utf-8")

    session = manager.load_session(session_id)

    assert session is not None
    rewritten = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert rewritten["schema_version"] == 2
    assert rewritten["session_id"] == session_id
    assert rewritten["created_at"] == payload["created_at"]
    assert rewritten["last_activity"] != payload["last_activity"]


def test_malformed_legacy_fields_warn_but_still_synthesize() -> None:
    payload = {
        "name": "2604141705-AbCd12",
        "created_at": "not-a-timestamp",
        "metadata": {
            "agent_name": ["dev"],
            "pinned": "yes",
            "last_history_by_agent": ["broken"],
        },
    }

    with pytest.warns(UserWarning) as recorded:
        snapshot = load_session_snapshot(payload)

    messages = [str(warning.message) for warning in recorded]
    assert any("created_at" in message for message in messages)
    assert any("agent_name" in message for message in messages)
    assert any("pinned" in message for message in messages)
    assert any("last_history_by_agent" in message for message in messages)
    assert snapshot.session_id == "2604141705-AbCd12"
    assert isinstance(snapshot.created_at, datetime)
    assert isinstance(snapshot.last_activity, datetime)
    assert snapshot.continuation.active_agent is None
    assert snapshot.metadata.pinned is False


def test_capture_session_snapshot_maps_runtime_state_for_all_known_agents(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session(
        metadata={
            "forked_from": "2604141600-ZzYyXx",
            "last_history_by_agent": {
                "foo": "history_foo.json",
                "bar": "history_bar.json",
            },
        }
    )
    session.info.history_files = ["history_foo.json", "history_bar.json"]

    foo_config = AgentConfig(
        "foo",
        instruction="template foo",
        model="config-foo",
        use_history=False,
        default_request_params=RequestParams(
            maxTokens=111,
            parallel_tool_calls=False,
        ),
    )
    bar_config = AgentConfig(
        "bar",
        instruction="template bar",
        model="config-bar",
        default_request_params=RequestParams(maxTokens=222),
    )
    bar_config.source_path = tmp_path / "cards" / "bar.md"

    foo_agent = _Agent(
        name="foo",
        instruction="resolved foo prompt",
        config=foo_config,
    )
    bar_agent = _Agent(
        name="bar",
        instruction="resolved bar prompt",
        config=bar_config,
        llm=_Llm(
            model_name="runtime-bar",
            provider_name="anthropic",
            request_params=RequestParams(
                maxTokens=4096,
                temperature=0.2,
                top_p=0.9,
                top_k=5,
                min_p=0.1,
                presence_penalty=0.3,
                frequency_penalty=0.4,
                repetition_penalty=1.1,
                use_history=True,
                parallel_tool_calls=True,
                max_iterations=7,
                tool_result_mode="selectable",
                streaming_timeout=12.5,
                service_tier="flex",
            ),
            overlay_manifest_path=tmp_path / "overlays" / "bar.yaml",
        ),
        usage_summary={
            "cumulative_input_tokens": 100,
            "cumulative_output_tokens": 25,
            "cumulative_billing_tokens": 130,
        },
        attached_mcp_servers=["zeta", "alpha"],
        child_agents={"child-b": object(), "child-a": object()},
    )

    identity = SessionSaveIdentity(
        manager=manager,
        session=session,
        created=False,
        acp_session_id="acp-123",
        session_cwd=tmp_path / "workspace",
        session_store_scope="workspace",
        session_store_cwd=tmp_path,
    )

    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", bar_agent),
        agent_registry=cast(
            "dict[str, AgentProtocol]",
            {"foo": foo_agent, "bar": bar_agent},
        ),
        identity=identity,
    )

    assert snapshot.continuation.active_agent == "bar"
    assert snapshot.continuation.cwd == str((tmp_path / "workspace"))
    assert snapshot.continuation.lineage.forked_from == "2604141600-ZzYyXx"
    assert snapshot.continuation.lineage.acp_session_id == "acp-123"
    assert set(snapshot.continuation.agents) == {"foo", "bar"}

    foo_snapshot = snapshot.continuation.agents["foo"]
    assert foo_snapshot.history_file == "history_foo.json"
    assert foo_snapshot.resolved_prompt == "resolved foo prompt"
    assert foo_snapshot.model == "config-foo"
    assert foo_snapshot.provider is None
    foo_params = foo_config.default_request_params
    assert foo_params is not None
    assert foo_snapshot.request_settings == SessionRequestSettingsSnapshot(
        max_tokens=foo_params.maxTokens,
        use_history=foo_params.use_history,
        parallel_tool_calls=foo_params.parallel_tool_calls,
        max_iterations=foo_params.max_iterations,
        tool_result_mode=foo_params.tool_result_mode,
        streaming_timeout=foo_params.streaming_timeout,
    )

    bar_snapshot = snapshot.continuation.agents["bar"]
    assert bar_snapshot.history_file == "history_bar.json"
    assert bar_snapshot.resolved_prompt == "resolved bar prompt"
    assert bar_snapshot.model == "runtime-bar"
    assert bar_snapshot.provider == "anthropic"
    assert bar_snapshot.request_settings == SessionRequestSettingsSnapshot(
        max_tokens=4096,
        temperature=0.2,
        top_p=0.9,
        top_k=5,
        min_p=0.1,
        presence_penalty=0.3,
        frequency_penalty=0.4,
        repetition_penalty=1.1,
        use_history=True,
        parallel_tool_calls=True,
        max_iterations=7,
        tool_result_mode="selectable",
        streaming_timeout=12.5,
        service_tier="flex",
    )
    assert bar_snapshot.card_provenance == [
        SessionCardProvenanceRef(ref=str((tmp_path / "cards" / "bar.md").resolve()))
    ]
    assert bar_snapshot.attachment_refs == [
        SessionAttachmentRef(ref="mcp_server:alpha"),
        SessionAttachmentRef(ref="mcp_server:zeta"),
        SessionAttachmentRef(ref="agent_tool:child-a"),
        SessionAttachmentRef(ref="agent_tool:child-b"),
    ]
    assert bar_snapshot.model_overlay_refs == [
        SessionModelOverlayRef(ref=str((tmp_path / "overlays" / "bar.yaml").resolve()))
    ]
    assert snapshot.analysis.usage_summary == SessionUsageSummarySnapshot(
        input_tokens=100,
        output_tokens=25,
        total_tokens=130,
    )


def test_capture_session_snapshot_preserves_existing_v2_fallback_values(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    persisted = SessionSnapshot(
        session_id=session.info.name,
        created_at=session.info.created_at,
        last_activity=session.info.last_activity,
        continuation=SessionContinuationSnapshot(
            cwd="/persisted/cwd",
            lineage=SessionLineageSnapshot(acp_session_id="persisted-acp"),
            agents={
                "foo": SessionAgentSnapshot(
                    history_file="history_foo.json",
                    model="persisted-model",
                    provider="persisted-provider",
                ),
                "bar": SessionAgentSnapshot(
                    history_file="history_bar.json",
                    resolved_prompt="persisted bar prompt",
                ),
            },
        ),
    )
    (session.directory / "session.json").write_text(
        json.dumps(persisted.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    foo_config = AgentConfig("foo", instruction="template foo", model=None)
    foo_agent = _Agent(
        name="foo",
        instruction="resolved foo prompt",
        config=foo_config,
    )
    identity = SessionSaveIdentity(
        manager=manager,
        session=session,
        created=False,
        acp_session_id=None,
        session_cwd=None,
        session_store_scope="workspace",
        session_store_cwd=tmp_path,
    )

    snapshot = capture_session_snapshot(
        session=session,
        active_agent=cast("AgentProtocol", foo_agent),
        agent_registry=None,
        identity=identity,
    )

    foo_snapshot = snapshot.continuation.agents["foo"]
    bar_snapshot = snapshot.continuation.agents["bar"]
    assert snapshot.continuation.cwd == "/persisted/cwd"
    assert snapshot.continuation.lineage.acp_session_id == "persisted-acp"
    assert foo_snapshot.history_file == "history_foo.json"
    assert foo_snapshot.model == "persisted-model"
    assert foo_snapshot.provider == "persisted-provider"
    assert bar_snapshot.history_file == "history_bar.json"
    assert bar_snapshot.resolved_prompt == "persisted bar prompt"


@pytest.mark.asyncio
async def test_save_history_writes_captured_snapshot_payload(tmp_path: Path) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()
    agent = _Agent(
        name="main",
        instruction="Resolved main prompt",
        config=AgentConfig("main", instruction="Template prompt", model="passthrough"),
        llm=_Llm(
            model_name="passthrough",
            provider_name="fast-agent",
            request_params=RequestParams(maxTokens=123, temperature=0.4),
        ),
        usage_summary={
            "cumulative_input_tokens": 11,
            "cumulative_output_tokens": 7,
            "cumulative_billing_tokens": 18,
        },
        message_history=[
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello save path")],
            ),
            PromptMessageExtended(
                role="assistant",
                content=[TextContent(type="text", text="saved")],
            ),
        ],
    )

    await session.save_history(cast("AgentProtocol", agent))

    payload = json.loads((session.directory / "session.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["continuation"]["active_agent"] == "main"
    assert payload["metadata"]["first_user_preview"] == "hello save path"

    agent_payload = payload["continuation"]["agents"]["main"]
    assert agent_payload["history_file"] == "history_main.json"
    assert agent_payload["resolved_prompt"] == "Resolved main prompt"
    assert agent_payload["model"] == "passthrough"
    assert agent_payload["provider"] == "fast-agent"
    assert agent_payload["request_settings"]["max_tokens"] == 123
    assert agent_payload["request_settings"]["temperature"] == 0.4
    assert payload["analysis"]["usage_summary"] == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }


@pytest.mark.asyncio
async def test_save_history_tracks_most_recent_active_agent_across_known_agents(
    tmp_path: Path,
) -> None:
    manager = SessionManager(
        cwd=tmp_path,
        environment_override=tmp_path / ".fast-agent",
        respect_env_override=False,
    )
    session = manager.create_session()

    foo_agent = _Agent(
        name="foo",
        instruction="Resolved foo prompt",
        config=AgentConfig("foo", instruction="Template foo", model="passthrough"),
        llm=_Llm(
            model_name="passthrough",
            provider_name="fast-agent",
            request_params=RequestParams(maxTokens=100),
        ),
        message_history=[
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello from foo")],
            ),
        ],
    )
    bar_agent = _Agent(
        name="bar",
        instruction="Resolved bar prompt",
        config=AgentConfig("bar", instruction="Template bar", model="passthrough"),
        llm=_Llm(
            model_name="passthrough",
            provider_name="fast-agent",
            request_params=RequestParams(maxTokens=200),
        ),
        message_history=[
            PromptMessageExtended(
                role="user",
                content=[TextContent(type="text", text="hello from bar")],
            ),
        ],
    )
    registry = cast("dict[str, AgentProtocol]", {"foo": foo_agent, "bar": bar_agent})

    await session.save_history(
        cast("AgentProtocol", foo_agent),
        agent_registry=registry,
    )
    await session.save_history(
        cast("AgentProtocol", bar_agent),
        agent_registry=registry,
    )

    payload = json.loads((session.directory / "session.json").read_text(encoding="utf-8"))
    continuation = payload["continuation"]
    assert continuation["active_agent"] == "bar"

    agents_payload = continuation["agents"]
    assert set(agents_payload) == {"foo", "bar"}
    assert agents_payload["foo"]["history_file"] == "history_foo.json"
    assert agents_payload["bar"]["history_file"] == "history_bar.json"
    assert agents_payload["foo"]["resolved_prompt"] == "Resolved foo prompt"
    assert agents_payload["bar"]["resolved_prompt"] == "Resolved bar prompt"
