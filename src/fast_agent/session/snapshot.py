"""Typed persisted session snapshot models and compatibility helpers."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.request_params import RequestParams
    from fast_agent.session.identity import SessionSaveIdentity
    from fast_agent.session.session_manager import Session, SessionInfo

SESSION_SNAPSHOT_SCHEMA_VERSION = 2

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_LEGACY_METADATA_KEYS = frozenset(
    {
        "acp_session_id",
        "agent_name",
        "cwd",
        "first_user_preview",
        "forked_from",
        "label",
        "last_history_by_agent",
        "pinned",
        "title",
    }
)


class SessionMetadataSnapshot(BaseModel):
    """Display/indexing metadata persisted with a session."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    label: str | None = None
    first_user_preview: str | None = None
    pinned: bool = False
    extras: dict[str, JsonValue] = Field(default_factory=dict)


class SessionLineageSnapshot(BaseModel):
    """Lineage metadata for forked or ACP-originated sessions."""

    model_config = ConfigDict(extra="forbid")

    forked_from: str | None = None
    acp_session_id: str | None = None


class SessionRequestSettingsSnapshot(BaseModel):
    """Persisted request-setting subset relevant to future turns."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    use_history: bool | None = None
    parallel_tool_calls: bool | None = None
    max_iterations: int | None = None
    tool_result_mode: Literal["postprocess", "passthrough", "selectable"] | None = None
    streaming_timeout: float | None = None
    service_tier: Literal["fast", "flex"] | None = None


class SessionCardProvenanceRef(BaseModel):
    """Reference to card provenance that affected a persisted session."""

    model_config = ConfigDict(extra="forbid")

    ref: str


class SessionAttachmentRef(BaseModel):
    """Reference to an attachment or attached resource."""

    model_config = ConfigDict(extra="forbid")

    ref: str


class SessionModelOverlayRef(BaseModel):
    """Reference to a model overlay affecting future-turn semantics."""

    model_config = ConfigDict(extra="forbid")

    ref: str


class SessionAgentSnapshot(BaseModel):
    """Persisted continuation state for a single agent."""

    model_config = ConfigDict(extra="forbid")

    history_file: str | None = None
    resolved_prompt: str | None = None
    model: str | None = None
    provider: str | None = None
    request_settings: SessionRequestSettingsSnapshot | None = None
    card_provenance: list[SessionCardProvenanceRef] = Field(default_factory=list)
    attachment_refs: list[SessionAttachmentRef] = Field(default_factory=list)
    model_overlay_refs: list[SessionModelOverlayRef] = Field(default_factory=list)


class SessionContinuationSnapshot(BaseModel):
    """Persisted state that defines the next turn."""

    model_config = ConfigDict(extra="forbid")

    active_agent: str | None = None
    cwd: str | None = None
    lineage: SessionLineageSnapshot = Field(default_factory=SessionLineageSnapshot)
    agents: dict[str, SessionAgentSnapshot] = Field(default_factory=dict)


class SessionUsageSummarySnapshot(BaseModel):
    """Small persisted usage summary for inspection."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class SessionTimingSummarySnapshot(BaseModel):
    """Small persisted timing summary for inspection."""

    model_config = ConfigDict(extra="forbid")

    duration_seconds: float | None = None


class SessionDiagnosticSnapshot(BaseModel):
    """Structured diagnostic entry kept for inspection only."""

    model_config = ConfigDict(extra="forbid")

    message: str
    details: dict[str, JsonValue] = Field(default_factory=dict)


class SessionAnalysisSnapshot(BaseModel):
    """Persisted analysis and diagnostics that are not live runtime truth."""

    model_config = ConfigDict(extra="forbid")

    usage_summary: SessionUsageSummarySnapshot | None = None
    timing_summary: SessionTimingSummarySnapshot | None = None
    provider_diagnostics: list[SessionDiagnosticSnapshot] = Field(default_factory=list)
    transport_diagnostics: list[SessionDiagnosticSnapshot] = Field(default_factory=list)


class SessionSnapshot(BaseModel):
    """Versioned persisted session snapshot.

    Phase 1 keeps SessionInfo as the public compatibility view while this model
    becomes the typed owner of ``session.json`` parsing and serialization.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = SESSION_SNAPSHOT_SCHEMA_VERSION
    session_id: str
    created_at: datetime
    last_activity: datetime
    metadata: SessionMetadataSnapshot = Field(default_factory=SessionMetadataSnapshot)
    continuation: SessionContinuationSnapshot = Field(default_factory=SessionContinuationSnapshot)
    analysis: SessionAnalysisSnapshot = Field(default_factory=SessionAnalysisSnapshot)


@runtime_checkable
class _AttachedMcpServerProvider(Protocol):
    def list_attached_mcp_servers(self) -> list[str]: ...


@runtime_checkable
class _AgentBackedToolRefProvider(Protocol):
    @property
    def agent_backed_tools(self) -> Mapping[str, object]: ...


def load_session_snapshot(payload: object) -> SessionSnapshot:
    """Load either a legacy v1 payload or an explicit v2 session snapshot."""
    payload_mapping = _as_object_mapping(payload)
    if payload_mapping is None:
        raise ValueError("Session snapshot payload must be a JSON object")

    raw_schema_version = payload_mapping.get("schema_version")
    if raw_schema_version is None:
        return synthesize_legacy_session_snapshot(payload_mapping)
    if raw_schema_version != SESSION_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported session snapshot schema version: {raw_schema_version!r}")
    return SessionSnapshot.model_validate(payload_mapping)


def synthesize_legacy_session_snapshot(payload: Mapping[str, object]) -> SessionSnapshot:
    """Synthesize a typed v2 snapshot from a legacy unversioned v1 payload."""
    session_id = _legacy_session_id(payload)
    now = datetime.now()

    raw_metadata = payload.get("metadata")
    metadata_mapping = _as_object_mapping(raw_metadata)
    if raw_metadata is not None and metadata_mapping is None:
        _warn_legacy_issue(
            session_id,
            "legacy session metadata must be an object; ignoring malformed metadata",
        )

    metadata = SessionMetadataSnapshot(
        title=_optional_str(metadata_mapping, "title", session_id),
        label=_optional_str(metadata_mapping, "label", session_id),
        first_user_preview=_optional_str(metadata_mapping, "first_user_preview", session_id),
        pinned=_optional_bool(metadata_mapping, "pinned", session_id, default=False),
        extras=_legacy_metadata_extras(metadata_mapping, session_id),
    )

    continuation = SessionContinuationSnapshot(
        active_agent=_optional_str(metadata_mapping, "agent_name", session_id),
        cwd=_optional_str(metadata_mapping, "cwd", session_id),
        lineage=SessionLineageSnapshot(
            forked_from=_optional_str(metadata_mapping, "forked_from", session_id),
            acp_session_id=_optional_str(metadata_mapping, "acp_session_id", session_id),
        ),
        agents=_legacy_agents(payload, metadata_mapping, session_id),
    )

    return SessionSnapshot(
        session_id=session_id,
        created_at=_legacy_timestamp(payload.get("created_at"), "created_at", session_id, now),
        last_activity=_legacy_timestamp(
            payload.get("last_activity"), "last_activity", session_id, now
        ),
        metadata=metadata,
        continuation=continuation,
    )


def snapshot_from_session_info(info: "SessionInfo") -> SessionSnapshot:
    """Build a typed snapshot from the compatibility-facing SessionInfo view."""
    metadata_dict = info.metadata if isinstance(info.metadata, dict) else {}
    metadata = SessionMetadataSnapshot(
        title=_typed_str(metadata_dict.get("title")),
        label=_typed_str(metadata_dict.get("label")),
        first_user_preview=_typed_str(metadata_dict.get("first_user_preview")),
        pinned=metadata_dict.get("pinned") is True,
        extras=_session_info_metadata_extras(metadata_dict, info.name),
    )

    agents = _agents_from_session_info(info)
    continuation = SessionContinuationSnapshot(
        active_agent=_typed_str(metadata_dict.get("agent_name")),
        cwd=_typed_str(metadata_dict.get("cwd")),
        lineage=SessionLineageSnapshot(
            forked_from=_typed_str(metadata_dict.get("forked_from")),
            acp_session_id=_typed_str(metadata_dict.get("acp_session_id")),
        ),
        agents=agents,
    )

    return SessionSnapshot(
        session_id=info.name,
        created_at=info.created_at,
        last_activity=info.last_activity,
        metadata=metadata,
        continuation=continuation,
        analysis=SessionAnalysisSnapshot(),
    )


def capture_session_snapshot(
    *,
    session: "Session",
    active_agent: "AgentProtocol",
    agent_registry: Mapping[str, "AgentProtocol"] | None,
    identity: "SessionSaveIdentity",
) -> SessionSnapshot:
    """Capture the authoritative persisted snapshot for the current runtime state."""
    snapshot = snapshot_from_session_info(session.info)
    existing_snapshot = _load_existing_session_snapshot(session)

    snapshot.continuation.active_agent = active_agent.name
    snapshot.continuation.cwd = _capture_continuation_cwd(
        compatibility_snapshot=snapshot,
        existing_snapshot=existing_snapshot,
        identity=identity,
    )
    snapshot.continuation.lineage = _capture_lineage_snapshot(
        compatibility_snapshot=snapshot,
        existing_snapshot=existing_snapshot,
        identity=identity,
    )
    snapshot.continuation.agents = _capture_agent_snapshots(
        session=session,
        active_agent=active_agent,
        agent_registry=agent_registry,
        compatibility_snapshot=snapshot,
        existing_snapshot=existing_snapshot,
    )
    snapshot.analysis = SessionAnalysisSnapshot(
        usage_summary=_capture_usage_summary(active_agent),
    )
    return snapshot


def session_info_from_snapshot(snapshot: SessionSnapshot) -> "SessionInfo":
    """Project a typed snapshot into the existing SessionInfo compatibility view."""
    from fast_agent.session.session_manager import SessionInfo

    metadata: dict[str, JsonValue | str | bool] = dict(snapshot.metadata.extras)
    if snapshot.metadata.title is not None:
        metadata["title"] = snapshot.metadata.title
    if snapshot.metadata.label is not None:
        metadata["label"] = snapshot.metadata.label
    if snapshot.metadata.first_user_preview is not None:
        metadata["first_user_preview"] = snapshot.metadata.first_user_preview
    if snapshot.metadata.pinned:
        metadata["pinned"] = True
    if snapshot.continuation.active_agent is not None:
        metadata["agent_name"] = snapshot.continuation.active_agent
    if snapshot.continuation.cwd is not None:
        metadata["cwd"] = snapshot.continuation.cwd
    if snapshot.continuation.lineage.forked_from is not None:
        metadata["forked_from"] = snapshot.continuation.lineage.forked_from
    if snapshot.continuation.lineage.acp_session_id is not None:
        metadata["acp_session_id"] = snapshot.continuation.lineage.acp_session_id

    history_files: list[str] = []
    history_map: dict[str, str] = {}
    for agent_name, agent_snapshot in snapshot.continuation.agents.items():
        history_file = agent_snapshot.history_file
        if history_file is None:
            continue
        history_map[agent_name] = history_file
        if history_file not in history_files:
            history_files.append(history_file)

    if history_map:
        metadata["last_history_by_agent"] = history_map

    return SessionInfo(
        name=snapshot.session_id,
        created_at=snapshot.created_at,
        last_activity=snapshot.last_activity,
        history_files=history_files,
        metadata=metadata,
    )


def clone_session_snapshot_for_fork(
    snapshot: SessionSnapshot,
    *,
    new_session_id: str,
    copied_history_files: Mapping[str, str],
    cloned_at: datetime,
    title: str | None = None,
) -> SessionSnapshot:
    """Clone a persisted snapshot for a forked local session."""
    cloned = snapshot.model_copy(deep=True)
    cloned.session_id = new_session_id
    cloned.created_at = cloned_at
    cloned.last_activity = cloned_at

    if title is not None:
        cloned.metadata.title = title

    cloned.continuation.lineage = SessionLineageSnapshot(
        forked_from=snapshot.session_id,
        acp_session_id=None,
    )

    for agent_snapshot in cloned.continuation.agents.values():
        history_file = agent_snapshot.history_file
        if history_file is None:
            continue
        agent_snapshot.history_file = copied_history_files.get(history_file)

    return cloned


def _legacy_session_id(payload: Mapping[str, object]) -> str:
    for key in ("name", "session_id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError("Legacy session payload is missing a usable session id")


def _legacy_timestamp(
    value: object,
    field_name: str,
    session_id: str,
    fallback: datetime,
) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            _warn_legacy_issue(
                session_id,
                f"legacy session {field_name!r} is not a valid ISO timestamp; using current time",
            )
            return fallback
    if value is not None:
        _warn_legacy_issue(
            session_id,
            f"legacy session {field_name!r} must be a string timestamp; using current time",
        )
    return fallback


def _legacy_metadata_extras(
    metadata: Mapping[str, object] | None,
    session_id: str,
) -> dict[str, JsonValue]:
    if metadata is None:
        return {}

    extras: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if key in _LEGACY_METADATA_KEYS:
            continue
        keep, normalized = _normalize_json_value(value, session_id, f"metadata.{key}")
        if keep:
            extras[key] = normalized
    return extras


def _legacy_agents(
    payload: Mapping[str, object],
    metadata: Mapping[str, object] | None,
    session_id: str,
) -> dict[str, SessionAgentSnapshot]:
    agents: dict[str, SessionAgentSnapshot] = {}

    raw_history_map = metadata.get("last_history_by_agent") if metadata is not None else None
    history_mapping = _as_object_mapping(raw_history_map)
    if history_mapping is not None:
        for agent_name, history_file in history_mapping.items():
            if not isinstance(history_file, str):
                _warn_legacy_issue(
                    session_id,
                    f"legacy last_history_by_agent[{agent_name!r}] must be a string; skipping entry",
                )
                continue
            agents[agent_name] = SessionAgentSnapshot(history_file=history_file)
        return agents

    if raw_history_map is not None:
        _warn_legacy_issue(
            session_id,
            "legacy last_history_by_agent must be an object; falling back to history_files",
        )

    raw_history_files = payload.get("history_files")
    if not isinstance(raw_history_files, list):
        if raw_history_files is not None:
            _warn_legacy_issue(
                session_id,
                "legacy history_files must be a list when present; ignoring malformed history_files",
            )
        return agents

    for history_file in raw_history_files:
        if not isinstance(history_file, str):
            _warn_legacy_issue(
                session_id,
                "legacy history_files contains a non-string entry; skipping entry",
            )
            continue
        if history_file.endswith("_previous.json"):
            continue
        agent_name = _history_agent_name(history_file)
        agents[agent_name] = SessionAgentSnapshot(history_file=history_file)

    return agents


def _optional_str(
    mapping: Mapping[str, object] | None,
    key: str,
    session_id: str,
) -> str | None:
    if mapping is None or key not in mapping:
        return None
    value = mapping[key]
    if value is None:
        return None
    if isinstance(value, str):
        return value
    _warn_legacy_issue(session_id, f"legacy metadata.{key} must be a string; ignoring value")
    return None


def _optional_bool(
    mapping: Mapping[str, object] | None,
    key: str,
    session_id: str,
    *,
    default: bool,
) -> bool:
    if mapping is None or key not in mapping:
        return default
    value = mapping[key]
    if isinstance(value, bool):
        return value
    if value is not None:
        _warn_legacy_issue(session_id, f"legacy metadata.{key} must be a boolean; using default")
    return default


def _session_info_metadata_extras(
    metadata: Mapping[str, object],
    session_id: str,
) -> dict[str, JsonValue]:
    extras: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if key in _LEGACY_METADATA_KEYS:
            continue
        keep, normalized = _normalize_json_value(value, session_id, f"metadata.{key}")
        if keep:
            extras[key] = normalized
    return extras


def _agents_from_session_info(info: "SessionInfo") -> dict[str, SessionAgentSnapshot]:
    metadata = info.metadata if isinstance(info.metadata, dict) else {}
    agents: dict[str, SessionAgentSnapshot] = {}

    history_map = metadata.get("last_history_by_agent")
    if isinstance(history_map, Mapping):
        for agent_name, history_file in history_map.items():
            if isinstance(agent_name, str) and isinstance(history_file, str):
                agents[agent_name] = SessionAgentSnapshot(history_file=history_file)

    for history_file in info.history_files:
        if history_file.endswith("_previous.json"):
            continue
        agent_name = _history_agent_name(history_file)
        agents[agent_name] = SessionAgentSnapshot(history_file=history_file)

    return agents


def _history_agent_name(filename: str) -> str:
    path = Path(filename)
    stem = path.stem
    if stem.startswith("history_"):
        stem = stem[len("history_") :]
    if stem.endswith("_previous"):
        stem = stem[: -len("_previous")]
    return stem or "agent"


def _typed_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _capture_continuation_cwd(
    *,
    compatibility_snapshot: SessionSnapshot,
    existing_snapshot: SessionSnapshot | None,
    identity: "SessionSaveIdentity",
) -> str | None:
    if identity.session_cwd is not None:
        return str(identity.session_cwd)
    if compatibility_snapshot.continuation.cwd is not None:
        return compatibility_snapshot.continuation.cwd
    if existing_snapshot is not None:
        return existing_snapshot.continuation.cwd
    return None


def _capture_lineage_snapshot(
    *,
    compatibility_snapshot: SessionSnapshot,
    existing_snapshot: SessionSnapshot | None,
    identity: "SessionSaveIdentity",
) -> SessionLineageSnapshot:
    compatibility_lineage = compatibility_snapshot.continuation.lineage
    existing_lineage = (
        existing_snapshot.continuation.lineage
        if existing_snapshot is not None
        else SessionLineageSnapshot()
    )
    return SessionLineageSnapshot(
        forked_from=compatibility_lineage.forked_from or existing_lineage.forked_from,
        acp_session_id=identity.acp_session_id or existing_lineage.acp_session_id,
    )


def _capture_agent_snapshots(
    *,
    session: "Session",
    active_agent: "AgentProtocol",
    agent_registry: Mapping[str, "AgentProtocol"] | None,
    compatibility_snapshot: SessionSnapshot,
    existing_snapshot: SessionSnapshot | None,
) -> dict[str, SessionAgentSnapshot]:
    agents = dict(agent_registry or {})
    agents[active_agent.name] = active_agent

    compatibility_agents = compatibility_snapshot.continuation.agents
    existing_agents = existing_snapshot.continuation.agents if existing_snapshot is not None else {}

    snapshots = dict(existing_agents)
    for agent_name, agent_snapshot in compatibility_agents.items():
        snapshots.setdefault(agent_name, agent_snapshot)

    for agent_name, agent in agents.items():
        snapshots[agent_name] = _capture_agent_snapshot(
            session=session,
            agent=agent,
            compatibility_snapshot=compatibility_agents.get(agent_name),
            existing_snapshot=existing_agents.get(agent_name),
        )
    return snapshots


def _capture_agent_snapshot(
    *,
    session: "Session",
    agent: "AgentProtocol",
    compatibility_snapshot: SessionAgentSnapshot | None,
    existing_snapshot: SessionAgentSnapshot | None,
) -> SessionAgentSnapshot:
    llm = agent.llm
    request_settings = _capture_request_settings_snapshot(agent)
    return SessionAgentSnapshot(
        history_file=_capture_history_file(
            session=session,
            agent_name=agent.name,
            compatibility_snapshot=compatibility_snapshot,
            existing_snapshot=existing_snapshot,
        ),
        resolved_prompt=agent.instruction,
        model=(
            llm.model_name
            if llm is not None and llm.model_name is not None
            else (
                agent.config.model
                if agent.config.model is not None
                else (existing_snapshot.model if existing_snapshot is not None else None)
            )
        ),
        provider=(
            llm.provider.config_name
            if llm is not None
            else (existing_snapshot.provider if existing_snapshot is not None else None)
        ),
        request_settings=(
            request_settings
            if request_settings is not None
            else (existing_snapshot.request_settings if existing_snapshot is not None else None)
        ),
        card_provenance=_capture_card_provenance(agent),
        attachment_refs=_capture_attachment_refs(agent),
        model_overlay_refs=_capture_model_overlay_refs(agent),
    )


def _capture_history_file(
    *,
    session: "Session",
    agent_name: str,
    compatibility_snapshot: SessionAgentSnapshot | None,
    existing_snapshot: SessionAgentSnapshot | None,
) -> str | None:
    metadata = session.info.metadata
    history_map = metadata.get("last_history_by_agent") if isinstance(metadata, dict) else None
    if isinstance(history_map, Mapping):
        history_file = history_map.get(agent_name)
        if isinstance(history_file, str):
            return history_file
    if compatibility_snapshot is not None and compatibility_snapshot.history_file is not None:
        return compatibility_snapshot.history_file
    if existing_snapshot is not None:
        return existing_snapshot.history_file
    return None


def _capture_request_settings_snapshot(
    agent: "AgentProtocol",
) -> SessionRequestSettingsSnapshot | None:
    llm = agent.llm
    if llm is not None:
        return _request_settings_snapshot_from_params(llm.default_request_params)
    return _request_settings_snapshot_from_params(agent.config.default_request_params)


def _request_settings_snapshot_from_params(
    params: "RequestParams | None",
) -> SessionRequestSettingsSnapshot | None:
    if params is None:
        return None

    snapshot = SessionRequestSettingsSnapshot(
        max_tokens=params.maxTokens,
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
        presence_penalty=params.presence_penalty,
        frequency_penalty=params.frequency_penalty,
        repetition_penalty=params.repetition_penalty,
        use_history=params.use_history,
        parallel_tool_calls=params.parallel_tool_calls,
        max_iterations=params.max_iterations,
        tool_result_mode=params.tool_result_mode,
        streaming_timeout=params.streaming_timeout,
        service_tier=params.service_tier,
    )
    return snapshot if snapshot.model_dump(exclude_none=True) else None


def _capture_card_provenance(agent: "AgentProtocol") -> list[SessionCardProvenanceRef]:
    source_path = agent.config.source_path
    if source_path is None:
        return []
    return [SessionCardProvenanceRef(ref=str(source_path.expanduser().resolve()))]


def _capture_attachment_refs(agent: "AgentProtocol") -> list[SessionAttachmentRef]:
    refs: list[SessionAttachmentRef] = []
    if isinstance(agent, _AttachedMcpServerProvider):
        refs.extend(
            SessionAttachmentRef(ref=f"mcp_server:{server_name}")
            for server_name in sorted(agent.list_attached_mcp_servers())
        )
    if isinstance(agent, _AgentBackedToolRefProvider):
        refs.extend(
            SessionAttachmentRef(ref=f"agent_tool:{child_name}")
            for child_name in sorted(agent.agent_backed_tools)
        )
    return refs


def _capture_model_overlay_refs(agent: "AgentProtocol") -> list[SessionModelOverlayRef]:
    llm = agent.llm
    if llm is None:
        return []
    overlay = llm.resolved_model.overlay
    if overlay is None:
        return []
    return [SessionModelOverlayRef(ref=str(overlay.manifest_path.expanduser().resolve()))]


def _capture_usage_summary(agent: "AgentProtocol") -> SessionUsageSummarySnapshot | None:
    usage_accumulator = agent.usage_accumulator
    if usage_accumulator is None:
        return None

    summary = usage_accumulator.get_summary()
    input_tokens = _summary_int(summary.get("cumulative_input_tokens"))
    output_tokens = _summary_int(summary.get("cumulative_output_tokens"))
    total_tokens = _summary_int(summary.get("cumulative_billing_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    snapshot = SessionUsageSummarySnapshot(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
    return snapshot if snapshot.model_dump(exclude_none=True) else None


def _summary_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None


def _load_existing_session_snapshot(session: "Session") -> SessionSnapshot | None:
    metadata_file = session.directory / "session.json"
    if not metadata_file.exists():
        return None
    try:
        with open(metadata_file, encoding="utf-8") as handle:
            return load_session_snapshot(json.load(handle))
    except Exception:
        return None


def _normalize_json_value(
    value: object,
    session_id: str,
    field_name: str,
) -> tuple[bool, JsonValue]:
    if value is None or isinstance(value, bool | int | float | str):
        return True, value
    if isinstance(value, list):
        normalized_list: list[JsonValue] = []
        for index, item in enumerate(value):
            keep, normalized_item = _normalize_json_value(
                item, session_id, f"{field_name}[{index}]"
            )
            if not keep:
                return False, None
            normalized_list.append(normalized_item)
        return True, normalized_list
    mapping = _as_object_mapping(value)
    if mapping is not None:
        normalized_dict: dict[str, JsonValue] = {}
        for key, item in mapping.items():
            keep, normalized_item = _normalize_json_value(item, session_id, f"{field_name}.{key}")
            if not keep:
                return False, None
            normalized_dict[key] = normalized_item
        return True, normalized_dict

    _warn_legacy_issue(
        session_id,
        f"{field_name} is not JSON-compatible and will be dropped from session extras",
    )
    return False, None


def _as_object_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None

    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        result[key] = item
    return result


def _warn_legacy_issue(session_id: str, message: str) -> None:
    warnings.warn(f"Session {session_id}: {message}", UserWarning, stacklevel=3)
