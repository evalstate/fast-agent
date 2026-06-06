from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompts.prompt_load import (
    load_transcript_into_agent,
    rehydrate_usage_from_history,
)

from .snapshot import (
    SessionAgentSnapshot,
    SessionAttachmentRef,
    SessionRequestSettingsSnapshot,
    SessionSnapshot,
    load_session_snapshot,
    snapshot_from_session_info,
)

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol

    from .session_manager import Session


@dataclass(slots=True, frozen=True)
class SessionHydrationPolicy:
    restore_transcript: bool = True
    restore_usage: bool = True
    restore_prompt: bool = True
    restore_runtime_state: bool = True

    @classmethod
    def for_refresh(cls) -> 'SessionHydrationPolicy':
        return cls(
            restore_transcript=True,
            restore_usage=True,
            restore_prompt=False,
            restore_runtime_state=False,
        )


@dataclass(slots=True)
class SessionHydrationWarning:
    code: str
    message: str
    agent_name: str | None = None
    ref: str | None = None


@dataclass(slots=True)
class SessionHydrationResult:
    session: Session
    snapshot: SessionSnapshot
    loaded_agents: dict[str, Path]
    restored_prompts: dict[str, str]
    skipped_agents: list[str]
    missing_history_files: list[str]
    warnings: list[SessionHydrationWarning] = field(default_factory=list)
    usage_notices: list[str] = field(default_factory=list)
    active_agent: str | None = None


@dataclass(slots=True)
class _HydrationState:
    loaded_agents: dict[str, Path] = field(default_factory=dict)
    restored_prompts: dict[str, str] = field(default_factory=dict)
    skipped_agents: list[str] = field(default_factory=list)
    missing_history_files: list[str] = field(default_factory=list)
    warnings: list[SessionHydrationWarning] = field(default_factory=list)
    usage_notices: list[str] = field(default_factory=list)


@runtime_checkable
class _McpAttachCapable(Protocol):
    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: object | None = None,
        options: object | None = None,
    ) -> object: ...

    def list_attached_mcp_servers(self) -> list[str]: ...


@runtime_checkable
class _AgentToolAttachCapable(Protocol):
    def add_agent_tool(
        self,
        child: object,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> str: ...


@runtime_checkable
class _AgentToolProvider(Protocol):
    @property
    def agent_backed_tools(self) -> Mapping[str, object]: ...


@runtime_checkable
class _NamedAgent(Protocol):
    @property
    def name(self) -> str: ...


class SessionHydrator:
    async def hydrate_session(
        self,
        *,
        session: Session,
        agents: Mapping[str, AgentProtocol],
        fallback_agent_name: str | None,
        policy: SessionHydrationPolicy | None = None,
    ) -> SessionHydrationResult:
        warnings: list[SessionHydrationWarning] = []
        snapshot = self._load_snapshot(session=session, warnings=warnings)
        agent_snapshots = self._select_agent_snapshots(
            session=session,
            snapshot=snapshot,
            agents=agents,
            fallback_agent_name=fallback_agent_name,
        )
        effective_policy = policy or SessionHydrationPolicy()
        state = _HydrationState(warnings=warnings)

        for agent_name, agent_snapshot in agent_snapshots.items():
            await self._hydrate_agent(
                session=session,
                agent_name=agent_name,
                agent_snapshot=agent_snapshot,
                agents=agents,
                policy=effective_policy,
                state=state,
            )

        active_agent = self._resolve_active_agent(
            snapshot=snapshot,
            agents=agents,
            loaded_agents=state.loaded_agents,
            fallback_agent_name=fallback_agent_name,
            warnings=state.warnings,
        )
        return SessionHydrationResult(
            session=session,
            snapshot=snapshot,
            loaded_agents=state.loaded_agents,
            restored_prompts=state.restored_prompts,
            skipped_agents=state.skipped_agents,
            missing_history_files=state.missing_history_files,
            warnings=state.warnings,
            usage_notices=state.usage_notices,
            active_agent=active_agent,
        )

    async def _hydrate_agent(
        self,
        *,
        session: Session,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        agents: Mapping[str, AgentProtocol],
        policy: SessionHydrationPolicy,
        state: _HydrationState,
    ) -> None:
        agent = agents.get(agent_name)
        if agent is None:
            self._record_missing_agent(agent_name, state)
            return

        if policy.restore_runtime_state:
            await self._restore_runtime_state(
                agent=agent,
                agent_name=agent_name,
                agent_snapshot=agent_snapshot,
                agents=agents,
                warnings=state.warnings,
            )
        if policy.restore_transcript:
            self._restore_agent_transcript(session, agent_name, agent, agent_snapshot, policy, state)
        if policy.restore_prompt:
            self._restore_agent_prompt(agent_name, agent, agent_snapshot, state)

    @staticmethod
    def _record_missing_agent(agent_name: str, state: _HydrationState) -> None:
        state.skipped_agents.append(agent_name)
        state.warnings.append(
            SessionHydrationWarning(
                code='missing-agent',
                message=f"Persisted agent {agent_name!r} is not available in this runtime",
                agent_name=agent_name,
            )
        )

    def _restore_agent_transcript(
        self,
        session: Session,
        agent_name: str,
        agent: AgentProtocol,
        agent_snapshot: SessionAgentSnapshot,
        policy: SessionHydrationPolicy,
        state: _HydrationState,
    ) -> None:
        history_file = agent_snapshot.history_file
        if history_file is None:
            return

        history_path = session.directory / history_file
        if not history_path.exists():
            state.missing_history_files.append(history_file)
            state.warnings.append(
                SessionHydrationWarning(
                    code='missing-history-file',
                    message=(
                        f"Persisted history file {history_file!r} is missing for agent"
                        f" {agent_name!r}"
                    ),
                    agent_name=agent_name,
                    ref=history_file,
                )
            )
            return

        try:
            notice = self._load_agent_transcript(agent, history_path, policy)
        except Exception as exc:
            state.warnings.append(
                SessionHydrationWarning(
                    code='history-load-failed',
                    message=f"Failed to restore history for agent {agent_name!r}: {exc}",
                    agent_name=agent_name,
                    ref=history_file,
                )
            )
            return

        state.loaded_agents[agent_name] = history_path
        if notice:
            state.usage_notices.append(notice)

    @staticmethod
    def _load_agent_transcript(
        agent: AgentProtocol,
        history_path: Path,
        policy: SessionHydrationPolicy,
    ) -> str | None:
        load_transcript_into_agent(agent, history_path)
        if policy.restore_usage and agent.usage_accumulator is not None:
            agent.usage_accumulator.reset()
        if policy.restore_usage:
            return rehydrate_usage_from_history(agent, history_path)
        return None

    @staticmethod
    def _restore_agent_prompt(
        agent_name: str,
        agent: AgentProtocol,
        agent_snapshot: SessionAgentSnapshot,
        state: _HydrationState,
    ) -> None:
        resolved_prompt = agent_snapshot.resolved_prompt
        if resolved_prompt is None:
            return
        try:
            agent.set_instruction(resolved_prompt)
        except Exception as exc:
            state.warnings.append(
                SessionHydrationWarning(
                    code='prompt-restore-failed',
                    message=f"Failed to restore prompt for agent {agent_name!r}: {exc}",
                    agent_name=agent_name,
                )
            )
        else:
            state.restored_prompts[agent_name] = agent.instruction

    async def _restore_runtime_state(
        self,
        *,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        agents: Mapping[str, AgentProtocol],
        warnings: list[SessionHydrationWarning],
    ) -> None:
        await self._restore_agent_model(
            agent=agent,
            agent_name=agent_name,
            agent_snapshot=agent_snapshot,
            warnings=warnings,
        )
        self._restore_agent_request_settings(agent, agent_name, agent_snapshot, warnings)
        await self._restore_mcp_attachments(agent, agent_name, agent_snapshot, warnings)
        self._restore_agent_tool_attachments(agent, agent_name, agent_snapshot, agents, warnings)

    async def _restore_agent_model(
        self,
        *,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        warnings: list[SessionHydrationWarning],
    ) -> None:
        model_spec = self._resolve_model_spec(
            agent=agent,
            agent_name=agent_name,
            agent_snapshot=agent_snapshot,
            warnings=warnings,
        )
        if model_spec is None:
            return
        try:
            model_result = agent.set_model(model_spec)
            if inspect.isawaitable(model_result):
                await model_result
        except Exception as exc:
            warnings.append(
                SessionHydrationWarning(
                    code='model-restore-failed',
                    message=f"Failed to restore model for agent {agent_name!r}: {exc}",
                    agent_name=agent_name,
                    ref=model_spec,
                )
            )

    def _restore_agent_request_settings(
        self,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        warnings: list[SessionHydrationWarning],
    ) -> None:
        request_settings = agent_snapshot.request_settings
        if request_settings is None:
            return
        try:
            self._apply_request_settings(agent, request_settings)
        except Exception as exc:
            warnings.append(
                SessionHydrationWarning(
                    code='request-settings-restore-failed',
                    message=f"Failed to restore request settings for agent {agent_name!r}: {exc}",
                    agent_name=agent_name,
                )
            )

    async def _restore_mcp_attachments(
        self,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        warnings: list[SessionHydrationWarning],
    ) -> None:
        server_names = self._persisted_attached_mcp_servers(agent_snapshot.attachment_refs)
        if not server_names or not isinstance(agent, _McpAttachCapable):
            return

        attached = set(agent.list_attached_mcp_servers())
        for server_name in server_names:
            if server_name in attached:
                continue
            try:
                await agent.attach_mcp_server(server_name=server_name)
            except Exception as exc:
                warnings.append(
                    SessionHydrationWarning(
                        code='attachment-restore-failed',
                        message=(
                            f"Failed to restore MCP attachment {server_name!r} for agent"
                            f" {agent_name!r}: {exc}"
                        ),
                        agent_name=agent_name,
                        ref=server_name,
                    )
                )
            else:
                attached.add(server_name)

    def _restore_agent_tool_attachments(
        self,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        agents: Mapping[str, AgentProtocol],
        warnings: list[SessionHydrationWarning],
    ) -> None:
        tool_names = self._persisted_attached_agent_tools(agent_snapshot.attachment_refs)
        if not tool_names or not isinstance(agent, _AgentToolAttachCapable):
            return

        attached_tool_names = self._attached_agent_tool_names(agent)
        for child_name in tool_names:
            if child_name in attached_tool_names:
                continue

            child_agent = agents.get(child_name)
            if child_agent is None:
                warnings.append(
                    SessionHydrationWarning(
                        code='attachment-restore-missing-agent',
                        message=(
                            f"Persisted agent tool {child_name!r} for agent {agent_name!r}"
                            " is not available in this runtime"
                        ),
                        agent_name=agent_name,
                        ref=child_name,
                    )
                )
                continue

            try:
                agent.add_agent_tool(child_agent)
            except Exception as exc:
                warnings.append(
                    SessionHydrationWarning(
                        code='attachment-restore-failed',
                        message=(
                            f"Failed to restore agent tool {child_name!r} for agent"
                            f" {agent_name!r}: {exc}"
                        ),
                        agent_name=agent_name,
                        ref=child_name,
                    )
                )
            else:
                attached_tool_names.add(child_name)

    def _resolve_model_spec(
        self,
        *,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        warnings: list[SessionHydrationWarning],
    ) -> str | None:
        overlay_spec = self._resolve_overlay_model_spec(
            agent=agent,
            agent_name=agent_name,
            agent_snapshot=agent_snapshot,
            warnings=warnings,
        )
        if overlay_spec is not None:
            return overlay_spec

        model_spec = agent_snapshot.model_spec
        if model_spec is not None:
            stripped_model_spec = model_spec.strip()
            if stripped_model_spec:
                return stripped_model_spec

        model_name = agent_snapshot.model
        if model_name is None:
            return None
        provider_name = agent_snapshot.provider
        if provider_name is not None and not model_name.startswith(f'{provider_name}.'):
            return f'{provider_name}.{model_name}'
        return model_name

    def _resolve_overlay_model_spec(
        self,
        *,
        agent: AgentProtocol,
        agent_name: str,
        agent_snapshot: SessionAgentSnapshot,
        warnings: list[SessionHydrationWarning],
    ) -> str | None:
        if not agent_snapshot.model_overlay_refs:
            return None

        overlay_ref = agent_snapshot.model_overlay_refs[0].ref
        overlay_path = Path(overlay_ref).expanduser().resolve()
        source_path = agent.config.source_path
        start_path = source_path.parent if source_path is not None else Path.cwd()

        try:
            from fast_agent.llm.model_overlays import load_model_overlay_registry

            registry = load_model_overlay_registry(start_path=start_path)
        except Exception as exc:
            warnings.append(
                SessionHydrationWarning(
                    code='overlay-restore-failed',
                    message=f"Failed to load model overlays for agent {agent_name!r}: {exc}",
                    agent_name=agent_name,
                    ref=overlay_ref,
                )
            )
            return None

        for overlay in registry.overlays:
            if overlay.manifest_path.expanduser().resolve() == overlay_path:
                return overlay.name

        warnings.append(
            SessionHydrationWarning(
                code='overlay-restore-missing',
                message=(
                    f"Persisted model overlay {overlay_ref!r} is not available for agent"
                    f" {agent_name!r}"
                ),
                agent_name=agent_name,
                ref=overlay_ref,
            )
        )
        return None

    def _apply_request_settings(
        self,
        agent: AgentProtocol,
        request_settings: SessionRequestSettingsSnapshot,
    ) -> None:
        params = self._base_request_params(agent)
        params.maxTokens = request_settings.max_tokens or params.maxTokens
        params.temperature = request_settings.temperature
        params.top_p = request_settings.top_p
        params.top_k = request_settings.top_k
        params.min_p = request_settings.min_p
        params.presence_penalty = request_settings.presence_penalty
        params.frequency_penalty = request_settings.frequency_penalty
        params.repetition_penalty = request_settings.repetition_penalty
        params.use_history = (
            request_settings.use_history
            if request_settings.use_history is not None
            else params.use_history
        )
        params.parallel_tool_calls = (
            request_settings.parallel_tool_calls
            if request_settings.parallel_tool_calls is not None
            else params.parallel_tool_calls
        )
        params.max_iterations = (
            request_settings.max_iterations
            if request_settings.max_iterations is not None
            else params.max_iterations
        )
        params.tool_result_mode = (
            request_settings.tool_result_mode
            if request_settings.tool_result_mode is not None
            else params.tool_result_mode
        )
        params.streaming_timeout = request_settings.streaming_timeout
        params.service_tier = request_settings.service_tier
        params.systemPrompt = agent.instruction

        agent.config.use_history = params.use_history
        agent.config.default_request_params = params.model_copy(deep=True)
        llm = agent.llm
        if llm is not None:
            llm.default_request_params = params.model_copy(deep=True)

    def _base_request_params(self, agent: AgentProtocol) -> RequestParams:
        llm = agent.llm
        if llm is not None:
            return llm.default_request_params.model_copy(deep=True)

        default_params = agent.config.default_request_params
        if default_params is not None:
            return default_params.model_copy(deep=True)

        return RequestParams(use_history=agent.config.use_history, systemPrompt=agent.instruction)

    def _persisted_attached_mcp_servers(
        self,
        attachment_refs: list[SessionAttachmentRef],
    ) -> list[str]:
        server_names: list[str] = []
        for attachment_ref in attachment_refs:
            ref = attachment_ref.ref
            if not ref.startswith('mcp_server:'):
                continue
            server_name = ref.split(':', 1)[1]
            if server_name and server_name not in server_names:
                server_names.append(server_name)
        return server_names

    def _persisted_attached_agent_tools(
        self,
        attachment_refs: list[SessionAttachmentRef],
    ) -> list[str]:
        tool_names: list[str] = []
        for attachment_ref in attachment_refs:
            ref = attachment_ref.ref
            if not ref.startswith('agent_tool:'):
                continue
            child_name = ref.split(':', 1)[1]
            if child_name and child_name not in tool_names:
                tool_names.append(child_name)
        return tool_names

    def _attached_agent_tool_names(self, agent: AgentProtocol) -> set[str]:
        if not isinstance(agent, _AgentToolProvider):
            return set()

        attached: set[str] = set()
        for tool_name, child_agent in agent.agent_backed_tools.items():
            if tool_name.startswith('agent__'):
                attached.add(tool_name.split('__', 1)[1])
            if isinstance(child_agent, _NamedAgent):
                attached.add(child_agent.name)
        return attached

    def _load_snapshot(
        self,
        *,
        session: Session,
        warnings: list[SessionHydrationWarning],
    ) -> SessionSnapshot:
        snapshot_path = session.directory / "session.json"
        try:
            with snapshot_path.open(encoding="utf-8") as handle:
                return load_session_snapshot(json.load(handle))
        except Exception as exc:
            warnings.append(
                SessionHydrationWarning(
                    code='snapshot-load-fallback',
                    message=f'Falling back to compatibility session info while loading snapshot: {exc}',
                    ref=str(snapshot_path),
                )
            )
            return snapshot_from_session_info(session.info)

    def _select_agent_snapshots(
        self,
        *,
        session: Session,
        snapshot: SessionSnapshot,
        agents: Mapping[str, AgentProtocol],
        fallback_agent_name: str | None,
    ) -> dict[str, SessionAgentSnapshot]:
        if snapshot.continuation.agents:
            return snapshot.continuation.agents

        metadata = session.info.metadata
        history_map = metadata.get('last_history_by_agent') if isinstance(metadata, dict) else None
        if isinstance(history_map, Mapping) and history_map:
            return snapshot_from_session_info(session.info).continuation.agents

        fallback_name = fallback_agent_name if fallback_agent_name in agents else None
        if fallback_name is None:
            fallback_name = next(iter(agents), None)
        if fallback_name is None:
            return {}

        history_path = session.latest_history_path(fallback_name)
        if history_path is None or not history_path.exists():
            return {}
        return {fallback_name: SessionAgentSnapshot(history_file=history_path.name)}

    def _resolve_active_agent(
        self,
        *,
        snapshot: SessionSnapshot,
        agents: Mapping[str, AgentProtocol],
        loaded_agents: Mapping[str, Path],
        fallback_agent_name: str | None,
        warnings: list[SessionHydrationWarning],
    ) -> str | None:
        persisted_active_agent = snapshot.continuation.active_agent
        if persisted_active_agent is not None:
            if persisted_active_agent in agents:
                return persisted_active_agent
            warnings.append(
                SessionHydrationWarning(
                    code='missing-active-agent',
                    message=(
                        f"Persisted active agent {persisted_active_agent!r} is not available in"
                        ' this runtime'
                    ),
                    agent_name=persisted_active_agent,
                )
            )

        if len(loaded_agents) == 1:
            return next(iter(loaded_agents))
        if fallback_agent_name is not None and fallback_agent_name in agents:
            return fallback_agent_name
        return next(iter(agents), None)
