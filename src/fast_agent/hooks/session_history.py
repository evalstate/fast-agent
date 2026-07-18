"""Session history hook for saving conversations after each turn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from fast_agent.context import get_current_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.session import extract_session_title, get_session_manager
from fast_agent.session.identity import (
    SessionSaveContext,
    SessionStoreScope,
    normalize_session_store_scope,
    resolve_session_for_save,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.hooks.hook_context import HookContext
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LlmAgentProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.session.session_manager import Session, SessionManager
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)


def _effective_use_history(ctx: "HookContext") -> bool:
    request_params = ctx.request_params
    if request_params is not None:
        return request_params.use_history
    return ctx.agent.config.use_history


@runtime_checkable
class _AttachedMcpServerProvider(Protocol):
    def list_attached_mcp_servers(self) -> list[str]: ...


@runtime_checkable
class _AgentBackedToolProvider(Protocol):
    @property
    def agent_backed_tools(self) -> Mapping[str, "LlmAgentProtocol"]: ...


@dataclass
class _SessionHistoryAgentProxy:
    """Delegate agent metadata while exposing a snapshot history for persistence."""

    agent: AgentProtocol
    message_history: list["PromptMessageExtended"]

    @property
    def name(self) -> str:
        return self.agent.name

    @property
    def config(self) -> "AgentConfig":
        return self.agent.config

    @property
    def instruction(self) -> str:
        return self.agent.instruction

    @property
    def llm(self) -> "FastAgentLLMProtocol | None":
        return self.agent.llm

    @property
    def usage_accumulator(self) -> "UsageAccumulator | None":
        return self.agent.usage_accumulator

    def list_attached_mcp_servers(self) -> list[str]:
        if isinstance(self.agent, _AttachedMcpServerProvider):
            return self.agent.list_attached_mcp_servers()
        return []

    @property
    def agent_backed_tools(self) -> Mapping[str, "LlmAgentProtocol"]:
        if isinstance(self.agent, _AgentBackedToolProvider):
            return self.agent.agent_backed_tools
        return {}


class _SessionInfoUpdateCapable(Protocol):
    async def send_session_info_update(
        self,
        *,
        title: str | None | object = ...,
        updated_at: str | None | object = ...,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _SessionHistoryContext:
    acp_session_id: str | None = None
    session_cwd: Path | None = None
    session_store_scope: SessionStoreScope = "workspace"
    session_store_cwd: Path | None = None
    session_manager: "SessionManager | None" = None
    resolved_prompts: dict[str, str] | None = None
    acp_context: _SessionInfoUpdateCapable | None = None


async def save_session_history(ctx: "HookContext") -> None:
    """Save the agent history into the active session after a turn completes."""
    current_context = get_current_context()
    config = current_context.config if current_context else None
    if config is not None and not config.session_history:
        return

    agent_config = ctx.agent.config
    if agent_config.tool_only:
        return

    if not _effective_use_history(ctx):
        return

    if not ctx.message_history:
        return

    history_agent = _SessionHistoryAgentProxy(
        agent=cast("AgentProtocol", ctx.agent),
        message_history=ctx.message_history,
    )
    session_context = _session_history_context(ctx)

    metadata: dict[str, object] = {"agent_name": ctx.agent_name}
    model_name = agent_config.model
    if model_name:
        metadata["model"] = model_name
    identity = resolve_session_for_save(
        current_session=None,
        get_manager=_session_manager_resolver(session_context),
        context=SessionSaveContext(
            acp_session_id=session_context.acp_session_id,
            session_cwd=session_context.session_cwd,
            session_store_scope=session_context.session_store_scope,
            session_store_cwd=session_context.session_store_cwd,
        ),
        seed_metadata=metadata,
    )
    manager = identity.manager
    session = identity.session

    previous_title = extract_session_title(session.info.metadata) if session else None

    # Mid-turn tool-loop checkpoints happen after every tool call; keep them
    # cheap (compact JSON, reused git state). Turn boundaries, cancellations,
    # and errors do full-fidelity saves.
    checkpoint = ctx.hook_type == "after_tool_loop_iteration"
    try:
        await manager.save_current_session(
            cast("AgentProtocol", history_agent),
            agent_registry=ctx.agent_registry,
            identity=identity,
            resolved_prompts=session_context.resolved_prompts,
            checkpoint=checkpoint,
        )
    except Exception as exc:
        logger.warning(
            "Failed to save session history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
        return

    await _send_session_info_update(
        acp_context=session_context.acp_context,
        session=session,
        previous_title=previous_title,
    )


def _session_history_context(ctx: "HookContext") -> _SessionHistoryContext:
    agent_context = ctx.context
    acp_context = agent_context.acp if agent_context else None
    if acp_context is None:
        return _SessionHistoryContext(
            session_cwd=Path.cwd().resolve(),
            session_manager=agent_context.session_manager if agent_context else None,
        )

    assert agent_context is not None
    return _SessionHistoryContext(
        acp_session_id=acp_context.session_id,
        session_cwd=_resolved_path(acp_context.session_cwd),
        session_store_scope=normalize_session_store_scope(acp_context.session_store_scope),
        session_store_cwd=_resolved_path(acp_context.session_store_cwd),
        session_manager=agent_context.session_manager,
        resolved_prompts=acp_context.resolved_instructions_snapshot() or None,
        acp_context=acp_context,
    )


def _session_manager_resolver(
    context: _SessionHistoryContext,
) -> "Callable[[Path | None], SessionManager]":
    manager = context.session_manager
    if manager is None:
        return lambda cwd: get_session_manager(cwd=cwd)

    def resolve(cwd: Path | None) -> SessionManager:
        if cwd is not None and cwd.resolve() != manager.workspace_dir:
            raise RuntimeError(
                "Session history save requested a different cwd than the active "
                "session manager."
            )
        return manager

    return resolve


def _resolved_path(raw_path: object | None) -> Path | None:
    if not raw_path:
        return None
    return Path(str(raw_path)).expanduser().resolve()


async def _send_session_info_update(
    *,
    acp_context: _SessionInfoUpdateCapable | None,
    session: "Session | None",
    previous_title: str | None,
) -> None:
    if acp_context is None or session is None:
        return
    try:
        new_title = extract_session_title(session.info.metadata)
        updated_at = session.info.last_activity.isoformat()
        if new_title != previous_title:
            await acp_context.send_session_info_update(title=new_title, updated_at=updated_at)
            return
        await acp_context.send_session_info_update(updated_at=updated_at)
    except Exception as exc:
        logger.warning(
            "Failed to send ACP session info update",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
