"""Session history hook for saving conversations after each turn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from fast_agent.context import get_current_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.session import extract_session_title, get_session_manager
from fast_agent.session.identity import SessionSaveContext, resolve_session_for_save

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookContext
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)


@dataclass
class _SessionHistoryAgentProxy:
    """Delegate agent metadata while exposing a snapshot history for persistence."""

    agent: AgentProtocol
    message_history: list["PromptMessageExtended"]

    def __getattr__(self, name: str) -> object:
        return getattr(self.agent, name)


async def save_session_history(ctx: "HookContext") -> None:
    """Save the agent history into the active session after a turn completes."""
    current_context = get_current_context()
    config = current_context.config if current_context else None
    if config is not None and not config.session_history:
        return

    agent_config = ctx.agent.config
    if agent_config.tool_only:
        return

    if not ctx.message_history:
        return

    history_agent = _SessionHistoryAgentProxy(
        agent=cast("AgentProtocol", ctx.agent),
        message_history=ctx.message_history,
    )
    acp_session_id: str | None = None
    session_cwd: Path | None = None
    session_store_scope: Literal["workspace", "app"] = "workspace"
    session_store_cwd: Path | None = None
    agent_context = ctx.context
    acp_context = agent_context.acp if agent_context else None
    if acp_context is not None:
        acp_session_id = acp_context.session_id
        raw_session_cwd = acp_context.session_cwd
        if raw_session_cwd:
            session_cwd = Path(str(raw_session_cwd)).expanduser().resolve()
        raw_session_store_scope = acp_context.session_store_scope
        if raw_session_store_scope == "app":
            session_store_scope = "app"
        elif raw_session_store_scope == "workspace":
            session_store_scope = "workspace"
        raw_session_store_cwd = acp_context.session_store_cwd
        if raw_session_store_cwd:
            session_store_cwd = Path(str(raw_session_store_cwd)).expanduser().resolve()

    metadata: dict[str, object] = {"agent_name": ctx.agent_name}
    model_name = agent_config.model
    if model_name:
        metadata["model"] = model_name
    identity = resolve_session_for_save(
        current_session=None,
        get_manager=lambda cwd: get_session_manager(cwd=cwd),
        context=SessionSaveContext(
            acp_session_id=acp_session_id,
            session_cwd=session_cwd,
            session_store_scope=session_store_scope,
            session_store_cwd=session_store_cwd,
        ),
        seed_metadata=metadata,
    )
    manager = identity.manager
    session = identity.session

    previous_title = extract_session_title(session.info.metadata) if session else None

    try:
        await manager.save_current_session(
            cast("AgentProtocol", history_agent),
            agent_registry=ctx.agent_registry,
            identity=identity,
        )
    except Exception as exc:
        logger.warning(
            "Failed to save session history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
        return

    if acp_context is None or session is None:
        return

    try:
        new_title = extract_session_title(session.info.metadata)
        if new_title != previous_title:
            await acp_context.send_session_info_update(
                title=new_title,
                updated_at=session.info.last_activity.isoformat(),
            )
        else:
            await acp_context.send_session_info_update(
                updated_at=session.info.last_activity.isoformat(),
            )
    except Exception as exc:
        logger.warning(
            "Failed to send ACP session info update",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
