"""Session history hook for saving conversations after each turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.context import get_current_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.session import extract_session_title, get_session_manager

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookContext
    from fast_agent.interfaces import AgentProtocol

logger = get_logger(__name__)


async def save_session_history(ctx: "HookContext") -> None:
    """Save the agent history into the active session after a turn completes."""
    context = get_current_context()
    config = context.config if context else None
    if config is not None and not getattr(config, "session_history", True):
        return

    agent_config = getattr(ctx.agent, "config", None)
    if agent_config and getattr(agent_config, "tool_only", False):
        return

    if not ctx.message_history:
        return

    manager = get_session_manager()
    acp_session_id = None
    agent_context = getattr(ctx.agent, "context", None)
    acp_context = getattr(agent_context, "acp", None) if agent_context else None
    if acp_context is not None:
        acp_session_id = getattr(acp_context, "session_id", None)
    session = manager.current_session
    if session is None:
        metadata: dict[str, object] = {"agent_name": ctx.agent_name}
        model_name = getattr(agent_config, "model", None) if agent_config else None
        if model_name:
            metadata["model"] = model_name
        if acp_session_id:
            manager.create_session_with_id(str(acp_session_id), metadata=metadata)
        else:
            manager.create_session(metadata=metadata)
        session = manager.current_session
    elif acp_session_id:
        session = manager.current_session
        if (
            session is not None
            and session.info.metadata.get("acp_session_id") != acp_session_id
        ):
            session.info.metadata["acp_session_id"] = acp_session_id
            session._save_metadata()

    previous_title = extract_session_title(session.info.metadata) if session else None

    try:
        await manager.save_current_session(cast("AgentProtocol", ctx.agent))
    except Exception as exc:
        logger.warning("Failed to save session history", exc_info=exc)
        return

    if acp_context is None or session is None:
        return

    new_title = extract_session_title(session.info.metadata)
    if not new_title or new_title == previous_title:
        return

    try:
        await acp_context.send_session_info_update(
            title=new_title,
            updated_at=session.info.last_activity.isoformat(),
        )
    except Exception as exc:
        logger.warning("Failed to send ACP session info update", exc_info=exc)
