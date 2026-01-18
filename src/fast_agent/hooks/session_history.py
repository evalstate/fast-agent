"""Session history hook for saving conversations after each turn."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.context import get_current_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.session import get_session_manager

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
    if manager.current_session is None:
        metadata: dict[str, object] = {"agent_name": ctx.agent_name}
        model_name = getattr(agent_config, "model", None) if agent_config else None
        if model_name:
            metadata["model"] = model_name
        manager.create_session(metadata=metadata)

    try:
        await manager.save_current_session(cast("AgentProtocol", ctx.agent))
    except Exception as exc:
        logger.warning("Failed to save session history", exc_info=exc)
