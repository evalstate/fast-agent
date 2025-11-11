"""
Factory for resolving tool permission handlers with proper precedence.

Similar to elicitation_factory, this provides a way to configure which
permission handler to use with a precedence order.
"""

from typing import Any, Optional

from fast_agent.acp.tool_permission import (
    ToolPermissionHandlerFnT,
    acp_tool_permission_handler,
    allow_all_tool_permission_handler,
    deny_all_tool_permission_handler,
)
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


def resolve_tool_permission_handler(
    agent_config: AgentConfig,
    app_config: Any,
    server_config: Any = None,
    is_acp_mode: bool = False,
) -> Optional[ToolPermissionHandlerFnT]:
    """
    Resolve tool permission handler with proper precedence.

    Precedence order:
    1. Agent decorator supplied (highest precedence)
    2. Server-specific config file setting
    3. Global config file setting
    4. ACP mode detection (if is_acp_mode=True, use acp_tool_permission_handler)
    5. Default allow-all handler (lowest precedence)

    Args:
        agent_config: Agent configuration from decorator
        app_config: Application configuration from YAML
        server_config: Server-specific configuration (optional)
        is_acp_mode: Whether the agent is running in ACP mode

    Returns:
        ToolPermissionHandlerFnT handler or None (no permission checking)
    """

    # 1. Decorator takes highest precedence
    if hasattr(agent_config, "tool_permission_handler") and agent_config.tool_permission_handler:
        logger.debug(
            f"Using decorator-provided tool permission handler for agent {agent_config.name}"
        )
        return agent_config.tool_permission_handler

    # 2. Check server-specific config first
    if server_config:
        permission_config = getattr(server_config, "tool_permission", {})
        if isinstance(permission_config, dict):
            mode = permission_config.get("mode")
        else:
            mode = getattr(permission_config, "mode", None)

        if mode:
            handler = _resolve_handler_from_mode(mode, agent_config.name, "server config")
            if handler is not None:
                return handler

    # 3. Check global config file
    permission_config = getattr(app_config, "tool_permission", {})
    if isinstance(permission_config, dict):
        mode = permission_config.get("mode")
    else:
        mode = getattr(permission_config, "mode", None)

    if mode:
        handler = _resolve_handler_from_mode(mode, agent_config.name, "global config")
        if handler is not None:
            return handler

    # 4. ACP mode detection
    if is_acp_mode:
        logger.debug(
            f"Using ACP tool permission handler (ACP mode) for agent {agent_config.name}"
        )
        return acp_tool_permission_handler

    # 5. Default to allow-all for non-ACP mode
    logger.debug(f"Using default allow-all tool permission handler for agent {agent_config.name}")
    return allow_all_tool_permission_handler


def _resolve_handler_from_mode(
    mode: str, agent_name: str, config_source: str
) -> Optional[ToolPermissionHandlerFnT]:
    """
    Resolve a handler from a configuration mode string.

    Args:
        mode: Mode string ("acp", "allow-all", "deny-all", "none")
        agent_name: Name of the agent (for logging)
        config_source: Source of the config (for logging)

    Returns:
        Handler function or None
    """
    if mode == "none":
        logger.debug(
            f"Tool permissions disabled by {config_source} for agent {agent_name}"
        )
        return allow_all_tool_permission_handler  # No checking, just allow
    elif mode == "acp":
        logger.debug(
            f"Using ACP tool permission handler ({config_source}) for agent {agent_name}"
        )
        return acp_tool_permission_handler
    elif mode == "allow-all":
        logger.debug(
            f"Using allow-all tool permission handler ({config_source}) for agent {agent_name}"
        )
        return allow_all_tool_permission_handler
    elif mode == "deny-all":
        logger.debug(
            f"Using deny-all tool permission handler ({config_source}) for agent {agent_name}"
        )
        return deny_all_tool_permission_handler
    else:
        logger.warning(
            f"Unknown tool permission mode '{mode}' in {config_source} for agent {agent_name}"
        )
        return None
