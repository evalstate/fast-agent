"""
Factory for resolving elicitation handlers with proper precedence.
"""

from typing import Any, Optional

from mcp.client.session import ElicitationFnT

from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.elicitation_handlers import (
    auto_cancel_elicitation_handler,
    forms_elicitation_handler,
)

logger = get_logger(__name__)


def resolve_elicitation_handler(
    agent_config: AgentConfig, app_config: Any, server_config: Any = None
) -> Optional[ElicitationFnT]:
    """Resolve elicitation handler with proper precedence.

    Precedence order:
    1. Agent decorator supplied (highest precedence)
    2. Server-specific config file setting
    3. Global config file setting
    4. Default forms handler (lowest precedence)

    Args:
        agent_config: Agent configuration from decorator
        app_config: Application configuration from YAML
        server_config: Server-specific configuration (optional)

    Returns:
        ElicitationFnT handler or None (no elicitation capability)
    """

    # 1. Decorator takes highest precedence
    if agent_config.elicitation_handler:
        logger.debug(f"Using decorator-provided elicitation handler for agent {agent_config.name}")
        return agent_config.elicitation_handler

    # 2. Check server-specific config first
    if server_config:
        elicitation_config = getattr(server_config, "elicitation", {})
        if isinstance(elicitation_config, dict):
            mode = elicitation_config.get("mode")
        else:
            mode = getattr(elicitation_config, "mode", None)

        if mode:
            if mode == "none":
                logger.debug(f"Elicitation disabled by server config for agent {agent_config.name}")
                return None  # Don't advertise elicitation capability
            elif mode == "auto_cancel":
                logger.debug(
                    f"Using auto-cancel elicitation handler (server config) for agent {agent_config.name}"
                )
                return auto_cancel_elicitation_handler
            else:  # "forms" or other
                logger.debug(
                    f"Using forms elicitation handler (server config) for agent {agent_config.name}"
                )
                return forms_elicitation_handler

    # 3. Check global config file
    elicitation_config = getattr(app_config, "elicitation", {})
    if isinstance(elicitation_config, dict):
        mode = elicitation_config.get("mode", "forms")
    else:
        mode = getattr(elicitation_config, "mode", "forms")

    if mode == "none":
        logger.debug(f"Elicitation disabled by global config for agent {agent_config.name}")
        return None  # Don't advertise elicitation capability
    elif mode == "auto_cancel":
        logger.debug(
            f"Using auto-cancel elicitation handler (global config) for agent {agent_config.name}"
        )
        return auto_cancel_elicitation_handler
    else:  # "forms" or default
        logger.debug(f"Using default forms elicitation handler for agent {agent_config.name}")
        return forms_elicitation_handler
