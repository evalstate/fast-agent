"""
Factory for resolving elicitation handlers with proper precedence.

Supports multiple modes:
- "forms": Interactive terminal forms (default for CLI)
- "auto-cancel": Automatically cancel all elicitation requests
- "acp-interactive": Interactive Q&A over ACP (for ACP sessions)
- "none": Don't advertise elicitation capability
"""

from typing import Any

from mcp.client.session import ElicitationFnT
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.elicitation_handlers import (
    auto_cancel_elicitation_handler,
    forms_elicitation_handler,
)

logger = get_logger(__name__)


async def acp_delegating_elicitation_handler(
    context: RequestContext[Any, Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """
    Elicitation handler that delegates to ACP handler if available.

    This handler checks if the MCP session's aggregator has an ACP elicitation
    handler registered. If so, it delegates to that handler for interactive Q&A.
    Otherwise, it falls back to the forms handler.

    This allows elicitations to work in ACP mode without requiring configuration
    changes - the ACP handler is automatically used when running under ACP.
    """
    # Try to get the aggregator from the session to check for ACP handler
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    if hasattr(context, "session") and isinstance(context.session, MCPAgentClientSession):
        session = context.session
        # Check if aggregator has an ACP elicitation handler
        aggregator = getattr(session, "_aggregator_ref", None)
        if aggregator is None:
            # Try to get it via agent reference
            agent = getattr(session, "_agent_ref", None)
            if agent:
                aggregator = getattr(agent, "_aggregator", None)

        if aggregator:
            acp_handler = getattr(aggregator, "_acp_elicitation_handler", None)
            if acp_handler:
                logger.debug(
                    "Delegating elicitation to ACP handler",
                    name="acp_elicitation_delegate",
                )
                return await acp_handler(context, params)

    # Fall back to forms handler
    logger.debug(
        "No ACP handler available, using forms handler",
        name="forms_elicitation_fallback",
    )
    return await forms_elicitation_handler(context, params)


def resolve_elicitation_handler(
    agent_config: AgentConfig, app_config: Any, server_config: Any = None
) -> ElicitationFnT | None:
    """Resolve elicitation handler with proper precedence.

    Precedence order:
    1. Agent decorator supplied (highest precedence)
    2. Server-specific config file setting
    3. Global config file setting
    4. Default delegating handler (lowest precedence)

    The delegating handler automatically uses ACP interactive mode when running
    under ACP, and falls back to terminal forms otherwise.

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
            handler = _get_handler_for_mode(mode, agent_config.name, "server config")
            if handler is not None or mode == "none":
                return handler

    # 3. Check global config file
    elicitation_config = getattr(app_config, "elicitation", {})
    if isinstance(elicitation_config, dict):
        mode = elicitation_config.get("mode", "forms")
    else:
        mode = getattr(elicitation_config, "mode", "forms")

    handler = _get_handler_for_mode(mode, agent_config.name, "global config")
    if handler is not None or mode == "none":
        return handler

    # 4. Default to delegating handler (auto-detects ACP mode)
    logger.debug(f"Using default delegating elicitation handler for agent {agent_config.name}")
    return acp_delegating_elicitation_handler


def _get_handler_for_mode(
    mode: str, agent_name: str, config_source: str
) -> ElicitationFnT | None:
    """
    Get the appropriate handler for a given elicitation mode.

    Args:
        mode: The elicitation mode string
        agent_name: Name of the agent (for logging)
        config_source: Where the mode was configured (for logging)

    Returns:
        ElicitationFnT handler or None for "none" mode
    """
    if mode == "none":
        logger.debug(f"Elicitation disabled by {config_source} for agent {agent_name}")
        return None
    elif mode == "auto-cancel":
        logger.debug(f"Using auto-cancel elicitation handler ({config_source}) for agent {agent_name}")
        return auto_cancel_elicitation_handler
    elif mode == "acp-interactive":
        # Explicit ACP interactive mode - use delegating handler
        logger.debug(
            f"Using ACP interactive elicitation handler ({config_source}) for agent {agent_name}"
        )
        return acp_delegating_elicitation_handler
    elif mode == "forms":
        # Explicit forms mode - use delegating handler (falls back to forms if not in ACP)
        logger.debug(f"Using forms elicitation handler ({config_source}) for agent {agent_name}")
        return acp_delegating_elicitation_handler
    else:
        # Unknown mode - use delegating handler as default
        logger.warning(
            f"Unknown elicitation mode '{mode}' for agent {agent_name}, using default"
        )
        return acp_delegating_elicitation_handler
