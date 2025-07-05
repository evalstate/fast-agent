"""
Predefined elicitation handlers for different use cases.
"""

import json
from typing import TYPE_CHECKING, Any

from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult, ErrorData

from mcp_agent.human_input.elicitation_handler import elicitation_input_callback
from mcp_agent.human_input.types import HumanInputRequest
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.helpers.server_config_helpers import get_server_config

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)


async def auto_cancel_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult | ErrorData:
    """Handler that automatically cancels all elicitation requests.
    
    Useful for production deployments where you want to advertise elicitation
    capability but automatically decline all requests.
    """
    logger.info(f"Auto-cancelling elicitation request: {params.message}")
    return ElicitResult(action="cancel")


async def forms_elicitation_handler(
    context: RequestContext["ClientSession", Any], params: ElicitRequestParams
) -> ElicitResult:
    """
    Interactive forms-based elicitation handler using enhanced input handler.
    """
    logger.info(f"Eliciting response for params: {params}")

    # Get server config for additional context
    server_config = get_server_config(context)
    server_name = server_config.name if server_config else "Unknown Server"
    server_info = (
        {"command": server_config.command} if server_config and server_config.command else None
    )

    # Get agent name - try multiple sources in order of preference
    agent_name: str | None = None

    # 1. Check if we have an MCPAgentClientSession in the context
    from mcp_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
    if hasattr(context, "session") and isinstance(context.session, MCPAgentClientSession):
        agent_name = context.session.agent_name

    # 2. If no agent name yet, use a sensible default
    if not agent_name:
        agent_name = "Unknown Agent"

    # Create human input request
    request = HumanInputRequest(
        prompt=params.message,
        description=f"Schema: {params.requestedSchema}" if params.requestedSchema else None,
        request_id=f"elicit_{id(params)}",
        metadata={
            "agent_name": agent_name,
            "server_name": server_name,
            "elicitation": True,
            "requested_schema": params.requestedSchema,
        },
    )

    try:
        # Call the enhanced elicitation handler
        response = await elicitation_input_callback(
            request=request,
            agent_name=agent_name,
            server_name=server_name,
            server_info=server_info,
        )

        # Check for special action responses
        response_data = response.response.strip()

        # Handle special responses
        if response_data == "__DECLINED__":
            return ElicitResult(action="decline")
        elif response_data == "__CANCELLED__":
            return ElicitResult(action="cancel")
        elif response_data == "__DISABLE_SERVER__":
            # Log that user wants to disable elicitation for this server
            logger.warning(f"User requested to disable elicitation for server: {server_name}")
            # For now, just cancel - in a full implementation, this would update server config
            return ElicitResult(action="cancel")

        # Parse response based on schema if provided
        if params.requestedSchema:
            # Check if the response is already JSON (from our form)
            try:
                # Try to parse as JSON first (from schema-driven form)
                content = json.loads(response_data)
                # Validate that all required fields are present
                required_fields = params.requestedSchema.get("required", [])
                for field in required_fields:
                    if field not in content:
                        logger.warning(f"Missing required field '{field}' in elicitation response")
                        return ElicitResult(action="decline")
            except json.JSONDecodeError:
                # Not JSON, try to handle as simple text response
                # This is a fallback for simple schemas or text-based responses
                properties = params.requestedSchema.get("properties", {})
                if len(properties) == 1:
                    # Single field schema - try to parse based on type
                    field_name = list(properties.keys())[0]
                    field_def = properties[field_name]
                    field_type = field_def.get("type")

                    if field_type == "boolean":
                        # Parse boolean values
                        if response_data.lower() in ["yes", "y", "true", "1"]:
                            content = {field_name: True}
                        elif response_data.lower() in ["no", "n", "false", "0"]:
                            content = {field_name: False}
                        else:
                            return ElicitResult(action="decline")
                    elif field_type == "string":
                        content = {field_name: response_data}
                    elif field_type in ["number", "integer"]:
                        try:
                            value = (
                                int(response_data)
                                if field_type == "integer"
                                else float(response_data)
                            )
                            content = {field_name: value}
                        except ValueError:
                            return ElicitResult(action="decline")
                    else:
                        # Unknown type, just pass as string
                        content = {field_name: response_data}
                else:
                    # Multiple fields but text response - can't parse reliably
                    logger.warning("Text response provided for multi-field schema")
                    return ElicitResult(action="decline")
        else:
            # No schema, just return the raw response
            content = {"response": response_data}

        # Return the response wrapped in ElicitResult with accept action
        return ElicitResult(action="accept", content=content)
    except (KeyboardInterrupt, EOFError, TimeoutError):
        # User cancelled or timeout
        return ElicitResult(action="cancel")