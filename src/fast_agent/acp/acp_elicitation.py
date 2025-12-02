"""
ACP Elicitation Handler.

Provides interactive Q&A style elicitation when running via ACP by converting
form schemas into conversational prompts and collecting responses via the
ACP message flow.
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

from acp.helpers import session_notification, update_agent_message_text
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult, ErrorData

from fast_agent.acp.acp_elicitation_state import (
    get_acp_elicitation_state,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.server_config_helpers import get_server_config

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)

# Marker text to indicate elicitation is active
ELICITATION_MARKER = "[ELICITATION_ACTIVE]"


def format_field_as_question(
    field_name: str,
    field_def: dict[str, Any],
    index: int,
) -> str:
    """
    Format a JSON Schema field definition as a human-readable question.

    Args:
        field_name: Name of the field
        field_def: JSON Schema field definition
        index: 1-based index for numbering

    Returns:
        Formatted question string
    """
    field_type = field_def.get("type", "string")
    title = field_def.get("title", field_name)
    description = field_def.get("description", "")
    default = field_def.get("default")

    # Build the question
    question_parts = [f"{index}. **{title}**"]

    if description:
        question_parts.append(f"   {description}")

    # Add type hints
    type_hints = []

    if field_type == "boolean":
        type_hints.append("Answer: yes/no")
    elif field_type == "integer":
        min_val = field_def.get("minimum")
        max_val = field_def.get("maximum")
        if min_val is not None or max_val is not None:
            range_str = f"Range: {min_val or '...'} to {max_val or '...'}"
            type_hints.append(range_str)
        type_hints.append("Type: whole number")
    elif field_type == "number":
        min_val = field_def.get("minimum")
        max_val = field_def.get("maximum")
        if min_val is not None or max_val is not None:
            range_str = f"Range: {min_val or '...'} to {max_val or '...'}"
            type_hints.append(range_str)
        type_hints.append("Type: number")
    elif field_type == "string":
        fmt = field_def.get("format")
        if fmt == "email":
            type_hints.append("Format: email address")
        elif fmt == "uri":
            type_hints.append("Format: URL")
        elif fmt == "date":
            type_hints.append("Format: date (YYYY-MM-DD)")
        elif fmt == "date-time":
            type_hints.append("Format: datetime")

        min_len = field_def.get("minLength")
        max_len = field_def.get("maxLength")
        if min_len or max_len:
            len_str = f"Length: {min_len or 0} to {max_len or 'unlimited'}"
            type_hints.append(len_str)

    # Handle enum/choices
    enum_values = field_def.get("enum")
    if enum_values:
        choices_str = ", ".join(str(v) for v in enum_values)
        type_hints.append(f"Options: {choices_str}")

    if default is not None:
        type_hints.append(f"Default: {default}")

    if type_hints:
        question_parts.append(f"   _({', '.join(type_hints)})_")

    return "\n".join(question_parts)


def format_elicitation_prompt(
    params: ElicitRequestParams,
    agent_name: str,
    server_name: str,
) -> tuple[str, list[str]]:
    """
    Format an elicitation request as a Q&A style prompt.

    Args:
        params: The elicitation parameters from MCP
        agent_name: Name of the requesting agent
        server_name: Name of the MCP server

    Returns:
        Tuple of (formatted prompt string, list of field names)
    """
    lines = []

    # Header
    lines.append("---")
    lines.append(f"**Elicitation Request** from `{server_name}`")
    lines.append("")

    # Message from server
    if params.message:
        lines.append(params.message)
        lines.append("")

    # Parse schema and format fields
    field_names: list[str] = []
    schema = params.requestedSchema

    if schema and isinstance(schema, dict):
        properties = schema.get("properties", {})
        required_fields = set(schema.get("required", []))

        if properties:
            lines.append("**Please provide the following information:**")
            lines.append("")

            for idx, (field_name, field_def) in enumerate(properties.items(), 1):
                field_names.append(field_name)
                is_required = field_name in required_fields

                question = format_field_as_question(field_name, field_def, idx)
                if is_required:
                    question = question.replace("**", "**[Required] ", 1)
                lines.append(question)
                lines.append("")
    else:
        # No schema, just a simple message prompt
        lines.append("**Please provide your response:**")
        lines.append("")
        field_names.append("response")

    # Instructions
    lines.append("---")
    lines.append("_Reply with your answers. Use this format:_")
    if len(field_names) > 1:
        lines.append("```")
        for fn in field_names:
            lines.append(f"{fn}: <your answer>")
        lines.append("```")
    else:
        lines.append("_Simply type your answer._")
    lines.append("")
    lines.append("_Type `/cancel` to cancel this request._")
    lines.append("---")

    return "\n".join(lines), field_names


def parse_elicitation_response(
    response_text: str,
    field_names: list[str],
    schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    Parse a user's response to an elicitation prompt.

    Args:
        response_text: The raw response text from the user
        field_names: List of expected field names
        schema: The original JSON Schema (for type coercion)

    Returns:
        Parsed response dict, or None if parsing failed
    """
    response_text = response_text.strip()

    # Check for cancellation
    if response_text.lower() in ("/cancel", "cancel", "/decline", "decline"):
        return None

    # Single field - just use the response as-is
    if len(field_names) == 1:
        field_name = field_names[0]
        value = coerce_value(response_text, field_name, schema)
        return {field_name: value}

    # Multiple fields - parse key: value format
    result: dict[str, Any] = {}
    current_field: str | None = None
    current_value_lines: list[str] = []

    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if line starts with a field name
        field_match = None
        for fn in field_names:
            # Match "field_name:" or "field_name :"
            if line.lower().startswith(f"{fn.lower()}:") or line.lower().startswith(
                f"{fn.lower()} :"
            ):
                field_match = fn
                # Extract value after the colon
                colon_idx = line.index(":")
                value_part = line[colon_idx + 1 :].strip()
                break

        if field_match:
            # Save previous field if any
            if current_field and current_value_lines:
                result[current_field] = coerce_value(
                    "\n".join(current_value_lines), current_field, schema
                )

            current_field = field_match
            current_value_lines = [value_part] if value_part else []
        elif current_field:
            # Continue previous field's value
            current_value_lines.append(line)
        else:
            # No field matched yet, try positional mapping
            # If user just provides values without field names
            pass

    # Save last field
    if current_field and current_value_lines:
        result[current_field] = coerce_value(
            "\n".join(current_value_lines), current_field, schema
        )

    # If we got no results, try positional parsing
    if not result and len(field_names) > 0:
        # Split by common delimiters
        parts = []
        for delim in ["\n", ",", ";", "|"]:
            if delim in response_text:
                parts = [p.strip() for p in response_text.split(delim) if p.strip()]
                break

        if not parts:
            parts = [response_text]

        # Map parts to fields positionally
        for i, fn in enumerate(field_names):
            if i < len(parts):
                result[fn] = coerce_value(parts[i], fn, schema)

    return result if result else None


def coerce_value(
    value: str,
    field_name: str,
    schema: dict[str, Any] | None,
) -> Any:
    """
    Coerce a string value to the appropriate type based on schema.

    Args:
        value: The string value to coerce
        field_name: Name of the field
        schema: The JSON Schema

    Returns:
        The coerced value
    """
    if not schema:
        return value

    properties = schema.get("properties", {})
    field_def = properties.get(field_name, {})
    field_type = field_def.get("type", "string")

    value = value.strip()

    if field_type == "boolean":
        return value.lower() in ("yes", "y", "true", "1", "on")
    elif field_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    elif field_type == "number":
        try:
            return float(value)
        except ValueError:
            return value

    return value


async def acp_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult | ErrorData:
    """
    ACP-aware elicitation handler that uses interactive Q&A.

    This handler is used when fast-agent is running via ACP. Instead of showing
    a terminal form, it sends a formatted prompt to the user and waits for their
    response via the ACP message flow.

    Args:
        context: The MCP request context
        params: The elicitation parameters

    Returns:
        ElicitResult with the user's response or action
    """
    logger.info(
        "ACP elicitation handler invoked",
        name="acp_elicitation_handler_start",
        message=params.message,
    )

    # Get server config for context
    server_config = get_server_config(context)
    server_name = server_config.name if server_config else "Unknown Server"

    # Get agent name from session
    agent_name: str = "Unknown Agent"
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    if hasattr(context, "session") and isinstance(context.session, MCPAgentClientSession):
        agent_name = context.session.agent_name or agent_name

    # Get session_id from context metadata
    session_id: str | None = None
    if hasattr(context, "session"):
        session = context.session
        # Check for ACP session ID in various places
        if hasattr(session, "_acp_session_id"):
            session_id = session._acp_session_id
        elif hasattr(session, "acp_session_id"):
            session_id = session.acp_session_id

    if not session_id:
        # Fall back to forms handler if no ACP session
        logger.warning(
            "No ACP session found, falling back to forms handler",
            name="acp_elicitation_no_session",
        )
        from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler

        return await forms_elicitation_handler(context, params)

    # Get the elicitation state manager
    state = get_acp_elicitation_state()

    # Check if there's an ACP connection for this session
    connection = state.get_connection(session_id)
    if not connection:
        logger.warning(
            "No ACP connection found for session, falling back to forms handler",
            name="acp_elicitation_no_connection",
            session_id=session_id,
        )
        from fast_agent.mcp.elicitation_handlers import forms_elicitation_handler

        return await forms_elicitation_handler(context, params)

    # Format the elicitation as a Q&A prompt
    prompt_text, field_names = format_elicitation_prompt(
        params, agent_name, server_name
    )

    # Generate a unique request ID
    request_id = f"elicit_{uuid.uuid4().hex[:8]}"

    try:
        # Start the elicitation and get the future to wait on
        response_future = await state.start_elicitation(
            session_id=session_id,
            request_id=request_id,
            params=params,
            agent_name=agent_name,
            server_name=server_name,
            field_names=field_names,
        )

        # Send the prompt to the user via ACP sessionUpdate
        message_chunk = update_agent_message_text(prompt_text)
        notification = session_notification(session_id, message_chunk)
        await connection.sessionUpdate(notification)

        logger.info(
            "Sent elicitation prompt via ACP",
            name="acp_elicitation_prompt_sent",
            session_id=session_id,
            request_id=request_id,
            field_count=len(field_names),
        )

        # Wait for the user's response (with timeout)
        timeout_seconds = 300  # 5 minute timeout
        try:
            response_data = await asyncio.wait_for(
                response_future, timeout=timeout_seconds
            )

            # Check for special responses
            if response_data.get("__action__") == "cancel":
                return ElicitResult(action="cancel")
            elif response_data.get("__action__") == "decline":
                return ElicitResult(action="decline")

            # Remove any internal action markers
            response_data.pop("__action__", None)

            logger.info(
                "Received elicitation response",
                name="acp_elicitation_response_received",
                session_id=session_id,
                request_id=request_id,
            )

            return ElicitResult(action="accept", content=response_data)

        except asyncio.TimeoutError:
            logger.warning(
                "Elicitation timed out",
                name="acp_elicitation_timeout",
                session_id=session_id,
                request_id=request_id,
            )
            return ElicitResult(action="cancel")

        except asyncio.CancelledError:
            logger.info(
                "Elicitation cancelled",
                name="acp_elicitation_cancelled",
                session_id=session_id,
                request_id=request_id,
            )
            return ElicitResult(action="cancel")

    except RuntimeError as e:
        logger.error(
            f"Error starting elicitation: {e}",
            name="acp_elicitation_error",
            session_id=session_id,
        )
        return ElicitResult(action="cancel")
