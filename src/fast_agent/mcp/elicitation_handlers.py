"""
Predefined elicitation handlers for different use cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.types import (
    ElicitRequestFormParams,
    ElicitRequestParams,
    ElicitRequestURLParams,
    ElicitResult,
    ErrorData,
)

from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp import ClientSession
    from mcp.shared.context import RequestContext

    from fast_agent.human_input.types import HumanInputResponse
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

ElicitationContent = dict[str, str | int | float | bool | list[str] | None]

logger = get_logger(__name__)
_BOOLEAN_RESPONSE_VALUES = {
    "yes": True,
    "y": True,
    "true": True,
    "1": True,
    "no": False,
    "n": False,
    "false": False,
    "0": False,
}


@dataclass(frozen=True, slots=True)
class _ElicitationContextInfo:
    agent_name: str
    server_name: str
    server_info: dict[str, str] | None
    session: MCPAgentClientSession | None


def _required_schema_fields(requested_schema: dict[str, Any]) -> list[str]:
    required = requested_schema.get("required", [])
    if not isinstance(required, list):
        return []
    return [field for field in required if isinstance(field, str)]


def _validate_required_fields(
    content: ElicitationContent,
    requested_schema: dict[str, Any],
) -> bool:
    for field in _required_schema_fields(requested_schema):
        if field not in content:
            logger.warning(f"Missing required field '{field}' in elicitation response")
            return False
    return True


def _parse_boolean_response(response_data: str) -> bool | None:
    return _BOOLEAN_RESPONSE_VALUES.get(strip_casefold(response_data))


def _parse_integer_response(response_data: str) -> int | None:
    try:
        return int(response_data)
    except ValueError:
        return None


def _parse_number_response(response_data: str) -> float | None:
    try:
        return float(response_data)
    except ValueError:
        return None


_SCALAR_FIELD_PARSERS: dict[str, Callable[[str], str | int | float | bool | None]] = {
    "boolean": _parse_boolean_response,
    "integer": _parse_integer_response,
    "number": _parse_number_response,
}


def _parse_single_field_value(
    response_data: str,
    field_type: object,
) -> str | int | float | bool | None:
    parser = _SCALAR_FIELD_PARSERS.get(field_type) if isinstance(field_type, str) else None
    if parser is not None:
        return parser(response_data)
    return response_data


def _parse_single_field_response(
    response_data: str,
    requested_schema: dict[str, Any],
) -> ElicitationContent | None:
    properties = requested_schema.get("properties", {})
    if not isinstance(properties, dict) or len(properties) != 1:
        logger.warning("Text response provided for multi-field schema")
        return None

    field_name = next(iter(properties))
    if not isinstance(field_name, str):
        return None
    field_def = properties[field_name]
    field_type = field_def.get("type") if isinstance(field_def, dict) else None
    value = _parse_single_field_value(response_data, field_type)
    if value is None and field_type in _SCALAR_FIELD_PARSERS:
        return None

    return {field_name: value}


def _parse_elicitation_content(
    response: "HumanInputResponse",
    requested_schema: dict[str, Any] | None,
) -> ElicitationContent | None:
    response_data = response.response.strip()
    if requested_schema is None:
        return {"response": response_data}

    try:
        loaded = json.loads(response_data)
    except json.JSONDecodeError:
        content = _parse_single_field_response(response_data, requested_schema)
    else:
        if not isinstance(loaded, dict):
            logger.warning("JSON elicitation response must be an object")
            return None
        content = loaded

    if content is None or not _validate_required_fields(content, requested_schema):
        return None
    return content


async def auto_cancel_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult | ErrorData:
    """Handler that automatically cancels all elicitation requests.

    Useful for production deployments where you want to advertise elicitation
    capability but automatically decline all requests.
    """
    del context
    logger.info(f"Auto-cancelling elicitation request: {params.message}")
    return ElicitResult(action="cancel")


def _mcp_agent_session(
    context: RequestContext["ClientSession", Any],
) -> MCPAgentClientSession | None:
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    session = context.session
    if isinstance(session, MCPAgentClientSession):
        return session
    return None


def _elicitation_context_info(
    context: RequestContext["ClientSession", Any],
) -> _ElicitationContextInfo:
    server_config = get_server_config(context)
    session = _mcp_agent_session(context)
    server_name = server_config.name if server_config and server_config.name else "Unknown Server"
    server_info = (
        {"command": server_config.command} if server_config and server_config.command else None
    )
    agent_name = session.agent_name if session and session.agent_name else "Unknown Agent"
    return _ElicitationContextInfo(
        agent_name=agent_name,
        server_name=server_name,
        server_info=server_info,
        session=session,
    )


def _handle_url_elicitation(
    params: ElicitRequestURLParams,
    context_info: _ElicitationContextInfo,
) -> ElicitResult:
    url = str(params.url)
    elicitation_id = str(params.elicitationId) if params.elicitationId is not None else None
    logger.info(
        f"URL elicitation from {context_info.server_name}: {url} "
        f"(elicitationId={elicitation_id})"
    )

    queued = False
    if context_info.session is not None:
        queued = context_info.session.queue_url_elicitation_for_active_request(
            message=params.message,
            url=url,
            elicitation_id=elicitation_id,
        )

    if not queued:
        from fast_agent.ui.console_display import ConsoleDisplay

        display = ConsoleDisplay()
        display.show_url_elicitation(
            message=params.message,
            url=url,
            server_name=context_info.server_name or "Unknown Server",
            agent_name=context_info.agent_name,
            elicitation_id=elicitation_id,
        )

    return ElicitResult(action="accept")


def _form_human_input_request(
    params: ElicitRequestFormParams,
    context_info: _ElicitationContextInfo,
):
    from fast_agent.human_input.types import HumanInputRequest

    requested_schema = params.requestedSchema
    return HumanInputRequest(
        prompt=params.message,
        description=f"Schema: {requested_schema}" if requested_schema else None,
        request_id=f"elicit_{id(params)}",
        metadata={
            "agent_name": context_info.agent_name,
            "server_name": context_info.server_name,
            "elicitation": True,
            "requested_schema": requested_schema,
        },
    )


def _disable_server_elicitation(server_name: str) -> None:
    logger.warning(
        f"User requested to disable elicitation for server: {server_name} — disabling for session"
    )
    try:
        from fast_agent.human_input.elicitation_state import elicitation_state

        elicitation_state.disable_server(server_name)
    except Exception:
        # Do not fail the flow if state update fails.
        pass


def _special_elicitation_action(response_data: str, server_name: str) -> ElicitResult | None:
    if response_data == "__DECLINED__":
        return ElicitResult(action="decline")
    if response_data == "__CANCELLED__":
        return ElicitResult(action="cancel")
    if response_data == "__DISABLE_SERVER__":
        _disable_server_elicitation(server_name)
        return ElicitResult(action="cancel")
    return None


async def _handle_form_elicitation(
    params: ElicitRequestFormParams,
    context_info: _ElicitationContextInfo,
) -> ElicitResult:
    from fast_agent.human_input.elicitation_handler import elicitation_input_callback

    response = await elicitation_input_callback(
        request=_form_human_input_request(params, context_info),
        agent_name=context_info.agent_name,
        server_name=context_info.server_name,
        server_info=context_info.server_info,
    )

    response_data = response.response.strip()
    if special_result := _special_elicitation_action(response_data, context_info.server_name):
        return special_result

    content = _parse_elicitation_content(response, params.requestedSchema)
    if content is None:
        return ElicitResult(action="decline")
    return ElicitResult(action="accept", content=content)


async def forms_elicitation_handler(
    context: RequestContext["ClientSession", Any], params: ElicitRequestParams
) -> ElicitResult:
    """
    Combined elicitation handler supporting both form and URL modes.

    For form mode: Uses interactive forms-based UI for data collection.
    For URL mode: Displays the URL inline for out-of-band user interaction.
    """
    logger.info(f"Eliciting response for params: {params}")
    context_info = _elicitation_context_info(context)
    if isinstance(params, ElicitRequestURLParams):
        return _handle_url_elicitation(params, context_info)

    try:
        return await _handle_form_elicitation(params, context_info)
    except (KeyboardInterrupt, EOFError, TimeoutError):
        return ElicitResult(action="cancel")
