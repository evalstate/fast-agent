from dataclasses import dataclass
from typing import Any

import pytest
from mcp.types import CallToolResult, ElicitRequestURLParams

from fast_agent.human_input.types import HumanInputResponse
from fast_agent.mcp.elicitation_handlers import (
    _parse_elicitation_content,
    forms_elicitation_handler,
)
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession


@dataclass
class _ContextWithSession:
    session: MCPAgentClientSession


def _response(value: str) -> HumanInputResponse:
    return HumanInputResponse(request_id="request", response=value)


def test_parse_elicitation_content_accepts_json_object_with_required_fields() -> None:
    content = _parse_elicitation_content(
        _response('{"name": "Ada", "age": 37}'),
        {"required": ["name"], "properties": {"name": {"type": "string"}}},
    )

    assert content == {"name": "Ada", "age": 37}


@pytest.mark.parametrize("payload", ['["Ada"]', '"Ada"', "42", "true"])
def test_parse_elicitation_content_rejects_non_object_json(payload: str) -> None:
    content = _parse_elicitation_content(
        _response(payload),
        {"required": ["name"], "properties": {"name": {"type": "string"}}},
    )

    assert content is None


def test_parse_elicitation_content_uses_single_field_text_fallback() -> None:
    content = _parse_elicitation_content(
        _response("yes"),
        {"required": ["confirmed"], "properties": {"confirmed": {"type": "boolean"}}},
    )

    assert content == {"confirmed": True}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("no", False),
        (" YES ", True),
        ("n", False),
        ("+42", 42),
        (".5", 0.5),
    ],
)
def test_parse_elicitation_content_coerces_single_field_text_values(
    payload: str,
    expected: bool | int | float,
) -> None:
    field_type = "boolean" if isinstance(expected, bool) else "integer"
    if isinstance(expected, float):
        field_type = "number"

    content = _parse_elicitation_content(
        _response(payload),
        {"required": ["value"], "properties": {"value": {"type": field_type}}},
    )

    assert content == {"value": expected}


@pytest.mark.parametrize(
    ("payload", "field_type"),
    [
        ("maybe", "boolean"),
        ("four", "integer"),
        ("many", "number"),
    ],
)
def test_parse_elicitation_content_rejects_invalid_typed_single_field_text(
    payload: str,
    field_type: str,
) -> None:
    content = _parse_elicitation_content(
        _response(payload),
        {"required": ["value"], "properties": {"value": {"type": field_type}}},
    )

    assert content is None


def test_parse_elicitation_content_keeps_unknown_single_field_type_as_text() -> None:
    content = _parse_elicitation_content(
        _response("2026-06-01"),
        {"required": ["value"], "properties": {"value": {"type": "date"}}},
    )

    assert content == {"value": "2026-06-01"}


@pytest.mark.asyncio
async def test_forms_handler_defers_url_elicitation_to_result_payload(capsys) -> None:
    session = object.__new__(MCPAgentClientSession)
    session.session_server_name = "session-server"
    session.server_config = None
    session.agent_name = "test-agent"
    session._pending_url_elicitations = []

    context: Any = _ContextWithSession(session=session)
    params = ElicitRequestURLParams(
        mode="url",
        message="Open browser to continue",
        url="https://example.com/continue",
        elicitationId="form-url-1",
    )

    result = await forms_elicitation_handler(context, params)
    assert result.action == "accept"

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    tool_result = CallToolResult(content=[], isError=False)
    session._attach_pending_url_elicitation_payload_for_request(
        tool_result,
        request_method="tools/call",
    )

    payload = MCPAgentClientSession.get_url_elicitation_required_payload(tool_result)
    assert payload is not None
    assert payload.server_name == "session-server"
    assert payload.request_method == "tools/call"
    assert len(payload.elicitations) == 1
    assert payload.elicitations[0].message == "Open browser to continue"
    assert payload.elicitations[0].url == "https://example.com/continue"
    assert payload.elicitations[0].elicitation_id == "form-url-1"
