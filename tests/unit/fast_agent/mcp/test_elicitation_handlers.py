from dataclasses import dataclass
from typing import Any

import pytest
from mcp_types import ElicitRequestURLParams, ElicitResult

from fast_agent.human_input.types import HumanInputResponse
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.elicitation_handlers import (
    _parse_elicitation_content,
)
from fast_agent.mcp.tool_result_metadata import (
    set_url_elicitation_required_payload,
    url_elicitation_required_payload,
)
from fast_agent.mcp.url_elicitation_required import (
    URLElicitationDisplayItem,
    URLElicitationRequiredDisplayPayload,
)


@dataclass
class _ContextWithSession:
    session: object


def _response(value: str) -> HumanInputResponse:
    return HumanInputResponse(request_id="request", response=value)


def test_parse_elicitation_content_accepts_json_object_with_required_fields() -> None:
    content = _parse_elicitation_content(
        _response('{"name": "Ada", "age": 37}'),
        {"required": ["name"], "properties": {"name": {"type": "string"}}},
    )

    assert content == {"name": "Ada", "age": 37}


def test_url_elicitation_payload_round_trips_on_builtin_exception() -> None:
    exc = Exception("url elicitation required")
    payload = URLElicitationRequiredDisplayPayload(
        server_name="session-server",
        request_method="tools/call",
        elicitations=[
            URLElicitationDisplayItem(
                message="Open browser to continue",
                url="https://example.com/continue",
                elicitation_id="form-url-1",
            )
        ],
        issues=[],
    )

    set_url_elicitation_required_payload(exc, payload)

    assert url_elicitation_required_payload(exc) is payload


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
    runtime = MCPClientCallbackRuntime(
        server_name="session-server",
        server_config=None,
        agent_name="test-agent",
    )

    context: Any = _ContextWithSession(session=object())
    params = ElicitRequestURLParams(
        mode="url",
        message="Open browser to continue",
        url="https://example.com/continue",
        elicitation_id="form-url-1",
    )

    callback = runtime.elicitation_callback
    assert callback is not None
    result = await callback(context, params)
    assert isinstance(result, ElicitResult)
    assert result.action == "accept"

    captured = capsys.readouterr()
    assert captured.out.strip() == ""

    items = runtime.consume_pending_url_elicitations()
    assert len(items) == 1
    assert items[0].message == "Open browser to continue"
    assert items[0].url == "https://example.com/continue"
    assert items[0].elicitation_id == "form-url-1"
