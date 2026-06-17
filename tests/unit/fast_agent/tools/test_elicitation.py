import json

import pytest

from fast_agent.constants import HUMAN_INPUT_TOOL_NAME
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.elicitation import (
    get_elicitation_fastmcp_tool,
    get_elicitation_input_callback,
    get_elicitation_tool,
    run_elicitation_form,
    set_elicitation_input_callback,
)
from fast_agent.utils.type_narrowing import is_str_object_dict


def _assert_str_object_dict(value: object) -> dict[str, object]:
    assert is_str_object_dict(value)
    return dict(value)


def _collect_schema_refs(value: object) -> list[str]:
    refs: list[str] = []
    if is_str_object_dict(value):
        mapping = _assert_str_object_dict(value)
        ref = mapping.get("$ref")
        if isinstance(ref, str):
            refs.append(ref)
        for child in mapping.values():
            refs.extend(_collect_schema_refs(child))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_schema_refs(item))
    return refs


@pytest.mark.asyncio
async def test_run_elicitation_form_builds_schema_from_field_spec() -> None:
    captured_request: dict[str, object] | None = None
    captured_agent_name: str | None = None
    captured_server_name: str | None = None
    captured_server_info: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request, captured_agent_name, captured_server_name, captured_server_info
        captured_request = request
        captured_agent_name = agent_name
        captured_server_name = server_name
        captured_server_info = server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "title": "Feedback",
                "description": "Collect reviewer feedback",
                "message": "Please fill in the form",
                "fields": [
                    {
                        "name": "summary",
                        "type": "text",
                        "label": "Summary",
                        "help": "High level overview",
                        "required": True,
                    },
                    {
                        "name": "priority",
                        "type": "radio",
                        "options": [
                            {"value": "high", "label": "High"},
                            {"value": "low", "label": "Low"},
                        ],
                    },
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    assert request["prompt"] == "Please fill in the form"
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    assert requested_schema["title"] == "Feedback"
    assert requested_schema["description"] == "Collect reviewer feedback"
    properties = _assert_str_object_dict(requested_schema["properties"])
    summary_property = _assert_str_object_dict(properties["summary"])
    priority_property = _assert_str_object_dict(properties["priority"])
    assert summary_property["description"] == "Summary - High level overview"
    assert priority_property["type"] == "string"
    assert priority_property["enum"] == ["high", "low"]
    assert requested_schema["required"] == ["summary"]
    assert captured_agent_name == "planner"
    assert captured_server_name == "__human_input__"
    assert captured_server_info is None


@pytest.mark.asyncio
async def test_run_elicitation_form_types_radio_options_from_values() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "fields": [
                    {
                        "name": "score",
                        "type": "radio",
                        "options": [
                            {"value": 1, "label": "Low"},
                            {"value": 2, "label": "High"},
                        ],
                    },
                    {
                        "name": "approved",
                        "type": "radio",
                        "options": [
                            {"value": True, "label": "Yes"},
                            {"value": False, "label": "No"},
                        ],
                    },
                    {
                        "name": "mixed",
                        "type": "radio",
                        "options": [{"value": "auto"}, {"value": 0}],
                    },
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    score_property = _assert_str_object_dict(properties["score"])
    approved_property = _assert_str_object_dict(properties["approved"])
    mixed_property = _assert_str_object_dict(properties["mixed"])
    assert score_property["type"] == "integer"
    assert score_property["enum"] == [1, 2]
    assert approved_property["type"] == "boolean"
    assert approved_property["enum"] == [True, False]
    assert "type" not in mixed_property
    assert mixed_property["enum"] == ["auto", 0]


@pytest.mark.asyncio
async def test_run_elicitation_form_does_not_type_radio_enum_with_unsupported_values() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "fields": [
                    {
                        "name": "choice",
                        "type": "radio",
                        "options": [
                            {"value": "ok"},
                            {"value": {"nested": True}},
                        ],
                    },
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    choice_property = _assert_str_object_dict(properties["choice"])
    assert "type" not in choice_property
    assert choice_property["enum"] == ["ok", {"nested": True}]


@pytest.mark.asyncio
async def test_run_elicitation_form_ignores_malformed_radio_options() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "fields": [
                    {"name": "choice", "type": "radio", "options": "yes"},
                    {"name": "mode", "type": "radio", "options": {"value": "auto"}},
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    choice_property = _assert_str_object_dict(properties["choice"])
    mode_property = _assert_str_object_dict(properties["mode"])
    assert choice_property == {"type": "string"}
    assert mode_property == {"type": "string"}


@pytest.mark.asyncio
async def test_run_elicitation_form_number_field_ignores_boolean_min_max() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "fields": [
                    {
                        "name": "count",
                        "type": "number",
                        "min": True,
                        "max": False,
                    }
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    count_property = _assert_str_object_dict(properties["count"])
    assert count_property["type"] == "number"
    assert "minimum" not in count_property
    assert "maximum" not in count_property


@pytest.mark.asyncio
async def test_run_elicitation_form_number_field_ignores_non_finite_min_max() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "fields": [
                    {
                        "name": "rating",
                        "type": "number",
                        "min": float("-inf"),
                        "max": float("nan"),
                    },
                    {
                        "name": "score",
                        "type": "number",
                        "min": 0,
                        "max": 10.5,
                    },
                ],
            },
            agent_name="planner",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    rating_property = _assert_str_object_dict(properties["rating"])
    score_property = _assert_str_object_dict(properties["score"])
    assert rating_property["type"] == "number"
    assert "minimum" not in rating_property
    assert "maximum" not in rating_property
    assert score_property["minimum"] == 0
    assert score_property["maximum"] == 10.5


@pytest.mark.asyncio
async def test_run_elicitation_form_accepts_schema_string_and_merges_overrides() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "done"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        result = await run_elicitation_form(
            {
                "schema": "```json\n"
                + json.dumps(
                    {
                        "type": "object",
                        "properties": {"email": {"type": "string", "format": "email"}},
                    }
                )
                + "\n```",
                "title": "Contact details",
                "description": "Used for follow-up",
                "message": "Share your email",
            },
            agent_name="reviewer",
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert result == "done"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    assert request["prompt"] == "Share your email"
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    assert requested_schema["title"] == "Contact details"
    assert requested_schema["description"] == "Used for follow-up"


@pytest.mark.asyncio
async def test_run_elicitation_form_rejects_more_than_seven_fields() -> None:
    fields = [{"name": f"field_{index}", "type": "text"} for index in range(8)]

    with pytest.raises(ValueError, match="maximum allowed is 7"):
        await run_elicitation_form({"fields": fields}, agent_name="planner")


def test_get_elicitation_tool_builds_sanitized_schema_without_refs() -> None:
    tool = get_elicitation_tool()

    assert tool.name == HUMAN_INPUT_TOOL_NAME
    refs = _collect_schema_refs(tool.inputSchema)
    assert refs == []
    properties = tool.inputSchema.get("properties")
    assert isinstance(properties, dict)
    assert "fields" in properties


@pytest.mark.asyncio
async def test_fastmcp_elicitation_tool_normalizes_explicit_fields() -> None:
    captured_request: dict[str, object] | None = None

    async def callback(
        request: dict[str, object],
        agent_name: str | None,
        server_name: str | None,
        server_info: dict[str, object] | None,
    ) -> str:
        nonlocal captured_request
        captured_request = request
        del agent_name, server_name, server_info
        return "accepted"

    previous_callback = get_elicitation_input_callback()
    set_elicitation_input_callback(callback)
    try:
        tool = get_elicitation_fastmcp_tool()
        result = await tool.run(
            {
                "title": "Approval",
                "fields": [
                    {
                        "name": "approved",
                        "type": "checkbox",
                        "label": "Approved",
                    }
                ],
            }
        )
    finally:
        set_elicitation_input_callback(previous_callback)

    assert get_text(result.content[0]) == "accepted"
    assert captured_request is not None
    request = _assert_str_object_dict(captured_request)
    metadata = _assert_str_object_dict(request["metadata"])
    requested_schema = _assert_str_object_dict(metadata["requested_schema"])
    properties = _assert_str_object_dict(requested_schema["properties"])
    assert "approved" in properties
