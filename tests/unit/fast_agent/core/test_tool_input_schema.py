from __future__ import annotations

from fast_agent.core.tool_input_schema import validate_tool_input_schema


def test_validate_tool_input_schema_accepts_none() -> None:
    result = validate_tool_input_schema(None)

    assert result.normalized is None
    assert result.errors == ()
    assert result.warnings == ()


def test_validate_tool_input_schema_normalizes_valid_mapping() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    result = validate_tool_input_schema(schema)

    assert result.normalized == schema
    assert result.errors == ()
    assert result.warnings == ()


def test_validate_tool_input_schema_reports_shape_errors() -> None:
    result = validate_tool_input_schema(
        {
            "type": "array",
            "properties": {"": {"type": "string"}},
            "required": ["missing", ""],
            "additionalProperties": "no",
        }
    )

    assert result.normalized is None
    assert "'type' must be 'object'" in result.errors
    assert "'properties' keys must be non-empty strings" in result.errors
    assert "'required[1]' must be a non-empty string" in result.errors
    assert "'required' references undefined properties: missing" in result.errors
    assert "'additionalProperties' must be a boolean or mapping" in result.errors


def test_validate_tool_input_schema_warns_for_required_description() -> None:
    result = validate_tool_input_schema(
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
    )

    assert result.normalized is not None
    assert result.errors == ()
    assert result.warnings == ("required property 'query' should include a description",)


def test_validate_tool_input_schema_normalizes_property_and_required_names() -> None:
    result = validate_tool_input_schema(
        {
            "type": "object",
            "properties": {" query ": {"type": "string"}},
            "required": [" query "],
        }
    )

    assert result.normalized is not None
    assert result.errors == ()
    assert result.warnings == ("required property 'query' should include a description",)
