from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator

from fast_agent.llm.provider.openai.schema_sanitizer import sanitize_tool_input_schema


@pytest.mark.parametrize("mapping", ["properties", "$defs", "definitions", "patternProperties"])
def test_default_named_schema_entries_are_preserved(mapping: str) -> None:
    schema = {
        "type": "object",
        mapping: {"default": {"type": "integer", "default": 3}},
    }
    if mapping == "properties":
        schema["required"] = ["default"]
        schema["additionalProperties"] = False
    original = deepcopy(schema)

    sanitized = sanitize_tool_input_schema(schema)

    assert sanitized[mapping]["default"] == {"type": "integer"}
    assert schema == original
    if mapping == "properties":
        Draft202012Validator(sanitized).validate({"default": 7})


@pytest.mark.parametrize("keyword", ["const", "enum", "examples"])
def test_instance_data_keeps_literal_default_keys(keyword: str) -> None:
    value = {"default": "literal", "nested": {"default": 3}}
    schema = {"type": "object", keyword: value if keyword == "const" else [value]}

    sanitized = sanitize_tool_input_schema(schema)

    Draft202012Validator(sanitized).validate(value)
    assert sanitized[keyword] == schema[keyword]
