from copy import deepcopy

from openai.lib._pydantic import _ensure_strict_json_schema, to_strict_json_schema
from pydantic import BaseModel

from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.schema_sanitizer import (
    sanitize_response_format_schema,
    sanitize_tool_input_schema,
    should_strip_tool_schema_defaults,
)
from fast_agent.llm.provider_types import Provider


class StructuredSample(BaseModel):
    name: str
    count: int = 3
    tags: dict[str, str]


def test_sanitize_tool_input_schema_removes_default_recursively() -> None:
    schema = {
        "type": "object",
        "properties": {
            "seed": {
                "type": "integer",
                "description": "Seed for reproducible generation",
                "default": 42,
            },
            "nested": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "default": 1,
                    }
                },
            },
        },
    }

    sanitized = sanitize_tool_input_schema(schema)

    seed_schema = sanitized["properties"]["seed"]
    nested_count_schema = sanitized["properties"]["nested"]["properties"]["count"]

    assert "default" not in seed_schema
    assert "default" not in nested_count_schema
    assert seed_schema["type"] == "integer"


def test_sanitize_response_format_schema_requires_all_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "context": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["value"],
    }

    sanitized = sanitize_response_format_schema(schema)

    assert sanitized["required"] == ["value", "context"]
    assert sanitized["additionalProperties"] is False
    assert "default" not in sanitized["properties"]["context"]


def test_sanitize_response_format_schema_matches_openai_sdk_strictifier() -> None:
    schema = StructuredSample.model_json_schema()
    expected = deepcopy(schema)

    sanitized = sanitize_response_format_schema(schema)
    _ensure_strict_json_schema(expected, path=(), root=expected)

    assert sanitized == expected
    assert sanitized == to_strict_json_schema(StructuredSample)
    assert sanitized["properties"]["tags"]["additionalProperties"] == {"type": "string"}
    assert schema["required"] == ["name", "tags"]


def test_openai_response_format_uses_strict_schema_for_raw_structured_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string"},
            "context": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["value"],
    }

    llm = OpenAILLM(Provider.OPENAI, model="gpt-5-mini")
    response_format = llm.schema_to_response_format(schema)
    strict_schema = response_format["json_schema"]["schema"]

    assert strict_schema["required"] == ["value", "context"]
    assert strict_schema["additionalProperties"] is False
    assert "default" not in strict_schema["properties"]["context"]
    assert schema["required"] == ["value"]


def test_should_strip_tool_schema_defaults_known_kimi_variants() -> None:
    assert should_strip_tool_schema_defaults("kimi25")
    assert should_strip_tool_schema_defaults("kimi26")
    assert should_strip_tool_schema_defaults("hf.moonshotai/Kimi-K2.5:fireworks-ai")
    assert should_strip_tool_schema_defaults("hf.moonshotai/Kimi-K2.6:novita")
    assert not should_strip_tool_schema_defaults("gpt-5-mini")


def test_adjust_schema_preserves_defaults_for_models_that_support_them() -> None:
    schema = {
        "type": "object",
        "properties": {
            "seed": {
                "description": "Seed for reproducible generation",
                "default": 42,
            }
        },
    }

    llm = OpenAILLM(Provider.OPENAI, model="gpt-5-mini")
    adjusted = llm.adjust_schema(schema, model_name="gpt-5-mini")

    seed_schema = adjusted["properties"]["seed"]
    assert seed_schema.get("default") == 42


def test_adjust_schema_strips_defaults_for_kimi25_variants() -> None:
    schema = {
        "type": "object",
        "properties": {
            "seed": {
                "description": "Seed for reproducible generation",
                "default": 42,
            }
        },
    }

    llm = OpenAILLM(Provider.OPENAI, model="kimi25")
    adjusted = llm.adjust_schema(schema, model_name="kimi25")

    seed_schema = adjusted["properties"]["seed"]
    assert "default" not in seed_schema
    assert seed_schema["type"] == "integer"
