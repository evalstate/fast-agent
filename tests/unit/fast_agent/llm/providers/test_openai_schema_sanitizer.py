from fast_agent.llm.provider.openai.llm_openai import OpenAILLM
from fast_agent.llm.provider.openai.schema_sanitizer import (
    sanitize_tool_input_schema,
    should_strip_tool_schema_defaults,
)
from fast_agent.llm.provider_types import Provider


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


def test_should_strip_tool_schema_defaults_known_kimi_variants() -> None:
    assert should_strip_tool_schema_defaults("kimi25")
    assert should_strip_tool_schema_defaults("hf.moonshotai/Kimi-K2.5:fireworks-ai")
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
