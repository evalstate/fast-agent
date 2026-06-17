from mcp.types import Tool

from fast_agent.tools.apply_patch_tool import (
    OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY,
    build_apply_patch_tool,
    extract_apply_patch_input,
    get_openai_responses_custom_tool_payload,
)


def test_extract_apply_patch_input_requires_non_empty_string() -> None:
    assert extract_apply_patch_input({"input": "  *** Begin Patch\n*** End Patch\n  "}) == (
        "*** Begin Patch\n*** End Patch"
    )
    assert extract_apply_patch_input(None) is None
    assert extract_apply_patch_input("not a mapping") is None  # ty: ignore[invalid-argument-type]
    assert extract_apply_patch_input({}) is None
    assert extract_apply_patch_input({"input": "   "}) is None
    assert extract_apply_patch_input({"input": 1}) is None


def test_get_openai_responses_custom_tool_payload_accepts_valid_meta() -> None:
    payload = get_openai_responses_custom_tool_payload(build_apply_patch_tool())

    assert payload is not None
    assert payload["type"] == "custom"
    assert payload["name"] == "apply_patch"
    assert payload["format"]["type"] == "grammar"
    assert payload["format"]["syntax"] == "lark"
    assert "*** Begin Patch" in payload["format"]["definition"]


def test_get_openai_responses_custom_tool_payload_rejects_invalid_meta() -> None:
    invalid_tool = Tool.model_validate(
        {
            "name": "bad_custom",
            "description": "invalid custom metadata",
            "inputSchema": {"type": "object"},
            "_meta": {
                OPENAI_RESPONSES_CUSTOM_TOOL_META_KEY: {
                    "type": "custom",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": 42,
                    },
                }
            },
        }
    )

    assert get_openai_responses_custom_tool_payload(invalid_tool) is None
