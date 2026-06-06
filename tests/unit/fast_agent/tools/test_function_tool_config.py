import pytest

from fast_agent.tools.function_tool_config import (
    FunctionToolSpec,
    function_tool_entrypoint,
    parse_function_tool_card_entry,
    serialize_function_tools,
)


def test_parse_function_tool_card_entry_normalizes_string_entrypoint() -> None:
    assert (
        parse_function_tool_card_entry("  tools.py:search  ", field_path="function_tools[0]")
        == "tools.py:search"
    )


def test_parse_function_tool_card_entry_rejects_blank_string_entrypoint() -> None:
    with pytest.raises(ValueError, match="'function_tools\\[0\\]' entries must be non-empty"):
        parse_function_tool_card_entry("   ", field_path="function_tools[0]")


def test_parse_function_tool_card_entry_normalizes_object_entrypoint() -> None:
    spec = parse_function_tool_card_entry(
        {"entrypoint": "  tools.py:search  ", "variant": "code"},
        field_path="function_tools[0]",
    )

    assert isinstance(spec, FunctionToolSpec)
    assert spec.entrypoint == "tools.py:search"
    assert spec.code_arg == "code"
    assert spec.language == "python"


def test_function_tool_entrypoint_normalizes_raw_values() -> None:
    assert function_tool_entrypoint("  tools.py:search  ") == "tools.py:search"
    assert function_tool_entrypoint("   ") is None
    assert function_tool_entrypoint({"entrypoint": "  tools.py:search  "}) == "tools.py:search"


def test_serialize_function_tools_normalizes_and_omits_blank_entrypoints() -> None:
    assert serialize_function_tools("  tools.py:search  ") == ["tools.py:search"]
    assert serialize_function_tools("   ") is None
    assert serialize_function_tools(
        [
            {"entrypoint": "  tools.py:search  ", "variant": "code"},
            {"entrypoint": "   ", "variant": "code"},
        ]
    ) == [{"entrypoint": "tools.py:search", "variant": "code"}]
