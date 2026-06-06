from __future__ import annotations

from fast_agent.tools.filesystem_tool_definitions import READ_TEXT_FILE_TOOL_NAME
from fast_agent.utils.tool_names import (
    EXECUTE_TOOL_KEYWORDS,
    EXECUTE_TOOL_NAME,
    is_read_text_file_tool_name,
    is_shell_execution_tool_name,
    matches_tool_name,
    normalize_tool_name,
)


def test_normalize_tool_name_uses_final_namespace_component() -> None:
    assert normalize_tool_name("server.execute") == "execute"
    assert normalize_tool_name("agent:shell") == "shell"
    assert normalize_tool_name("mcp/tools/bash") == "bash"
    assert normalize_tool_name("mcp/server.tools:execute") == "execute"
    assert normalize_tool_name("server__read_text_file") == "read_text_file"
    assert normalize_tool_name("mcp/server.tools:runner/execute") == "execute"


def test_normalize_tool_name_preserves_single_underscores() -> None:
    assert normalize_tool_name("read_text_file") == "read_text_file"


def test_normalize_tool_name_strips_surrounding_whitespace() -> None:
    assert normalize_tool_name(" server.execute ") == "execute"
    assert normalize_tool_name(" SERVER.EXECUTE ") == "execute"
    assert normalize_tool_name("   ") == ""


def test_normalize_tool_name_returns_empty_final_namespace_component() -> None:
    assert normalize_tool_name("server.tools/") == ""


def test_is_shell_execution_tool_name_matches_exact_execution_aliases() -> None:
    assert is_shell_execution_tool_name(EXECUTE_TOOL_NAME)
    assert is_shell_execution_tool_name("server.execute")
    assert is_shell_execution_tool_name("server__execute")
    assert is_shell_execution_tool_name("agent:shell")
    assert not is_shell_execution_tool_name("read_text_file")


def test_is_read_text_file_tool_name_matches_namespaced_and_legacy_aliases() -> None:
    assert is_read_text_file_tool_name(READ_TEXT_FILE_TOOL_NAME)
    assert is_read_text_file_tool_name(f"server.{READ_TEXT_FILE_TOOL_NAME}")
    assert is_read_text_file_tool_name(f"server__{READ_TEXT_FILE_TOOL_NAME}")
    assert is_read_text_file_tool_name(f"server.tools/foo__{READ_TEXT_FILE_TOOL_NAME}")
    assert not is_read_text_file_tool_name("write_text_file")


def test_matches_tool_name_matches_canonical_and_legacy_aliases() -> None:
    assert matches_tool_name("apply_patch", "apply_patch")
    assert matches_tool_name("server.apply_patch", "apply_patch")
    assert matches_tool_name("server__apply_patch", "apply_patch")
    assert not matches_tool_name("not_apply_patch", "apply_patch")


def test_matches_tool_name_rejects_empty_normalized_names() -> None:
    assert not matches_tool_name("server.", "")
    assert not matches_tool_name("server.", "tools/")
    assert not matches_tool_name("", "")
    assert not matches_tool_name(None, "tools/")


def test_matches_tool_name_normalizes_canonical_name() -> None:
    assert matches_tool_name("server.apply_patch", "tools/APPLY_PATCH")
    assert matches_tool_name("server__apply_patch", "tools.apply_patch")


def test_execute_tool_keywords_include_shell_command_aliases() -> None:
    assert {EXECUTE_TOOL_NAME, "run", "exec", "command", "bash", "shell"}.issubset(
        EXECUTE_TOOL_KEYWORDS
    )
