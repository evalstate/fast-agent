"""Shared tool-name normalization and alias sets."""

from __future__ import annotations

from fast_agent.tools.filesystem_tool_definitions import READ_TEXT_FILE_TOOL_NAME
from fast_agent.utils.action_normalization import normalize_action_token

EXECUTE_TOOL_NAME = "execute"
SHELL_EXECUTION_TOOL_NAMES = frozenset({EXECUTE_TOOL_NAME, "bash", "shell"})
SHELL_BUILTIN_TOOL_NAMES = frozenset(
    {
        "bash",
        "zsh",
        "sh",
        "pwsh",
        "powershell",
        "cmd",
        "shell",
    }
)
EXECUTE_TOOL_KEYWORDS = (EXECUTE_TOOL_NAME, "run", "exec", "command", "bash", "shell")
TOOL_NAME_NAMESPACE_SEPARATORS = ("/", ".", ":", "__")


def _last_namespace_separator(tool_name: str) -> tuple[int, str]:
    return max(
        ((tool_name.rfind(separator), separator) for separator in TOOL_NAME_NAMESPACE_SEPARATORS),
        key=lambda item: item[0],
    )


def normalize_tool_name(tool_name: str | None) -> str:
    """Normalize MCP-style namespaced tool names to their final component."""
    normalized = normalize_action_token(tool_name)
    if not normalized:
        return ""
    separator_index, separator = _last_namespace_separator(normalized)
    if separator_index < 0:
        return normalized
    return normalized[separator_index + len(separator) :]


def is_shell_execution_tool_name(tool_name: str | None) -> bool:
    return normalize_tool_name(tool_name) in SHELL_EXECUTION_TOOL_NAMES


def matches_tool_name(tool_name: str | None, canonical_name: str) -> bool:
    normalized = normalize_tool_name(tool_name)
    canonical = normalize_tool_name(canonical_name)
    return bool(normalized and canonical and normalized == canonical)


def is_read_text_file_tool_name(tool_name: str | None) -> bool:
    return matches_tool_name(tool_name, READ_TEXT_FILE_TOOL_NAME)
