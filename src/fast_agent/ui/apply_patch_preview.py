from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.patch.errors import ParseError
from fast_agent.patch.parser import (
    BEGIN_PATCH_MARKER,
    END_PATCH_MARKER,
    ParseMode,
    parse_patch_text,
)
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold
from fast_agent.utils.tool_names import is_shell_execution_tool_name

DEFAULT_PATCH_PREVIEW_MAX_LINES = 120
_PREVIEW_STRIPPED_PREFIX_STYLES = (
    ("apply_patch preview:", "bold white"),
    ("*** ", "cyan"),
    ("@@", "yellow"),
    ("+", "green"),
    ("-", "red"),
)
_PREVIEW_EXACT_LINE_STYLES = {
    "other args:": "bold magenta",
}
_SHELL_SYNTAX_LANGUAGES = {
    "pwsh": "powershell",
    "powershell": "powershell",
    "cmd": "batch",
}

if TYPE_CHECKING:
    from collections.abc import Mapping

type JsonObject = dict[str, object]


@dataclass(frozen=True)
class PatchSummary:
    file_count: int
    operation_counts: dict[str, int]
    summary: str


@dataclass(frozen=True)
class ApplyPatchPreview:
    is_valid: bool
    summary: str
    rendered_patch: str
    file_count: int
    operation_counts: dict[str, int]


def is_shell_execution_tool(tool_name: str | None) -> bool:
    return is_shell_execution_tool_name(tool_name)


def _normalize_shell_executable(value: str | None) -> str:
    if not value:
        return ""
    normalized = strip_casefold(value).strip("'\"")
    if not normalized:
        return ""
    name = PureWindowsPath(normalized).name
    if name.endswith(".exe"):
        return name[:-4]
    return name


def shell_syntax_language(
    shell_name: str | None,
    *,
    shell_path: str | None = None,
) -> str:
    normalized = _normalize_shell_executable(shell_name)
    if not normalized:
        normalized = _normalize_shell_executable(shell_path)

    return _SHELL_SYNTAX_LANGUAGES.get(normalized, "bash")


def _apply_patch_begin_index(command: str) -> int | None:
    if not command or not re.search(r"\bapply_patch\b", command):
        return None

    begin_index = command.find(BEGIN_PATCH_MARKER)
    if begin_index < 0:
        return None
    return begin_index


def extract_apply_patch_text(command: str) -> str | None:
    begin_index = _apply_patch_begin_index(command)
    if begin_index is None:
        return None
    end_index = command.find(END_PATCH_MARKER, begin_index)
    if end_index < 0:
        return None

    patch_text = command[begin_index : end_index + len(END_PATCH_MARKER)].strip()
    return patch_text or None


def extract_partial_apply_patch_text(command: str) -> str | None:
    begin_index = _apply_patch_begin_index(command)
    if begin_index is None:
        return None

    end_index = command.find(END_PATCH_MARKER, begin_index)
    if end_index >= 0:
        patch_text = command[begin_index : end_index + len(END_PATCH_MARKER)]
    else:
        patch_text = command[begin_index:]
    patch_text = patch_text.strip()
    return patch_text or None


def summarize_patch(patch_text: str) -> PatchSummary | None:
    if not patch_text:
        return None

    try:
        parsed = parse_patch_text(patch_text, ParseMode.LENIENT)
    except ParseError:
        return None

    operation_counts = {"add": 0, "update": 0, "delete": 0}
    file_paths: set[str] = set()
    for hunk in parsed.hunks:
        operation_counts[hunk.kind] = operation_counts.get(hunk.kind, 0) + 1
        file_paths.add(str(hunk.path))

    file_count = len(file_paths)
    operations = ", ".join(
        format_count(operation_counts[k], k)
        for k in ("add", "update", "delete")
        if operation_counts[k]
    )
    if not operations:
        operations = "no operations"
    summary = f"apply_patch preview: {format_count(file_count, 'file')} ({operations})"

    return PatchSummary(
        file_count=file_count,
        operation_counts=operation_counts,
        summary=summary,
    )


def render_patch_preview(patch_text: str, max_lines: int | None = None) -> str:
    lines = patch_text.splitlines()
    if max_lines is None or len(lines) <= max_lines:
        return patch_text

    if max_lines <= 0:
        omitted = len(lines)
        return f"(+{format_count(omitted, 'more line')})"

    visible = lines[:max_lines]
    omitted = len(lines) - max_lines
    visible.append(f"(+{format_count(omitted, 'more line')})")
    return "\n".join(visible)


def extract_non_command_args(tool_args: Mapping[str, object]) -> JsonObject:
    return {key: value for key, value in tool_args.items() if key != "command"}


def _append_other_args(parts: list[str], other_args: Mapping[str, object] | None) -> None:
    if not other_args:
        return
    try:
        other_args_text = json.dumps(other_args, indent=2, ensure_ascii=True, sort_keys=True)
    except (TypeError, ValueError):
        other_args_text = str(dict(other_args))
    parts.append("other args:")
    parts.append(other_args_text)


def format_apply_patch_preview(
    preview: ApplyPatchPreview,
    *,
    other_args: Mapping[str, object] | None = None,
) -> str:
    parts: list[str] = [preview.summary, preview.rendered_patch]
    _append_other_args(parts, other_args)
    return "\n".join(parts)


def format_partial_apply_patch_preview(
    patch_text: str,
    *,
    other_args: Mapping[str, object] | None = None,
    max_lines: int | None = DEFAULT_PATCH_PREVIEW_MAX_LINES,
) -> str:
    preview = build_apply_patch_preview_from_input(patch_text, max_lines=max_lines)
    if preview is not None:
        return format_apply_patch_preview(preview, other_args=other_args)

    parts = [
        "apply_patch preview: streaming patch (partial)",
        render_patch_preview(patch_text, max_lines=max_lines),
    ]
    _append_other_args(parts, other_args)
    return "\n".join(parts)


def _preview_line_style(line: str) -> str | None:
    raw = line.rstrip("\n")
    if not raw:
        return None
    stripped = raw.lstrip()

    exact_style = _PREVIEW_EXACT_LINE_STYLES.get(stripped)
    if exact_style is not None:
        return exact_style

    for prefix, style in _PREVIEW_STRIPPED_PREFIX_STYLES:
        if stripped.startswith(prefix):
            return style

    if stripped.startswith("(+") and (
        stripped.endswith("more line)") or stripped.endswith("more lines)")
    ):
        return "dim"
    if raw.startswith(" "):
        return "dim"
    return None


def style_apply_patch_preview_text(
    text: str,
    *,
    default_style: str | None = "dim",
) -> Text:
    styled = Text()
    for line in text.splitlines(keepends=True):
        style = _preview_line_style(line)
        styled.append(line, style=style or default_style)
    return styled


def build_apply_patch_preview_from_input(
    patch_text: str,
    *,
    max_lines: int | None = DEFAULT_PATCH_PREVIEW_MAX_LINES,
) -> ApplyPatchPreview | None:
    if not patch_text:
        return None

    patch_summary = summarize_patch(patch_text)
    if patch_summary is None:
        return None

    rendered_patch = render_patch_preview(patch_text, max_lines=max_lines)
    return ApplyPatchPreview(
        is_valid=True,
        summary=patch_summary.summary,
        rendered_patch=rendered_patch,
        file_count=patch_summary.file_count,
        operation_counts=patch_summary.operation_counts,
    )


def build_apply_patch_preview(
    command: str,
    *,
    max_lines: int | None = DEFAULT_PATCH_PREVIEW_MAX_LINES,
) -> ApplyPatchPreview | None:
    patch_text = extract_apply_patch_text(command)
    if patch_text is None:
        return None

    return build_apply_patch_preview_from_input(patch_text, max_lines=max_lines)


def build_partial_apply_patch_preview(
    command: str,
    *,
    other_args: Mapping[str, object] | None = None,
    max_lines: int | None = DEFAULT_PATCH_PREVIEW_MAX_LINES,
) -> str | None:
    patch_text = extract_partial_apply_patch_text(command)
    if patch_text is None:
        return None
    return format_partial_apply_patch_preview(
        patch_text,
        other_args=other_args,
        max_lines=max_lines,
    )
