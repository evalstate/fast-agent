from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from rich.text import Text

from fast_agent.patch.parser import (
    BEGIN_PATCH_MARKER,
    END_PATCH_MARKER,
    ParseMode,
    parse_patch_text,
)

DEFAULT_PATCH_PREVIEW_MAX_LINES = 120
_SHELL_TOOL_ALIASES = frozenset({"execute", "bash", "shell"})


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


def normalize_tool_name(tool_name: str | None) -> str:
    if not tool_name:
        return ""
    normalized = tool_name.lower()
    for sep in ("/", ".", ":"):
        if sep in normalized:
            normalized = normalized.rsplit(sep, 1)[-1]
    return normalized


def is_shell_execution_tool(tool_name: str | None) -> bool:
    return normalize_tool_name(tool_name) in _SHELL_TOOL_ALIASES


def shell_syntax_language(
    shell_name: str | None,
    *,
    shell_path: str | None = None,
) -> str:
    normalized = normalize_tool_name(shell_name)
    if not normalized and shell_path:
        normalized = normalize_tool_name(shell_path)

    if normalized in {"pwsh", "powershell"}:
        return "powershell"
    if normalized == "cmd":
        return "batch"
    return "bash"


def extract_apply_patch_text(command: str) -> str | None:
    if not command or not re.search(r"\bapply_patch\b", command):
        return None

    begin_index = command.find(BEGIN_PATCH_MARKER)
    if begin_index < 0:
        return None

    end_index = command.find(END_PATCH_MARKER, begin_index)
    if end_index < 0:
        return None

    patch_text = command[begin_index : end_index + len(END_PATCH_MARKER)].strip()
    return patch_text or None


def extract_partial_apply_patch_text(command: str) -> str | None:
    if not command or not re.search(r"\bapply_patch\b", command):
        return None

    begin_index = command.find(BEGIN_PATCH_MARKER)
    if begin_index < 0:
        return None

    end_index = command.find(END_PATCH_MARKER, begin_index)
    if end_index >= 0:
        patch_text = command[begin_index : end_index + len(END_PATCH_MARKER)]
    else:
        patch_text = command[begin_index:]
    patch_text = patch_text.strip()
    return patch_text or None


def _summary_noun(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def summarize_patch(patch_text: str) -> PatchSummary | None:
    if not patch_text:
        return None

    try:
        parsed = parse_patch_text(patch_text, ParseMode.LENIENT)
    except Exception:
        return None

    operation_counts = {"add": 0, "update": 0, "delete": 0}
    file_paths: set[str] = set()
    for hunk in parsed.hunks:
        operation_counts[hunk.kind] = operation_counts.get(hunk.kind, 0) + 1
        file_paths.add(str(hunk.path))

    file_count = len(file_paths)
    operations = ", ".join(
        _summary_noun(operation_counts[k], k) for k in ("add", "update", "delete") if operation_counts[k]
    )
    if not operations:
        operations = "no operations"
    summary = f"apply_patch preview: {_summary_noun(file_count, 'file')} ({operations})"

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
        return f"(+{omitted} more lines)"

    visible = lines[:max_lines]
    omitted = len(lines) - max_lines
    visible.append(f"(+{omitted} more lines)")
    return "\n".join(visible)


def extract_non_command_args(tool_args: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in tool_args.items() if key != "command"}


def _append_other_args(parts: list[str], other_args: Mapping[str, Any] | None) -> None:
    if not other_args:
        return
    try:
        other_args_text = json.dumps(other_args, indent=2, ensure_ascii=True, sort_keys=True)
    except Exception:
        other_args_text = str(dict(other_args))
    parts.append("other args:")
    parts.append(other_args_text)


def format_apply_patch_preview(
    preview: ApplyPatchPreview,
    *,
    other_args: Mapping[str, Any] | None = None,
) -> str:
    parts: list[str] = [preview.summary, preview.rendered_patch]
    _append_other_args(parts, other_args)
    return "\n".join(parts)


def format_partial_apply_patch_preview(
    patch_text: str,
    *,
    other_args: Mapping[str, Any] | None = None,
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
    if stripped.startswith("apply_patch preview:"):
        return "bold white"
    if stripped == "other args:":
        return "bold magenta"
    if stripped.startswith("*** "):
        return "cyan"
    if stripped.startswith("@@"):
        return "yellow"
    if stripped.startswith("+"):
        return "green"
    if stripped.startswith("-"):
        return "red"
    if stripped.startswith("(+") and stripped.endswith("more lines)"):
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
    other_args: Mapping[str, Any] | None = None,
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
