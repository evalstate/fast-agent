"""Concise unified-diff previews for ``edit_file`` tool calls."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import unified_diff
from typing import TYPE_CHECKING

from fast_agent.ui.apply_patch_preview import DEFAULT_PATCH_PREVIEW_MAX_LINES, render_patch_preview

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class EditFilePreview:
    path: str
    old_string: str
    new_string: str
    replace_all: bool
    partial: bool = False


def build_edit_file_preview(arguments: Mapping[str, object]) -> EditFilePreview | None:
    path = arguments.get("path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    replace_all = arguments.get("replace_all", False)
    if (
        not isinstance(path, str)
        or not path.strip()
        or not isinstance(old_string, str)
        or not old_string
        or not isinstance(new_string, str)
        or not isinstance(replace_all, bool)
    ):
        return None
    return EditFilePreview(
        path=path,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )


def build_partial_edit_file_preview(
    *,
    path: str | None,
    old_string: str | None,
    new_string: str | None,
) -> EditFilePreview | None:
    if not path or not old_string or new_string is None:
        return None
    return EditFilePreview(
        path=path,
        old_string=old_string,
        new_string=new_string,
        replace_all=False,
        partial=True,
    )


def format_edit_file_preview(
    preview: EditFilePreview,
    *,
    max_lines: int | None = DEFAULT_PATCH_PREVIEW_MAX_LINES,
) -> str:
    detail = " (all matches)" if preview.replace_all else ""
    if preview.partial:
        detail += " (partial)"
    diff_lines = unified_diff(
        preview.old_string.splitlines(),
        preview.new_string.splitlines(),
        fromfile=preview.path,
        tofile=preview.path,
        n=0,
        lineterm="",
    )
    diff_text = "\n".join(diff_lines)
    return "\n".join(
        (
            f"edit_file preview: {preview.path}{detail}",
            render_patch_preview(diff_text, max_lines=max_lines),
        )
    )
