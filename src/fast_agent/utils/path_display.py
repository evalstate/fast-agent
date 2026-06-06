"""Shared path display helpers."""

from __future__ import annotations

import os
from pathlib import Path

from fast_agent.utils.text import strip_to_none


def format_relative_path(path: Path, *, cwd: Path | None = None) -> str:
    base = cwd or Path.cwd()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def format_home_relative_path(path: Path | str, *, home: Path | None = None) -> str:
    home_dir = (home or Path.home()).expanduser().resolve()
    expanded = Path(path).expanduser()
    try:
        relative = expanded.resolve().relative_to(home_dir)
    except ValueError:
        return str(expanded)
    if relative.as_posix() == ".":
        return "~"
    return f"~/{relative.as_posix()}"


def format_working_directory(path: Path, *, cwd: Path | None = None) -> str:
    base = cwd or Path.cwd()
    display = format_relative_path(path, cwd=base)
    if display != ".":
        return display
    return _format_current_directory_name(base)


def _format_current_directory_name(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    if len(parts) == 1:
        return parts[0]
    return str(path)


def left_truncate_with_ellipsis(text: str, max_length: int, *, ellipsis: str = "…") -> str:
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    suffix_length = max_length - len(ellipsis)
    if suffix_length <= 0:
        return ellipsis[:max_length]
    return f"{ellipsis}{text[-suffix_length:]}"


def _display_path(path_text: str) -> Path:
    normalized = os.path.normpath(path_text.replace("\\", "/"))
    return Path(normalized)


def format_parent_current_path(path_text: str) -> str:
    normalized_text = strip_to_none(path_text)
    if normalized_text is None:
        return ""

    path = _display_path(normalized_text)
    normalized = str(path)
    current = path.name or normalized
    parent = path.parent.name
    if parent:
        return f"{parent}/{current}"
    return current


def fit_path_for_display(path_text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""

    normalized_text = strip_to_none(path_text)
    if normalized_text is None:
        return ""

    compact = format_parent_current_path(normalized_text)
    if len(compact) <= max_length:
        return compact

    current = _display_path(normalized_text).name or normalized_text
    if len(current) <= max_length:
        return current

    return left_truncate_with_ellipsis(current, max_length)
