"""Shared argument parsing helpers for filesystem tools."""

from __future__ import annotations

import errno
from dataclasses import dataclass
from typing import Any

from fast_agent.utils.numeric import positive_int_or_none


@dataclass(frozen=True, slots=True)
class ReadTextFileArguments:
    payload: dict[str, Any]
    path: str
    line: int | None
    limit: int | None


@dataclass(frozen=True, slots=True)
class WriteTextFileArguments:
    payload: dict[str, Any]
    path: str
    content: str


_PERMISSION_ERRNOS = {errno.EACCES, errno.EPERM}


def is_permission_error(exc: OSError) -> bool:
    """Return whether an OS error represents a filesystem permission denial."""
    return isinstance(exc, PermissionError) or exc.errno in _PERMISSION_ERRNOS


def permission_denied_message(path: str) -> str:
    return f"Permission denied for file: {path}."


def coerce_tool_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """Return a tool argument payload or raise a user-facing ValueError."""
    if not isinstance(arguments, dict):
        raise ValueError("Error: arguments must be a dict")
    return arguments


def coerce_positive_int_argument(value: Any, field: str) -> int | None:
    """Return a positive integer argument or raise a user-facing ValueError."""
    if value is None:
        return None
    parsed = positive_int_or_none(value)
    if parsed is None:
        raise ValueError(
            f"Error: '{field}' argument must be an integer greater than or equal to 1"
        )
    return parsed


def coerce_required_string_argument(
    value: Any,
    field: str,
    *,
    allow_empty: bool = False,
    strip: bool = False,
) -> str:
    """Return a required string argument or raise a user-facing ValueError."""
    if not isinstance(value, str):
        raise ValueError(f"Error: '{field}' argument is required and must be a string")

    resolved = value.strip() if strip else value
    if not allow_empty and not resolved:
        raise ValueError(f"Error: '{field}' argument is required and must be a string")
    return resolved


def coerce_optional_string_argument(
    value: Any,
    field: str,
    *,
    include_argument_word: bool = True,
    empty_as_none: bool = False,
    strip: bool = False,
) -> str | None:
    """Return an optional string argument or raise a user-facing ValueError."""
    if value is None:
        return None
    if not isinstance(value, str):
        subject = f"'{field}' argument" if include_argument_word else f"'{field}'"
        raise ValueError(f"Error: {subject} must be a string")
    resolved = value.strip() if strip else value
    if empty_as_none and not resolved:
        return None
    return resolved


def parse_read_text_file_arguments(
    arguments: dict[str, Any] | None,
) -> ReadTextFileArguments:
    """Parse shared read_text_file arguments for ACP and local runtimes."""
    payload = coerce_tool_arguments(arguments)
    return ReadTextFileArguments(
        payload=payload,
        path=coerce_required_string_argument(payload.get("path"), "path", strip=True),
        line=coerce_positive_int_argument(payload.get("line"), "line"),
        limit=coerce_positive_int_argument(payload.get("limit"), "limit"),
    )


def parse_write_text_file_arguments(
    arguments: dict[str, Any] | None,
) -> WriteTextFileArguments:
    """Parse shared write_text_file arguments for ACP and local runtimes."""
    payload = coerce_tool_arguments(arguments)
    return WriteTextFileArguments(
        payload=payload,
        path=coerce_required_string_argument(payload.get("path"), "path", strip=True),
        content=coerce_required_string_argument(
            payload.get("content"),
            "content",
            allow_empty=True,
        ),
    )
