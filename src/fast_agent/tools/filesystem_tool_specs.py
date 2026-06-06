"""Shared table helpers for filesystem runtime tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.types import CallToolResult, Tool

FilesystemToolHandler = Callable[[dict[str, Any] | None, str | None], Awaitable["CallToolResult"]]


@dataclass(frozen=True, slots=True)
class FilesystemToolSpec:
    name: str
    enabled: Callable[[], bool]
    tool: Callable[[], Tool]
    handler: FilesystemToolHandler


def enabled_tool_specs(specs: Iterable[FilesystemToolSpec]) -> tuple[FilesystemToolSpec, ...]:
    return tuple(spec for spec in specs if spec.enabled())


def enabled_tool_spec(
    specs: Iterable[FilesystemToolSpec],
    tool_name: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> FilesystemToolSpec | None:
    resolved_name = aliases.get(tool_name, tool_name) if aliases else tool_name
    return next(
        (spec for spec in specs if spec.name == resolved_name and spec.enabled()),
        None,
    )
