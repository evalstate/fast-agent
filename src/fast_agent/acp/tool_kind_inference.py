"""ACP tool-kind inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fast_agent.utils.text import strip_casefold
from fast_agent.utils.tool_names import EXECUTE_TOOL_KEYWORDS, EXECUTE_TOOL_NAME

if TYPE_CHECKING:
    from acp.schema import ToolKind


@dataclass(frozen=True, slots=True)
class ToolKindKeywordGroup:
    kind: "ToolKind"
    keywords: tuple[str, ...]


TOOL_KIND_KEYWORD_GROUPS: tuple[ToolKindKeywordGroup, ...] = (
    ToolKindKeywordGroup("fetch", ("fetch", "download", "http", "request", "curl")),
    ToolKindKeywordGroup("read", ("read", "get", "list", "show", "cat")),
    ToolKindKeywordGroup("edit", ("write", "edit", "update", "modify", "patch", "create")),
    ToolKindKeywordGroup("delete", ("delete", "remove", "clear", "clean", "rm")),
    ToolKindKeywordGroup("move", ("move", "rename", "mv", "copy", "cp")),
    ToolKindKeywordGroup("search", ("search", "find", "query", "grep", "locate")),
    ToolKindKeywordGroup(EXECUTE_TOOL_NAME, EXECUTE_TOOL_KEYWORDS),
    ToolKindKeywordGroup("think", ("think", "plan", "reason", "analyze")),
)

_WORD_PATTERN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+")


def _tool_name_words(tool_name: str) -> set[str]:
    parts = re.split(r"[^A-Za-z0-9]+", tool_name)
    words: set[str] = set()
    for part in parts:
        if not part:
            continue
        matches = _WORD_PATTERN.findall(part)
        if matches:
            words.update(strip_casefold(match) for match in matches)
        else:
            words.add(strip_casefold(part))
    return words


def infer_tool_kind(tool_name: str, arguments: dict[str, Any] | None = None) -> ToolKind:
    del arguments

    words = _tool_name_words(tool_name)
    for group in TOOL_KIND_KEYWORD_GROUPS:
        if any(keyword in words for keyword in group.keywords):
            return group.kind

    return "other"
