"""Instruction directive for opt-in built-in subagents."""

from __future__ import annotations

import re
from dataclasses import dataclass

SUBAGENT_DIRECTIVE = "fast-agent-subagents"

_DIRECTIVE_OPEN = re.compile(
    rf"(?m)^[ \t]*(?:"
    rf"{SUBAGENT_DIRECTIVE}[ \t]*(?:\r?\n|$)"
    rf"|<!--[ \t]*{SUBAGENT_DIRECTIVE}(?:"
    rf"[ \t]*(?P<single_close>--!?>)[ \t]*(?:\r?\n|$)"
    rf"|[ \t]*(?P<body_start>\r?\n)"
    rf"))"
)
_DIRECTIVE_CLOSE = re.compile(
    r"(?m)^(?P<standalone>[ \t]*--!?>[ \t]*(?:\r?\n|$))"
    r"|(?P<inline>--!?>)[ \t]*(?P<inline_line_end>\r?\n|$)"
)


@dataclass(frozen=True, slots=True)
class SubagentDirectiveResult:
    instruction: str
    subagent_instruction: str
    found: bool


def resolve_subagent_directive(instruction: str) -> SubagentDirectiveResult:
    """Project standalone directives for the parent and built-in children."""
    parent_parts: list[str] = []
    subagent_parts: list[str] = []
    cursor = 0
    found = False
    single_closes = {
        match.start("single_close")
        for match in _DIRECTIVE_OPEN.finditer(instruction)
        if match.group("single_close") is not None
    }
    close_matches = (
        match
        for match in _DIRECTIVE_CLOSE.finditer(instruction)
        if match.group("inline") is None or match.start("inline") not in single_closes
    )
    close_match = next(close_matches, None)

    for open_match in _DIRECTIVE_OPEN.finditer(instruction):
        if open_match.start() < cursor:
            continue

        block_end = open_match.end()
        body = ""
        if open_match.group("body_start") is not None:
            while close_match is not None and close_match.start() < block_end:
                close_match = next(close_matches, None)
            if close_match is None:
                continue

            body = instruction[block_end : close_match.start()]
            if close_match.group("inline") is not None:
                body = body.rstrip(" \t") + (close_match.group("inline_line_end") or "")
            block_end = close_match.end()
            close_match = next(close_matches, None)

        prefix = instruction[cursor : open_match.start()]
        parent_parts.extend((prefix, body))
        subagent_parts.append(prefix)
        cursor = block_end
        found = True

    remainder = instruction[cursor:]
    parent_parts.append(remainder)
    subagent_parts.append(remainder)
    return SubagentDirectiveResult(
        instruction="".join(parent_parts),
        subagent_instruction="".join(subagent_parts),
        found=found,
    )
