"""Instruction directive for opt-in built-in subagents."""

from __future__ import annotations

import re
from dataclasses import dataclass

SUBAGENT_DIRECTIVE = "fast-agent-subagents"

_DIRECTIVE_BLOCK = re.compile(
    rf"(?ms)^[ \t]*(?:"
    rf"{SUBAGENT_DIRECTIVE}[ \t]*(?:\r?\n|$)"
    rf"|<!--[ \t]*{SUBAGENT_DIRECTIVE}(?:"
    rf"[ \t]*-->[ \t]*(?:\r?\n|$)"
    rf"|[ \t]*\r?\n(?P<body>.*?)(?=^[ \t]*-->[ \t]*(?:\r?\n|$))"
    rf"^[ \t]*-->[ \t]*(?:\r?\n|$)"
    rf"))"
)


@dataclass(frozen=True, slots=True)
class SubagentDirectiveResult:
    instruction: str
    subagent_instruction: str
    found: bool


def resolve_subagent_directive(instruction: str) -> SubagentDirectiveResult:
    """Project standalone directives for the parent and built-in children."""
    resolved, count = _DIRECTIVE_BLOCK.subn(
        lambda match: match.group("body") or "",
        instruction,
    )
    subagent_instruction = _DIRECTIVE_BLOCK.sub("", instruction)
    return SubagentDirectiveResult(
        instruction=resolved,
        subagent_instruction=subagent_instruction,
        found=count > 0,
    )
