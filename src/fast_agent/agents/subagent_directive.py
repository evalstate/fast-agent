"""Instruction directive for opt-in built-in subagents."""

from __future__ import annotations

import re
from dataclasses import dataclass

SUBAGENT_DIRECTIVE = "fast-agent-subagents"

_DIRECTIVE_LINE = re.compile(
    rf"(?m)^[ \t]*(?:{SUBAGENT_DIRECTIVE}|<!--\s*{SUBAGENT_DIRECTIVE}\s*-->)[ \t]*(?:\r?\n|$)"
)


@dataclass(frozen=True, slots=True)
class SubagentDirectiveResult:
    instruction: str
    found: bool


def resolve_subagent_directive(instruction: str) -> SubagentDirectiveResult:
    """Remove exact standalone directives and report whether one was present."""
    resolved, count = _DIRECTIVE_LINE.subn("", instruction)
    return SubagentDirectiveResult(instruction=resolved, found=count > 0)
