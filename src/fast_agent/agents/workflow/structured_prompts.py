"""Helpers for workflow agents that re-parse generated text as structured output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.core.prompt import Prompt

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


def structured_reparse_prompt(response_text: str, *, source: str) -> "PromptMessageExtended":
    """Build a prompt asking a child agent to coerce workflow output to a schema."""
    return Prompt.user(
        f"Convert this {source} response to the requested structured schema:\n\n"
        f"{response_text}"
    )
