"""Prompt UI shared state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit.history import InMemoryHistory


@dataclass
class PromptUiState:
    """Mutable state used by interactive prompt UI helpers."""

    agent_histories: dict[str, "InMemoryHistory"] = field(default_factory=dict)
    available_agents: set[str] = field(default_factory=set)
    in_multiline_mode: bool = False
    last_copyable_output: str | None = None
    copy_notice: str | None = None
    copy_notice_until: float = 0.0
    startup_notices: list[str] = field(default_factory=list)
    help_message_shown: bool = False


DEFAULT_PROMPT_UI_STATE = PromptUiState()
