"""Wizard stage definitions and state management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class WizardStage(Enum):
    """Setup wizard stages."""

    WELCOME = "welcome"
    TOKEN_CHECK = "token_check"
    TOKEN_GUIDE = "token_guide"
    TOKEN_VERIFY = "token_verify"
    MODEL_SELECT = "model_select"
    MCP_CONNECT = "mcp_connect"
    CONFIRM = "confirm"
    COMPLETE = "complete"


@dataclass
class WizardState:
    """Tracks wizard progress and collected data."""

    stage: WizardStage = WizardStage.WELCOME
    token_verified: bool = False
    hf_username: str | None = None
    selected_model: str | None = None
    selected_model_display: str | None = None
    mcp_load_on_start: bool = False
    error_message: str | None = None
    # Track if this is the first message (to show welcome)
    first_message: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "stage": self.stage.value,
            "token_verified": self.token_verified,
            "hf_username": self.hf_username,
            "selected_model": self.selected_model,
            "selected_model_display": self.selected_model_display,
            "mcp_load_on_start": self.mcp_load_on_start,
            "error_message": self.error_message,
        }

    def clear_error(self) -> None:
        """Clear any error message."""
        self.error_message = None
