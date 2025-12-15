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
    error_message: str | None = None
    # Track if this is the first message (to show welcome)
    first_message: bool = True
    # Track if we're waiting for login tool result
    awaiting_login_result: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "stage": self.stage.value,
            "token_verified": self.token_verified,
            "hf_username": self.hf_username,
            "selected_model": self.selected_model,
            "selected_model_display": self.selected_model_display,
            "error_message": self.error_message,
        }

    def clear_error(self) -> None:
        """Clear any error message."""
        self.error_message = None
