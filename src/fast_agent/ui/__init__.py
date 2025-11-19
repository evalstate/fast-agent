"""UI utilities and primitives for interactive console features.

Design goals:
- Keep import side-effects minimal to avoid circular imports.
- Make primitives easy to access with lazy attribute loading.
"""

from typing import Any

__all__ = [
    "ElicitationForm",
    "show_simple_elicitation_form",
    "form_dialog",
    "ELICITATION_STYLE",
]


def __getattr__(name: str) -> Any:
    """Lazy attribute loader to avoid importing heavy modules at package import time."""
    if name == "ELICITATION_STYLE":
        from .elicitation_style import ELICITATION_STYLE as _STYLE

        return _STYLE
    if name in ("ElicitationForm", "show_simple_elicitation_form", "form_dialog"):
        from .elicitation_form import (
            ElicitationForm as _Form,
        )
        from .elicitation_form import (
            show_simple_elicitation_form as _show,
        )

        if name == "ElicitationForm":
            return _Form
        if name == "show_simple_elicitation_form":
            return _show
        if name == "form_dialog":
            return _show
    raise AttributeError(name)
