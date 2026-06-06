"""UI utilities and primitives for interactive console features.

Design goals:
- Keep import side-effects minimal to avoid circular imports.
- Make primitives easy to access with lazy attribute loading.
"""

from importlib import import_module
from typing import Any, NamedTuple

__all__ = [
    "ELICITATION_STYLE",
    "ElicitationForm",
    "form_dialog",
    "show_simple_elicitation_form",
]


class _LazyExport(NamedTuple):
    module: str
    attribute: str


_LAZY_EXPORTS = {
    "ELICITATION_STYLE": _LazyExport("fast_agent.ui.elicitation_style", "ELICITATION_STYLE"),
    "ElicitationForm": _LazyExport("fast_agent.ui.elicitation_form", "ElicitationForm"),
    "show_simple_elicitation_form": _LazyExport(
        "fast_agent.ui.elicitation_form",
        "show_simple_elicitation_form",
    ),
    "form_dialog": _LazyExport(
        "fast_agent.ui.elicitation_form",
        "show_simple_elicitation_form",
    ),
}


def __getattr__(name: str) -> Any:
    """Lazy attribute loader to avoid importing heavy modules at package import time."""
    if lazy_export := _LAZY_EXPORTS.get(name):
        return getattr(import_module(lazy_export.module), lazy_export.attribute)
    raise AttributeError(name)
