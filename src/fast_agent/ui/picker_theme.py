from __future__ import annotations

from prompt_toolkit.styles import Style

PICKER_STYLE_MAP: dict[str, str] = {
    "selected": "reverse",
    "active": "ansigreen",
    "attention": "ansiyellow",
    "inactive": "ansibrightblack",
    "muted": "ansibrightblack",
    "focus": "ansicyan",
}


def build_picker_style() -> Style:
    """Return the shared picker style used by interactive selection UIs."""

    return Style.from_dict(PICKER_STYLE_MAP)
