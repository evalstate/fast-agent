"""Utilities for escaping template placeholders."""

from __future__ import annotations

_ESCAPED_OPEN = "__FAST_AGENT_ESCAPED_OPEN__"
_ESCAPED_CLOSE = "__FAST_AGENT_ESCAPED_CLOSE__"


def protect_escaped_braces(text: str) -> str:
    """Protect escaped braces so template substitution ignores them.

    Use ``\\{{`` or ``\\}}`` in templates to render literal braces.
    """
    return text.replace(r"\{{", _ESCAPED_OPEN).replace(r"\}}", _ESCAPED_CLOSE)


def restore_escaped_braces(text: str, *, keep_escape: bool = False) -> str:
    """Restore escaped braces after template substitution."""
    if keep_escape:
        return text.replace(_ESCAPED_OPEN, r"\{{").replace(_ESCAPED_CLOSE, r"\}}")
    return text.replace(_ESCAPED_OPEN, "{{").replace(_ESCAPED_CLOSE, "}}")
