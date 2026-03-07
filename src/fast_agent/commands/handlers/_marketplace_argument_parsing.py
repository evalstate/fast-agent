"""Shared argument parsers for marketplace-style command handlers."""

from __future__ import annotations

import shlex


def parse_update_argument(argument: str | None) -> tuple[str | None, bool, bool, str | None]:
    """Parse update command arguments into selector/flags/error tuple."""
    if argument is None:
        return None, False, False, None

    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        return None, False, False, f"Invalid update arguments: {exc}"

    selector: str | None = None
    force = False
    yes = False
    for token in tokens:
        if token == "--force":
            force = True
            continue
        if token == "--yes":
            yes = True
            continue
        if token.startswith("--"):
            return None, False, False, f"Unknown option: {token}"
        if selector is not None:
            return None, False, False, "Only one selector is allowed."
        selector = token

    return selector, force, yes, None
