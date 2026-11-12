"""Minimal release-version comparison (``X.Y.Z`` prefixes only)."""

from __future__ import annotations

import operator
import re

_RELEASE = re.compile(r"^\s*(\d+(?:\.\d+)*)")
_CLAUSE = re.compile(r"\s*(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)\s*")
_OPERATORS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
}


def release_tuple(version: str) -> tuple[int, ...] | None:
    """Leading numeric release segments of ``version``; ``None`` when absent."""
    match = _RELEASE.match(version)
    return tuple(int(part) for part in match.group(1).split(".")) if match else None


def compare_releases(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    width = max(len(left), len(right))
    padded_left = left + (0,) * (width - len(left))
    padded_right = right + (0,) * (width - len(right))
    return (padded_left > padded_right) - (padded_left < padded_right)


def requirement_satisfied(requirement: str, version: str) -> bool:
    """Check ``version`` against comma-separated clauses such as ``">=0.10.2,<0.12"``.

    Unparseable requirements are treated as unsatisfied so nothing is offered on a guess.
    """
    current = release_tuple(version)
    if current is None:
        return False
    for clause in requirement.split(","):
        match = _CLAUSE.fullmatch(clause)
        if match is None:
            return False
        op, bound = match.groups()
        bound_tuple = tuple(int(part) for part in bound.split("."))
        if not _OPERATORS[op](compare_releases(current, bound_tuple), 0):
            return False
    return True
