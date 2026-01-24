"""
Errors for the openai/codex apply_patch port (Apache 2.0).
"""

from __future__ import annotations

from dataclasses import dataclass


class ApplyPatchError(Exception):
    """Base error for apply_patch failures."""


class ParseError(ApplyPatchError):
    """Base error for parsing failures."""


@dataclass(frozen=True)
class InvalidPatchError(ParseError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class InvalidHunkError(ParseError):
    message: str
    line_number: int

    def __str__(self) -> str:
        return f"invalid hunk at line {self.line_number}, {self.message}"


@dataclass(frozen=True)
class IoError(ApplyPatchError):
    context: str
    source: str

    def __str__(self) -> str:
        return f"{self.context}: {self.source}"
