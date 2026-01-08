"""Shared helpers for MCP transport error handling."""

from __future__ import annotations


def wrap_transport_error(message: str, exc: Exception) -> Exception:
    wrapped = RuntimeError(f"{message} Cause: {exc}")
    wrapped.__cause__ = exc
    return wrapped
