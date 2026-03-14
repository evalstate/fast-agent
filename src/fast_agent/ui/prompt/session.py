"""Compatibility wrapper for prompt input helpers.

Prefer ``fast_agent.ui.prompt.input`` for prompt-toolkit input orchestration.
``session`` is reserved for persisted chat/thread session concepts.
"""

from fast_agent.ui.prompt import input as _input

__all__ = getattr(_input, "__all__", [])


def __getattr__(name: str):
    return getattr(_input, name)
