"""Turn-scoped input handoff from interactive surfaces to the tool loop.

The callback belongs to one agent instance. Delegated agents inherit context but
must not consume input addressed to their parent. Non-interactive runs have no
source and need no terminal/UI dependencies.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class _InputSource:
    agent: object
    take_messages: Callable[[], list[str]]


_source: ContextVar[_InputSource | None] = ContextVar("queued_input_source", default=None)


@contextmanager
def queued_input_source(agent: object, take_messages: Callable[[], list[str]]) -> Generator[None]:
    token = _source.set(_InputSource(agent, take_messages))
    try:
        yield
    finally:
        _source.reset(token)


def take_queued_messages(agent: object) -> list[str]:
    source = _source.get()
    if source is None or source.agent is not agent:
        return []
    return source.take_messages()
