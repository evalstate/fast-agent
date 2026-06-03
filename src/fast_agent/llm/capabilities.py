"""Helpers for reading optional LLM capabilities from typed and custom LLMs."""

from __future__ import annotations

from inspect import getattr_static
from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.interfaces import FastAgentLLMProtocol


T = TypeVar("T")


def read_capability(
    llm: "FastAgentLLMProtocol | object | None",
    attribute_name: str,
    getter: "Callable[[FastAgentLLMProtocol], T]",
    *,
    default: T,
) -> T:
    if llm is None:
        return default

    candidate = cast("FastAgentLLMProtocol", llm)
    try:
        getattr_static(candidate, attribute_name)
    except AttributeError:
        return default
    return getter(candidate)


def read_bool_capability(
    llm: "FastAgentLLMProtocol | object | None",
    attribute_name: str,
    getter: "Callable[[FastAgentLLMProtocol], bool]",
) -> bool:
    return read_capability(llm, attribute_name, getter, default=False) is True
