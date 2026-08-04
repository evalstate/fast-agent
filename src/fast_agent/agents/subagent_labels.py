"""Display labels for built-in subagent invocations."""

from __future__ import annotations

from random import choice
from re import fullmatch
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, StringConstraints

if TYPE_CHECKING:
    from collections.abc import Callable

_ADJECTIVES = ("brisk", "calm", "clever", "eager", "gentle", "swift")
_ANIMALS = ("badger", "falcon", "otter", "panda", "raven", "tiger")
_LABEL_PATTERN = r"^[A-Za-z0-9](?:[A-Za-z0-9 _-]*[A-Za-z0-9])?$"


def _strip_label(value: object) -> object:
    return value.strip() if isinstance(value, str) else value


type SubagentLabel = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=32,
        pattern=_LABEL_PATTERN,
    ),
    BeforeValidator(_strip_label),
]


def generate_subagent_label() -> str:
    """Return a friendly, non-durable label for an unnamed child."""
    return f"{choice(_ADJECTIVES)}-{choice(_ANIMALS)}"


def resolve_subagent_label(
    requested_label: str | None,
    *,
    used_labels: set[str],
    generator: Callable[[], str],
) -> str:
    """Choose an unused display label, suffixing duplicates within one parent."""
    base = requested_label if requested_label is not None else generator()
    label = base
    for suffix_number in range(2, len(used_labels) + 3):
        if label not in used_labels:
            used_labels.add(label)
            return label
        suffix = f"-{suffix_number}"
        label = f"{base[: 32 - len(suffix)]}{suffix}"
    raise RuntimeError("Unable to resolve a unique subagent label")


def requested_subagent_display_label(label: object) -> str:
    """Return a safe request-panel label before tool-boundary validation."""
    if not isinstance(label, str):
        return "subagent"
    label = label.strip()
    if len(label) > 32 or fullmatch(_LABEL_PATTERN, label) is None:
        return "subagent"
    return label
