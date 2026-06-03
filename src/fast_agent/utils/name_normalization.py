"""Shared string normalization helpers for user-facing names."""

from __future__ import annotations

import re

from fast_agent.utils.text import strip_casefold

_PROVIDER_KEY_SEPARATOR_RE = re.compile(r"[-_\s]+")


def normalize_provider_key(value: str) -> str:
    return strip_casefold(_PROVIDER_KEY_SEPARATOR_RE.sub("", value))
