"""Provider prompt-cache retention facts shared by scheduling and display."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

# OpenAI prompt caching docs guarantee at least 30 minutes for GPT-5.6 and later,
# including later major versions and model variants such as ``-mini``.
# https://platform.openai.com/docs/guides/prompt-caching
OPENAI_MINIMUM_CACHE_TTL_SECONDS = 30 * 60
_OPENAI_MINIMUM_CACHE_TTL_MODEL_VERSION = (5, 6)

# The Codex backend keeps a conversation cached for as long as its websocket
# connection lives: unused sessions still read from cache after 50 minutes, and
# a connection is replaced within the hour.
CODEX_OBSERVED_CACHE_TTL_SECONDS = 50 * 60

# Gemini 4 uses Google implicit caching (no fixed documented TTL; capacity-based
# eviction). Conservative retention floors observed on Copilot Chat Completions
# (2026-11-25):
# - gemini-4-neon: hit through 35m
# - gemini-4-hydrogen: hit through 20m, missed at 30m
# - gemini-4-carbon: hit through 15m
# - gemini-4-argon: hit at 4m, missed at 5m and 6m
COPILOT_GEMINI_OBSERVED_CACHE_TTL_SECONDS: Mapping[str, int] = MappingProxyType(
    {
        "gemini-4-neon": 30 * 60,
        "gemini-4-hydrogen": 15 * 60,
        "gemini-4-carbon": 15 * 60,
        "gemini-4-argon": 4 * 60,
    }
)

# Google AI Studio implicit caching is enabled by default for Gemini 2.5 and
# later (https://aistudio.google.com/docs/caching.md) with no fixed documented
# TTL. Live probes (2026-11-25) across Gemini 4 and 3.8 Flash hit through ~4-6m
# and were consistently cold by 10m (with some 4-5m evictions), so schedule
# against a conservative 4-minute retention (pings every 3m 15s).
GOOGLE_GEMINI_OBSERVED_CACHE_TTL_SECONDS = 4 * 60
_GOOGLE_IMPLICIT_CACHE_MIN_VERSION = (2, 5)

# Sent instead of new user input when a provider cannot refresh a cache without
# generating; the reply is discarded.
CACHE_KEEP_WARM_MESSAGE = (
    "[automated cache keep-warm request - not from the user] "
    "Do not continue the task and do not call tools. Reply with exactly: OK"
)


class PromptCacheWarmError(Exception):
    """A keep-warm request could not be made or did not report usage."""


def uses_openai_minimum_cache_ttl(model: str) -> bool:
    match = re.search(r"gpt-(\d+)(?:\.(\d+))?", model.lower())
    if match is None:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _OPENAI_MINIMUM_CACHE_TTL_MODEL_VERSION


def copilot_gemini_observed_cache_ttl(model: str) -> int | None:
    return COPILOT_GEMINI_OBSERVED_CACHE_TTL_SECONDS.get(model.lower())


def uses_google_implicit_cache(model: str) -> bool:
    match = re.search(r"gemini-(\d+)(?:\.(\d+))?", model.lower())
    if match is None:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= _GOOGLE_IMPLICIT_CACHE_MIN_VERSION
