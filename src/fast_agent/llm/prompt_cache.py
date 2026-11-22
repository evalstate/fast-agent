"""Provider prompt-cache retention facts shared by scheduling and display."""

from __future__ import annotations

import re

# OpenAI prompt caching docs guarantee at least 30 minutes for GPT-5.6 and later,
# including later major versions and model variants such as ``-mini``.
# https://platform.openai.com/docs/guides/prompt-caching
OPENAI_MINIMUM_CACHE_TTL_SECONDS = 30 * 60
_OPENAI_MINIMUM_CACHE_TTL_MODEL_VERSION = (5, 6)

# The Codex backend keeps a conversation cached for as long as its websocket
# connection lives: unused sessions still read from cache after 50 minutes, and
# a connection is replaced within the hour.
CODEX_OBSERVED_CACHE_TTL_SECONDS = 50 * 60

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
