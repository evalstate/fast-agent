"""
Context compaction for managing LLM conversation history.

This module provides utilities to compact conversation history when it exceeds
token limits, using either truncation (removing older messages) or summarization
(condensing history into a summary).
"""

from fast_agent.llm.compaction.compaction import ContextCompaction
from fast_agent.llm.compaction.token_estimation import estimate_tokens_for_messages
from fast_agent.llm.compaction.types import ContextCompactionMode

__all__ = [
    "ContextCompaction",
    "ContextCompactionMode",
    "estimate_tokens_for_messages",
]
