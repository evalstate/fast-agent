"""Shared type definitions for fast-agent.

This module contains type definitions that can be safely imported from anywhere
without causing circular import issues. It has minimal dependencies.
"""

# Re-export common types for convenience
from fast_agent.types.llm_stop_reason import LlmStopReason

__all__ = [
    "LlmStopReason",
]
