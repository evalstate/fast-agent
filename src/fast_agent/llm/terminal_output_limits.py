from __future__ import annotations

from fast_agent.constants import (
    DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
    MAX_TERMINAL_OUTPUT_BYTE_LIMIT,
    TERMINAL_BYTES_PER_TOKEN,
    TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO,
    TERMINAL_OUTPUT_TOKEN_RATIO,
)
from fast_agent.llm.model_database import ModelDatabase


def calculate_terminal_output_limit_for_model(model_name: str | None) -> int:
    if not model_name:
        return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

    max_tokens = ModelDatabase.get_max_output_tokens(model_name)
    if not max_tokens:
        return DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT

    terminal_token_budget = max(int(max_tokens * TERMINAL_OUTPUT_TOKEN_RATIO), 1)
    terminal_token_budget = max(
        int(terminal_token_budget * (1 - TERMINAL_OUTPUT_TOKEN_HEADROOM_RATIO)), 1
    )
    terminal_byte_budget = int(terminal_token_budget * TERMINAL_BYTES_PER_TOKEN)

    terminal_byte_budget = min(terminal_byte_budget, MAX_TERMINAL_OUTPUT_BYTE_LIMIT)
    return max(DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT, terminal_byte_budget)
