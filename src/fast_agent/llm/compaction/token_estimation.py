"""
Token estimation utilities for context compaction.

This module provides utilities to estimate token counts for messages
without making API calls. This is useful for pre-emptive compaction
decisions before sending messages to the LLM.

The estimation uses a simple character-to-token ratio which works
reasonably well across different tokenizers (BPE, SentencePiece, etc.).
"""

from typing import TYPE_CHECKING

from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.usage_tracking import UsageAccumulator

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)

# Average characters per token across common tokenizers
# This is a reasonable approximation that works for English text
# BPE tokenizers (GPT, Claude) average ~3.5-4 chars per token
# We use 4.0 as a slightly conservative estimate
DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_tokens_for_text(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """
    Estimate token count for a text string.

    Args:
        text: The text to estimate tokens for
        chars_per_token: Average characters per token (default 4.0)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_tokens_for_messages(
    messages: list["PromptMessageExtended"],
    system_prompt: str | None = None,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """
    Estimate token count for a list of messages.

    This provides a rough estimate without calling the provider's tokenizer.
    The estimate includes:
    - System prompt (if provided)
    - Message content (text, embedded resources)
    - Role prefixes (small overhead per message)
    - Tool results (if present)

    Args:
        messages: List of PromptMessageExtended objects
        system_prompt: Optional system prompt to include in count
        chars_per_token: Average characters per token

    Returns:
        Estimated total token count
    """
    total_chars = 0

    # Add system prompt
    if system_prompt:
        total_chars += len(system_prompt)

    # Process each message
    for message in messages:
        # Add role overhead (e.g., "user:", "assistant:")
        total_chars += len(message.role) + 2  # role + ": "

        # Process content blocks
        if message.content:
            for block in message.content:
                total_chars += _estimate_block_chars(block)

        # Process tool results
        if message.tool_results:
            for tool_id, tool_result in message.tool_results.items():
                # Add tool ID overhead
                total_chars += len(tool_id) + 10  # rough overhead for formatting

                if tool_result.content:
                    for block in tool_result.content:
                        total_chars += _estimate_block_chars(block)

    return estimate_tokens_for_text("x" * total_chars, chars_per_token)


def _estimate_block_chars(block) -> int:
    """
    Estimate character count for a content block.

    Args:
        block: A content block (TextContent, ImageContent, etc.)

    Returns:
        Estimated character count
    """
    if isinstance(block, TextContent):
        return len(block.text) if block.text else 0

    if isinstance(block, TextResourceContents):
        return len(block.text) if hasattr(block, "text") and block.text else 0

    if isinstance(block, ImageContent):
        # Images are typically tokenized as a fixed number of tokens
        # depending on size. Use a conservative estimate.
        # Most vision models use 85-170 tokens per image tile
        return 500 * DEFAULT_CHARS_PER_TOKEN  # ~500 tokens for average image

    if isinstance(block, EmbeddedResource):
        resource = getattr(block, "resource", None)
        if resource and isinstance(resource, TextResourceContents):
            return len(resource.text) if hasattr(resource, "text") and resource.text else 0
        # Non-text embedded resources (images, PDFs)
        return 1000 * DEFAULT_CHARS_PER_TOKEN  # ~1000 tokens estimate

    # Unknown block type - return small overhead
    return 50


def get_current_context_tokens(usage_accumulator: UsageAccumulator | None) -> int:
    """
    Get the current context token count from the usage accumulator.

    This is more accurate than estimation because it uses the actual
    token counts reported by the provider after each turn.

    Args:
        usage_accumulator: The usage accumulator from the LLM

    Returns:
        Current context tokens, or 0 if not available
    """
    if usage_accumulator is None:
        return 0
    return usage_accumulator.current_context_tokens


def get_context_headroom(
    usage_accumulator: UsageAccumulator | None,
    limit: int | None = None,
) -> int | None:
    """
    Calculate remaining token headroom before hitting the limit.

    Args:
        usage_accumulator: The usage accumulator from the LLM
        limit: Optional explicit limit. If not provided, uses model's context window.

    Returns:
        Remaining tokens before limit, or None if limit unknown
    """
    if usage_accumulator is None:
        return None

    current = usage_accumulator.current_context_tokens

    if limit is not None:
        return max(0, limit - current)

    window = usage_accumulator.context_window_size
    if window is None:
        return None

    return max(0, window - current)
