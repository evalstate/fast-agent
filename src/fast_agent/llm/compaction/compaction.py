"""
Context compaction manager for LLM conversations.

This module provides the main compaction logic for managing conversation history
when it exceeds token limits. It supports two strategies:

1. TRUNCATE: Remove older messages while preserving system context
2. SUMMARIZE: Use the LLM to create a summary of the conversation
"""

from typing import TYPE_CHECKING, Protocol

from mcp.types import TextContent

from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.compaction.token_estimation import (
    estimate_tokens_for_messages,
    get_current_context_tokens,
)
from fast_agent.llm.compaction.types import ContextCompactionMode

if TYPE_CHECKING:
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended, RequestParams

logger = get_logger(__name__)


class CompactionLLMProtocol(Protocol):
    """
    Protocol for the LLM interface needed by compaction.

    This is a minimal interface that allows compaction to work with
    any LLM that can generate text responses.
    """

    @property
    def usage_accumulator(self) -> "UsageAccumulator | None": ...

    @property
    def model_name(self) -> str | None: ...

    async def generate(
        self,
        messages: list["PromptMessageExtended"],
        request_params: "RequestParams | None" = None,
    ) -> "PromptMessageExtended": ...


class ContextCompaction:
    """
    Manages context window by compacting message history when limits are exceeded.

    This class operates on PromptMessageExtended objects (the normalized message
    format) and is provider-agnostic. The actual token counting is delegated to
    the usage tracking infrastructure.
    """

    # Default summarization prompt
    SUMMARIZATION_PROMPT = """You are a conversation summarizer. Your task is to create a concise summary \
of the following agentic workflow.

Keep in mind that the summary should help the agent continue where it left off. Include:
- Key decisions and actions taken
- Important results or findings
- Current state and any pending tasks
- Things that worked and things that didn't

The conversation may contain tool calls and their results.

--- Conversation ---
{conversation}

--- Summary ---
Provide a clear, structured summary that preserves the essential context:"""

    @classmethod
    async def compact_if_needed(
        cls,
        messages: list["PromptMessageExtended"],
        mode: ContextCompactionMode | None,
        limit: int | None,
        llm: CompactionLLMProtocol | None = None,
        system_prompt: str | None = None,
    ) -> tuple[list["PromptMessageExtended"], bool]:
        """
        Compact messages if the token limit is exceeded.

        Args:
            messages: The conversation history to potentially compact
            mode: The compaction strategy (NONE, TRUNCATE, or SUMMARIZE)
            limit: Token limit threshold. If None, no compaction is performed.
            llm: LLM instance (required for SUMMARIZE mode)
            system_prompt: System prompt to include in token estimation

        Returns:
            Tuple of (possibly compacted messages, whether compaction occurred)
        """
        # Early exit if compaction is disabled
        if not mode or mode == ContextCompactionMode.NONE:
            return messages, False

        if not limit:
            return messages, False

        if not messages:
            return messages, False

        # Get current token count
        current_tokens = cls._get_token_count(messages, system_prompt, llm)

        logger.debug(
            "Compaction check",
            current_tokens=current_tokens,
            limit=limit,
            mode=mode,
            message_count=len(messages),
        )

        # Check if compaction is needed
        if current_tokens <= limit:
            return messages, False

        logger.info(
            "Context limit exceeded, compacting",
            current_tokens=current_tokens,
            limit=limit,
            mode=mode,
        )

        # Apply the appropriate strategy
        if mode == ContextCompactionMode.TRUNCATE:
            compacted = cls._truncate(
                messages=messages,
                limit=limit,
                system_prompt=system_prompt,
                llm=llm,
            )
        elif mode == ContextCompactionMode.SUMMARIZE:
            if llm is None:
                logger.warning("SUMMARIZE mode requires LLM, falling back to TRUNCATE")
                compacted = cls._truncate(
                    messages=messages,
                    limit=limit,
                    system_prompt=system_prompt,
                    llm=llm,
                )
            else:
                compacted = await cls._summarize(
                    messages=messages,
                    limit=limit,
                    llm=llm,
                    system_prompt=system_prompt,
                )
        else:
            logger.warning(f"Unknown compaction mode: {mode}, returning original messages")
            return messages, False

        final_tokens = cls._get_token_count(compacted, system_prompt, llm)
        logger.info(
            "Compaction complete",
            original_messages=len(messages),
            compacted_messages=len(compacted),
            original_tokens=current_tokens,
            final_tokens=final_tokens,
        )

        return compacted, True

    @classmethod
    def _get_token_count(
        cls,
        messages: list["PromptMessageExtended"],
        system_prompt: str | None,
        llm: CompactionLLMProtocol | None,
    ) -> int:
        """
        Get token count for messages.

        Uses the usage accumulator if available for accurate counts,
        falls back to estimation otherwise.
        """
        # Try to use actual token count from usage accumulator
        if llm and llm.usage_accumulator:
            actual = get_current_context_tokens(llm.usage_accumulator)
            if actual > 0:
                return actual

        # Fall back to estimation
        return estimate_tokens_for_messages(messages, system_prompt)

    @classmethod
    def _truncate(
        cls,
        messages: list["PromptMessageExtended"],
        limit: int,
        system_prompt: str | None,
        llm: CompactionLLMProtocol | None,
    ) -> list["PromptMessageExtended"]:
        """
        Truncate history by removing older non-template messages.

        Preserves:
        - Messages marked as templates (is_template=True)
        - The most recent messages that fit within the limit
        """
        from fast_agent.types import PromptMessageExtended  # noqa: I001, TC001

        # Separate template messages from conversation messages
        template_messages: list[PromptMessageExtended] = []
        conversation_messages: list[PromptMessageExtended] = []

        for msg in messages:
            if msg.is_template:
                template_messages.append(msg)
            else:
                conversation_messages.append(msg)

        # Start with templates
        result = list(template_messages)

        # If no conversation messages, nothing to truncate
        if not conversation_messages:
            return result

        # Remove messages from the front (oldest first) until we're under the limit
        # But always keep at least the last message
        while len(conversation_messages) > 1:
            candidate = result + conversation_messages
            tokens = cls._get_token_count(candidate, system_prompt, llm)

            if tokens <= limit:
                break

            # Remove the oldest conversation message
            removed = conversation_messages.pop(0)
            logger.debug(
                "Removed message during truncation",
                role=removed.role,
                content_preview=removed.first_text()[:50] if removed.first_text() else "",
            )

        result.extend(conversation_messages)
        return result

    @classmethod
    async def _summarize(
        cls,
        messages: list["PromptMessageExtended"],
        limit: int,
        llm: CompactionLLMProtocol,
        system_prompt: str | None,
    ) -> list["PromptMessageExtended"]:
        """
        Summarize history using the LLM to condense older messages.

        The resulting history will contain:
        - Template messages (preserved)
        - A single user message with the summary
        """
        from fast_agent.types import PromptMessageExtended, RequestParams

        # Separate template messages from conversation messages
        template_messages: list[PromptMessageExtended] = []
        conversation_messages: list[PromptMessageExtended] = []

        for msg in messages:
            if msg.is_template:
                template_messages.append(msg)
            else:
                conversation_messages.append(msg)

        # If too few messages to summarize, fall back to truncation
        if len(conversation_messages) < 2:
            logger.debug("Too few messages to summarize, using truncation")
            return cls._truncate(messages, limit, system_prompt, llm)

        # Format conversation for summarization
        conversation_text = cls._format_conversation_for_summary(conversation_messages)

        # Generate summary using the LLM
        summary_prompt = cls.SUMMARIZATION_PROMPT.format(conversation=conversation_text)

        # Create a temporary message for the summary request
        summary_request = PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text=summary_prompt)],
        )

        # Make the summary call without history
        try:
            summary_response = await llm.generate(
                messages=[summary_request],
                request_params=RequestParams(use_history=False, maxTokens=2048),
            )
            summary_text = summary_response.first_text() or "Unable to generate summary."
        except Exception as e:
            logger.error(f"Error generating summary: {e}, falling back to truncation")
            return cls._truncate(messages, limit, system_prompt, llm)

        # Create a summary message
        summary_message = PromptMessageExtended(
            role="user",
            content=[
                TextContent(
                    type="text",
                    text=f"[Previous conversation summary]\n\n{summary_text}\n\n[End of summary - continuing conversation]",
                )
            ],
        )

        # Return templates + summary
        result = list(template_messages)
        result.append(summary_message)

        return result

    @classmethod
    def _format_conversation_for_summary(
        cls, messages: list["PromptMessageExtended"]
    ) -> str:
        """Format messages into a text representation for summarization."""
        lines: list[str] = []

        for msg in messages:
            role = msg.role.upper()
            text = msg.first_text() or "[non-text content]"

            # Truncate very long messages
            if len(text) > 2000:
                text = text[:2000] + "... [truncated]"

            lines.append(f"{role}: {text}")

            # Include tool results summary
            if msg.tool_results:
                for tool_id, result in msg.tool_results.items():
                    result_text = ""
                    if result.content:
                        for block in result.content:
                            if hasattr(block, "text"):
                                result_text = block.text[:500] if len(block.text) > 500 else block.text
                                break
                    lines.append(f"  TOOL[{tool_id}]: {result_text or '[result]'}")

        return "\n".join(lines)
