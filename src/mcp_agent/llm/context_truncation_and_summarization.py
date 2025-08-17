"""
Context truncation manager for LLM conversations.
"""
# region Imports -- External Imports
from typing import List, Optional

## Types
from mcp.types import (
    TextContent,
)

## Context
from mcp_agent.context_dependent import ContextDependent

## Core
from mcp_agent.core.agent_types import ContextTruncationMode

# endregion
# region Imports -- Internal Imports
## LLM
from mcp_agent.llm.augmented_llm import AugmentedLLM

## Logging
from mcp_agent.logging.logger import get_logger

## MCP
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# endregion

logger = get_logger(__name__)

class ContextTruncation(ContextDependent):
    """
    Manages the context window of an LLM by truncating the message history
    when it exceeds a specified token limit.

    Takes PromptMessageMultipart and truncates those. It does not deal with provider-specific
    formats. Provider-specific formats should be converted somewhere else!
    """

    @classmethod
    async def truncate_if_required(
        cls,
        messages: List[PromptMessageMultipart],
        truncation_mode: Optional[ContextTruncationMode],
        limit: Optional[int],
        model_name: str,
        system_prompt: str,
        provider: AugmentedLLM,
    ) -> List[PromptMessageMultipart]:
        """
        Checks if truncation is needed and applies the specified strategy.
        This is a synchronous wrapper for easier integration.
        """

        logger.warning(f"""
🔍 TRUNCATION DEBUG: truncate_if_required() called
                            
📊 Messages count: {len(messages)}
📝 Truncation mode: {truncation_mode}
🎯 Token limit: {limit}
🤖 Model name: {model_name}
💭 System prompt length: {len(system_prompt) if system_prompt else 0}

""")

        # Debug: Log detailed message analysis
        total_chars = 0
        for i, msg in enumerate(messages):
            msg_chars = sum(len(content.text) if hasattr(content, 'text') else len(str(content)) for content in msg.content)
            total_chars += msg_chars
            logger.warning(f"📨 Message {i}: role={msg.role}, chars={msg_chars}")
            if i < 2:  # Log first 2 messages in detail
                logger.warning(f"    Content preview: {str(msg.content)[:200]}...")

        logger.warning(f"📏 Total character count across all messages: {total_chars}")

        # Use provider's native token counting for accuracy
        logger.warning(f"🧮 Calling provider.get_token_count() with {len(messages)} messages...")
        estimated_tokens = provider.get_token_count(messages, system_prompt)
        logger.warning(f"🎯 CRITICAL: estimated_tokens returned: {estimated_tokens}")
        logger.warning(f"🔍 Ratio: {estimated_tokens}/{total_chars} = {estimated_tokens/total_chars if total_chars > 0 else 0:.4f} tokens per char")


        ############## RETURN EARLY IF NO TRUNCATION OR NO LIMIT ###############
        if not truncation_mode or truncation_mode == ContextTruncationMode.NONE or not limit:
            logger.warning(f"🚫 EARLY RETURN: No truncation (mode={truncation_mode}, limit={limit})")
            return messages

        ############# NO NEED TO TRUNCATE IF LIMIT NOT CROSSED ##############
        if estimated_tokens <= limit:
            logger.warning(f"✅ NO TRUNCATION NEEDED: {estimated_tokens} <= {limit} tokens")
            return messages

        ########## IF ARGUMENTS VALID: ACT UPON TRUNCATION MODE ##########
        logger.warning(f"🚨 TRUNCATION TRIGGERED: {estimated_tokens} > {limit} tokens!")
        logger.warning(f"🔄 Applying {truncation_mode} mode...")
        
        if truncation_mode == ContextTruncationMode.SUMMARIZE:
            logger.warning("📝 Starting SUMMARIZE truncation...")
            result = await cls._summarize_and_truncate(messages, limit, provider)
            logger.warning(f"✅ SUMMARIZE complete, returning {len(result)} messages")
            return result
        
        elif truncation_mode == ContextTruncationMode.REMOVE:
            logger.warning("✂️ Starting REMOVE truncation...")
            result = cls._truncate(messages, limit, system_prompt, provider)
            logger.warning(f"✅ REMOVE complete, returning {len(result)} messages")
            return result
        
        ######## FALLBACK - RETURN ORIGINAL MESSAGES ############
        logger.warning(f"⚠️ FALLBACK: Unknown truncation mode {truncation_mode}, returning original messages")
        return messages

    @classmethod
    async def _summarize_and_truncate(
        cls, messages: List[PromptMessageMultipart], max_tokens: int, provider: AugmentedLLM
    ) -> List[PromptMessageMultipart]:
        """(Private) Truncates history by summarizing older messages."""
        logger.info(f"Context has exceeded {max_tokens} tokens. Applying summarization.")

        conversation_messages = [m for m in messages if m.role != "system"]

        summary_string = await cls._summarize_messages(
            messages_to_summarize=conversation_messages,
            provider=provider
        )

        summary = PromptMessageMultipart(
            role="user",
            content=[TextContent(type="text", text=summary_string)],
        )

        return [summary]

    @classmethod
    async def _summarize_messages(
        cls, 
        messages_to_summarize: List[PromptMessageMultipart],
        provider: AugmentedLLM,
    ) -> str:
        """Uses the provided LLM to summarize a list of messages."""
        logger.info("Summarizing older messages...")
        
        # Format the conversation history into a single string for the prompt
        conversation_text = "\n".join(
            f"{msg.role}: {msg.first_text()}" for msg in messages_to_summarize
        )
        
        prompt = (
            "You are a conversation summarizer. Your task is to create a concise summary "
            "of the following agentic workflow. "
            "Keep in mind that it should help the agent "
            "to get back to work where it left off -- meaning that "
            "you can roughly outline the steps that were taken already, "
            "things that worked, things that did not work, and tasks that "
            "were already completed.  "
            "The conversation may contain tool call results. \n\n"
            
            f"--- Conversation ---\n{conversation_text}\n\n--- Summary ---"
        )

        # Call the provider's API call method. This helps to make the ContextTruncation class provider-agnostic.
        summary_string = await provider.execute_simple_api_call(
            message_string=prompt
        )
        
        return summary_string


    @classmethod
    def _needs_truncation(
        cls, 
        messages: List[PromptMessageMultipart], 
        max_tokens: int, 
        system_prompt: str | None = None,
        provider: AugmentedLLM | None = None
    ) -> bool:
        """Check if the context needs to be truncated."""
        if not max_tokens:
            return False
        
        if provider:
            current_tokens = provider.get_token_count(messages, system_prompt)
        else:
            # This shouldn't happen since provider should always be provided
            logger.warning("No provider available for token counting - this may cause inaccurate counts")
            return False
        
        return current_tokens > max_tokens

    @classmethod
    def _truncate(
        cls, 
        messages: List[PromptMessageMultipart], 
        max_tokens: int, 
        system_prompt: str | None = None, 
        provider: AugmentedLLM | None = None
    ) -> List[PromptMessageMultipart]:
        """(Private) Truncates history by removing the oldest messages."""
        if provider:
            initial_tokens = provider.get_token_count(messages, system_prompt)
        else:
            # This shouldn't happen since provider should always be provided
            logger.warning("No provider available for token counting in truncate")
            return messages
            
        logger.warning(
            f"Context ({initial_tokens} tokens) has exceeded the limit of {max_tokens} tokens. "
            "Applying simple truncation (remove)."
        )
        
        truncated_messages = list(messages)
        
        # Loop until the token count is within the limit
        while len(truncated_messages) >= 1 and cls._needs_truncation(
            messages=truncated_messages, max_tokens=max_tokens, system_prompt=system_prompt, provider=provider
        ):
            
            # Find the first non-system message to remove
            for i, msg in enumerate(truncated_messages):
                if msg.role != "system":
                    truncated_messages.pop(i)
                    break 
            else:  # No non-system messages left to remove
                break 
        
        if provider:
            final_tokens = provider.get_token_count(truncated_messages, system_prompt)
        else:
            # This shouldn't happen since provider should always be provided
            final_tokens = 0
        logger.info(f"Simple truncation complete. New token count: {final_tokens}")

        return truncated_messages