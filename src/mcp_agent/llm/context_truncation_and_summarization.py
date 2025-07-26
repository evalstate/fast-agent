"""
Context truncation manager for LLM conversations.
"""
# region Imports -- External Imports
import json
from typing import List, Optional

import tiktoken

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
truncate_if_required()
                            
messages: {messages}
truncation_mode: {str(truncation_mode)}
limit: {str(limit)}
model_name: {model_name}
system_prompt: {system_prompt}

""")
        

        logger.warning(f"""

limit: {str(limit)}

""")

        estimated_tokens = cls._estimate_tokens(
            messages=messages, 
            model=model_name,
        )
        logger.warning(f"[006] estimated_tokens: {estimated_tokens}")


        ############# NO NEED TO TRUNCATE IF LIMIT NOT CROSSED ##############
        if estimated_tokens <= limit:
            return messages

        ############## RETURN EARLY IF NO TRUNCATION OR NO LIMIT ###############
        if not truncation_mode or truncation_mode == ContextTruncationMode.NONE or not limit:
            return messages
        
        elif not cls._needs_truncation(messages, limit, model_name):
            return messages

        ########## IF ARGUMENTS VALID: ACT UPON TRUNCATION MODE ##########
        if truncation_mode == ContextTruncationMode.SUMMARIZE:
            return await cls._summarize_and_truncate(messages, limit, model_name, provider=provider)
        
        elif truncation_mode == ContextTruncationMode.REMOVE:
            logger.warning("[004]")
            return cls._truncate(messages, limit, model_name)
        
        ######## RETURN TRUNCATED MESSAGES ############
        return messages

    @classmethod
    async def _summarize_and_truncate(
        cls, messages: List[PromptMessageMultipart], max_tokens: int, model: str, provider: AugmentedLLM
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
    def _estimate_tokens(
        cls, 
        messages: List[PromptMessageMultipart], 
        model: str, 
        system_prompt: Optional[str] = None,
    ) -> int:
        """Estimate the number of tokens for a list of messages using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(f"Model {model} not found. Using cl100k_base tokenizer.")
            encoding = tiktoken.get_encoding("cl100k_base")

        total_tokens = len(encoding.encode(system_prompt)) if system_prompt else 0

        for message_index, message in enumerate(messages):
            # Assuming first_text() gets the main text content

            num_tokens_current_message = cls._estimate_tokens_from_message(message=message, encoding=encoding)
            total_tokens += num_tokens_current_message
            logger.info(f"[007] Message {message_index} with text {str(message)[:1000]} has {num_tokens_current_message} tokens.")

        logger.info("[008] Adding an overhead of 4 tokens per message.")
        total_tokens += len(messages) * 4 # Approximation for message overhead
        return total_tokens

    @classmethod
    def _estimate_tokens_from_message(
            cls,
            message: PromptMessageMultipart,
            encoding: tiktoken.Encoding
    ) -> int:
        """Estimates tokens for a single, potentially multi-paar message."""

        tokens = 4  # Base overhead.

        if not message.content:
            return tokens
        
        for block in message.content:
            block_type = getattr(block, 'type', '').lower()

            if block_type == "text":
                text = getattr(block, 'text', '')
                if text:
                    tokens += len(encoding.encode(text))

            elif block_type == "tool_use":
                tool_name = getattr(block, 'name', '')
                tool_args = getattr(block, 'input', {})
                tool_id = getattr(block, 'id', '')

                if tool_name:
                    tokens += len(encoding.encode(tool_name))
                if tool_args:
                    args_json = json.dumps(tool_args)
                    tokens += len(encoding.encode(args_json))
                if tool_id:
                    tokens += len(encoding.encode(tool_id))
                
                tokens += 10  # Add overhead for tool use structure

        
            elif block_type == "tool_result":
                # Count tool result ID
                tool_use_id = getattr(block, 'tool_use_id', '')
                if tool_use_id:
                    tokens += len(encoding.encode(tool_use_id))
                
                # Count tool result content
                tool_content = getattr(block, 'content', [])
                if isinstance(tool_content, str):
                    tokens += len(encoding.encode(tool_content))
                elif isinstance(tool_content, list):
                    for nested_block in tool_content:
                        nested_type = getattr(nested_block, 'type', '').lower()
                        if nested_type == "text":
                            nested_text = getattr(nested_block, 'text', '')
                            if nested_text:
                                tokens += len(encoding.encode(nested_text))
                        # Add more nested block types as needed
                
                # Add overhead for tool result structure
                tokens += 10
                
            elif block_type == "image":
                # Images have a fixed token cost in most models
                tokens += 85  # Anthropic's typical image token cost
                
            elif block_type == "resource":
                # Estimate tokens for embedded resources
                resource = getattr(block, 'resource', None)
                if resource:
                    # Count resource text content
                    resource_text = getattr(resource, 'text', '')
                    if resource_text:
                        tokens += len(encoding.encode(resource_text))
                    else:
                        # For binary resources, add a fixed cost
                        tokens += 20
            
            else:
                # Unknown block type - add minimal overhead
                logger.warning(f"Unknown block type in token estimation: {block_type}")
                tokens += 5
        
        return tokens

    @classmethod
    def _needs_truncation(
        cls, messages: List[PromptMessageMultipart], max_tokens: int, model: str, system_prompt: str | None = None
    ) -> bool:
        """Check if the context needs to be truncated."""
        if not max_tokens:
            return False
        current_tokens = cls._estimate_tokens(messages, model, system_prompt)
        return current_tokens > max_tokens

    @classmethod
    def _truncate(
        cls, messages: List[PromptMessageMultipart], max_tokens: int, model: str, system_prompt: str | None = None
    ) -> List[PromptMessageMultipart]:
        """(Private) Truncates history by removing the oldest messages."""
        initial_tokens = cls._estimate_tokens(messages, model, system_prompt)
        logger.warning(
            f"Context ({initial_tokens} tokens) has exceeded the limit of {max_tokens} tokens. "
            "Applying simple truncation (remove)."
        )
        
        truncated_messages = list(messages)
        
        # Loop until the token count is within the limit
        while len(truncated_messages) >= 1 and cls._needs_truncation(
            messages=truncated_messages, max_tokens=max_tokens, model=model, system_prompt=system_prompt
        ):
            
            # Find the first non-system message to remove
            for i, msg in enumerate(truncated_messages):
                if msg.role != "system":
                    truncated_messages.pop(i)
                    break 
            else:  # No non-system messages left to remove
                break 
        
        final_tokens = cls._estimate_tokens(truncated_messages, model, system_prompt)
        logger.info(f"Simple truncation complete. New token count: {final_tokens}")

        return truncated_messages