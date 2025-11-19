"""
Utility functions for OpenAI integration with MCP.

This file provides backward compatibility with the existing API while
delegating to the proper implementations in the providers/ directory.
"""

from typing import Any, Dict, Union

from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)

from fast_agent.llm.provider.openai.multipart_converter_openai import OpenAIConverter
from fast_agent.llm.provider.openai.openai_multipart import (
    openai_to_extended,
)
from fast_agent.types import PromptMessageExtended


def openai_message_to_prompt_message_multipart(
    message: Union[ChatCompletionMessage, Dict[str, Any]],
) -> PromptMessageExtended:
    """
    Convert an OpenAI ChatCompletionMessage to a PromptMessageExtended.

    Args:
        message: The OpenAI message to convert (can be an actual ChatCompletionMessage
                or a dictionary with the same structure)

    Returns:
        A PromptMessageExtended representation
    """
    return openai_to_extended(message)


def openai_message_param_to_prompt_message_multipart(
    message_param: ChatCompletionMessageParam,
) -> PromptMessageExtended:
    """
    Convert an OpenAI ChatCompletionMessageParam to a PromptMessageExtended.

    Args:
        message_param: The OpenAI message param to convert

    Returns:
        A PromptMessageExtended representation
    """
    return openai_to_extended(message_param)


def prompt_message_multipart_to_openai_message_param(
    multipart: PromptMessageExtended,
) -> ChatCompletionMessageParam:
    """
    Convert a PromptMessageExtended to an OpenAI ChatCompletionMessageParam.

    Args:
        multipart: The PromptMessageExtended to convert

    Returns:
        An OpenAI ChatCompletionMessageParam representation
    """
    # convert_to_openai now returns a list, return the first element for backward compatibility
    messages = OpenAIConverter.convert_to_openai(multipart)
    return messages[0] if messages else {"role": multipart.role, "content": ""}
