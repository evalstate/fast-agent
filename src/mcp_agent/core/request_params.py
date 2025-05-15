"""
Request parameters definitions for LLM interactions.
"""

from typing import Any, Dict, List

from mcp import SamplingMessage
from mcp.types import CreateMessageRequestParams
from pydantic import Field


class RequestParams(CreateMessageRequestParams):
    """
    Parameters to configure the AugmentedLLM 'generate' requests.
    """

    messages: List[SamplingMessage] = Field(exclude=True, default=[])
    """
    Ignored. 'messages' are removed from CreateMessageRequestParams 
    to avoid confusion with the 'message' parameter on 'generate' method.
    """

    maxTokens: int = 2048
    """The maximum number of tokens to sample, as requested by the server."""

    model: str | None = None
    """
    The model to use for the LLM generation. This can only be set during Agent creation.
    If specified, this overrides the 'modelPreferences' selection criteria.
    """

    use_history: bool = True
    """
    Agent/LLM maintains conversation history. Does not include applied Prompts
    """

    max_iterations: int = 20
    """
    The maximum number of tool calls allowed in a conversation turn
    """

    parallel_tool_calls: bool = True
    """
    Whether to allow simultaneous tool calls
    """
    response_format: Any | None = None
    """
    Override response format for structured calls. Prefer sending pydantic model - only use in exceptional circumstances
    """

    template_vars: Dict[str, Any] = Field(default_factory=dict)
    """
    Optional dictionary of template variables for dynamic templates. Currently only works for TensorZero inference backend
    """
