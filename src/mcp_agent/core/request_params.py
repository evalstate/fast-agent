"""
Request parameters definitions for LLM interactions.
"""

from typing import Any, List, Dict

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
    The model to use for the LLM generation.
    If specified, this overrides the 'modelPreferences' selection criteria.
    """

    use_history: bool = True
    """
    Include the message history in the generate request.
    """

    max_iterations: int = 10
    """
    The maximum number of iterations to run the LLM for.
    """

    parallel_tool_calls: bool = True
    """
    Whether to allow multiple tool calls per iteration.
    Also known as multi-step tool use.
    """
    response_format: Any | None = None
    """
    Override response format for structured calls. Prefer sending pydantic model - only use in exceptional circumstances
    """

    t0_system_template_vars: Dict[str, Any] = Field(default_factory=dict)
    """
    Optional dictionary of template variables for TensorZero system prompts.
    These variables can be used to pass dynamoc fields to T0 system prompts on the fly.
    """
