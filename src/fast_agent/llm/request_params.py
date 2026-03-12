"""
Request parameters definitions for LLM interactions.
"""

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from mcp import SamplingMessage
from mcp.types import CreateMessageRequestParams
from pydantic import AliasChoices, Field

from fast_agent.constants import DEFAULT_MAX_ITERATIONS, DEFAULT_STREAMING_TIMEOUT

if TYPE_CHECKING:
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
else:
    ToolExecutionHandler = Any


ResponseMode: TypeAlias = Literal["inherit", "postprocess", "passthrough"]
ToolResultMode: TypeAlias = Literal["postprocess", "passthrough", "selectable"]


def response_mode_to_tool_result_mode(response_mode: ResponseMode) -> ToolResultMode | None:
    """Map a per-call response override to a concrete tool result mode."""
    if response_mode == "inherit":
        return None
    return response_mode


def tool_result_mode_allows_response_mode(tool_result_mode: ToolResultMode) -> bool:
    """Return whether a tool should expose a per-call response mode switch."""
    return tool_result_mode == "selectable"


def tool_result_mode_is_passthrough(tool_result_mode: ToolResultMode) -> bool:
    """Return whether the effective tool handling should bypass postprocessing."""
    return tool_result_mode == "passthrough"


class RequestParams(CreateMessageRequestParams):
    """
    Parameters to configure the FastAgentLLM 'generate' requests.
    """

    messages: list[SamplingMessage] = Field(exclude=True, default=[])
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

    max_iterations: int = DEFAULT_MAX_ITERATIONS
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

    template_vars: dict[str, Any] = Field(default_factory=dict)
    """
    Optional dictionary of template variables for dynamic templates. Currently only works for TensorZero inference backend
    """

    mcp_metadata: dict[str, Any] | None = None
    """
    Metadata to pass through to MCP tool calls via the _meta field.
    """

    tool_execution_handler: ToolExecutionHandler | None = Field(default=None, repr=False)
    """
    Internal per-request tool execution handler (not sent to LLM providers).
    """

    emit_loop_progress: bool = False
    """
    Emit monotonic progress updates for the internal tool loop when supported.
    """

    tool_result_mode: ToolResultMode = "postprocess"
    """
    Control how tool results are returned.

    - ``postprocess``: run a post-tool LLM synthesis step.
    - ``passthrough``: return tool results directly as assistant output.
    - ``selectable``: expose a per-call ``response_mode`` switch; absent an override,
      execution behaves like ``postprocess``.
    """

    streaming_timeout: float | None = DEFAULT_STREAMING_TIMEOUT
    """
    Maximum idle time in seconds to wait between streaming events. Set to None
    to disable idle timeout enforcement.
    """

    top_p: float | None = Field(
        default=None,
        validation_alias=AliasChoices("top_p", "topP"),
    )
    """Optional nucleus sampling parameter (provider support varies)."""

    top_k: int | None = Field(
        default=None,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    """Optional top-k sampling parameter (provider support varies)."""

    min_p: float | None = Field(
        default=None,
        validation_alias=AliasChoices("min_p", "minP"),
    )
    """Optional minimum probability threshold for sampling (provider support varies)."""

    presence_penalty: float | None = Field(
        default=None,
        validation_alias=AliasChoices("presence_penalty", "presencePenalty"),
    )
    """Optional presence penalty (provider support varies)."""

    frequency_penalty: float | None = Field(
        default=None,
        validation_alias=AliasChoices("frequency_penalty", "frequencyPenalty"),
    )
    """Optional frequency penalty (provider support varies)."""

    repetition_penalty: float | None = Field(
        default=None,
        validation_alias=AliasChoices("repetition_penalty", "repetitionPenalty"),
    )
    """Optional repetition penalty (provider support varies)."""

    service_tier: Literal["fast", "flex"] | None = None
    """Responses-family service tier override (fast/priority or flex)."""
