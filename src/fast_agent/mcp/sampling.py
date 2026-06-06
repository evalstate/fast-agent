"""
This simplified implementation directly converts between MCP types and PromptMessageExtended.
Supports "sampling with tools" as per MCP specification.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp import ClientSession
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    CreateMessageResultWithTools,
    TextContent,
)

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.model_resolution import (
    HARDCODED_DEFAULT_MODEL,
    get_context_cli_model_override,
    get_context_model_references,
    resolve_model_reference,
    resolve_model_spec,
)
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.sampling_converter import SamplingConverter
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.context import Context
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)


@runtime_checkable
class _NamedSamplingSession(Protocol):
    session_server_name: str | None


@dataclass(frozen=True, slots=True)
class _SamplingModelSelection:
    model: str
    api_key: str | None
    app_context: "Context | None"


def create_sampling_llm(
    params: CreateMessageRequestParams, model_string: str, api_key: str | None
) -> FastAgentLLMProtocol:
    """
    Create an LLM instance for sampling without tools support.
    This utility function creates a minimal LLM instance based on the model string.

    Args:
        mcp_ctx: The MCP ClientSession
        model_string: The model to use (e.g. "passthrough", "claude-3-5-sonnet-latest")

    Returns:
        An initialized LLM instance ready to use
    """
    from fast_agent.llm.model_factory import ModelFactory

    app_context = None
    try:
        from fast_agent.context import get_current_context

        app_context = get_current_context()
    except Exception:
        logger.warning("App context not available for sampling call")

    agent = LlmAgent(
        config=sampling_agent_config(params),
        context=app_context,
    )

    # Create the LLM using the factory
    factory = ModelFactory.create_factory(model_string)
    llm = factory(agent=agent, api_key=api_key)

    # Attach the LLM to the agent
    agent._llm = llm

    return llm


def _current_app_context() -> "Context | None":
    try:
        from fast_agent.context import get_current_context

        return get_current_context()
    except Exception:
        return None


def _sampling_server_name(context: RequestContext[ClientSession, Any]) -> str:
    session = context.session
    if isinstance(session, _NamedSamplingSession) and session.session_server_name:
        return session.session_server_name
    return "unknown"


def _start_sampling_notification(server_name: str) -> None:
    try:
        from fast_agent.ui import notification_tracker

        notification_tracker.start_sampling(server_name)
    except Exception:
        # Don't let notification tracking break sampling
        pass


def _end_sampling_notification(server_name: str) -> None:
    try:
        from fast_agent.ui import notification_tracker

        notification_tracker.end_sampling(server_name)
    except Exception:
        # Don't let notification tracking break sampling
        pass


def resolve_auto_sampling_enabled(app_context: "Context | None") -> bool:
    if app_context is None or app_context.config is None:
        return True
    return app_context.config.auto_sampling


def _configured_sampling_model(context: RequestContext[ClientSession, Any]) -> str | None:
    server_config = get_server_config(context)
    if server_config and server_config.sampling:
        return server_config.sampling.model
    return None


def _agent_sampling_overrides(
    context: RequestContext[ClientSession, Any],
) -> tuple[str | None, str | None]:
    from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

    if not isinstance(context.session, MCPAgentClientSession):
        return None, None

    model = context.session.agent_model
    api_key = context.session.api_key
    if model:
        logger.debug(f"Using agent's model for sampling: {model}")
    if api_key:
        logger.debug("Using agent's API key override for sampling")
    return model, api_key


def _default_sampling_model(app_context: Any | None) -> str | None:
    try:
        resolved_model = resolve_model_spec(
            app_context,
            cli_model=get_context_cli_model_override(app_context),
            hardcoded_default=HARDCODED_DEFAULT_MODEL,
        )
        if resolved_model.model:
            logger.debug(
                f"Using {resolved_model.source} model for sampling: {resolved_model.model}"
            )
        return resolved_model.model
    except Exception as e:
        logger.debug(f"Could not resolve default model for sampling: {e}")
        return None


def _select_sampling_model(
    context: RequestContext[ClientSession, Any],
) -> _SamplingModelSelection:
    app_context = _current_app_context()
    model = _configured_sampling_model(context)
    api_key: str | None = None

    if model is None and resolve_auto_sampling_enabled(app_context):
        model, api_key = _agent_sampling_overrides(context)
        if model is None:
            model = _default_sampling_model(app_context)

    if model is None:
        raise ValueError(
            "No model configured for sampling (server config, agent model, or system default)"
        )

    resolved = resolve_model_reference(model, get_context_model_references(app_context))
    return _SamplingModelSelection(model=resolved, api_key=api_key, app_context=app_context)


def _sampling_response(
    llm_response: "PromptMessageExtended",
    *,
    model: str,
    has_tools: bool,
) -> CreateMessageResult | CreateMessageResultWithTools:
    if has_tools and llm_response.stop_reason == LlmStopReason.TOOL_USE:
        content_blocks = SamplingConverter.llm_response_to_sampling_content(llm_response)
        return CreateMessageResultWithTools(
            role=llm_response.role,
            content=content_blocks,
            model=model,
            stopReason="toolUse",
        )

    return CreateMessageResult(
        role=llm_response.role,
        content=TextContent(type="text", text=llm_response.first_text()),
        model=model,
        stopReason=LlmStopReason.END_TURN.value,
    )


async def sample(
    context: RequestContext[ClientSession, Any], params: CreateMessageRequestParams
) -> CreateMessageResult | CreateMessageResultWithTools:
    """
    Handle sampling requests from the MCP protocol using SamplingConverter.

    This function:
    1. Extracts the model from the request
    2. Uses SamplingConverter to convert types
    3. Calls the LLM's generate method (with tools if provided)
    4. Returns the result as CreateMessageResult or CreateMessageResultWithTools

    Supports "sampling with tools" per MCP specification. When tools are provided
    and the LLM wants to use them, returns CreateMessageResultWithTools with
    stopReason="toolUse". The MCP server is responsible for executing tools
    and sending follow-up requests with tool results.

    Args:
        context: The MCP RequestContext containing the ClientSession
        params: The sampling request parameters (may include tools and toolChoice)

    Returns:
        CreateMessageResult for final answers, or
        CreateMessageResultWithTools when the LLM wants to use tools
    """
    # Get server name for notification tracking
    server_name = _sampling_server_name(context)
    _start_sampling_notification(server_name)

    model: str | None = None
    try:
        selection = _select_sampling_model(context)
        model = selection.model

        # Create an LLM instance
        llm = create_sampling_llm(params, model, selection.api_key)

        # Extract all messages from the request params
        if not params.messages:
            raise ValueError("No messages provided")

        # Convert all SamplingMessages to PromptMessageExtended objects
        conversation = SamplingConverter.convert_messages(params.messages)

        # Extract request parameters using our converter
        request_params = SamplingConverter.extract_request_params(params)

        # Check if tools are provided in the request
        tools = params.tools if params.tools else None
        has_tools = bool(tools)

        # Call LLM with tools if provided
        llm_response: PromptMessageExtended = await llm.generate(
            conversation, request_params, tools=tools
        )

        # Log response (truncate for brevity)
        response_text = llm_response.first_text()
        log_text = response_text[:50] if response_text else "<no text>"
        logger.info(f"Complete sampling request: {log_text}...")

        return _sampling_response(llm_response, model=model, has_tools=has_tools)
    except Exception as e:
        logger.error(f"Error in sampling: {e!s}")
        return SamplingConverter.error_result(
            error_message=f"Error in sampling: {e!s}", model=model
        )
    finally:
        _end_sampling_notification(server_name)


def sampling_agent_config(
    params: CreateMessageRequestParams | None = None,
) -> AgentConfig:
    """
    Build a sampling AgentConfig based on request parameters.

    Args:
        params: Optional CreateMessageRequestParams that may contain a system prompt

    Returns:
        An initialized AgentConfig for use in sampling
    """
    # Use systemPrompt from params if available, otherwise use default
    instruction = "You are a helpful AI Agent."
    if params and params.systemPrompt is not None:
        instruction = params.systemPrompt

    return AgentConfig(name="sampling_agent", instruction=instruction)
