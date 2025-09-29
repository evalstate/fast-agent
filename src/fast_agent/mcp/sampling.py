"""
This simplified implementation directly converts between MCP types and PromptMessageExtended.
"""

from typing import TYPE_CHECKING

from mcp import ClientSession
from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core.logging.logger import get_logger
from fast_agent.interfaces import FastAgentLLMProtocol
from fast_agent.llm.sampling_converter import SamplingConverter
from fast_agent.mcp.helpers.server_config_helpers import get_server_config
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)


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


async def sample(mcp_ctx: ClientSession, params: CreateMessageRequestParams) -> CreateMessageResult:
    """
    Handle sampling requests from the MCP protocol using SamplingConverter.

    This function:
    1. Extracts the model from the request
    2. Uses SamplingConverter to convert types
    3. Calls the LLM's generate_prompt method
    4. Returns the result as a CreateMessageResult

    Args:
        mcp_ctx: The MCP ClientSession
        params: The sampling request parameters

    Returns:
        A CreateMessageResult containing the LLM's response
    """
    # Get server name for notification tracking
    server_name = "unknown"
    if hasattr(mcp_ctx, "session") and hasattr(mcp_ctx.session, "session_server_name"):
        server_name = mcp_ctx.session.session_server_name or "unknown"

    # Start tracking sampling operation
    try:
        from fast_agent.ui import notification_tracker
        notification_tracker.start_sampling(server_name)
    except Exception:
        # Don't let notification tracking break sampling
        pass

    model: str | None = None
    api_key: str | None = None
    try:
        # Extract model from server config using type-safe helper
        server_config = get_server_config(mcp_ctx)

        # First priority: explicitly configured sampling model
        if server_config and hasattr(server_config, "sampling") and server_config.sampling:
            model = server_config.sampling.model

        # Second priority: auto_sampling fallback (if enabled at application level)
        if model is None:
            # Check if auto_sampling is enabled
            auto_sampling_enabled = False
            try:
                from fast_agent.context import get_current_context

                app_context = get_current_context()
                if app_context and app_context.config:
                    auto_sampling_enabled = getattr(app_context.config, "auto_sampling", True)
            except Exception as e:
                logger.debug(f"Could not get application config: {e}")
                auto_sampling_enabled = True  # Default to enabled

            if auto_sampling_enabled:
                # Import here to avoid circular import
                from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession

                # Try agent's model first (from the session)
                if hasattr(mcp_ctx, "session") and isinstance(
                    mcp_ctx.session, MCPAgentClientSession
                ):
                    if mcp_ctx.session.agent_model:
                        model = mcp_ctx.session.agent_model
                        logger.debug(f"Using agent's model for sampling: {model}")
                    if mcp_ctx.session.api_key:
                        api_key = mcp_ctx.session.api_key
                        logger.debug(f"Using agent's API KEY for sampling: {api_key}")

                # Fall back to system default model
                if model is None:
                    try:
                        if app_context and app_context.config and app_context.config.default_model:
                            model = app_context.config.default_model
                            logger.debug(f"Using system default model for sampling: {model}")
                    except Exception as e:
                        logger.debug(f"Could not get system default model: {e}")

        if model is None:
            raise ValueError(
                "No model configured for sampling (server config, agent model, or system default)"
            )

        # Create an LLM instance
        llm = create_sampling_llm(params, model, api_key)

        # Extract all messages from the request params
        if not params.messages:
            raise ValueError("No messages provided")

        # Convert all SamplingMessages to PromptMessageExtended objects
        conversation = SamplingConverter.convert_messages(params.messages)

        # Extract request parameters using our converter
        request_params = SamplingConverter.extract_request_params(params)

        llm_response: PromptMessageExtended = await llm.generate(conversation, request_params)
        logger.info(f"Complete sampling request : {llm_response.first_text()[:50]}...")

        return CreateMessageResult(
            role=llm_response.role,
            content=TextContent(type="text", text=llm_response.first_text()),
            model=model,
            stopReason=LlmStopReason.END_TURN.value,
        )
    except Exception as e:
        logger.error(f"Error in sampling: {str(e)}")
        return SamplingConverter.error_result(
            error_message=f"Error in sampling: {str(e)}", model=model
        )
    finally:
        # End tracking sampling operation
        try:
            from fast_agent.ui import notification_tracker
            notification_tracker.end_sampling(server_name)
        except Exception:
            # Don't let notification tracking break sampling
            pass


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
