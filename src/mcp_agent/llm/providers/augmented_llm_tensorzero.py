import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID

# Attempt to import tensorzero, provide guidance if missing
try:
    from tensorzero import (
        AsyncTensorZeroGateway,
        ChatChunk,
        ChatInferenceResponse,
        ContentBlock,
        JsonChunk,
        JsonInferenceResponse,
        Usage,
    )
    from tensorzero.exceptions import TensorZeroError
except ImportError:

    raise ImportError("""Could not import the 'tensorzero' library.
        Please install it to use the TensorZero provider:

            pip install tensorzero

        Ensure you also have the T0_URI environment variable set or configured
        in your fastagent.secrets.yaml.""")

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ProviderKeyError, ProviderModelError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.mcp.response import AugmentedLLMResponse


class TensorZeroAugmentedLLM(AugmentedLLMProtocol):
    """
    AugmentedLLM implementation for interacting with a TensorZero Gateway.
    """

    def __init__(
        self,
        agent: Agent,
        model: str,  # Expected to be the T0 function name
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ):
        """
        Initializes the TensorZero Augmented LLM.

        Args:
            agent: The agent instance this LLM is attached to.
            model: The TensorZero function name to use for inference.
            request_params: Default request parameters.
            **kwargs: Additional keyword arguments (e.g., episode_id).
        """
        super().__init__(agent, model, request_params, **kwargs)
        self.agent = agent
        self.t0_function_name = model  # model string is the T0 function name
        self.request_params = request_params or RequestParams()
        self._episode_id: Optional[UUID] = kwargs.get("episode_id")

        t0_config = self.agent.app.context.secrets.get("tensorzero", {})
        t0_uri = t0_config.get("uri")

        if not t0_uri:
            raise ProviderKeyError(
                "'uri' not found under 'tensorzero' key in secrets. Please update "
                "fastagent.secrets.yaml like:\n\n"
                "tensorzero:\n"
                "  uri: <your_t0_uri>\n"
            )

        try:
            self.gateway = AsyncTensorZeroGateway(t0_uri)
        except Exception as e:
             # Use single quotes for the f-string
            raise ProviderModelError(f'Failed to initialize TensorZero Gateway: {e}')

    def _map_messages_to_t0(
        self, messages: List[PromptMessageMultipart]
    ) -> Dict[str, Any]:
        """Maps fast-agent messages to the T0 input format."""
        t0_messages = []
        system_prompt = None

        for msg in messages:
            role = msg.role.value  # e.g., 'user', 'assistant', 'system', 'tool'

            if role == "system":
                # T0 handles system prompts separately
                if isinstance(msg.content, str):
                    system_prompt = msg.content
                else:
                    # Handle cases where system prompt might have complex content (though unusual)
                    # For now, only take the first text part if available.
                    first_text = next(
                        (part.text for part in msg.content if part.type == "text"), None
                    )
                    if first_text:
                        system_prompt = first_text
                    else:
                        self.agent.app.context.logger.warning(
                            "Complex system prompt content type not fully supported for T0, ignoring."
                        )
                continue  # Skip adding system messages to the main list

            t0_content: List[Dict[str, Any]] = []
            if isinstance(msg.content, str):
                t0_content.append({"type": "text", "text": msg.content})
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.type == "text":
                        t0_content.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        # Map image_url to T0's image type with url
                        t0_content.append({"type": "image", "url": part.image_url.url})
                    elif part.type == "image_data":
                        # Check if the necessary attributes exist
                        if hasattr(part, 'image_data') and hasattr(part.image_data, 'data') and hasattr(part.image_data, 'mime_type'):
                            t0_content.append({
                                "type": "image",
                                "data": part.image_data.data,
                                "mime_type": part.image_data.mime_type
                            })
                        else:
                            self.agent.app.context.logger.warning(
                                f"Image data part type missing expected attributes (data, mime_type): {part}"
                            )
                    elif part.type == "tool_call":
                        # T0 expects tool calls from the assistant
                        if role == "assistant":
                           t0_content.append(
                                {
                                    "type": "tool_call",
                                    "id": part.id,
                                    "name": part.name,
                                    # T0 expects arguments as dict, fast-agent has string
                                    "arguments": json.loads(part.arguments or '{}'),
                                }
                           )
                        else:
                             self.agent.app.context.logger.warning(
                                f"Tool call content found for non-assistant role '{role}', skipping for T0."
                            )
                    elif part.type == "tool_result":
                         # T0 expects tool results from the user role
                        if role == "user" or role == "tool": # Map fast-agent 'tool' role to T0 'user' role here
                             role = "user" # Force role to user for T0 tool results
                             t0_content.append(
                                 {
                                     "type": "tool_result",
                                     "tool_call_id": part.tool_call_id,
                                     "name": part.name,
                                     # T0 expects 'result', fast-agent has 'content'
                                     "result": part.content,
                                 }
                             )
                        else:
                              self.agent.app.context.logger.warning(
                                f"Tool result content found for non-user/tool role '{role}', skipping for T0."
                            )
                    # Add other type mappings here if needed (e.g., arguments)

            if t0_content:
                 # Ensure role is compatible with T0 ('user', 'assistant')
                 # Map 'tool' role from fast-agent to 'user' for tool results as handled above
                 if role == "tool":
                     role = "user" # Already handled if content was tool_result, redundant but safe

                 if role not in ["user", "assistant"]:
                     self.agent.app.context.logger.warning(
                            f"Unsupported role '{role}' for T0, mapping to 'user'. Content: {t0_content}"
                        )
                     role = "user" # Default fallback

                 t0_messages.append({"role": role, "content": t0_content})


        t0_input = {"messages": t0_messages}
        if system_prompt:
            t0_input["system"] = system_prompt

        self.agent.app.context.logger.debug(f"Mapped fast-agent messages to T0 input: {t0_input}")
        return t0_input

    def _map_params_to_t0(
        self, request_params: RequestParams, **kwargs: Any
    ) -> Dict[str, Any]:
        """Maps fast-agent request parameters to T0 'params' format."""
        t0_params: Dict[str, Any] = {}
        chat_completion_params: Dict[str, Any] = {}

        if request_params.temperature is not None:
            chat_completion_params["temperature"] = request_params.temperature
        if request_params.max_tokens is not None:
            chat_completion_params["max_tokens"] = request_params.max_tokens
        if request_params.top_p is not None:
            chat_completion_params["top_p"] = request_params.top_p
        # TODO: Map other common params if needed (frequency_penalty, presence_penalty, seed)

        # Handle response format (JSON mode)
        if request_params.response_format == {"type": "json_object"}:
            chat_completion_params["json_mode"] = "on"  # Using string enum value

        # Handle tools - T0 guide suggests tools are often part of the function config,
        # but can be overridden/specified in params.
        if request_params.tools:
             chat_completion_params["tools"] = [tool.model_dump() for tool in request_params.tools]

             # Handle tool_choice
             if request_params.tool_choice:
                 if isinstance(request_params.tool_choice, str):
                      chat_completion_params["tool_choice"] = request_params.tool_choice # e.g., "auto", "none", "required"
                 elif hasattr(request_params.tool_choice, 'function'):
                      # Map specific function choice if needed by T0 format
                      chat_completion_params["tool_choice"] = {
                          "type": "function",
                          "function": {"name": request_params.tool_choice.function.name}
                      }


        if chat_completion_params:
             t0_params["chat_completion"] = chat_completion_params

        # Add any other T0 specific params from kwargs if needed

        return t0_params


    def _adapt_t0_usage(self, usage: Optional[Usage]) -> Optional[Dict[str, int]]:
        """Adapts T0 Usage object to fast-agent dictionary format."""
        if not usage:
            return None
        return {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        }

    def _adapt_t0_finish_reason(self, reason: Optional[str]) -> Optional[str]:
         """Maps T0 finish reason string."""
         # T0 reasons seem compatible: stop, length, tool_call, content_filter
         return reason

    def _adapt_t0_response_content(
        self, content_blocks: List[ContentBlock]
    ) -> tuple[Optional[str], Optional[dict]]:
        """Extracts text and function call from T0 content blocks."""
        text_content = None
        function_call = None

        # Assuming primarily one type of response block or tool call first
        if content_blocks:
            first_block = content_blocks[0]
            if first_block.type == "text":
                text_content = first_block.text
            elif first_block.type == "tool_call":
                # Map T0 ToolCall to fast-agent function_call dict
                function_call = {
                    "id": first_block.id, # Pass ID for multi-turn
                    "name": first_block.name,
                    # fast-agent expects arguments as JSON string
                    "arguments": json.dumps(first_block.arguments or {}),
                }
            elif first_block.type == "thought":
                 # Log thoughts if needed, but don't typically return as primary content
                 self.agent.app.context.logger.debug(f"T0 thought: {first_block.text}")
                 # Check subsequent blocks for actual content
                 if len(content_blocks) > 1:
                     return self._adapt_t0_response_content(content_blocks[1:])


        return text_content, function_call


    def _adapt_t0_non_streaming(
        self, t0_response: Union[ChatInferenceResponse, JsonInferenceResponse]
    ) -> AugmentedLLMResponse:
        """Adapts a non-streaming T0 response to AugmentedLLMResponse."""
        self._episode_id = t0_response.episode_id # Store for potential follow-up calls
        text_content = None
        function_call = None
        raw_response = None # Store raw if needed

        if isinstance(t0_response, ChatInferenceResponse):
            text_content, function_call = self._adapt_t0_response_content(t0_response.content)
            raw_response = t0_response # Or serialize if needed
        elif isinstance(t0_response, JsonInferenceResponse):
            # Assuming JSON mode means the primary content is the JSON string/object
            if t0_response.output:
                 text_content = t0_response.output.raw # Return raw JSON string
                 # Alternatively, could return parsed if fast-agent expects dict
                 # text_content = json.dumps(t0_response.output.parsed)
            raw_response = t0_response # Or serialize

        return AugmentedLLMResponse(
            id=str(t0_response.inference_id),
            model=t0_response.variant_name, # Use the actual variant T0 used
            provider_name="tensorzero",
            role="assistant",
            content=text_content,
            function_call=function_call,
            usage=self._adapt_t0_usage(t0_response.usage),
            finish_reason=self._adapt_t0_finish_reason(t0_response.finish_reason.value if t0_response.finish_reason else None),
            raw_response=raw_response,
            episode_id=str(self._episode_id) # Pass episode ID back
        )


    def _adapt_t0_streaming_chunk(
        self, t0_chunk: Union[ChatChunk, JsonChunk]
    ) -> AugmentedLLMResponse:
        """Adapts a streaming T0 chunk to AugmentedLLMResponse."""
        self._episode_id = t0_chunk.episode_id # Store for potential follow-up calls
        delta_content = None
        delta_function_call_name = None
        delta_function_call_args = None
        function_call_id = None # Needed for tool call chunks
        is_final_chunk = t0_chunk.usage is not None # Usage only in final chunk

        if isinstance(t0_chunk, ChatChunk):
            if t0_chunk.content:
                first_chunk_content = t0_chunk.content[0]
                if first_chunk_content.type == "text":
                    delta_content = first_chunk_content.text
                elif first_chunk_content.type == "tool_call":
                    # Streamed tool calls provide incremental raw_name and raw_arguments
                    function_call_id = first_chunk_content.id
                    # Only populate deltas if they exist in the chunk
                    if hasattr(first_chunk_content, 'raw_name') and first_chunk_content.raw_name:
                        delta_function_call_name = first_chunk_content.raw_name
                    if hasattr(first_chunk_content, 'raw_arguments') and first_chunk_content.raw_arguments:
                         delta_function_call_args = first_chunk_content.raw_arguments
                elif first_chunk_content.type == "thought":
                     # Log thoughts, but don't yield as primary content delta
                     self.agent.app.context.logger.debug(f"T0 thought chunk: {first_chunk_content.text}")


        elif isinstance(t0_chunk, JsonChunk):
            # Treat JSON chunks as text deltas for now
            delta_content = t0_chunk.raw

        # Construct function call delta if applicable
        function_call_delta = None
        if delta_function_call_name or delta_function_call_args:
             function_call_delta = {"id": function_call_id} # Include ID
             if delta_function_call_name:
                 function_call_delta["name"] = delta_function_call_name
             if delta_function_call_args:
                 function_call_delta["arguments"] = delta_function_call_args


        return AugmentedLLMResponse(
            id=str(t0_chunk.inference_id),
            model=t0_chunk.variant_name,
            provider_name="tensorzero",
            role="assistant",
            delta_content=delta_content,
            # Use delta_function_call for streaming
            delta_function_call=function_call_delta,
            # Only populate these in the final chunk
            usage=self._adapt_t0_usage(t0_chunk.usage) if is_final_chunk else None,
            finish_reason=self._adapt_t0_finish_reason(t0_chunk.finish_reason.value if t0_chunk.finish_reason else None) if is_final_chunk else None,
            raw_response=t0_chunk if is_final_chunk else None, # Optionally include final chunk raw
            episode_id=str(self._episode_id) # Pass episode ID back
        )

    async def generate(
        self,
        messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[AugmentedLLMResponse, None]:
        """
        Generate responses using the TensorZero Gateway.

        Args:
            messages: The list of prompt messages.
            request_params: Request parameters overriding defaults.
            **kwargs: Additional arguments (e.g., episode_id for continuation).

        Yields:
            AugmentedLLMResponse objects (one for non-streaming, multiple for streaming).
        """
        merged_params = self.request_params.merge(request_params)
        stream = merged_params.stream

        t0_input = self._map_messages_to_t0(messages)
        t0_params = self._map_params_to_t0(merged_params, **kwargs)

        # Get episode_id from kwargs if provided for continuation, else use stored one
        current_episode_id = kwargs.get("episode_id") or self._episode_id

        try:
            if stream:
                response_stream = await self.gateway.inference(
                    function_name=self.t0_function_name,
                    input=t0_input,
                    params=t0_params,
                    stream=True,
                    episode_id=current_episode_id,
                )
                async for chunk in response_stream:
                    yield self._adapt_t0_streaming_chunk(chunk)
            else:
                response = await self.gateway.inference(
                    function_name=self.t0_function_name,
                    input=t0_input,
                    params=t0_params,
                    stream=False,
                    episode_id=current_episode_id,
                )
                yield self._adapt_t0_non_streaming(response)

        except TensorZeroError as e:
             self.agent.app.context.logger.error(f"TensorZero API Error: {e}")
             raise ProviderModelError(f"TensorZero API Error: {e}")
        except json.JSONDecodeError as e:
             self.agent.app.context.logger.error(f"Failed to decode JSON in tool call/result mapping: {e}")
             raise ProviderModelError(f"Failed to decode JSON in tool mapping: {e}")
        except Exception as e:
             self.agent.app.context.logger.error(f"Unexpected error during TensorZero inference: {e}")
             raise ProviderModelError(f"Unexpected error during TensorZero inference: {e}")

    # --- Protocol properties ---

    @property
    def provider(self) -> str:
        return "tensorzero"

    async def shutdown(self):
        """Perform any cleanup if necessary by closing the TensorZero gateway client."""

        if hasattr(self, 'gateway') and self.gateway is not None:
            try:
                await self.gateway.close()
                self.agent.app.context.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                 self.agent.app.context.logger.error(f"Error closing TensorZero Gateway client: {e}")

    def get_caller_id(self) -> str:
        return f"{self.agent.name} <{self.__class__.__name__}>" 