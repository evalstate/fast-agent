import json
from typing import Any, Dict, List, Optional, Union

from tensorzero.types import ChatChunk, JsonChunk, TensorZeroError
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    JsonInferenceResponse,
)

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams

from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.memory import SimpleMemory 
from mcp_agent.llm.provider_types import Provider
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp.types import (
    TextContent,
    ImageContent,
    EmbeddedResource
)


class TensorZeroAugmentedLLM(AugmentedLLM[Any, Any]):
    """
    AugmentedLLM implementation for TensorZero using its native API.
    Inherits from AugmentedLLM for history and display integration.
    Uses LAZY INITIALIZATION for the gateway client.
    Overrides _apply_prompt_provider_specific.
    """

    def __init__(self, agent: Agent, model: str, request_params: Optional[RequestParams] = None, **kwargs: Any):
        """
        Initialize TensorZero LLM. 'model' is the T0 function name.
        Calls super().__init__ to set up base functionality.
        """
        self.t0_function_name: str = model
        self._episode_id: Optional[str] = kwargs.get("episode_id")

        super().__init__(agent=agent, model=model, provider=Provider.TENSORZERO, request_params=request_params, **kwargs)

        self.gateway: Optional[AsyncTensorZeroGateway] = None
        self._resolved_url: Optional[str] = None
        self.t0_system_template_vars: Dict[str, Any] = {}

        self.logger.info(f"TensorZero LLM provider initialized for function '{self.t0_function_name}' (client pending).")
        if not isinstance(self.history, SimpleMemory):
             self.logger.warning(f"History object type is {type(self.history)}. Storing PromptMessageMultipart might fail.")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Set default parameters. Ensures use_history=True for base class."""
        func_name = kwargs.get("model", self.t0_function_name or "unknown_t0_function")
        return RequestParams(
            model=func_name,
            systemPrompt=self.instruction,
            maxTokens=4096,
            use_history=True,
            max_iterations=10,
        )

    async def _ensure_gateway_initialized(self) -> AsyncTensorZeroGateway:
        """Initializes the native T0 gateway client if not already done."""
        if self.gateway is None:
            self.logger.debug("First use: Initializing AsyncTensorZeroGateway client...")
            try:
                if not self.context:
                     raise ModelConfigError("Context not found for lazy T0 client initialization.")
                base_url = None
                if self.context.config and hasattr(self.context.config, 'tensorzero') and self.context.config.tensorzero:
                    t0_config = self.context.config.tensorzero
                    base_url = getattr(t0_config, 'base_url', None)
                self._resolved_url = base_url
                if not self._resolved_url:
                    raise ModelConfigError(
                        "TensorZero native base URL not configured.",
                        "Configure 'base_url' in fastagent.config.yaml"
                    )
                self.gateway = AsyncTensorZeroGateway(base_url=self._resolved_url)
                self.logger.info(f"TensorZero Gateway client initialized for URL: {self._resolved_url}")
            except Exception as e:
                 self.logger.error(f"Failed to initialize T0 Gateway lazily: {e}")
                 raise ModelConfigError(f"Failed to initialize T0 Gateway lazily: {e}") from e
        assert self.gateway is not None
        return self.gateway

    # --- Implement the required abstract method --- #
    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
    ) -> PromptMessageMultipart:
        """
        Provider-specific implementation called by base generate().
        Handles native T0 API call, history management, streaming aggregation/display.
        Returns the *final* aggregated message.
        """
        gateway = await self._ensure_gateway_initialized()
        merged_params = self.get_request_params(request_params)

        final_messages_to_send: List[PromptMessageMultipart] = []
        if merged_params.use_history:
             try:
                  history = self.history.get()
                  if isinstance(history, list):
                       valid_history = [m for m in history if isinstance(m, PromptMessageMultipart)]
                       final_messages_to_send.extend(valid_history)
                       self.logger.debug(f"Retrieved {len(valid_history)} messages from history.")
                  else:
                       self.logger.warning(f"History retrieval returned unexpected type: {type(history)}")
             except Exception as e:
                  self.logger.error(f"Error retrieving history: {e}")
        else:
             self.logger.debug("History usage disabled by request params.")

        final_messages_to_send.extend(multipart_messages)
        self.logger.debug(f"Total messages being sent to provider: {len(final_messages_to_send)}")


        current_episode_id = self._episode_id

        # Prepare native T0 input using the COMBINED list
        t0_input = self._prepare_t0_input(final_messages_to_send, merged_params)
        final_assembled_message: Optional[PromptMessageMultipart] = None

        try:
            response_iter_or_completion = await gateway.inference(
                function_name=self.t0_function_name,
                input=t0_input,
                stream=False,
                episode_id=current_episode_id,
            )
            completion = response_iter_or_completion
            final_assembled_message = self._adapt_t0_native_completion(completion)

            display_text = final_assembled_message.all_text()
            if display_text and display_text != "<no text>":
                title = f"ASSISTANT/{self.t0_function_name}"
                await self.show_assistant_message(message_text=display_text, title=title)


            if final_assembled_message and merged_params.use_history:
                 messages_to_add_to_history = multipart_messages + [final_assembled_message]
                 try:
                      if hasattr(self.history, 'add'):
                           # Try adding the list directly (depends on Memory implementation)
                           self.history.add(messages_to_add_to_history)
                           self.logger.debug(f"Added {len(messages_to_add_to_history)} messages to history object.")
                      elif hasattr(self.history, 'set'): # Fallback? Check SimpleMemory source
                           current_history = self.history.get() or []
                           self.history.set(current_history + messages_to_add_to_history)
                           self.logger.debug(f"Set history object with {len(messages_to_add_to_history)} new messages.")
                      else:
                           self.logger.warning("self.history object does not have an 'add' or 'set' method.")
                 except Exception as e:
                      self.logger.error(f"Failed to add messages to history: {e}")

            return final_assembled_message or PromptMessageMultipart(role="assistant", content=[])

        except TensorZeroError as e:
             self.logger.error(f"T0 Error in _apply_prompt: {e}")
             error_content = TextContent(type="text", text=f"Error communicating with TensorZero: {e}")
             return PromptMessageMultipart(role="assistant", content=[error_content])
        except Exception as e:
             import traceback
             self.logger.error(f"Unexpected Error in _apply_prompt: {e}\n{traceback.format_exc()}")
             error_content = TextContent(type="text", text=f"Unexpected error: {e}")
             return PromptMessageMultipart(role="assistant", content=[error_content])

    def _prepare_t0_input(
        self,
        messages: List[PromptMessageMultipart],
        merged_params: RequestParams
    ) -> Dict[str, Any]:
        """Prepares the 'input' dictionary using instance template vars and metadata overrides."""
        t0_system_object = self.t0_system_template_vars.copy()
        metadata_args = None
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
            metadata_args = merged_params.metadata.get("tensorzero_arguments")
        if isinstance(metadata_args, dict):
            t0_system_object.update(metadata_args)
            self.logger.debug(f"Merged tensorzero_arguments from metadata: {metadata_args}")

        t0_messages = []
        for msg in messages:
            if msg.role != "system":
                t0_content = []
                for part in msg.content:
                    if isinstance(part, TextContent): t0_content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        if hasattr(part, "url") and part.url: t0_content.append({"type": "image", "url": part.url})
                        elif hasattr(part, "data") and hasattr(part, "mime_type"): t0_content.append({"type": "image", "data": part.data, "mime_type": part.mime_type})
                if t0_content: t0_messages.append({"role": msg.role, "content": t0_content})

        # Handle system prompt separately
        final_input = {
            "messages": t0_messages,
            "system": t0_system_object
        }
        self.logger.debug(f"Prepared T0 input with system args: {list(t0_system_object.keys())}")
        return final_input

    def _adapt_t0_native_completion(self, completion: Union[ChatInferenceResponse, JsonInferenceResponse]) -> PromptMessageMultipart:
        """Adapts a non-streaming native T0 response to PromptMessageMultipart."""
        usage_data = None
        t0_episode_id_str = str(completion.episode_id) if completion.episode_id else None

        metadata = {
            "provider_name": "tensorzero",
            "t0_inference_id": str(completion.inference_id),
            "t0_episode_id": t0_episode_id_str, # Use the string version
            "t0_variant_name": completion.variant_name,
            "t0_usage": usage_data,
            "t0_finish_reason": completion.finish_reason.value if completion.finish_reason else None,
            "raw_response": completion, # Store raw object
        }

        # Update instance episode ID if changed
        if t0_episode_id_str and self._episode_id != t0_episode_id_str:
             self.logger.warning(f"Updating stored episode_id to: {t0_episode_id_str}")
             self._episode_id = t0_episode_id_str

        content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        tool_calls_data = []
        
        for block in completion.content:
            block_type = getattr(block, 'type', 'UNKNOWN')
            if block_type == "text":
                text_val = getattr(block, 'text', None)
                if text_val is not None:
                    new_part = TextContent(type="text", text=text_val)
                    content_parts.append(new_part)
            elif block_type == "image":
                # TODO: ... (image handling) ...
                pass
            elif block_type == "tool_call":
                # TODO: ... (tool call handling) ...
                tool_call_info = {
                    "id": getattr(block, 'id', None),
                    "name": getattr(block, 'name', None),
                    "arguments": json.dumps(getattr(block, 'arguments', {}) or {})
                }
                tool_calls_data.append(tool_call_info)
            elif block_type == "thought":
                thought_text = getattr(block, 'text', None)
                metadata["t0_thought"] = thought_text
            else:
                self.logger.warning(f"T0 Adapt: Skipping unknown block type: {block_type}")

        # Assign tool call metadata (outside the loop)
        if tool_calls_data:
                metadata["function_call"] = tool_calls_data[0]
                metadata["all_function_calls"] = tool_calls_data

        elif isinstance(completion, JsonInferenceResponse):
            # TODO: ... (JSON handling) ...
            pass

        return PromptMessageMultipart(role="assistant", content=content_parts, metadata=metadata)

    def _adapt_t0_native_streaming_chunk(self, chunk: Union[ChatChunk, JsonChunk]) -> PromptMessageMultipart:
        metadata = {...}
        delta_content_parts = []
        return PromptMessageMultipart(role="assistant", content=delta_content_parts, metadata=metadata)


    async def shutdown(self):
        """Close the T0 gateway client if initialized."""
        if self.gateway:
            try:
                await self.gateway.close()
                self.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                 self.logger.error(f"Error closing TensorZero Gateway client: {e}")
