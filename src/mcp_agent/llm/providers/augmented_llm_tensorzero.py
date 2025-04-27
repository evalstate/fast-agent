import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from tensorzero.types import ChatChunk, JsonChunk, TensorZeroError
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    JsonInferenceResponse,
)


from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError, ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.provider_types import Provider

from mcp_agent.llm.providers.multipart_converter_openai import OpenAIConverter
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import AugmentedLLMProtocol
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

from mcp.types import (
    Role,
    TextContent,
    ImageContent, 
    EmbeddedResource
)

from mcp_agent.ui.console_display import ConsoleDisplay


class TensorZeroAugmentedLLM(AugmentedLLMProtocol):
    """
    AugmentedLLM implementation for TensorZero using its native API.
    Uses LAZY INITIALIZATION for the gateway client.
    """

    def __init__(self, agent: Agent, model: str, request_params: Optional[RequestParams] = None, **kwargs: Any):
        """
        Initialize TensorZero LLM placeholders. Client is created lazily.
        'model' is the T0 function name (e.g., 'chat').
        """
        self.agent = agent
        self.logger = get_logger(f"{__name__}.{self.agent.name}")
        self.request_params = request_params or RequestParams()
        self.t0_function_name: str = model
        self._episode_id: Optional[str] = kwargs.get("episode_id")
        self.instruction = agent.instruction

        # Initialize gateway client to None - will be created lazily
        self.gateway: Optional[AsyncTensorZeroGateway] = None
        self._resolved_uri: Optional[str] = None # Store resolved URI after lazy init

        self.display: Optional[ConsoleDisplay] = None
        context = self.agent.context # Use direct access now
        if context and context.config:
             try:
                  self.display = ConsoleDisplay(config=context.config)
                  self.logger.debug("ConsoleDisplay manually initialized.")
             except Exception as e:
                  self.logger.error(f"Failed to manually initialize ConsoleDisplay: {e}")
        else:
             # This case means context is missing during init, log it.
             self.logger.warning("Context or config missing during init, cannot initialize ConsoleDisplay.")
        # --- Manually initialize display --- END

        self.logger.info(f"TensorZero LLM provider initialized for function '{self.t0_function_name}' (client pending).")

    async def _ensure_gateway_initialized(self) -> AsyncTensorZeroGateway:
        """Initializes the gateway client if not already done."""
        if self.gateway is None:
            self.logger.debug("First use: Initializing AsyncTensorZeroGateway client...")
            try:
                # --- Resolve URI and Create Client ---
                context = self.agent.context # Use direct context access
                if not context:
                    raise ModelConfigError("Agent context not available for lazy T0 client initialization.")

                # Resolve URI (checking config then secrets)
                base_url = None
                uri = None
                if context.config and hasattr(context.config, 'tensorzero') and context.config.tensorzero:
                     t0_config = context.config.tensorzero
                     base_url = getattr(t0_config, 'base_url', None)
                     if not base_url: uri = getattr(t0_config, 'uri', None)
                elif hasattr(context, 'secrets') and context.secrets:
                     t0_secrets = context.secrets.get("tensorzero", {})
                     uri = t0_secrets.get("uri")

                self._resolved_uri = base_url or uri
                if not self._resolved_uri:
                    raise ModelConfigError(
                        "TensorZero native base URL/URI not configured.",
                        "Configure 'base_url' or 'uri' under 'tensorzero' in config/secrets."
                    )

                # API key resolution (optional for T0 native)
                # Add logic here if T0 native gateway requires API key authentication
                # api_key = self._resolve_api_key_lazy(context)

                self.gateway = AsyncTensorZeroGateway(base_url=self._resolved_uri)
                self.logger.info(f"TensorZero Gateway client initialized for URI: {self._resolved_uri}")
                # --- End Resolve URI and Create Client ---

            except ModelConfigError as e: # Catch config errors during lazy init
                 self.logger.error(f"Configuration error during lazy T0 client init: {e}")
                 raise # Re-raise config errors
            except Exception as e:
                 self.logger.error(f"Failed to initialize AsyncTensorZeroGateway lazily: {e}")
                 raise ModelConfigError(f"Failed to initialize AsyncTensorZeroGateway lazily: {e}") from e

        # Should not be None here due to checks/exceptions above
        assert self.gateway is not None
        return self.gateway

    def _prepare_t0_input(
        self, messages: List[PromptMessageMultipart], merged_params: RequestParams
    ) -> Dict[str, Any]:
        """Prepares the 'input' dictionary for the native T0 API call."""
        system_prompt_args = None
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
            system_prompt_args = merged_params.metadata.get("tensorzero_arguments")

        base_instruction = self.instruction or "You are a helpful assistant."
        t0_system_object = {
            "BASE_INSTRUCTIONS": base_instruction,
            "DISCLAIMER_TEXT": "", "BIBLE": "",
            "USER_PORTFOLIO_DESCRIPTION": "", "USER_PROFILE_DATA": "{}", "CONTEXT": ""
        }
        if isinstance(system_prompt_args, dict): t0_system_object.update(system_prompt_args)

        t0_messages = []
        for msg in messages:
            print(f"Processing message: {msg}")
            if msg.role != "system":
                t0_content = []
                for part in msg.content:
                    if isinstance(part, TextContent): t0_content.append({"type": "text", "text": part.text})
                    elif isinstance(part, ImageContent):
                        if hasattr(part, "url") and part.url: t0_content.append({"type": "image", "url": part.url})
                        elif hasattr(part, "data") and hasattr(part, "mime_type"): t0_content.append({"type": "image", "data": part.data, "mime_type": part.mime_type})
                if t0_content:
                    t0_messages.append({"role": msg.role, "content": t0_content})

        return {"messages": t0_messages, "system": t0_system_object}

    # --- Helper for Merging --- #
    def _merge_request_params(self, override_params: Optional[RequestParams]) -> RequestParams:
        """Merges provided params with the instance defaults."""
        # Start with a copy of instance defaults
        base_dump = self.request_params.model_dump(exclude_unset=True)

        if override_params:
            # Get override values, excluding unset
            override_dump = override_params.model_dump(exclude_unset=True)
            # Update base with override values
            base_dump.update(override_dump)

        # Create new RequestParams from merged dict
        return RequestParams(**base_dump)

    # --- AugmentedLLMProtocol Implementation --- #

    async def generate(
        self,
        messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ) -> PromptMessageMultipart:
        """Generate responses using the native TensorZero Gateway API."""
        gateway = await self._ensure_gateway_initialized()
        merged_params = self._merge_request_params(request_params)
        stream = False
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
             stream = merged_params.metadata.pop('stream', False)

        if stream:
            # --- Stream Aggregation Logic --- START
            self.logger.debug("Streaming requested, consuming stream to return single aggregated message...")
            final_message: Optional[PromptMessageMultipart] = None
            all_content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
            last_metadata: Optional[Dict[str, Any]] = None
            tool_call_assembly: Dict[str, Dict[str, Any]] = {}
            async for chunk_message in self.stream_generate(messages, merged_params, **kwargs):
                all_content_parts.extend(chunk_message.content)
                last_metadata = chunk_message.metadata
                if chunk_message.metadata and "delta_function_call" in chunk_message.metadata:
                    delta = chunk_message.metadata["delta_function_call"]
                    deltas_to_process = delta if isinstance(delta, list) else [delta]
                    for d in deltas_to_process:
                        if isinstance(d, dict) and d.get('id'):
                            call_id = d['id']
                            if call_id not in tool_call_assembly:
                                 tool_call_assembly[call_id] = {"id": call_id, "name": "", "arguments": ""}
                            if d.get('name'): tool_call_assembly[call_id]["name"] = d['name']
                            if d.get('arguments'): tool_call_assembly[call_id]["arguments"] += d['arguments']

            if not all_content_parts and not tool_call_assembly:
                self.logger.warning("T0 stream yielded no content or tool calls during aggregation.")
                final_assembled_message = PromptMessageMultipart(role="assistant", content=[], metadata=last_metadata)
            else:
                if tool_call_assembly:
                    assembled_calls = list(tool_call_assembly.values())
                    if last_metadata is None: last_metadata = {}
                    for call in assembled_calls:
                         if isinstance(call.get("arguments"), dict):
                              call["arguments"] = json.dumps(call["arguments"], ensure_ascii=False)
                         elif not isinstance(call.get("arguments"), str):
                              call["arguments"] = ""
                    last_metadata["all_function_calls"] = assembled_calls
                    if assembled_calls:
                        last_metadata["function_call"] = assembled_calls[0]
                final_assembled_message = PromptMessageMultipart(
                    role="assistant",
                    content=all_content_parts,
                    metadata=last_metadata
                )

            # --- Display Assembled Message (use self.display) --- START
            display_text = final_assembled_message.all_text()
            if display_text and display_text != "<no text>" and self.display:
                 title = f"ASSISTANT/{self.t0_function_name}"
                 await self.display.show_assistant_message(message_text=display_text, title=title)
            # --- Display Assembled Message --- END
            self.logger.debug("Finished aggregating stream for generate().")
            print(f"!!! T0 Gen (Stream): Returning aggregated message: {final_assembled_message}")
            return final_assembled_message
            # --- Stream Aggregation Logic --- END
        else:
            # --- Non-Streaming Logic --- START
            gateway = await self._ensure_gateway_initialized()
            current_episode_id = kwargs.get("episode_id") or self._episode_id
            t0_input = self._prepare_t0_input(messages, merged_params)
            self.logger.debug(f"Calling T0 native inference (non-streaming)... InputKeys={list(t0_input.keys())}")
            try:
                print("!!! T0 Gen (Non-Stream): About to call gateway.inference")
                completion = await gateway.inference(
                    function_name=self.t0_function_name,
                    input=t0_input,
                    stream=False,
                    episode_id=current_episode_id,
                )
                print("!!! T0 Gen (Non-Stream): gateway.inference call completed")
                final_message = self._adapt_t0_native_completion(completion)

                # --- Display Assembled Message (use self.display) --- START
                display_text = final_message.all_text()
                if display_text and display_text != "<no text>" and self.display:
                    title = f"ASSISTANT/{self.t0_function_name}"
                    await self.display.show_assistant_message(message_text=display_text, title=title)
                # --- Display Assembled Message --- END

                print(f"!!! T0 Gen (Non-Stream): Returning message: {final_message}")
                return final_message
            except TensorZeroError as e:
                self.logger.error(f"T0 Native API Error: {e}")
                raise ModelConfigError(f"TensorZero Native API Error: {e}") from e
            except Exception as e:
                import traceback
                self.logger.error(f"Unexpected T0 Generate Error: {e}\n{traceback.format_exc()}")
                raise ModelConfigError(f"Unexpected T0 Generate Error: {e}") from e
            # --- Non-Streaming Logic --- END

    async def stream_generate(
        self,
        messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[PromptMessageMultipart, None]:
        """Streaming-specific generate method that yields chunks as they arrive."""
        gateway = await self._ensure_gateway_initialized()
        merged_params = self._merge_request_params(request_params)
        stream = True # This method always streams
        current_episode_id = kwargs.get("episode_id") or self._episode_id
        t0_input = self._prepare_t0_input(messages, merged_params)

        self.logger.debug(f"Calling T0 native inference (streaming)... InputKeys={list(t0_input.keys())}")
        try:
            response_iter = await gateway.inference(
                function_name=self.t0_function_name,
                input=t0_input,
                stream=stream, # Pass True
                episode_id=current_episode_id,
            )
            async for chunk in response_iter:
                yield self._adapt_t0_native_streaming_chunk(chunk)

        except TensorZeroError as e:
            self.logger.error(f"T0 Native API Error (stream): {e}")
            raise ModelConfigError(f"TensorZero Native API Error: {e}") from e
        except Exception as e:
            import traceback
            self.logger.error(f"Unexpected T0 Stream Generate Error: {e}\n{traceback.format_exc()}")
            raise ModelConfigError(f"Unexpected T0 Stream Generate Error: {e}") from e


    def _adapt_t0_native_completion(self, completion: Union[ChatInferenceResponse, JsonInferenceResponse]) -> PromptMessageMultipart:
        """Adapts a non-streaming native T0 response to PromptMessageMultipart."""
        # Add prints back for debugging
        print(f"!!! T0 Adapt: Adapting completion object: Type={type(completion)}")

        # --- Adapt Usage --- START
        usage_data = None
        if completion.usage:
             try:
                  usage_data = {
                       "input_tokens": completion.usage.input_tokens,
                       "output_tokens": completion.usage.output_tokens,
                  }
             except AttributeError:
                  self.logger.warning("Could not access expected tokens attributes on T0 Usage object.")
                  usage_data = str(completion.usage)
        # --- Adapt Usage --- END

        print(f"!!! T0 Adapt: Adapted usage data: {usage_data}")

        metadata = {
            "provider_name": "tensorzero",
            "t0_inference_id": str(completion.inference_id),
            "t0_episode_id": str(completion.episode_id),
            "t0_variant_name": completion.variant_name,
            "t0_usage": usage_data,
            "t0_finish_reason": completion.finish_reason.value if completion.finish_reason else None,
            "raw_response": completion,
        }
        print(f"!!! T0 Adapt: Initial metadata created.")

        # Update instance episode ID
        if completion.episode_id and str(completion.episode_id) != self._episode_id:
            self._episode_id = str(completion.episode_id)
            self.logger.debug(f"Updated episode_id: {self._episode_id}")

        content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        tool_calls_data = None # Initialize

        if isinstance(completion, ChatInferenceResponse):
            # print("!!! DEBUG: Completion is ChatInferenceResponse")
            tool_calls_data = []
            if hasattr(completion, 'content') and completion.content:
                 # print(f"!!! DEBUG: Processing {len(completion.content)} content blocks...")
                 for block in completion.content:
                     # print(f"!!! DEBUG: Processing block: Type={getattr(block, 'type', 'N/A')}, Value={block}")
                     block_type = getattr(block, 'type', None)
                     if block_type == "text":
                         text_content = getattr(block, 'text', None)
                         if text_content is not None:
                              content_parts.append(TextContent(type="text", text=text_content))
                              # print(f"!!! DEBUG: Added TextContent: {text_content[:100]}...")
                         else:
                              # print("!!! DEBUG: Text block missing 'text' attribute.")
                              pass
                     elif block_type == "image":
                         # Handle image stream - might be complex, adapt simply for now
                         img_args = {"type": "image"}
                         if hasattr(block, 'url') and block.url: img_args["url"] = block.url
                         if hasattr(block, 'data') and hasattr(block, 'mime_type'): img_args["data"] = block.data
                         if hasattr(block, 'mime_type'): img_args["mime_type"] = block.mime_type
                         if "url" in img_args or "data" in img_args: content_parts.append(ImageContent(**img_args))
                     elif block_type == "tool_call":
                         tool_call_info = {
                             "id": getattr(block, 'id', None),
                             "name": getattr(block, 'name', None),
                             "arguments": json.dumps(getattr(block, 'arguments', {}) or {})
                         }
                         tool_calls_data.append(tool_call_info)
                         # print(f"!!! DEBUG: Added Tool Call to data list: {tool_call_info}")
                     elif block_type == "thought":
                         thought_text = getattr(block, 'text', None)
                         metadata["t0_thought"] = thought_text
                         # print(f"!!! DEBUG: Added Thought to metadata: {thought_text}")
                     else:
                          # print(f"!!! DEBUG: Skipping unknown block type: {block_type}")
                          pass
            else:
                 # print("!!! DEBUG: ChatInferenceResponse has no 'content' attribute or content is empty.")
                 pass

            if tool_calls_data:
                metadata["function_call"] = tool_calls_data[0]
                metadata["all_function_calls"] = tool_calls_data
                # print(f"!!! DEBUG: Final tool calls in metadata: {tool_calls_data}")

        elif isinstance(completion, JsonInferenceResponse):
            # print("!!! DEBUG: Completion is JsonInferenceResponse")
            if completion.output and completion.output.raw:
                content_parts.append(TextContent(type="text", text=completion.output.raw))
                # print(f"!!! DEBUG: Added JSON content: {completion.output.raw[:100]}...")
            metadata["json_output_parsed"] = completion.output.parsed if completion.output else None
            # print(f"!!! DEBUG: Parsed JSON: {metadata['json_output_parsed']}")

        # print(f"!!! DEBUG: Final content parts count: {len(content_parts)}")
        # print(f"!!! DEBUG: Final metadata: {metadata}")
        # print(f"!!! DEBUG: Returning PromptMessageMultipart with content: {content_parts}")
        # Use string literal for role
        return PromptMessageMultipart(role="assistant", content=content_parts, metadata=metadata)


    def _adapt_t0_native_streaming_chunk(self, chunk: Union[ChatChunk, JsonChunk]) -> PromptMessageMultipart:
        """Adapts a native T0 streaming chunk to PromptMessageMultipart."""
        metadata = {
            "provider_name": "tensorzero",
            "t0_inference_id": str(chunk.inference_id),
            "t0_episode_id": str(chunk.episode_id),
            "t0_variant_name": chunk.variant_name,
        }
        # Update instance episode ID
        if chunk.episode_id and str(chunk.episode_id) != self._episode_id:
             self._episode_id = str(chunk.episode_id)
             self.logger.debug(f"Updated episode_id from chunk: {self._episode_id}")

        delta_content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        is_final_chunk = hasattr(chunk, "usage") and chunk.usage is not None

        if isinstance(chunk, ChatChunk):
             if chunk.content:
                 # chunk.content here is List[ContentBlockChunk]
                 for content_chunk in chunk.content:
                      if hasattr(content_chunk, 'type'): # Basic check for safety
                           if content_chunk.type == "text":
                                delta_content_parts.append(TextContent(type="text", text=getattr(content_chunk, 'text', '')))
                           elif content_chunk.type == "image":
                                # Handle image stream - might be complex, adapt simply for now
                                img_args = {"type": "image"}
                                if hasattr(content_chunk, 'url'): img_args["url"] = content_chunk.url
                                if hasattr(content_chunk, 'data'): img_args["data"] = content_chunk.data
                                if hasattr(content_chunk, 'mime_type'): img_args["mime_type"] = content_chunk.mime_type
                                if "url" in img_args or "data" in img_args: delta_content_parts.append(ImageContent(**img_args))
                           elif content_chunk.type == "tool_call":
                                # Store partial tool call info in metadata
                                partial_call = {"id": getattr(content_chunk, 'id', None)}
                                if hasattr(content_chunk, 'raw_name') and content_chunk.raw_name: partial_call["name"] = content_chunk.raw_name
                                if hasattr(content_chunk, 'raw_arguments') and content_chunk.raw_arguments: partial_call["arguments"] = content_chunk.raw_arguments
                                metadata["delta_function_call"] = partial_call
                           elif content_chunk.type == "thought":
                               metadata["t0_thought"] = getattr(content_chunk, 'text', '')

        elif isinstance(chunk, JsonChunk):
             if chunk.raw: delta_content_parts.append(TextContent(type="text", text=chunk.raw))

        # Add final chunk info
        if is_final_chunk:
            # --- Adapt Usage --- START
            usage_data = None
            if hasattr(chunk, 'usage') and chunk.usage:
                 try:
                      # Access attributes directly
                      usage_data = {
                           "input_tokens": chunk.usage.input_tokens,
                           "output_tokens": chunk.usage.output_tokens,
                      }
                 except AttributeError:
                      self.logger.warning("Could not access expected tokens attributes on T0 streaming Usage object.")
                      usage_data = str(chunk.usage) # Fallback
            # --- Adapt Usage --- END

            metadata["t0_usage"] = usage_data # Use adapted usage data
            metadata["t0_finish_reason"] = chunk.finish_reason.value if chunk.finish_reason else None
            # Store the raw chunk object itself
            metadata["raw_response"] = chunk

        return PromptMessageMultipart(role="assistant", content=delta_content_parts, metadata=metadata)


    @property
    def provider(self) -> str:
        return Provider.TENSORZERO.value

    async def shutdown(self):
        """Perform any cleanup if necessary by closing the client."""
        if self.gateway:
            try:
                await self.gateway.close()
                self.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                 self.logger.error(f"Error closing TensorZero Gateway client: {e}")

    def get_caller_id(self) -> str:
        return f"{self.agent.name} <{self.__class__.__name__}>"
