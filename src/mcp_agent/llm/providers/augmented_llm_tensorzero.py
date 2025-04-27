import json
import inspect # For checking async generator
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from uuid import UUID

from tensorzero.types import ChatChunk, JsonChunk, TensorZeroError
from tensorzero import (
    AsyncTensorZeroGateway,
    ChatInferenceResponse,
    JsonInferenceResponse,
)

from mcp_agent.agents.agent import Agent
from mcp_agent.context import Context
from mcp_agent.core.exceptions import ModelConfigError, ProviderKeyError
from mcp_agent.core.request_params import RequestParams

from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.memory import SimpleMemory # For type checking history
from mcp_agent.llm.provider_types import Provider
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart
from mcp_agent.core.prompt import Prompt # For error message fallback
from mcp.types import (
    Role, TextContent, ImageContent, EmbeddedResource
)

# Use Any for generic types for simplicity with base class
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
        self.t0_function_name: str = model # Store T0 function name early
        self._episode_id: Optional[str] = kwargs.get("episode_id")

        # Call the base class constructor FIRST
        super().__init__(agent=agent, model=model, provider=Provider.TENSORZERO, request_params=request_params, **kwargs)

        # Gateway client is initialized lazily
        self.gateway: Optional[AsyncTensorZeroGateway] = None
        self._resolved_uri: Optional[str] = None

        self.logger.info(f"TensorZero LLM provider initialized for function '{self.t0_function_name}' (client pending).")
        # Check history type after super init
        if not isinstance(self.history, SimpleMemory):
             self.logger.warning(f"History object type is {type(self.history)}. Storing PromptMessageMultipart might fail.")

    # _initialize_default_params is called by base __init__
    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Set default parameters. Ensures use_history=True for base class."""
        func_name = kwargs.get("model", self.t0_function_name or "unknown_t0_function")
        return RequestParams(
            model=func_name,
            systemPrompt=self.instruction,
            maxTokens=4096,
            use_history=True, # MUST be True for self.history to be used
            max_iterations=10,
        )

    async def _ensure_gateway_initialized(self) -> AsyncTensorZeroGateway:
        """Initializes the native T0 gateway client if not already done."""
        if self.gateway is None:
            self.logger.debug("First use: Initializing AsyncTensorZeroGateway client...")
            try:
                if not self.context:
                     raise ModelConfigError("Context not found for lazy T0 client initialization.")
                base_url = None; uri = None
                if self.context.config and hasattr(self.context.config, 'tensorzero') and self.context.config.tensorzero:
                    t0_config = self.context.config.tensorzero
                    base_url = getattr(t0_config, 'base_url', None)
                    if not base_url: uri = getattr(t0_config, 'uri', None)
                elif hasattr(self.context, 'secrets') and self.context.secrets:
                     t0_secrets = self.context.secrets.get("tensorzero", {})
                     uri = t0_secrets.get("uri")
                self._resolved_uri = base_url or uri
                if not self._resolved_uri:
                    raise ModelConfigError(
                        "TensorZero native base URL/URI not configured.",
                        "Configure 'base_url' or 'uri' under 'tensorzero' in config/secrets."
                    )
                self.gateway = AsyncTensorZeroGateway(base_url=self._resolved_uri)
                self.logger.info(f"TensorZero Gateway client initialized for URI: {self._resolved_uri}")
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

        # --- History Management --- START
        final_messages_to_send: List[PromptMessageMultipart] = []
        if merged_params.use_history:
             try:
                  # Retrieve history (expects List[PromptMessageMultipart] or compatible)
                  history = self.history.get() # Base class provides self.history
                  if isinstance(history, list):
                       # Basic check if items seem okay - might need more robust validation
                       valid_history = [m for m in history if isinstance(m, PromptMessageMultipart)]
                       final_messages_to_send.extend(valid_history)
                       self.logger.debug(f"Retrieved {len(valid_history)} messages from history.")
                  else:
                       self.logger.warning(f"History retrieval returned unexpected type: {type(history)}")
             except Exception as e:
                  self.logger.error(f"Error retrieving history: {e}")
        else:
             self.logger.debug("History usage disabled by request params.")

        # Add current turn's messages (passed into this method)
        final_messages_to_send.extend(multipart_messages)
        self.logger.debug(f"Total messages being sent to provider: {len(final_messages_to_send)}")
        # --- History Management --- END

        # Determine streaming preference
        stream = False
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
             stream = merged_params.metadata.pop('stream', False)

        current_episode_id = self._episode_id

        # Prepare native T0 input using the COMBINED list
        t0_input = self._prepare_t0_input(final_messages_to_send, merged_params)

        self.logger.debug(f"Calling T0 native inference. Stream={stream}, InputKeys={list(t0_input.keys())}\") # Log WITHOUT ParamKeys")

        final_assembled_message: Optional[PromptMessageMultipart] = None

        try:
            response_iter_or_completion = await gateway.inference(
                function_name=self.t0_function_name,
                input=t0_input,
                stream=stream,
                episode_id=current_episode_id,
            )

            if stream:
                # --- Stream Aggregation & Display --- #
                all_content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
                last_metadata: Optional[Dict[str, Any]] = None
                tool_call_assembly: Dict[str, Dict[str, Any]] = {}
                full_text_for_display = "" # Not needed if displaying chunk by chunk

                async for chunk in response_iter_or_completion:
                    chunk_message = self._adapt_t0_native_streaming_chunk(chunk)
                    all_content_parts.extend(chunk_message.content)
                    last_metadata = chunk_message.metadata

                    # Display Delta
                    chunk_text = chunk_message.last_text()
                    if chunk_text and chunk_text != "<no text>":
                        title = f"ASSISTANT/{self.t0_function_name} (Streaming)"
                        print(f"!!! T0 APPLY (Stream Chunk): About to display delta: '{chunk_text[:50]}...'")
                        await self.show_assistant_message(message_text=chunk_text, title=title, is_delta=True)
                        print(f"!!! T0 APPLY (Stream Chunk): Finished display call.")

                    # Assemble Tool Calls from deltas
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

                # Assemble Final Message Object
                if not all_content_parts and not tool_call_assembly:
                     final_assembled_message = PromptMessageMultipart(role="assistant", content=[], metadata=last_metadata)
                else:
                     if tool_call_assembly:
                         assembled_calls = list(tool_call_assembly.values())
                         if last_metadata is None: last_metadata = {}
                         for call in assembled_calls: # Ensure args are strings
                              if not isinstance(call.get('arguments'), str): call['arguments'] = json.dumps(call.get('arguments',{}))
                         last_metadata["all_function_calls"] = assembled_calls
                         if assembled_calls:
                              last_metadata["function_call"] = assembled_calls[0]
                     final_assembled_message = PromptMessageMultipart(role="assistant", content=all_content_parts, metadata=last_metadata)

                self.logger.debug("Finished processing stream.")
                # --- End Stream Handling ---
            else:
                # --- Non-Streaming Handling --- #
                completion = response_iter_or_completion
                final_assembled_message = self._adapt_t0_native_completion(completion)

                # Display final message
                display_text = final_assembled_message.all_text()
                print(f"!!! T0 APPLY (Non-Stream): Text to display: '{display_text[:50]}...'")
                if display_text and display_text != "<no text>":
                    title = f"ASSISTANT/{self.t0_function_name}"
                    print(f"!!! T0 APPLY (Non-Stream): About to call self.show_assistant_message...")
                    await self.show_assistant_message(message_text=display_text, title=title)
                    print(f"!!! T0 APPLY (Non-Stream): Finished display call.")
                else:
                    print(f"!!! T0 APPLY (Non-Stream): Display condition not met (Text='{display_text}').")

            # --- Update History --- START
            # Add the original input messages for this turn PLUS the final response
            if final_assembled_message and merged_params.use_history:
                 # multipart_messages contains the input for THIS turn
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
            # --- Update History --- END

            # Return the final message (or empty if error occurred before assignment)
            return final_assembled_message or PromptMessageMultipart(role="assistant", content=[])

        # Error Handling
        except TensorZeroError as e:
             self.logger.error(f"T0 Error in _apply_prompt: {e}")
             error_content = TextContent(type="text", text=f"Error communicating with TensorZero: {e}")
             return PromptMessageMultipart(role="assistant", content=[error_content])
        except Exception as e:
             import traceback
             self.logger.error(f"Unexpected Error in _apply_prompt: {e}\n{traceback.format_exc()}")
             error_content = TextContent(type="text", text=f"Unexpected error: {e}")
             return PromptMessageMultipart(role="assistant", content=[error_content])

    # --- Native T0 Input/Param/Response Helpers --- #
    def _prepare_t0_input(self, messages: List[PromptMessageMultipart], merged_params: RequestParams) -> Dict[str, Any]:
        """Prepares the 'input' dictionary for the native T0 API call."""
        system_prompt_args = None
        if merged_params.metadata and isinstance(merged_params.metadata, dict): system_prompt_args = merged_params.metadata.get("tensorzero_arguments")
        base_instruction = self.instruction or "You are a helpful assistant."
        t0_system_object = {"BASE_INSTRUCTIONS": base_instruction, "DISCLAIMER_TEXT": "", "BIBLE": "", "USER_PORTFOLIO_DESCRIPTION": "", "USER_PROFILE_DATA": "{}", "CONTEXT": ""}
        if isinstance(system_prompt_args, dict): t0_system_object.update(system_prompt_args)
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
        return {"messages": t0_messages, "system": t0_system_object}

    def _adapt_t0_native_completion(self, completion: Union[ChatInferenceResponse, JsonInferenceResponse]) -> PromptMessageMultipart:
        """Adapts a non-streaming native T0 response to PromptMessageMultipart."""
        # print(f"!!! T0 Adapt: Adapting completion object: Type={type(completion)}")
        # ... (usage adaptation) ...
        # print(f"!!! T0 Adapt: Adapted usage data: {usage_data}")
        # ... (metadata setup) ...
        metadata = { ... } # Assume metadata setup is ok
        # print(f"!!! T0 Adapt: Initial metadata created.")
        # ... (episode ID update) ...

        content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        tool_calls_data = [] # Initialize always

        if isinstance(completion, ChatInferenceResponse):
            print(f"!!! T0 Adapt: Is ChatInferenceResponse. Has content attr? {hasattr(completion, 'content')}") # ADD
            if hasattr(completion, 'content') and completion.content:
                print(f"!!! T0 Adapt: completion.content = {completion.content}") # ADD
                for block in completion.content:
                     block_type = getattr(block, 'type', 'UNKNOWN')
                     print(f"!!! T0 Adapt: Processing block of type: {block_type}") # ADD
                     if block_type == "text":
                         text_val = getattr(block, 'text', None)
                         print(f"!!! T0 Adapt: Found text block, text value: '{text_val}'") # ADD
                         if text_val is not None:
                              new_part = TextContent(type="text", text=text_val)
                              print(f"!!! T0 Adapt: Created TextContent part: {new_part}") # ADD
                              content_parts.append(new_part)
                              print(f"!!! T0 Adapt: Appended part. content_parts now: {content_parts}") # ADD
                     elif block_type == "image":
                          # ... (image handling) ...
                          pass
                     elif block_type == "tool_call":
                          tool_call_info = {
                              "id": getattr(block, 'id', None),
                              "name": getattr(block, 'name', None),
                              "arguments": json.dumps(getattr(block, 'arguments', {}) or {})
                          }
                          tool_calls_data.append(tool_call_info)
                          # print(f"!!! T0 Adapt: Added Tool Call to data list: {tool_call_info}")
                     elif block_type == "thought":
                          thought_text = getattr(block, 'text', None)
                          metadata["t0_thought"] = thought_text
                          # print(f"!!! T0 Adapt: Added Thought to metadata: {thought_text}")
                     else:
                          print(f"!!! T0 Adapt: Skipping unknown block type: {block_type}") # ADD
            else:
                print("!!! T0 Adapt: No content found in ChatInferenceResponse.") # ADD

            # Assign tool call metadata (outside the loop)
            if tool_calls_data:
                 metadata["function_call"] = tool_calls_data[0]
                 metadata["all_function_calls"] = tool_calls_data
                 # print(f"!!! T0 Adapt: Final tool calls in metadata: {tool_calls_data}")

        elif isinstance(completion, JsonInferenceResponse):
            # ... (JSON handling)
            pass

        print(f"!!! T0 Adapt: Final content_parts before return: {content_parts}") # Keep this
        return PromptMessageMultipart(role="assistant", content=content_parts, metadata=metadata)

    def _adapt_t0_native_streaming_chunk(self, chunk: Union[ChatChunk, JsonChunk]) -> PromptMessageMultipart:
        # (Implementation remains the same, ensures self._episode_id is updated)
        metadata = {...}
        delta_content_parts = []
        # ... process chunk.content ...
        return PromptMessageMultipart(role="assistant", content=delta_content_parts, metadata=metadata)

    # --- Base Class Method Overrides (if any specific needed, otherwise inherit) --- #
    # Example: If T0 needs specific tool result format
    # def convert_function_results_to_provider_format(...)

    # --- Protocol properties --- #
    # Provider property inherited from AugmentedLLM
    # @property
    # def provider(self) -> str:

    async def shutdown(self):
        """Close the T0 gateway client if initialized."""
        if self.gateway:
            try:
                await self.gateway.close()
                self.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                 self.logger.error(f"Error closing TensorZero Gateway client: {e}")

    # get_caller_id is inherited from AugmentedLLM
    # def get_caller_id(self) -> str:
