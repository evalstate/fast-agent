import json
from typing import Any, Dict, List, Optional, Union, cast, AsyncIterator, Tuple

from tensorzero.types import (
    ChatChunk,
    JsonChunk,
    TensorZeroError,
)

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
    EmbeddedResource,
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
)

from mcp_agent.mcp.helpers.content_helpers import get_text


# Helper function to convert block objects to dictionaries safely
def block_to_dict(block: Any) -> Dict[str, Any]:
    if hasattr(block, "model_dump"):
        try:
            return block.model_dump(mode="json")  # Use json mode for better serialization
        except:
            pass
    if hasattr(block, "__dict__"):
        try:
            return vars(block)
        except:
            pass
    # Fallback for unknown block types - represent minimally
    # Ensure basic types are handled correctly
    if isinstance(block, (str, int, float, bool, list, dict, type(None))):
        # This case shouldn't happen if blocks are objects, but as a safeguard
        return {"type": "raw", "content": block}
    return {"type": getattr(block, "type", "unknown"), "content": str(block)}


class TensorZeroAugmentedLLM(AugmentedLLM[Any, Any]):
    """
    AugmentedLLM implementation for TensorZero using its native API.
    Inherits from AugmentedLLM for history and display integration.
    Uses LAZY INITIALIZATION for the gateway client.
    Overrides _apply_prompt_provider_specific.
    """

    def __init__(
        self,
        agent: Agent,
        model: str,
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ):
        """
        Initialize TensorZero LLM. 'model' is the T0 function name.
        Calls super().__init__ to set up base functionality.
        """
        self.t0_function_name: str = model
        self._episode_id: Optional[str] = kwargs.get("episode_id")

        # Store the expected input keys if provided (useful for validation)
        self._t0_input_keys = kwargs.get("t0_input_keys", None)

        super().__init__(
            agent=agent,
            model=model,
            provider=Provider.TENSORZERO,
            request_params=request_params,
            **kwargs,
        )

        self.gateway: Optional[AsyncTensorZeroGateway] = None
        self._resolved_url: Optional[str] = None
        # This now specifically holds variables for the T0 function's 'input' dictionary
        self.t0_system_template_vars: Dict[str, Any] = kwargs.get(
            "t0_system_template_vars", kwargs.get("t0_function_input_vars", {})
        )

        self.logger.info(
            f"TensorZero LLM provider initialized for function '{self.t0_function_name}' (client pending). Input keys from t0_system_template_vars: {list(self.t0_system_template_vars.keys())}"
        )
        if not isinstance(self.history, SimpleMemory):
            self.logger.warning(
                f"History object type is {type(self.history)}. Storing PromptMessageMultipart might fail."
            )

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Set default parameters. Ensures use_history=True for base class."""
        func_name = kwargs.get("model", self.t0_function_name or "unknown_t0_function")
        return RequestParams(
            model=func_name,
            systemPrompt=self.instruction,
            maxTokens=4096,
            use_history=True,
            max_iterations=10,
            parallel_tool_calls=True,
        )

    async def _ensure_gateway_initialized(self) -> AsyncTensorZeroGateway:
        """Initializes the native T0 gateway client if not already done."""
        if self.gateway is None:
            self.logger.debug("First use: Initializing AsyncTensorZeroGateway client...")
            try:
                if not self.context:
                    raise ModelConfigError("Context not found for lazy T0 client initialization.")
                base_url = None
                if (
                    self.context.config
                    and hasattr(self.context.config, "tensorzero")
                    and self.context.config.tensorzero
                ):
                    t0_config = self.context.config.tensorzero
                    base_url = getattr(t0_config, "base_url", None)
                self._resolved_url = base_url
                if not self._resolved_url:
                    raise ModelConfigError(
                        "TensorZero native base URL not configured.",
                        "Configure 'base_url' in fastagent.config.yaml",
                    )
                self.gateway = AsyncTensorZeroGateway(base_url=self._resolved_url)
                self.logger.info(
                    f"TensorZero Gateway client initialized for URL: {self._resolved_url}"
                )
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
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        gateway = await self._ensure_gateway_initialized()
        merged_params = self.get_request_params(request_params)

        # --- Prepare Messages (including history and potential tool results from previous turn) ---
        history_messages: List[PromptMessageMultipart] = []
        if merged_params.use_history:
            try:
                retrieved_history = self.history.get()
                if isinstance(retrieved_history, list):
                    valid_history = [
                        m for m in retrieved_history if isinstance(m, PromptMessageMultipart)
                    ]
                    history_messages.extend(valid_history)
                    self.logger.debug(f"Retrieved {len(valid_history)} messages from history.")
                else:
                    self.logger.warning(
                        f"History retrieval returned unexpected type: {type(retrieved_history)}"
                    )
            except Exception as e:
                self.logger.error(f"Error retrieving history: {e}")
        else:
            self.logger.debug("History usage disabled by request params.")

        all_messages_for_api = history_messages + multipart_messages
        # Format messages, including potential tool results embedded by the framework
        t0_formatted_messages = self._format_messages_for_t0(all_messages_for_api)

        # --- Prepare Function Input (schema vars + messages) ---
        t0_system_vars = self._prepare_t0_function_input(merged_params)
        t0_api_input_dict = {
            "system": t0_system_vars,
            "messages": t0_formatted_messages,  # Includes formatted results if present
        }

        # --- Prepare Tools (as top-level arg) ---
        available_tools: Optional[List[Dict[str, Any]]] = await self._prepare_t0_tools()
        use_parallel_calls = merged_params.parallel_tool_calls if available_tools else False

        # --- Prepare other args ---
        current_episode_id = self._episode_id
        final_assembled_message: Optional[PromptMessageMultipart] = None

        # --- Single API Call (No Loop) ---
        try:
            self.logger.debug(f"Calling T0 inference for '{self.t0_function_name}'...")
            # --- Print Inference Payload ---
            payload_to_print = {
                "function_name": self.t0_function_name,
                "input": t0_api_input_dict,
                "additional_tools": available_tools,
                "parallel_tool_calls": use_parallel_calls,
                "episode_id": current_episode_id,
            }
            print(
                f"\n*** T0 Inference Payload (Iteration 1): ***\n{json.dumps(payload_to_print, indent=2)}\n*** End Payload ***\n"
            )

            response_iter_or_completion = await gateway.inference(
                function_name=self.t0_function_name,
                input=t0_api_input_dict,
                additional_tools=available_tools,
                parallel_tool_calls=use_parallel_calls,
                stream=False,
                episode_id=current_episode_id,
            )

            # --- Print Raw Gateway Response ---
            print(
                f"\n*** Raw T0 Gateway Response ({type(response_iter_or_completion)}): ***\n{response_iter_or_completion}\n*** End Raw Response ***\n"
            )

            # --- Process Response ---
            if isinstance(
                response_iter_or_completion, (ChatInferenceResponse, JsonInferenceResponse)
            ):
                completion = response_iter_or_completion
                # Adapt response: executes tools if requested, attaches results
                final_assembled_message = await self._adapt_t0_native_completion(
                    completion, available_tools
                )
            else:
                self.logger.error(f"Unexpected response type: {type(response_iter_or_completion)}")
                final_assembled_message = PromptMessageMultipart(
                    role="assistant",
                    content=[
                        TextContent(
                            type="text", text="Unexpected response type from TensorZero API"
                        )
                    ],
                )

        # --- Error Handling ---
        except TensorZeroError as e:
            # ... (improved error logging logic remains the same) ...
            error_details = ""
            detail = getattr(e, "detail", None)
            # ... (parsing detail/args) ...
            self.logger.error(f"T0 Error in _apply_prompt (HTTP {e.status_code}){error_details}")
            error_content = TextContent(
                type="text",
                text=f"Error communicating with TensorZero (HTTP {e.status_code}){error_details}",
            )
            return PromptMessageMultipart(role="assistant", content=[error_content])
        except Exception as e:
            # ... (generic exception handling remains the same) ...
            import traceback

            self.logger.error(f"Unexpected Error in _apply_prompt: {e}\n{traceback.format_exc()}")
            error_content = TextContent(type="text", text=f"Unexpected error: {e}")
            return PromptMessageMultipart(role="assistant", content=[error_content])

        # --- Update History & Display ---
        if final_assembled_message and merged_params.use_history:
            try:
                # Get the full history list to append to
                current_history = self.history.get() or []
                # Start building the messages for this turn to add to history
                history_to_add = multipart_messages + [final_assembled_message]

                # Check if tool results need to be added to history
                executed_results = getattr(final_assembled_message, "_executed_tool_results", None)
                if executed_results:
                    # Append the CallToolResult objects directly to the list
                    history_to_add.extend(executed_results)
                    self.logger.debug(
                        f"Adding {len(executed_results)} executed tool results to history update."
                    )
                    # Clean up the temporary attribute now that results are handled
                    delattr(final_assembled_message, "_executed_tool_results")

                # Combine with previous history and set
                self.history.set(current_history + history_to_add)
                self.logger.debug(
                    f"Updated self.history. Total messages: {len(current_history + history_to_add)}"
                )
            except Exception as e:
                self.logger.error(f"Failed to update self.history: {e}")

        # Display only the main assistant message content
        display_text = (
            final_assembled_message.all_text() if final_assembled_message else "<No Response>"
        )
        if display_text and display_text != "<no text>":
            title = f"ASSISTANT/{self.t0_function_name}"
            await self.show_assistant_message(message_text=display_text, title=title)

        return final_assembled_message or PromptMessageMultipart(role="assistant", content=[])

    def _get_text_from_call_tool_result(self, result: CallToolResult) -> str:
        """Helper to extract combined text from a CallToolResult's content list."""
        texts = []
        if result.content:
            for part in result.content:
                text = get_text(part)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    # Modified to handle embedding CallToolResult content
    def _format_messages_for_t0(
        self, messages: List[PromptMessageMultipart]
    ) -> List[Dict[str, Any]]:
        """Formats PromptMessageMultipart list (which might contain embedded tool results) into T0's API dict format."""
        t0_messages = []
        for msg in messages:
            if msg.role == "system":
                continue

            t0_content_blocks = []
            for part in msg.content:
                if isinstance(part, TextContent):
                    t0_content_blocks.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    if hasattr(part, "data") and part.data:
                        t0_content_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": getattr(part, "mimeType", "image/png"),
                                    "data": part.data,
                                },
                            }
                        )
                elif isinstance(part, CallToolResult):
                    # Found a tool result embedded in the message content
                    tool_use_id = getattr(part, "_t0_tool_use_id_temp", None)
                    tool_name = getattr(part, "_t0_tool_name_temp", None)

                    if tool_use_id and tool_name:
                        result_content_str = self._get_text_from_call_tool_result(part)
                        try:
                            json_result = json.dumps(result_content_str)
                        except TypeError:
                            json_result = json.dumps(str(result_content_str))

                        t0_content_blocks.append(
                            {
                                "type": "tool_result",
                                "id": tool_use_id,
                                "name": tool_name,
                                "result": json_result,
                            }
                        )
                    else:
                        self.logger.warning(
                            "Found CallToolResult in message content without required temp attributes (_t0_tool_use_id_temp, _t0_tool_name_temp)"
                        )
                # Add other content types like EmbeddedResource if needed

            if t0_content_blocks:
                valid_role = msg.role if msg.role in ["user", "assistant"] else "user"
                if valid_role != msg.role:
                    self.logger.warning(
                        f"Mapping message role '{msg.role}' to '{valid_role}' for T0."
                    )
                # Ensure content is always a list, even if only one block
                content_value = (
                    t0_content_blocks
                    if isinstance(t0_content_blocks, list)
                    else [t0_content_blocks]
                )
                t0_messages.append({"role": valid_role, "content": content_value})
        return t0_messages

    def _prepare_t0_function_input(self, merged_params: RequestParams) -> Dict[str, Any]:
        """Prepares the 'input' dictionary matching the function's expected schema (excluding messages)."""
        t0_func_input = self.t0_system_template_vars.copy()
        metadata_args = None
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
            metadata_args = merged_params.metadata.get("tensorzero_arguments")
        if isinstance(metadata_args, dict):
            if self._t0_input_keys:
                for key, value in metadata_args.items():
                    if key in self._t0_input_keys:
                        t0_func_input[key] = value
                    else:
                        self.logger.warning(
                            f"Ignoring metadata key '{key}' not in expected function input keys."
                        )
            else:
                t0_func_input.update(metadata_args)
            self.logger.debug(f"Merged tensorzero_arguments from metadata: {metadata_args}")

        self.logger.debug(
            f"Prepared T0 function input dict (excluding messages): {list(t0_func_input.keys())}"
        )
        return t0_func_input

    async def _prepare_t0_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches and formats tools, removing top-level 'type'."""
        formatted_tools: List[Dict[str, Any]] = []
        try:
            tools_response = await self.aggregator.list_tools()
            if tools_response and hasattr(tools_response, "tools") and tools_response.tools:
                for mcp_tool in tools_response.tools:
                    if (
                        not isinstance(mcp_tool.inputSchema, dict)
                        or mcp_tool.inputSchema.get("type") != "object"
                    ):
                        self.logger.warning(
                            f"Tool '{mcp_tool.name}' has potentially invalid parameters schema format for T0 (must be JSON schema object). Skipping."
                        )
                        continue

                    # Updated structure: remove 'type'
                    t0_tool_dict = {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description if mcp_tool.description else "",
                        "parameters": mcp_tool.inputSchema,
                    }
                    formatted_tools.append(t0_tool_dict)

                self.logger.debug(
                    f"Fetched and formatted {len(formatted_tools)} tools for TensorZero additional_tools"
                )
                return formatted_tools if formatted_tools else None
            else:
                self.logger.debug("No tools found via aggregator or response was empty.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to fetch or format tools: {e}")
            return None

    # Return type is now just PromptMessageMultipart
    async def _adapt_t0_native_completion(
        self,
        completion: Union[ChatInferenceResponse, JsonInferenceResponse],
        available_tools_for_display: Optional[List[Dict[str, Any]]] = None,
    ) -> PromptMessageMultipart:
        usage_data = None
        if hasattr(completion, "usage") and completion.usage:
            usage_data = {
                "input_tokens": getattr(completion.usage, "input_tokens", 0),
                "output_tokens": getattr(completion.usage, "output_tokens", 0),
            }
            self.logger.debug(f"T0 Usage: {usage_data}")

        t0_episode_id_str = str(completion.episode_id) if completion.episode_id else None
        if t0_episode_id_str and self._episode_id != t0_episode_id_str:
            self.logger.warning(
                f"Updating stored episode_id from {self._episode_id} to: {t0_episode_id_str}"
            )
            self._episode_id = t0_episode_id_str

        content_parts_this_turn: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        executed_tool_results: List[CallToolResult] = []

        if isinstance(completion, ChatInferenceResponse) and hasattr(completion, "content"):
            for block in completion.content:
                block_type = getattr(block, "type", "UNKNOWN")

                if block_type == "text":
                    text_val = getattr(block, "text", None)
                    if text_val is not None:
                        content_parts_this_turn.append(TextContent(type="text", text=text_val))

                elif block_type == "tool_call":
                    # --- Store raw block as dict in content ---
                    # tool_call_block_dict = block_to_dict(block)
                    # REMOVED: content_parts_this_turn.append(tool_call_block_dict) # Add the dict directly

                    # --- Execute the tool ---
                    tool_use_id = getattr(block, "id", None)
                    tool_name = getattr(block, "name", None)
                    # Get arguments, ensure it's a dict
                    tool_input_raw = getattr(block, "arguments", None)
                    tool_input = {}  # Define tool_input before conditional assignment
                    if isinstance(tool_input_raw, dict):
                        tool_input = tool_input_raw
                    elif tool_input_raw is not None:
                        # Attempt to parse if it's a string, otherwise default to empty dict
                        try:
                            tool_input = (
                                json.loads(str(tool_input_raw))
                                if isinstance(tool_input_raw, str)
                                else {}
                            )
                        except:
                            tool_input = {}
                    # else: tool_input remains {}

                    if tool_use_id and tool_name:
                        self.show_tool_call(
                            available_tools_for_display, tool_name, json.dumps(tool_input)
                        )
                        # Ensure arguments is dict before passing
                        mcp_tool_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=tool_name, arguments=tool_input or {}
                            ),
                        )
                        try:
                            result: CallToolResult = await self.call_tool(
                                mcp_tool_request, tool_use_id
                            )
                            # Store temporary attributes needed by _format_messages_for_t0
                            setattr(result, "_t0_tool_use_id_temp", tool_use_id)
                            setattr(result, "_t0_tool_name_temp", tool_name)
                            executed_tool_results.append(result)
                            self.show_oai_tool_result(str(result))
                        except Exception as e:
                            # ... (Error handling, attach temp attributes to error_result) ...
                            error_result = CallToolResult(
                                isError=True,
                                content=[
                                    TextContent(
                                        type="text",
                                        text=f"Error executing tool {tool_name}: {str(e)}",
                                    )
                                ],
                            )
                            setattr(error_result, "_t0_tool_use_id_temp", tool_use_id)
                            setattr(error_result, "_t0_tool_name_temp", tool_name)
                            executed_tool_results.append(error_result)
                            self.show_oai_tool_result(
                                f"ERROR: {self._get_text_from_call_tool_result(error_result)}"
                            )

                elif block_type == "thought":
                    thought_text = getattr(block, "text", None)
                    self.logger.debug(f"T0 thought: {thought_text}")
                else:
                    self.logger.warning(f"T0 Adapt: Skipping unknown block type: {block_type}")

        elif isinstance(completion, JsonInferenceResponse):
            self.logger.debug("JsonInferenceResponse received, extracting content")
            try:
                response_dict = {}
                for attr_name in dir(completion):
                    if not attr_name.startswith("_") and attr_name not in (
                        "episode_id",
                        "inference_id",
                        "variant_name",
                        "finish_reason",
                        "usage",
                    ):
                        attr_value = getattr(completion, attr_name)
                        if not callable(attr_value):
                            response_dict[attr_name] = attr_value
                json_text = json.dumps(response_dict, indent=2)
                content_parts_this_turn.append(TextContent(type="text", text=json_text))
            except Exception as e:
                self.logger.error(f"Error processing JsonInferenceResponse: {e}")
                content_parts_this_turn.append(
                    TextContent(type="text", text=f"Error processing JSON response: {str(e)}")
                )

        # --- Construct Final Message ---
        final_message = PromptMessageMultipart(
            role="assistant",
            content=content_parts_this_turn,  # Includes text AND raw tool_call dicts
        )

        # Attach executed results if any
        if executed_tool_results:
            setattr(final_message, "_executed_tool_results", executed_tool_results)
            self.logger.debug(
                f"Attaching {len(executed_tool_results)} executed tool results to final message."
            )

        return final_message

    def _adapt_t0_native_streaming_chunk(
        self, chunk: Union[ChatChunk, JsonChunk]
    ) -> PromptMessageMultipart:
        """Handle streaming chunks (placeholder implementation)"""
        # TODO: Implement streaming adaptation if needed
        metadata = {}
        delta_content_parts = []
        return PromptMessageMultipart(role="assistant", content=delta_content_parts)

    async def shutdown(self):
        """Close the T0 gateway client if initialized."""
        if self.gateway:
            try:
                await self.gateway.close()
                self.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                self.logger.error(f"Error closing TensorZero Gateway client: {e}")
