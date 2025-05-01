import json
from typing import Any, Dict, List, Optional, Union, cast, AsyncIterator

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
            systemPrompt=self.instruction,  # Still useful for base class logic, though not sent directly to T0 here
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
        """
        Provider-specific implementation called by base generate().
        Handles native T0 API call, history management, streaming aggregation/display.
        Separates function input, messages, and tools as top-level args for gateway.inference.
        """
        gateway = await self._ensure_gateway_initialized()
        merged_params = self.get_request_params(request_params)

        # --- Prepare Messages (including history) ---
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

        all_messages = history_messages + multipart_messages
        self.logger.debug(f"Total messages to format for provider: {len(all_messages)}")
        t0_formatted_messages = self._format_messages_for_t0(all_messages)

        # --- Prepare Function Input (schema vars only) ---
        # _prepare_t0_function_input already returns the dict for the 'system' key
        t0_system_vars = self._prepare_t0_function_input(merged_params)

        # --- Combine into the final 'input' dictionary for the API call ---
        t0_api_input_dict = {"system": t0_system_vars, "messages": t0_formatted_messages}

        # --- Prepare Tools (as top-level arg) ---
        available_tools: Optional[List[Dict[str, Any]]] = await self._prepare_t0_tools()
        use_parallel_calls = merged_params.parallel_tool_calls if available_tools else False

        # --- Prepare other args ---
        current_episode_id = self._episode_id
        final_assembled_message: Optional[PromptMessageMultipart] = None

        try:
            # --- Call T0 Inference ---
            self.logger.debug(
                f"Calling T0 inference for '{self.t0_function_name}' with input keys: {list(t0_api_input_dict.keys())}, {len(available_tools) if available_tools else 0} tools."
            )
            response_iter_or_completion = await gateway.inference(
                function_name=self.t0_function_name,
                input=t0_api_input_dict,  # Pass the combined input dict with 'system' and 'messages'
                additional_tools=available_tools,  # Pass tools separately (top-level)
                parallel_tool_calls=use_parallel_calls,  # Pass parallel calls separately (top-level)
                stream=False,
                episode_id=current_episode_id,
            )

            # --- Process Response ---
            if isinstance(
                response_iter_or_completion, (ChatInferenceResponse, JsonInferenceResponse)
            ):
                completion = response_iter_or_completion
                final_assembled_message = await self._adapt_t0_native_completion(
                    completion, available_tools
                )
            else:
                self.logger.error(f"Unexpected response type: {type(response_iter_or_completion)}")
                error_content = TextContent(
                    type="text", text=f"Unexpected response type from TensorZero API"
                )
                return PromptMessageMultipart(role="assistant", content=[error_content])

            # --- Display and History ---
            display_text = final_assembled_message.all_text()
            if display_text and display_text != "<no text>":
                title = f"ASSISTANT/{self.t0_function_name}"
                await self.show_assistant_message(message_text=display_text, title=title)

            if final_assembled_message and merged_params.use_history:
                self.logger.debug("Returning final message; history update handled by base class.")
                # Restore history update logic:
                try:
                    # Combine the messages sent to the API with the final response
                    # Note: all_messages already includes history + current user messages
                    updated_history_content = all_messages + [final_assembled_message]
                    # Filter out any potential None values if final_assembled_message was None
                    valid_history_to_set = [m for m in updated_history_content if m is not None]
                    self.history.set(valid_history_to_set)
                    self.logger.debug(
                        f"Updated self.history with {len(valid_history_to_set)} total messages."
                    )
                except Exception as e:
                    self.logger.error(f"Failed to update self.history: {e}")

            return final_assembled_message or PromptMessageMultipart(role="assistant", content=[])

        except TensorZeroError as e:
            # Improved error logging - Try e.detail
            error_details = ""
            detail = getattr(e, "detail", None)
            if detail:
                try:
                    if isinstance(detail, str):
                        error_json = json.loads(detail)
                    elif isinstance(detail, dict):
                        error_json = detail
                    else:
                        error_json = {}
                    nested_error = error_json.get(
                        "error", detail if isinstance(detail, str) else str(detail)
                    )
                    error_details = f": {nested_error}"
                except (json.JSONDecodeError, TypeError):
                    error_details = f": {detail}"  # Fallback to raw detail
            elif e.args:  # Fallback to args
                error_details = f": {e.args[0]}" if e.args else ""

            self.logger.error(f"T0 Error in _apply_prompt (HTTP {e.status_code}){error_details}")
            error_content = TextContent(
                type="text",
                text=f"Error communicating with TensorZero (HTTP {e.status_code}){error_details}",
            )
            return PromptMessageMultipart(role="assistant", content=[error_content])
        except Exception as e:
            import traceback

            self.logger.error(f"Unexpected Error in _apply_prompt: {e}\n{traceback.format_exc()}")
            error_content = TextContent(type="text", text=f"Unexpected error: {e}")
            return PromptMessageMultipart(role="assistant", content=[error_content])

    def _get_text_from_call_tool_result(self, result: CallToolResult) -> str:
        """Helper to extract combined text from a CallToolResult's content list."""
        texts = []
        if result.content:
            for part in result.content:
                text = get_text(part)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    def _format_messages_for_t0(
        self, messages: List[PromptMessageMultipart]
    ) -> List[Dict[str, Any]]:
        """Formats PromptMessageMultipart list into T0's expected message format."""
        t0_messages = []
        for msg in messages:
            if msg.role == "system":
                continue

            t0_content_blocks = []
            # Check if the message represents a tool result (via temporary attribute)
            tool_use_id = getattr(msg, "_t0_tool_use_id_temp", None)
            is_error = getattr(msg, "_t0_is_error_temp", False)

            if tool_use_id:
                # If it's a tool result message, format it as tool_result
                # Ensure msg is actually a CallToolResult before casting
                if isinstance(msg, CallToolResult):
                    result_content_str = self._get_text_from_call_tool_result(msg)
                    t0_content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_content_str,
                            "is_error": is_error,
                        }
                    )
                    # Clean up temporary attributes
                    try:
                        delattr(msg, "_t0_tool_use_id_temp")
                        delattr(msg, "_t0_is_error_temp")
                    except AttributeError:
                        pass  # Ignore if already deleted
                else:
                    self.logger.warning(
                        f"Message had tool ID attribute but was not CallToolResult: {type(msg)}"
                    )
            else:
                # Otherwise, format regular content parts
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

            if t0_content_blocks:
                valid_role = msg.role if msg.role in ["user", "assistant"] else "user"
                if valid_role != msg.role:
                    self.logger.warning(
                        f"Mapping message role '{msg.role}' to '{valid_role}' for T0."
                    )
                t0_messages.append({"role": valid_role, "content": t0_content_blocks})

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
        """Fetches and formats tools for the additional_tools parameter as dictionaries."""
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

                    # Create dictionary directly, ensuring name, description, and parameters are top-level
                    t0_tool_dict = {
                        "type": "function",
                        "name": mcp_tool.name,
                        "description": mcp_tool.description if mcp_tool.description else "",
                        "parameters": mcp_tool.inputSchema,  # Moved parameters to top level
                        # Removed nested "function" dictionary
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

    async def _adapt_t0_native_completion(
        self,
        completion: Union[ChatInferenceResponse, JsonInferenceResponse],
        available_tools_for_display: Optional[List[Dict[str, Any]]] = None,
    ) -> PromptMessageMultipart:
        """Adapts a non-streaming native T0 response to PromptMessageMultipart."""
        # --- Metadata Extraction (REMOVED - Cannot set metadata on PromptMessageMultipart) ---
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

        # --- Content and Tool Call Processing ---
        content_parts: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        executed_tool_results_for_next_turn: List[CallToolResult] = []

        if isinstance(completion, ChatInferenceResponse) and hasattr(completion, "content"):
            for block in completion.content:
                block_type = getattr(block, "type", "UNKNOWN")

                if block_type == "text":
                    text_val = getattr(block, "text", None)
                    if text_val is not None:
                        content_parts.append(TextContent(type="text", text=text_val))

                elif (
                    block_type == "tool_call"
                ):  # T0 uses 'tool_call' for requests (corrected from tool_use)
                    tool_use_id = getattr(block, "id", None)
                    tool_name = getattr(block, "name", None)
                    tool_input = (
                        getattr(block, "arguments", {}) or {}
                    )  # T0 uses 'arguments' for input

                    if tool_use_id and tool_name:
                        # Pass the received tools list to the display function
                        self.show_tool_call(
                            available_tools_for_display, tool_name, json.dumps(tool_input)
                        )
                        mcp_tool_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=tool_name, arguments=tool_input),
                        )
                        try:
                            result: CallToolResult = await self.call_tool(
                                mcp_tool_request, tool_use_id
                            )
                            # Add temporary attributes needed for formatting the *next* message
                            setattr(result, "_t0_tool_use_id_temp", tool_use_id)
                            setattr(result, "_t0_is_error_temp", False)
                            executed_tool_results_for_next_turn.append(result)
                            self.show_oai_tool_result(str(result))
                        except Exception as e:
                            self.logger.error(
                                f"Error executing tool {tool_name} (id: {tool_use_id}): {e}"
                            )
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
                            setattr(error_result, "_t0_is_error_temp", True)
                            executed_tool_results_for_next_turn.append(error_result)
                            self.show_oai_tool_result(
                                f"ERROR: {self._get_text_from_call_tool_result(error_result)}"
                            )

                elif block_type == "thought":
                    thought_text = getattr(block, "text", None)
                    # Store thought in logger context? Or discard?
                    self.logger.debug(f"T0 thought: {thought_text}")
                else:
                    self.logger.warning(f"T0 Adapt: Skipping unknown block type: {block_type}")

        elif isinstance(completion, JsonInferenceResponse):
            self.logger.debug("JsonInferenceResponse received, extracting content")
            try:
                # Revert to manual attribute extraction for JSON response
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
                content_parts.append(TextContent(type="text", text=json_text))
            except Exception as e:
                self.logger.error(f"Error processing JsonInferenceResponse: {e}")
                content_parts.append(
                    TextContent(type="text", text=f"Error processing JSON response: {str(e)}")
                )

        # --- Construct Final Message ---
        final_message = PromptMessageMultipart(
            role="assistant",
            content=content_parts,
        )

        return final_message

    # --- Placeholder for streaming adapter ---
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
