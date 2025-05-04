import json
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from tensorzero import AsyncTensorZeroGateway
from tensorzero.types import (
    ChatInferenceResponse,
    JsonInferenceResponse,
    TensorZeroError,
)

from mcp_agent.agents.agent import Agent
from mcp_agent.core.exceptions import ModelConfigError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.memory import Memory, SimpleMemory
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.multipart_converter_tensorzero import TensorZeroConverter
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart


class TensorZeroAugmentedLLM(AugmentedLLM[Dict[str, Any], Any]):
    """
    AugmentedLLM implementation for TensorZero using its native API.
    Uses the Converter pattern for message formatting.
    Implements multi-turn tool calling logic, storing API dicts in history.
    """

    def __init__(
        self,
        agent: Agent,
        model: str,
        request_params: Optional[RequestParams] = None,
        **kwargs: Any,
    ):
        self.t0_function_name: str = model
        self._episode_id: Optional[str] = kwargs.get("episode_id")
        self._t0_input_keys = kwargs.get("t0_input_keys", None)
        super().__init__(
            agent=agent,
            model=model,
            provider=Provider.TENSORZERO,
            request_params=request_params,
            **kwargs,
        )
        self.history: Memory[Dict[str, Any]] = SimpleMemory[Dict[str, Any]]()
        self.gateway: Optional[AsyncTensorZeroGateway] = None
        self._resolved_url: Optional[str] = None
        self.t0_system_template_vars: Dict[str, Any] = kwargs.get("t0_system_template_vars", {})
        self.logger.info(
            f"TensorZero LLM provider initialized for function '{self.t0_function_name}'. History type: {type(self.history)}"
        )

    @staticmethod
    def block_to_dict(block: Any) -> Dict[str, Any]:
        if hasattr(block, "model_dump"):
            try:
                dumped = block.model_dump(mode="json")
                if dumped:
                    return dumped
            except Exception:
                pass
        if hasattr(block, "__dict__"):
            try:
                block_vars = vars(block)
                if block_vars:
                    return block_vars
            except Exception:
                pass
        if isinstance(block, (str, int, float, bool, list, dict, type(None))):
            return {"type": "raw", "content": block}

        # Basic attribute extraction as fallback
        d = {"type": getattr(block, "type", "unknown")}
        for attr in ["id", "name", "text", "arguments"]:
            if hasattr(block, attr):
                d[attr] = getattr(block, attr)
        if len(d) == 1 and d.get("type") == "unknown":
            d["content"] = str(block)
        return d

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        func_name = kwargs.get("model", self.t0_function_name or "unknown_t0_function")
        return RequestParams(
            model=func_name,
            systemPrompt=self.instruction,
            maxTokens=4096,
            use_history=True,
            max_iterations=10,  # Max iterations for tool use loop
            parallel_tool_calls=True,
        )

    async def _initialize_gateway(self) -> AsyncTensorZeroGateway:
        if self.gateway is None:
            self.logger.debug("Initializing AsyncTensorZeroGateway client...")
            try:
                if not self.context:
                    raise ModelConfigError("Context not found")
                base_url = None
                if (
                    self.context.config
                    and hasattr(self.context.config, "tensorzero")
                    and self.context.config.tensorzero
                ):
                    base_url = getattr(self.context.config.tensorzero, "base_url", None)
                if not base_url:
                    raise ModelConfigError("TensorZero base URL not configured")

                resolved_url = str(base_url)
                self._resolved_url = resolved_url

                gateway_instance = await AsyncTensorZeroGateway.build_http(gateway_url=resolved_url)  # type: ignore
                self.gateway = gateway_instance
                self.logger.info(f"TensorZero Gateway client initialized for URL: {resolved_url}")
            except Exception as e:
                self.logger.error(f"Failed to initialize T0 Gateway: {e}")
                raise ModelConfigError(f"Failed to initialize T0 Gateway lazily: {e}")

        return self.gateway

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        gateway = await self._initialize_gateway()
        merged_params = self.get_request_params(request_params)

        # [1] Retrieve history
        current_api_messages: List[Dict[str, Any]] = []
        if merged_params.use_history:
            try:
                current_api_messages = self.history.get() or []
                self.logger.debug(
                    f"Retrieved {len(current_api_messages)} API dict messages from history."
                )
            except Exception as e:
                self.logger.error(f"Error retrieving history: {e}")

        # [2] Convert *new* incoming PromptMessageMultipart messages to API dicts
        for msg in multipart_messages:
            msg_dict = TensorZeroConverter.convert_mcp_to_t0_message(msg)
            if msg_dict:
                current_api_messages.append(msg_dict)

        t0_system_vars = self._prepare_t0_system_params(merged_params)
        if t0_system_vars:
            t0_api_input_dict = {"system": t0_system_vars}
        else:
            t0_api_input_dict = {}
        available_tools: Optional[List[Dict[str, Any]]] = await self._prepare_t0_tools()

        # [3] Initialize storage arrays for the text content of the assistant message reply and, optionally, tool calls and results, and begin inference loop
        final_assistant_message: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        last_executed_results: Optional[List[CallToolResult]] = None

        for i in range(merged_params.max_iterations):
            use_parallel_calls = merged_params.parallel_tool_calls if available_tools else False
            current_episode_id = self._episode_id

            try:
                self.logger.debug(
                    f"Calling T0 inference (Iteration {i + 1}/{merged_params.max_iterations})..."
                )
                t0_api_input_dict["messages"] = current_api_messages  # type: ignore

                # [4] Call the T0 inference API
                response_iter_or_completion = await gateway.inference(
                    function_name=self.t0_function_name,
                    input=t0_api_input_dict,
                    additional_tools=available_tools,
                    parallel_tool_calls=use_parallel_calls,
                    stream=False,
                    episode_id=current_episode_id,
                )

                if not isinstance(
                    response_iter_or_completion, (ChatInferenceResponse, JsonInferenceResponse)
                ):
                    self.logger.error(
                        f"Unexpected T0 response type: {type(response_iter_or_completion)}"
                    )
                    final_assistant_message = [
                        TextContent(type="text", text="Unexpected response type")
                    ]
                    break  # Exit loop

                # [5] quick check to confirm that episode_id is present and being used correctly by T0
                completion = response_iter_or_completion
                if completion.episode_id:  #
                    self._episode_id = str(completion.episode_id)
                    if self._episode_id != current_episode_id and current_episode_id is not None:
                        raise Exception(
                            f"Episode ID mismatch: {self._episode_id} != {current_episode_id}"
                        )

                # [6] Adapt T0 inference response to a format compatible with the broader framework
                (
                    content_parts_this_turn,  # Text/Image content ONLY
                    executed_results_this_iter,  # Results from THIS iteration
                    raw_tool_call_blocks,
                ) = await self._adapt_t0_native_completion(completion, available_tools)

                last_executed_results = (
                    executed_results_this_iter  # Track results from this iteration
                )

                # [7] If a text message was returned from the assistant, format that message using the multipart_converter_tensorzero.py helper methods and add this to the current list of API messages
                assistant_api_content = []
                for part in content_parts_this_turn:
                    api_part = TensorZeroConverter._convert_content_part(part)
                    if api_part:
                        assistant_api_content.append(api_part)
                if raw_tool_call_blocks:
                    assistant_api_content.extend(
                        [self.block_to_dict(b) for b in raw_tool_call_blocks]
                    )

                if assistant_api_content:
                    assistant_api_message_dict = {
                        "role": "assistant",
                        "content": assistant_api_content,
                    }
                    current_api_messages.append(assistant_api_message_dict)
                elif executed_results_this_iter:
                    self.logger.debug(
                        "Assistant turn contained only tool calls, no API message added."
                    )

                final_assistant_message = content_parts_this_turn

                # [8] If there were no tool calls we're done. If not, format the tool results and add them to the current list of API messages
                if not executed_results_this_iter:
                    self.logger.debug(f"Iteration {i + 1}: No tool calls detected. Finishing loop.")
                    break
                else:
                    user_message_with_results = (
                        TensorZeroConverter.convert_tool_results_to_t0_user_message(
                            executed_results_this_iter
                        )
                    )
                    if user_message_with_results:
                        current_api_messages.append(user_message_with_results)
                    else:
                        self.logger.error("Converter failed to format tool results, breaking loop.")
                        break

                # Check max iterations: TODO: implement logic in the future to handle this dynamically, checking for the presence of a tool call in the last iteration
                if i == merged_params.max_iterations - 1:
                    self.logger.warning(f"Max iterations ({merged_params.max_iterations}) reached.")
                    break

            # --- Error Handling for Inference Call ---
            except TensorZeroError as e:
                error_details = getattr(e, "detail", str(e.args[0] if e.args else e))
                self.logger.error(f"T0 Error (HTTP {e.status_code}): {error_details}")
                error_content = TextContent(type="text", text=f"T0 Error: {error_details}")
                return PromptMessageMultipart(role="assistant", content=[error_content])
            except Exception as e:
                import traceback

                self.logger.error(f"Unexpected Error: {e}\n{traceback.format_exc()}")
                error_content = TextContent(type="text", text=f"Unexpected error: {e}")
                return PromptMessageMultipart(role="assistant", content=[error_content])

        # [9] Construct the final assistant message and update history
        final_message_to_return = PromptMessageMultipart(
            role="assistant", content=final_assistant_message
        )

        if merged_params.use_history:
            try:
                # Store the final list of API DICTIONARIES in history
                self.history.set(current_api_messages)
                self.logger.debug(
                    f"Updated self.history with {len(current_api_messages)} API message dicts."
                )
            except Exception as e:
                self.logger.error(f"Failed to update self.history after loop: {e}")

        # [10] Post final assistant message to display
        display_text = final_message_to_return.all_text()
        if display_text and display_text != "<no text>":
            title = f"ASSISTANT/{self.t0_function_name}"
            await self.show_assistant_message(message_text=display_text, title=title)

        elif not final_assistant_message and last_executed_results:
            self.logger.debug("Final assistant turn involved only tool calls, no text to display.")

        return final_message_to_return

    def _prepare_t0_system_params(self, merged_params: RequestParams) -> Dict[str, Any]:
        """Prepares the 'system' dictionary part of the main input."""
        t0_func_input = self.t0_system_template_vars.copy()
        metadata_args = None
        if merged_params.metadata and isinstance(merged_params.metadata, dict):
            metadata_args = merged_params.metadata.get("tensorzero_arguments")
        if isinstance(metadata_args, dict):
            t0_func_input.update(metadata_args)
            self.logger.debug(f"Merged tensorzero_arguments from metadata: {metadata_args}")
        return t0_func_input

    async def _prepare_t0_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Fetches and formats tools for the additional_tools parameter."""
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
                            f"Tool '{mcp_tool.name}' has invalid parameters schema. Skipping."
                        )
                        continue
                    t0_tool_dict = {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description if mcp_tool.description else "",
                        "parameters": mcp_tool.inputSchema,
                    }
                    formatted_tools.append(t0_tool_dict)
                return formatted_tools if formatted_tools else None
        except Exception as e:
            self.logger.error(f"Failed to fetch or format tools: {e}")
        return None

    async def _adapt_t0_native_completion(
        self,
        completion: Union[ChatInferenceResponse, JsonInferenceResponse],
        available_tools_for_display: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[
        List[Union[TextContent, ImageContent, EmbeddedResource]],  # Text/Image content ONLY
        List[CallToolResult],  # Executed results
        List[Any],  # Raw tool_call blocks
    ]:
        content_parts_this_turn: List[Union[TextContent, ImageContent, EmbeddedResource]] = []
        executed_tool_results: List[CallToolResult] = []
        raw_tool_call_blocks_from_t0: List[Any] = []

        if isinstance(completion, ChatInferenceResponse) and hasattr(completion, "content"):
            for block in completion.content:
                block_type = getattr(block, "type", "UNKNOWN")

                if block_type == "text":
                    text_val = getattr(block, "text", None)
                    if text_val is not None:
                        content_parts_this_turn.append(TextContent(type="text", text=text_val))

                elif block_type == "tool_call":
                    raw_tool_call_blocks_from_t0.append(block)
                    tool_use_id = getattr(block, "id", None)
                    tool_name = getattr(block, "name", None)
                    tool_input_raw = getattr(block, "arguments", None)
                    tool_input = {}
                    if isinstance(tool_input_raw, dict):
                        tool_input = tool_input_raw
                    elif isinstance(tool_input_raw, str):
                        try:
                            tool_input = json.loads(tool_input_raw)
                        except json.JSONDecodeError:
                            tool_input = {}
                    elif tool_input_raw is not None:
                        tool_input = {}

                    if tool_use_id and tool_name:
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
                            setattr(result, "_t0_tool_use_id_temp", tool_use_id)
                            setattr(result, "_t0_tool_name_temp", tool_name)
                            setattr(result, "_t0_is_error_temp", False)
                            executed_tool_results.append(result)
                            self.show_oai_tool_result(str(result))
                        except Exception as e:
                            self.logger.error(
                                f"Error executing tool {tool_name} (id: {tool_use_id}): {e}"
                            )
                            error_text = f"Error executing tool {tool_name}: {str(e)}"
                            error_result = CallToolResult(
                                isError=True, content=[TextContent(type="text", text=error_text)]
                            )
                            setattr(error_result, "_t0_tool_use_id_temp", tool_use_id)
                            setattr(error_result, "_t0_tool_name_temp", tool_name)
                            setattr(error_result, "_t0_is_error_temp", True)
                            executed_tool_results.append(error_result)
                            self.show_oai_tool_result(f"ERROR: {error_text}")

                elif block_type == "thought":
                    thought_text = getattr(block, "text", None)
                    self.logger.debug(f"T0 thought: {thought_text}")
                # TODO: Add handling for 'image' block type from completion if needed
                else:
                    self.logger.warning(f"T0 Adapt: Skipping unknown block type: {block_type}")

        elif isinstance(completion, JsonInferenceResponse):
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

        return content_parts_this_turn, executed_tool_results, raw_tool_call_blocks_from_t0

    async def shutdown(self):
        """Close the T0 gateway client if initialized."""
        if self.gateway:
            try:
                await self.gateway.close()
                self.logger.debug("TensorZero Gateway client closed.")
            except Exception as e:
                self.logger.error(f"Error closing TensorZero Gateway client: {e}")
