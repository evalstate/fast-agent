# region External Imports
## External Imports -- General Imports
import json
from typing import List, Optional, Tuple, Type

## External Imports -- Provider-Specific Imports
from google import genai
from google.genai import (
    errors,
    types,
)

## External Imports -- MCP
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    TextContent,
)
from rich.text import Text

# endregion
# region Internal Imports
## Internal -- Core
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams

## Internal -- LLM
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.google_converter import GoogleConverter
from mcp_agent.llm.usage_tracking import TurnUsage

## Internal -- MCP
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

#endregion

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"


class GoogleNativeAugmentedLLM(AugmentedLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    GOOGLE_EXCLUDE_FIELDS = {
        # Add fields that should not be passed directly from RequestParams to google.genai config
        AugmentedLLM.PARAM_MESSAGES,  # Handled by contents
        AugmentedLLM.PARAM_MODEL,  # Handled during client/call setup
        AugmentedLLM.PARAM_SYSTEM_PROMPT,  # Handled by system_instruction in config
        # AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS, # Handled by tool_config in config
        AugmentedLLM.PARAM_USE_HISTORY,  # Handled by AugmentedLLM base / this class's logic
        AugmentedLLM.PARAM_MAX_ITERATIONS,  # Handled by this class's loop
        # Add any other OpenAI-specific params not applicable to google.genai
    }.union(AugmentedLLM.BASE_EXCLUDE_FIELDS)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE, **kwargs)
        self._google_client = self._initialize_google_client()
        self._converter = GoogleConverter()

    def _initialize_google_client(self) -> genai.Client:
        """
        Initializes the google.genai client.

        Reads Google API key or Vertex AI configuration from context config.
        """
        try:
            if not self._api_key(): #  _api_key() from base class.
                raise ProviderKeyError(
                    "Google API key not found.", "Please configure your Google API key."
                )
            
            if ( #  Check if Vertex or Gemini API
                self.context
                and self.context.config
                and hasattr(self.context.config, "google")
                and hasattr(self.context.config.google, "vertex_ai")
                and self.context.config.google.vertex_ai.enabled
            ):
                vertex_config = self.context.config.google.vertex_ai
                return genai.Client(
                    vertexai=True,
                    project=vertex_config.project_id,
                    location=vertex_config.location,
                    # Add other Vertex AI specific options if needed
                    # http_options=types.HttpOptions(api_version='v1') # Example for v1 API
                )
            else:
                # Default to Gemini Developer API
                return genai.Client(
                    api_key=self._api_key(),
                    # http_options=types.HttpOptions(api_version='v1') # Example for v1 API
                )
        except Exception as e:
            # Catch potential initialization errors and raise ProviderKeyError
            raise ProviderKeyError("Failed to initialize Google GenAI client.", str(e)) from e

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Google-specific default parameters."""
        chosen_model = kwargs.get("model", DEFAULT_GOOGLE_MODEL)

        return RequestParams(
            model=chosen_model,
            systemPrompt=self.instruction,  # System instruction will be mapped in _google_completion
            parallel_tool_calls=True,  # Assume parallel tool calls are supported by default with native API
            max_iterations=20,
            use_history=True,
            maxTokens=65536,  # Default max tokens for Google models
        )

    async def _completion_orchestrator(
            self,
            messages_for_turn: List[types.Content],
            params: RequestParams,
            structured_model: Optional[Type[ModelT]] = None,

    ) -> Tuple[List[ContentBlock], List[types.Content]]:
        """
        Orchestrates the agentic loop of API calls and tool use for a single turn.

        This method does not modify self.history directly.
        """
        all_content_responses: List[ContentBlock] = []
        turn_conversation_history = list(messages_for_turn)

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=params.model)

            # 1. Prepare the request for the API
            available_tools = await self.aggregator.list_tools()
            google_tools = self._converter.convert_to_google_tools(available_tools.tools)
            payload = self._prepare_request_payload(
                conversation_history=turn_conversation_history,
                params=params,
                tools=google_tools,
                structured_model=structured_model,
            )

            # 2. Execute the API call
            api_response = await self._execute_api_call(payload)
            if not api_response.candidates:
                self.logger.warning("No candidates returned from Gemini API.")
                break

            # 3. Process the response to determine the next action
            candidate = api_response.candidates[0]
            action, content_blocks, assistant_message= self._process_response(candidate)
            turn_conversation_history.append(assistant_message)

            # 4. Execute the determined action
            if action == self.ACTIONS.STOP:
                all_content_responses.extend(content_blocks)
                if any(isinstance(c, TextContent) and c.text for c in content_blocks):
                    await self.show_assistant_message("".join(c.text for c in content_blocks if isinstance(c, TextContent)))
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is '{candidate.finish_reason}'")
                break # Correctly breaks the loop

            # ADD THIS BLOCK
            elif action == self.ACTIONS.CONTINUE_WITH_TOOLS:
                tool_requests = [block for block in content_blocks if isinstance(block, CallToolRequestParams)]
                tool_results_for_api = await self._execute_tool_calls(tool_requests, available_tools)
                turn_conversation_history.extend(tool_results_for_api)
                # Loop continues to the next iteration

            else:
                self.logger.warning(f"Exceeded max iterations ({params.max_iterations}) without stopping.")

        new_messages = turn_conversation_history[len(messages_for_turn):] # Return the final content and the new messages generated during this turn.
        return all_content_responses, new_messages

    # --------------------------------------------------------------------------
    # Helper Methods (New & Refactored)
    # --------------------------------------------------------------------------

    def _prepare_request_payload(
        self,
        conversation_history: List[types.Content],
        params: RequestParams,
        tools: List[types.Tool],
        structured_model: Optional[Type[ModelT]] = None,
    ) -> dict:
        """Assembles the final dictionary of arguments for the Gemini API call."""
        config = self._converter.convert_request_params_to_google_config(params)
        tool_config = None

        if tools: 
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        
        if structured_model:
            config.response_mime_type = "application/json"
            config.response_schema = structured_model.model_json_schema()
            tools = None  # In JSON mode, no other tools are used.

        return {
            "model": params.model,
            "contents": conversation_history,
            "generation_config": config,
            "tools": tools,
            "tool_config": tool_config,
            "system_instruction": params.systemPrompt or self.instruction
        }
    
    async def _execute_api_call(
            self,
            payload: dict
    ) -> genai.types.GenerateContentResponse:
        """Executes the raw API call and handles usage tracking."""

        try:
            model_instance = self._google_client
            if payload.get("model"):
                model_instance = genai.GenerativeModel(payload["model"])  # Create a model instance with the specific model for this call

            api_response = await model_instance.generate_content_async(**payload)

            if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
                turn_usage = TurnUsage.from_google(
                    api_response.usage_metadata,
                    payload["model"],
                )
                self._finalize_turn_usage(turn_usage=turn_usage)
            
            return api_response

        except errors.GoogleAPICallError as e:
            self.logger.error(f"Google API Error: {e}")
            raise ProviderKeyError(f"Google API Error: {e.message}", str(e)) from e

        except Exception as e:
            self.logger.error(f"Error during Google generate_content call: {e}")
            raise e

    def _process_response(
            self, 
            candidate: types.Candidate,
    ) -> Tuple[str, List[ContentBlock], types.Content]:
        """Parses a response candidate to extract content and determine the next action."""

        assistant_message_content_parts = self._converter.convert_from_google_content(candidate.content)  # Convert the raw assistant message for internal use.
        raw_assistant_message = candidate.content  # Keep the raw assistant message to append to the turn's history.

        text_blocks = [block for block in assistant_message_content_parts if isinstance(block, TextContent)]
        tool_requests = [block for block in assistant_message_content_parts if isinstance(block, CallToolRequestParams)]

        if candidate.finish_reason == "TOOL_USE" and tool_requests:  # Determine next action
            return self.ACTIONS.CONTINUE_WITH_TOOLS, tool_requests, raw_assistant_message
        
        else: 
            return self.ACTIONS.STOP, text_blocks, raw_assistant_message

    async def _execute_tool_calls(
            self,
            tool_requests: List[CallToolRequestParams],
            available_tools,
    ) -> List[types.Content]:
        """Manages the execution of tool calls and converts results for the API."""
        tool_results_for_next_turn = []

        if tool_requests:
            await self.show_assistant_message(Text("Assistant requested tool calls...", style="dim green italic"))

        for tool_call_params in tool_requests:
            tool_call_request = CallToolRequest(method="tools/call", params=tool_call_params)

            self.show_tool_call(
                available_tools=available_tools.tools,
                tool_name=tool_call_request.params.name,
                tool_args=str(tool_call_request.params.arguments),
            )

            result = await self.call_tool(tool_call_request, None)
            self.show_tool_result(result)

            tool_results_for_next_turn.append((tool_call_params.name, result))
        
        return self._converter.convert_function_results_to_google(tool_results_for_next_turn)

    # --------------------------------------------------------------------------
    # Main Entry Points
    # --------------------------------------------------------------------------

    async def _apply_prompt_provider_specific(
        self, 
        multipart_messages: List[PromptMessageMultipart],
        request_params: Optional[RequestParams] = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Applies a prompt, handling history and generating a response if the last message is from the user."""

        self._reset_turn_tool_calls()

        params = self.get_request_params(request_params)

        # 1. Prepare messages for the current turn
        self.history.extend(multipart_messages, is_prompt=is_template)
        messages_for_turn = self._converter.convert_to_google_content(
            self.history.get(include_completion_history=params.use_history)
        )

        last_message_role = multipart_messages[-1].role if multipart_messages else None
        if last_message_role != "user":
            return multipart_messages[-1]
        
        # 2. Call the orchestrator
        final_content, new_history_messages = await self._completion_orchestrator(
            messages_for_turn=messages_for_turn,
            params=params
        )

        # 3. Update history with the generated messages (is_prompt=False)
        new_multipart_messages = self._converter._converter_convert_from_google_content_list(new_history_messages)
        self.history.extend(new_multipart_messages, is_prompt=False)

        self._log_chat_finished(model=params.model)
        return Prompt.assistant(*final_content)

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[Optional[ModelT], PromptMessageMultipart]:
        """
        Applies a prompt and generates a structured (JSON) response by callingthe orchestrator.

        Handles structured output for Gemini models using response_schema and response_mime_type.
        """
        params = self.get_request_params(request_params)
        self.history.extend(multipart_messages, is_prompt=False)

        messages_for_turn = self._converter.convert_to_google_content(
            self.history.get(include_completion_history=params.use_history)
        )

        final_content, new_history_messages = await self._completion_orchestrator(
            messages_for_turn=messages_for_turn,
            params=params,
            structured_model=model,
        )

        new_multipart_messages = self._converter.convert_from_google_content_list(new_history_messages)
        self.history.extend(new_multipart_messages, is_prompt=False)

        assistant_msg = Prompt.assistant(*final_content)

        # Parse and validate the response
        if final_content and isinstance(final_content[0], TextContent):
            text_response = final_content[0].text
            try:
                json_data = json.loads(text_response)
                validated_model = model.model_validate(json_data)
                return validated_model, assistant_msg
            
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Failed to parse or validate structured response: {e}")
                return None, assistant_msg

        return None, assistant_msg

    # --------------------------------------------------------------------------
    # Pro and Post Tool Call
    # --------------------------------------------------------------------------

    async def pre_tool_call(self, tool_call_id: str | None, request: CallToolRequest):
        """
        Hook called before a tool call.

        Args:
            tool_call_id: The ID of the tool call.
            request: The CallToolRequest object.

        Returns:
            The modified CallToolRequest object.
        """
        # Currently a pass-through, can add Google-specific logic if needed
        return request

    async def post_tool_call(
        self, tool_call_id: str | None, request: CallToolRequest, result: CallToolResult
    ):
        """
        Hook called after a tool call.

        Args:
            tool_call_id: The ID of the tool call.
            request: The original CallToolRequest object.
            result: The CallToolResult object.

        Returns:
            The modified CallToolResult object.
        """
        # Currently a pass-through, can add Google-specific logic if needed
        return result
