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
from mcp_agent.llm.context_truncation_and_summarization import ContextTruncation
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.google_converter import GoogleConverter
from mcp_agent.llm.usage_tracking import TurnUsage

## Internal -- MCP
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

#endregion

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"


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
        AugmentedLLM.PARAM_CONTEXT_TRUNCATION_MODE,
        AugmentedLLM.PARAM_CONTEXT_TRUNCATION_LENGTH_LIMIT,
    }.union(AugmentedLLM.BASE_EXCLUDE_FIELDS)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE, **kwargs)
        self.client = self._initialize_client()

    def _initialize_client(self) -> None:
        """
        Returns a Google client.
        """
        api_key = self._api_key()

        if not api_key:
            raise ProviderKeyError(
                "Google API key not found.", "Please configure your GOOGLE_API_KEY environment variable for the standard API path."
            )
        
        if (
            self.context
            and self.context.config
            and hasattr(self.context.config, "google")
            and hasattr(self.context.config.google, "vertex_ai")
            and self.context.config.google.vertex_ai.enabled
        ):

            self.logger.debug("Using Vertex AI path.")
            vertex_config = self.context.config.google.vertex_ai
            client = genai.Client(
                vertexai=True,
                project=vertex_config.project_id,
                location=vertex_config.location,
            )

        else:

            self.logger.debug("Using standard Gemini API path for model.")
            client = genai.Client(api_key=api_key)

        return client

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Anthropic-specific default parameters"""
        base_params = super()._initialize_default_params(kwargs)  # Get base defaults from parent (includes ModelDatabase lookup)
        base_params.model = kwargs.get("model", DEFAULT_GOOGLE_MODEL)  # Override with Anthropic-specific settings

        return base_params

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
        available_tools = await self.aggregator.list_tools()
        google_tools = GoogleConverter.convert_to_google_tools(available_tools.tools)

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=params.model)

            # 1. Prepare the request for the API
            payload = await self._prepare_request_payload(
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
            action, content_blocks, assistant_message = self._process_response(candidate)
            turn_conversation_history.append(assistant_message)

            # 4. Execute the determined action
            if action == self.ACTIONS.STOP:
                all_content_responses.extend(content_blocks)
                if any(isinstance(c, TextContent) and c.text for c in content_blocks):
                    text_to_show = "".join(c.text for c in content_blocks if isinstance(c, TextContent))
                    await self.show_assistant_message(text_to_show)
                self.logger.debug(f"Iteration {i}: Stopping because finish_reason is '{candidate.finish_reason}'")
                break 

            elif action == self.ACTIONS.CONTINUE_WITH_TOOLS:
                tool_requests = [block for block in content_blocks if isinstance(block, CallToolRequestParams)]
                tool_results_for_api = await self._execute_tool_calls(tool_requests, available_tools)
                turn_conversation_history.extend(tool_results_for_api)


        else:
            self.logger.warning(f"Exceeded max iterations ({params.max_iterations}) without stopping.")

        new_messages = turn_conversation_history[len(messages_for_turn):] # Return the final content and the new messages generated during this turn.
        return all_content_responses, new_messages

    # --------------------------------------------------------------------------
    # Helper Methods 
    # --------------------------------------------------------------------------

    async def _prepare_request_payload(
        self,
        conversation_history: List[types.Content],
        params: RequestParams,
        tools: List[types.Tool],
        structured_model: Optional[Type[ModelT]] = None,
    ) -> dict:
        """Assembles the final dictionary of arguments for the Gemini API call, applying truncation first."""
        
        # 1. Convert from Google's native format to Multipart format for processing
        multipart_history: List[PromptMessageMultipart] = GoogleConverter.convert_from_google_content_list(conversation_history)
        
        # 2. Apply truncation logic
        if hasattr(params, "context_truncation_mode") and params.context_truncation_mode:
            token_limit_for_truncation = params.context_truncation_length_limit or params.maxTokens

            truncated_multipart = await ContextTruncation.truncate_if_required(
                messages=multipart_history,
                truncation_mode=params.context_truncation_mode,
                limit=token_limit_for_truncation,
                model_name=params.model,
                system_prompt=params.systemPrompt or self.instruction,
                provider=self,
            )
            if len(truncated_multipart) < len(multipart_history):
                self.logger.info(f"History truncated from {len(multipart_history)} to {len(truncated_multipart)} messages.")
                multipart_history = truncated_multipart

        # 3. Convert the final (potentially truncated) history back to Google's format
        final_contents_for_api = GoogleConverter.convert_to_google_content(multipart_history)

        config = GoogleConverter.convert_request_params_to_google_config(params)
        tool_config = None

        if tools: 
            tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        
        if structured_model:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = types.Schema(
                type=types.Type.OBJECT,
                properties={k: types.Schema(**v) for k, v in structured_model.model_json_schema()['properties'].items()},
                required=structured_model.model_json_schema().get('required', [])
            ) 
            tools = None  # In JSON mode, no other tools are used.
    

        return {
            "model": params.model,
            "contents": final_contents_for_api,
            "generation_config": config,
            "tools": tools,
            "tool_config": tool_config,
            "system_instruction": params.systemPrompt or self.instruction
        }
    
    async def execute_simple_api_call(self, message_string, max_tokens=2_000) -> str:
        model = self.default_request_params.model
        payload = {
            "model_name": model,
            "contents": message_string,
        }

        api_response = await self._execute_api_call(
            payload=payload
        )

        response_string = api_response.text

        return response_string

    async def _execute_api_call(
        self,
        payload: dict
    ) -> types.GenerateContentResponse:
        """
        Executes the API call, choosing the correct model instantiation
        pattern for either Vertex AI or the standard Gemini API.
        """
        model_name = payload["model"]
        contents = payload["contents"]

        if hasattr(payload, "generation_config"):
            generation_config = payload["generation_config"]
            generation_config.tools = payload["tools"]
            generation_config.tool_config = payload["tool_config"]
            generation_config.system_instruction = payload["system_instruction"]
        else:
            generation_config = {}

        try:

            # The rest of the execution is the same for both paths
            api_response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=generation_config,
                
            )

            if hasattr(api_response, "usage_metadata") and api_response.usage_metadata:
                turn_usage = TurnUsage.from_google(
                    usage=api_response.usage_metadata,
                    model=model_name,
                )
                self._finalize_turn_usage(turn_usage=turn_usage)
            
            return api_response

        except errors.APIError as e:
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

        assistant_message_content_parts = GoogleConverter.convert_from_google_content(candidate.content)  # Convert the raw assistant message for internal use.
        raw_assistant_message = candidate.content  # Keep the raw assistant message to append to the turn's history.

        text_blocks = [block for block in assistant_message_content_parts if isinstance(block, TextContent)]
        tool_requests = [block for block in assistant_message_content_parts if isinstance(block, CallToolRequestParams)]

        # This is problematic. I found that the Google Gemini API provides:
        #         api_response.candidates[0].finish_reason = FinishReason.STOP
        # even when function calls are present.
        # 
        # if candidate.finish_reason == "TOOL_USE" and tool_requests:  # Determine next action
        #     return self.ACTIONS.CONTINUE_WITH_TOOLS, tool_requests, raw_assistant_message

        if tool_requests:
            return self.ACTIONS.CONTINUE_WITH_TOOLS, tool_requests, raw_assistant_message
        
        if candidate.finish_reason.lower() != "stop":
            self.logger.warning(f"Stopping Gemini iteration even though finish_reason == {candidate.finish_reason.lower()} because no tool calls are present.")
            

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
        
        return GoogleConverter.convert_function_results_to_google(tool_results_for_next_turn)

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
        messages_for_turn = GoogleConverter.convert_to_google_content(
            messages=self.history.get(include_completion_history=params.use_history),
            
        )

        last_message_role = multipart_messages[-1].role if multipart_messages else None
        if last_message_role != "user":
            return multipart_messages[-1]
        
        # 2. Call the orchestrator
        final_content, new_history_messages = await self._completion_orchestrator(
            messages_for_turn=messages_for_turn,
            params=params,
        )

        # 3. Update history with the generated messages (is_prompt=False)
        new_multipart_messages = GoogleConverter.convert_from_google_content_list(new_history_messages)
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

        initial_google_messages = GoogleConverter.convert_to_google_content(
            self.history.get(include_completion_history=params.use_history)
        )

        final_content, new_history_messages = await self._completion_orchestrator(
            messages_for_turn=initial_google_messages,
            params=params,
            structured_model=model,
        )

        new_multipart_messages = GoogleConverter.convert_from_google_content_list(new_history_messages)
        self.history.extend(new_multipart_messages, is_prompt=False)

        assistant_msg = Prompt.assistant(*final_content)

        
        if final_content and isinstance(final_content[0], TextContent):  # Parse and validate the response
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

