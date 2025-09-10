import secrets
from typing import Dict, List

# Import necessary types and client from google.genai
from google import genai
from google.genai import (
    errors,  # For error handling
    types,
)
from mcp import Tool as McpTool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)

from fast_agent.llm.fastagent_llm import FastAgentLLM

# Import the new converter class
from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini25"


# Define Google-specific parameter exclusions if necessary
GOOGLE_EXCLUDE_FIELDS = {
    # Add fields that should not be passed directly from RequestParams to google.genai config
    FastAgentLLM.PARAM_MESSAGES,  # Handled by contents
    FastAgentLLM.PARAM_MODEL,  # Handled during client/call setup
    FastAgentLLM.PARAM_SYSTEM_PROMPT,  # Handled by system_instruction in config
    # AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS, # Handled by tool_config in config
    FastAgentLLM.PARAM_USE_HISTORY,  # Handled by AugmentedLLM base / this class's logic
    FastAgentLLM.PARAM_MAX_ITERATIONS,  # Handled by this class's loop
    # Add any other OpenAI-specific params not applicable to google.genai
    FastAgentLLM.PARAM_MCP_METADATA,
}.union(FastAgentLLM.BASE_EXCLUDE_FIELDS)


class GoogleNativeLLM(FastAgentLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE, **kwargs)
        # Initialize the google.genai client
        self._google_client = self._initialize_google_client()
        # Initialize the converter
        self._converter = GoogleConverter()

    def _initialize_google_client(self) -> genai.Client:
        """
        Initializes the google.genai client.

        Reads Google API key or Vertex AI configuration from context config.
        """
        try:
            # Example: Authenticate using API key from config
            api_key = self._api_key()  # Assuming _api_key() exists in base class
            if not api_key:
                # Handle case where API key is missing
                raise ProviderKeyError(
                    "Google API key not found.", "Please configure your Google API key."
                )

            # Check for Vertex AI configuration
            if (
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
                    api_key=api_key,
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
            # Include other relevant default parameters
        )

    async def _google_completion(
        self,
        message: List[types.Content] | None,
        request_params: RequestParams | None = None,
        tools: List[McpTool] | None = None,
        *,
        response_mime_type: str | None = None,
        response_schema: object | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using Google's generate_content API and available tools.
        """
        request_params = self.get_request_params(request_params=request_params)
        responses: List[ContentBlock] = []

        # Build conversation history from stored provider-specific messages
        # and the provided message for this turn (no implicit conversion here).
        # We store provider-native Content objects in history.
        # Start with prompts + (optionally) accumulated conversation messages
        base_history: List[types.Content] = self.history.get(
            include_completion_history=request_params.use_history
        )
        # Make a working copy and add the provided turn message(s) if present
        conversation_history: List[types.Content] = list(base_history)
        if message:
            conversation_history.extend(message)

        self.logger.debug(f"Google completion requested with messages: {conversation_history}")
        self._log_chat_progress(self.chat_turn(), model=request_params.model)

        available_tools: List[types.Tool] = (
            self._converter.convert_to_google_tools(tools or []) if tools else []
        )

        # 2. Prepare generate_content arguments
        generate_content_config = self._converter.convert_request_params_to_google_config(
            request_params
        )

        # Apply structured output config OR tool calling (mutually exclusive)
        if response_schema or response_mime_type:
            # Structured output mode: disable tool use
            if response_mime_type:
                generate_content_config.response_mime_type = response_mime_type
            if response_schema is not None:
                generate_content_config.response_schema = response_schema
        elif available_tools:
            # Tool calling enabled only when not doing structured output
            generate_content_config.tools = available_tools
            generate_content_config.tool_config = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )

        # 3. Call the google.genai API
        try:
            # Use the async client
            api_response = await self._google_client.aio.models.generate_content(
                model=request_params.model,
                contents=conversation_history,  # Full conversational context for this turn
                config=generate_content_config,
            )
            self.logger.debug("Google generate_content response:", data=api_response)

            # Track usage if response is valid and has usage data
            if (
                hasattr(api_response, "usage_metadata")
                and api_response.usage_metadata
                and not isinstance(api_response, BaseException)
            ):
                try:
                    turn_usage = TurnUsage.from_google(
                        api_response.usage_metadata, request_params.model
                    )
                    self._finalize_turn_usage(turn_usage)

                except Exception as e:
                    self.logger.warning(f"Failed to track usage: {e}")

        except errors.APIError as e:
            # Handle specific Google API errors
            self.logger.error(f"Google API Error: {e.code} - {e.message}")
            raise ProviderKeyError(f"Google API Error: {e.code}", e.message or "") from e
        except Exception as e:
            self.logger.error(f"Error during Google generate_content call: {e}")
            # Decide how to handle other exceptions - potentially re-raise or return an error message
            raise e

        # 4. Process the API response
        if not api_response.candidates:
            # No response from the model, we're done
            self.logger.debug("No candidates returned.")

        candidate = api_response.candidates[0]  # Process the first candidate

        # Convert the model's response content to fast-agent types
        model_response_content_parts = self._converter.convert_from_google_content(
            candidate.content
        )
        stop_reason = LlmStopReason.END_TURN
        tool_calls: Dict[str, CallToolRequest] | None = None
        # Add model's response to the working conversation history for this turn
        conversation_history.append(candidate.content)

        # Extract and process text content and tool calls
        assistant_message_parts = []
        tool_calls_to_execute = []

        for part in model_response_content_parts:
            if isinstance(part, TextContent):
                responses.append(part)  # Add text content to the final responses to be returned
                assistant_message_parts.append(
                    part
                )  # Collect text for potential assistant message display
            elif isinstance(part, CallToolRequestParams):
                # This is a function call requested by the model
                # If in structured mode, ignore tool calls per either-or rule
                if response_schema or response_mime_type:
                    continue
                tool_calls_to_execute.append(part)  # Collect tool calls to execute

        if tool_calls_to_execute:
            stop_reason = LlmStopReason.TOOL_USE
            tool_calls = {}
            for tool_call_params in tool_calls_to_execute:
                # Convert to CallToolRequest and execute
                tool_call_request = CallToolRequest(method="tools/call", params=tool_call_params)
                hex_string = secrets.token_hex(3)[:5]
                tool_calls[hex_string] = tool_call_request

            self.logger.debug("Tool call results processed.")
        else:
            stop_reason = self._map_finish_reason(getattr(candidate, "finish_reason", None))

        # 6. Persist conversation state to provider-native history (exclude prompt messages)
        if request_params.use_history:
            # History store separates prompt vs conversation messages; keep prompts as-is
            prompt_messages = self.history.get(include_completion_history=False)
            # messages after prompts are the true conversation history
            new_messages = conversation_history[len(prompt_messages) :]
            self.history.set(new_messages, is_prompt=False)

        self._log_chat_finished(model=request_params.model)  # Use model from request_params
        return Prompt.assistant(*responses, stop_reason=stop_reason, tool_calls=tool_calls)

    #        return responses  # Return the accumulated responses (fast-agent content types)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[McpTool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Applies the prompt messages and potentially calls the LLM for completion.
        """

        request_params = self.get_request_params(request_params=request_params)

        # Determine the last message
        last_message = multipart_messages[-1]

        # Add previous messages (excluding the last user message) to provider-native history
        # If last is assistant, we add all messages and return it directly (no inference).
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )

        if messages_to_add:
            # Convert prior messages to google.genai Content and store in provider history
            converted_prior = self._converter.convert_to_google_content(messages_to_add)
            self.history.extend(converted_prior, is_prompt=is_template)

        if last_message.role == "assistant":
            # No generation required; the provided assistant message is the output
            return last_message

        # Build the provider-native message list for this turn from the last user message
        # This must handle tool results as function responses before any additional user content.
        turn_messages: List[types.Content] = []

        # 1) Convert tool results (if any) to google function responses
        if last_message.tool_results:
            # Map correlation IDs back to tool names using the last assistant tool_calls
            # found in our high-level message history
            id_to_name: Dict[str, str] = {}
            for prev in reversed(self._message_history):
                if prev.role == "assistant" and prev.tool_calls:
                    for call_id, call in prev.tool_calls.items():
                        try:
                            id_to_name[call_id] = call.params.name
                        except Exception:
                            pass
                    break

            tool_results_pairs = []
            for call_id, result in last_message.tool_results.items():
                tool_name = id_to_name.get(call_id, "tool")
                tool_results_pairs.append((tool_name, result))

            if tool_results_pairs:
                turn_messages.extend(
                    self._converter.convert_function_results_to_google(tool_results_pairs)
                )

        # 2) Convert any direct user content in the last message
        if last_message.content:
            user_contents = self._converter.convert_to_google_content([last_message])
            # convert_to_google_content returns a list; preserve order after tool responses
            turn_messages.extend(user_contents)

        # If we somehow have no provider-native parts, ensure we send an empty user content
        if not turn_messages:
            turn_messages.append(types.Content(role="user", parts=[types.Part.from_text("")]))

        # Delegate to the native completion with explicit turn messages
        return await self._google_completion(
            turn_messages, request_params=request_params, tools=tools
        )

    def _map_finish_reason(self, finish_reason: object) -> LlmStopReason:
        """Map Google finish reasons to LlmStopReason robustly."""
        # Normalize to string if it's an enum-like object
        reason = None
        try:
            reason = str(finish_reason) if finish_reason is not None else None
        except Exception:
            reason = None

        if not reason:
            return LlmStopReason.END_TURN

        # Extract last token after any dots or enum prefixes
        key = reason.split(".")[-1].upper()

        if key in {"STOP"}:
            return LlmStopReason.END_TURN
        if key in {"MAX_TOKENS", "LENGTH"}:
            return LlmStopReason.MAX_TOKENS
        if key in {
            "PROHIBITED_CONTENT",
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "SPII",
            "IMAGE_SAFETY",
        }:
            return LlmStopReason.SAFETY
        if key in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"}:
            return LlmStopReason.ERROR
        # Some SDKs include OTHER, LANGUAGE, GROUNDING, UNSPECIFIED, etc.
        return LlmStopReason.ERROR

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages,
        model,
        request_params=None,
    ):
        """
        Handles structured output for Gemini models using response_schema and response_mime_type,
        keeping provider-native (google.genai) history consistent with non-structured calls.
        """
        import json

        # Determine the last message and add prior messages to provider-native history
        last_message = multipart_messages[-1] if multipart_messages else None
        messages_to_add = (
            multipart_messages
            if last_message and last_message.role == "assistant"
            else multipart_messages[:-1]
        )
        if messages_to_add:
            converted_prior = self._converter.convert_to_google_content(messages_to_add)
            self.history.extend(converted_prior, is_prompt=False)

        # If the last message is an assistant message, attempt to parse its JSON and return
        if last_message and last_message.role == "assistant":
            assistant_text = last_message.last_text()
            if assistant_text:
                try:
                    json_data = json.loads(assistant_text)
                    validated_model = model.model_validate(json_data)
                    return validated_model, last_message
                except (json.JSONDecodeError, Exception) as e:
                    self.logger.warning(
                        f"Failed to parse assistant message as structured response: {e}"
                    )
                    return None, last_message

        # Prepare request params
        request_params = self.get_request_params(request_params)

        # Build schema for structured output
        schema = None
        try:
            schema = model.model_json_schema()
        except Exception:
            pass
        response_schema = model if schema is None else schema

        # Convert the last user message to provider-native content for the current turn
        turn_messages: List[types.Content] = []
        if last_message:
            turn_messages = self._converter.convert_to_google_content([last_message])

        # Delegate to unified completion with structured options enabled (no tools)
        assistant_msg = await self._google_completion(
            turn_messages,
            request_params=request_params,
            tools=None,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Parse using shared helper for consistency
        parsed, _ = self._structured_from_multipart(assistant_msg, model)
        return parsed, assistant_msg
