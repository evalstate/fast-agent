from typing import List

# Import necessary types and client from google.genai
from google import genai
from google.genai import (
    errors,  # For error handling
    types,
)
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
)
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.prompt import Prompt
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider

# Import the new converter class
from mcp_agent.llm.providers.google_converter import GoogleConverter
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"


class GoogleNativeAugmentedLLM(AugmentedLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages,
        model,
        request_params=None,
    ):
        """
        Handles structured output for Gemini models using response_schema and response_mime_type.
        """
        import json

        # Check if the last message is from assistant
        if multipart_messages and multipart_messages[-1].role == "assistant":
            last_message = multipart_messages[-1]

            # Extract text content from the assistant message
            assistant_text = last_message.first_text()

            if assistant_text:
                try:
                    json_data = json.loads(assistant_text)
                    validated_model = model.model_validate(json_data)

                    # Update history with all messages including the assistant message
                    self.history.extend(multipart_messages, is_prompt=False)

                    # Return the validated model and the assistant message
                    return validated_model, last_message

                except (json.JSONDecodeError, Exception) as e:
                    self.logger.warning(
                        f"Failed to parse assistant message as structured response: {e}"
                    )
                    # Return None and the assistant message on failure
                    self.history.extend(multipart_messages, is_prompt=False)
                    return None, last_message

        # Prepare request params
        request_params = self.get_request_params(request_params)
        # Convert Pydantic model to schema dict for Gemini
        schema = None
        try:
            schema = model.model_json_schema()
        except Exception:
            pass

        # Set up Gemini config for structured output
        def _get_schema_type(model):
            # Try to get the type annotation for the model (for list[...] etc)
            # Fallback to dict schema if not available
            try:
                return model
            except Exception:
                return None

        # Use the schema as a dict or as a type, as Gemini supports both
        response_schema = _get_schema_type(model)
        if schema is not None:
            response_schema = schema

        # Set config for structured output
        generate_content_config = self._converter.convert_request_params_to_google_config(
            request_params
        )
        generate_content_config.response_mime_type = "application/json"
        generate_content_config.response_schema = response_schema

        # Convert messages to google.genai format
        conversation_history = self._converter.convert_to_google_content(multipart_messages)

        # Call Gemini API
        try:
            api_response = await self._google_client.aio.models.generate_content(
                model=request_params.model,
                contents=conversation_history,
                config=generate_content_config,
            )
        except Exception as e:
            self.logger.error(f"Error during Gemini structured call: {e}")
            # Return None and a dummy assistant message
            return None, Prompt.assistant(f"Error: {e}")

        # Parse the response as JSON and validate against the model
        if not api_response.candidates or not api_response.candidates[0].content.parts:
            return None, Prompt.assistant("No structured response returned.")

        # Try to extract the JSON from the first part
        text = None
        for part in api_response.candidates[0].content.parts:
            if part.text:
                text = part.text
                break
        if text is None:
            return None, Prompt.assistant("No structured text returned.")

        try:
            json_data = json.loads(text)
            validated_model = model.model_validate(json_data)
            # Update LLM history with user and assistant messages for correct history tracking
            # Add user message(s)
            for msg in multipart_messages:
                self.history.append(msg)
            # Add assistant message
            assistant_msg = Prompt.assistant(text)
            self.history.append(assistant_msg)
            return validated_model, assistant_msg
        except Exception as e:
            self.logger.warning(f"Failed to parse structured response: {e}")
            # Still update history for consistency
            for msg in multipart_messages:
                self.history.append(msg)
            assistant_msg = Prompt.assistant(text)
            self.history.append(assistant_msg)
            return None, assistant_msg

    # Define Google-specific parameter exclusions if necessary
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
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using Google's generate_content API and available tools.
        """
        request_params = self.get_request_params(request_params=request_params)
        responses: List[TextContent | ImageContent | EmbeddedResource] = []

        # Load full conversation history if use_history is true
        if request_params.use_history:
            # Get history from self.history and convert to google.genai format
            conversation_history = self._converter.convert_to_google_content(
                self.history.get(include_completion_history=True)
            )
        else:
            # If not using history, start with an empty list
            conversation_history = []

        self.logger.debug(f"Google completion requested with messages: {conversation_history}")
        self._log_chat_progress(
            self.chat_turn(), model=request_params.model
        )  # Log chat progress at the start of completion

        # Keep track of the number of messages in history before this turn
        initial_history_length = len(conversation_history)

        for i in range(request_params.max_iterations):
            # 1. Get available tools
            aggregator_response = await self.aggregator.list_tools()
            available_tools = self._converter.convert_to_google_tools(
                aggregator_response.tools
            )  # Convert fast-agent tools to google.genai tools

            # 2. Prepare generate_content arguments
            generate_content_config = self._converter.convert_request_params_to_google_config(
                request_params
            )

            # Add tools and tool_config to generate_content_config if tools are available
            if available_tools:
                generate_content_config.tools = available_tools
                # Set tool_config mode to AUTO to allow the model to decide when to call tools
                generate_content_config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                )

            # 3. Call the google.genai API
            try:
                # Use the async client
                api_response = await self._google_client.aio.models.generate_content(
                    model=request_params.model,
                    contents=conversation_history,  # Pass the current turn's conversation history
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
                        self.usage_accumulator.add_turn(turn_usage)

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
                self.logger.debug(f"Iteration {i}: No candidates returned.")
                break

            candidate = api_response.candidates[0]  # Process the first candidate

            # Convert the model's response content to fast-agent types
            model_response_content_parts = self._converter.convert_from_google_content(
                candidate.content
            )

            # Add model's response to conversation history for potential next turn
            # This is for the *internal* conversation history of this completion call
            # to handle multi-turn tool use within one _google_completion call.
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
                    tool_calls_to_execute.append(part)  # Collect tool calls to execute

            # Display assistant message if there is text content
            if assistant_message_parts:
                # Combine text parts for display
                assistant_text = "".join(
                    [p.text for p in assistant_message_parts if isinstance(p, TextContent)]
                )
                # Display the assistant message. If there are tool calls, indicate that.
                if tool_calls_to_execute:
                    tool_names = ", ".join([tc.name for tc in tool_calls_to_execute])
                    display_text = Text(
                        f"{assistant_text}\nAssistant requested tool calls: {tool_names}",
                        style="dim green italic",
                    )
                    await self.show_assistant_message(display_text, tool_names)
                else:
                    await self.show_assistant_message(Text(assistant_text))

            # 5. Handle tool calls if any
            if tool_calls_to_execute:
                tool_results = []
                for tool_call_params in tool_calls_to_execute:
                    # Convert to CallToolRequest and execute
                    tool_call_request = CallToolRequest(
                        method="tools/call", params=tool_call_params
                    )
                    self.show_tool_call(
                        aggregator_response.tools,  # Pass fast-agent tool definitions for display
                        tool_call_request.params.name,
                        str(
                            tool_call_request.params.arguments
                        ),  # Convert dict to string for display
                    )

                    # Execute the tool call. google.genai does not provide a tool_call_id, pass None.
                    result = await self.call_tool(tool_call_request, None)
                    self.show_oai_tool_result(
                        str(result.content)
                    )  # Use show_oai_tool_result for consistency

                    tool_results.append((tool_call_params.name, result))  # Store name and result

                    # Add tool result content to the overall responses to be returned
                    responses.extend(result.content)

                # Convert tool results back to google.genai format and add to conversation_history for the next turn
                tool_response_google_contents = self._converter.convert_function_results_to_google(
                    tool_results
                )
                conversation_history.extend(tool_response_google_contents)

                self.logger.debug(f"Iteration {i}: Tool call results processed.")
            else:
                # If no tool calls, check finish reason to stop or continue
                # google.genai finish reasons: STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER
                if candidate.finish_reason in ["STOP", "MAX_TOKENS", "SAFETY"]:
                    self.logger.debug(
                        f"Iteration {i}: Stopping because finish_reason is '{candidate.finish_reason}'"
                    )
                    # Display message if stopping due to max tokens
                    if (
                        candidate.finish_reason == "MAX_TOKENS"
                        and request_params
                        and request_params.maxTokens is not None
                    ):
                        message_text = Text(
                            f"the assistant has reached the maximum token limit ({request_params.maxTokens})",
                            style="dim green italic",
                        )
                        await self.show_assistant_message(message_text)
                    break  # Exit the loop if a stopping condition is met
                # If no tool calls and no explicit stopping reason, the model might be done.
                # Break to avoid infinite loops if the model doesn't explicitly stop or call tools.
                self.logger.debug(
                    f"Iteration {i}: No tool calls and no explicit stop reason, breaking."
                )
                break

        # 6. Update history after all iterations are done (or max_iterations reached)
        # Only add the new messages generated during this completion turn to history
        if request_params.use_history:
            new_google_messages = conversation_history[initial_history_length:]
            new_multipart_messages = self._converter.convert_from_google_content_list(
                new_google_messages
            )
            self.history.extend(new_multipart_messages)

        self._log_chat_finished(model=request_params.model)  # Use model from request_params
        return responses  # Return the accumulated responses (fast-agent content types)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """
        Applies the prompt messages and potentially calls the LLM for completion.
        """
        request_params = self.get_request_params(
            request_params=request_params
        )  # Get request params

        # Add incoming messages to history before calling completion
        # This ensures the current user message is part of the history for the API call
        self.history.extend(multipart_messages, is_prompt=is_template)

        last_message_role = multipart_messages[-1].role if multipart_messages else None

        if last_message_role == "user":
            # If the last message is from the user, call the LLM for a response
            # _google_completion will now load history internally
            responses = await self._google_completion(request_params=request_params)

            # History update is now handled within _google_completion
            pass

            return Prompt.assistant(*responses)  # Return combined responses as an assistant message
        else:
            # If the last message is not from the user (e.g., assistant), no completion is needed for this step
            # The messages have already been added to history by the calling code/framework
            return multipart_messages[-1]  # Return the last message as is

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
