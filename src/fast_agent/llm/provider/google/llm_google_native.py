import json
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

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.prompt import Prompt
from fast_agent.llm.fastagent_llm import FastAgentLLM

# Import the new converter class
from fast_agent.llm.provider.google.google_converter import GoogleConverter
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

# Define default model and potentially other Google-specific defaults
DEFAULT_GOOGLE_MODEL = "gemini25"


# Define Google-specific parameter exclusions if necessary
GOOGLE_EXCLUDE_FIELDS = {
    # Add fields that should not be passed directly from RequestParams to google.genai config
    FastAgentLLM.PARAM_MESSAGES,  # Handled by contents
    FastAgentLLM.PARAM_MODEL,  # Handled during client/call setup
    FastAgentLLM.PARAM_SYSTEM_PROMPT,  # Handled by system_instruction in config
    FastAgentLLM.PARAM_USE_HISTORY,  # Handled by FastAgentLLM base / this class's logic
    FastAgentLLM.PARAM_MAX_ITERATIONS,  # Handled by this class's loop
    FastAgentLLM.PARAM_MCP_METADATA,
}.union(FastAgentLLM.BASE_EXCLUDE_FIELDS)


class GoogleNativeLLM(FastAgentLLM[types.Content, types.Content]):
    """
    Google LLM provider using the native google.genai library.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, provider=Provider.GOOGLE, **kwargs)
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

    async def _stream_generate_content(
        self,
        *,
        model: str,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
        client: genai.Client,
    ) -> types.GenerateContentResponse | None:
        """Stream Gemini responses and return the final aggregated completion."""
        try:
            response_stream = await client.aio.models.generate_content_stream(
                model=model,
                contents=contents,
                config=config,
            )
        except AttributeError:
            # Older SDKs might not expose streaming; fall back to non-streaming.
            return None
        except errors.APIError:
            raise
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Google streaming failed during setup; falling back to non-streaming",
                exc_info=exc,
            )
            return None

        return await self._consume_google_stream(response_stream, model=model)

    async def _consume_google_stream(
        self,
        response_stream,
        *,
        model: str,
    ) -> types.GenerateContentResponse | None:
        """Consume the async streaming iterator and aggregate the final response."""
        estimated_tokens = 0
        timeline: List[tuple[str, int | None, str]] = []
        tool_streams: Dict[int, Dict[str, str]] = {}
        active_tool_index: int | None = None
        tool_counter = 0
        usage_metadata = None
        last_chunk: types.GenerateContentResponse | None = None

        try:
            async for chunk in response_stream:
                last_chunk = chunk
                if getattr(chunk, "usage_metadata", None):
                    usage_metadata = chunk.usage_metadata

                if not getattr(chunk, "candidates", None):
                    continue

                candidate = chunk.candidates[0]
                content = getattr(candidate, "content", None)
                if content is None or not getattr(content, "parts", None):
                    continue

                for part in content.parts:
                    if getattr(part, "text", None):
                        text = part.text or ""
                        if text:
                            if timeline and timeline[-1][0] == "text":
                                prev_type, prev_index, prev_text = timeline[-1]
                                timeline[-1] = (prev_type, prev_index, prev_text + text)
                            else:
                                timeline.append(("text", None, text))
                            estimated_tokens = self._update_streaming_progress(
                                text,
                                model,
                                estimated_tokens,
                            )
                            self._notify_tool_stream_listeners(
                                "text",
                                {
                                    "chunk": text,
                                    "streams_arguments": False,
                                },
                            )

                    if getattr(part, "function_call", None):
                        function_call = part.function_call
                        name = getattr(function_call, "name", None) or "tool"
                        args = getattr(function_call, "args", None) or {}

                        if active_tool_index is None:
                            active_tool_index = tool_counter
                            tool_counter += 1
                            tool_use_id = f"tool_{self.chat_turn()}_{active_tool_index}"
                            tool_streams[active_tool_index] = {
                                "name": name,
                                "tool_use_id": tool_use_id,
                                "buffer": "",
                            }
                            self._notify_tool_stream_listeners(
                                "start",
                                {
                                    "tool_name": name,
                                    "tool_use_id": tool_use_id,
                                    "index": active_tool_index,
                                    "streams_arguments": False,
                                },
                            )
                            timeline.append(("tool_call", active_tool_index, ""))

                        stream_info = tool_streams.get(active_tool_index)
                        if not stream_info:
                            continue

                        try:
                            serialized_args = json.dumps(args, separators=(",", ":"))
                        except Exception:
                            serialized_args = str(args)

                        previous = stream_info.get("buffer", "")
                        if isinstance(previous, str) and serialized_args.startswith(previous):
                            delta = serialized_args[len(previous) :]
                        else:
                            delta = serialized_args
                        stream_info["buffer"] = serialized_args

                        if delta:
                            self._notify_tool_stream_listeners(
                                "delta",
                                {
                                    "tool_name": stream_info["name"],
                                    "tool_use_id": stream_info["tool_use_id"],
                                    "index": active_tool_index,
                                    "chunk": delta,
                                    "streams_arguments": False,
                                },
                            )

                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason:
                    finish_value = str(finish_reason).split(".")[-1].upper()
                    if finish_value in {"FUNCTION_CALL", "STOP"} and active_tool_index is not None:
                        stream_info = tool_streams.get(active_tool_index)
                        if stream_info:
                            self._notify_tool_stream_listeners(
                                "stop",
                                {
                                    "tool_name": stream_info["name"],
                                    "tool_use_id": stream_info["tool_use_id"],
                                    "index": active_tool_index,
                                    "streams_arguments": False,
                                },
                            )
                        active_tool_index = None
        finally:
            stream_close = getattr(response_stream, "aclose", None)
            if callable(stream_close):
                try:
                    await stream_close()
                except Exception:
                    pass

        if active_tool_index is not None:
            stream_info = tool_streams.get(active_tool_index)
            if stream_info:
                self._notify_tool_stream_listeners(
                    "stop",
                    {
                        "tool_name": stream_info["name"],
                        "tool_use_id": stream_info["tool_use_id"],
                        "index": active_tool_index,
                        "streams_arguments": False,
                    },
                )

        if not timeline and last_chunk is None:
            return None

        final_parts: List[types.Part] = []
        for entry_type, index, payload in timeline:
            if entry_type == "text":
                final_parts.append(types.Part.from_text(text=payload))
            elif entry_type == "tool_call" and index is not None:
                stream_info = tool_streams.get(index)
                if not stream_info:
                    continue
                buffer = stream_info.get("buffer", "")
                try:
                    args_obj = json.loads(buffer) if buffer else {}
                except json.JSONDecodeError:
                    args_obj = {"__raw": buffer}
                final_parts.append(
                    types.Part.from_function_call(
                        name=str(stream_info.get("name") or "tool"),
                        args=args_obj,
                    )
                )

        final_content = types.Content(role="model", parts=final_parts)

        if last_chunk is not None:
            final_response = last_chunk.model_copy(deep=True)
            if getattr(final_response, "candidates", None):
                final_candidate = final_response.candidates[0]
                final_candidate.content = final_content
            else:
                final_response.candidates = [types.Candidate(content=final_content)]
        else:
            final_response = types.GenerateContentResponse(
                candidates=[types.Candidate(content=final_content)]
            )

        if usage_metadata:
            final_response.usage_metadata = usage_metadata

        return final_response

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
        client = self._initialize_google_client()
        try:
            # Use the async client
            api_response = None
            streaming_supported = response_schema is None and response_mime_type is None
            if streaming_supported:
                api_response = await self._stream_generate_content(
                    model=request_params.model,
                    contents=conversation_history,
                    config=generate_content_config,
                    client=client,
                )
            if api_response is None:
                api_response = await client.aio.models.generate_content(
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
        finally:
            try:
                await client.aio.aclose()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

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
            # Convert prior messages to google.genai Content
            converted_prior = self._converter.convert_to_google_content(messages_to_add)
            # Only persist prior context when history is enabled; otherwise inline later
            if request_params.use_history:
                self.history.extend(converted_prior, is_prompt=is_template)
            else:
                # Prepend prior context directly to the turn message list
                # This keeps the single-turn chain intact without relying on provider memory
                pass

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

        # If not using provider history, include prior messages inline for this turn
        if messages_to_add and not request_params.use_history:
            prior_contents = self._converter.convert_to_google_content(messages_to_add)
            turn_messages = prior_contents + turn_messages

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
