import json
import logging
from typing import TypeVar, Generic, Optional, List, Any, Dict, Union, Tuple, Type # Added Tuple, Type
from io import BytesIO # Added for file uploads

from openai import OpenAI, AsyncOpenAI
from openai.types.beta.threads.run import Run
from openai.types.beta.threads.message import Message
# Used for constructing message content for OAI
from openai.types.beta.threads.message_content_part_param import (
    MessageContentPartParam,
    TextContentBlockParam,
    ImageFileContentBlockParam,
    ImageURLContentBlockParam,
)
from openai.types.beta.threads.message_create_params import MessageCreateParams
from openai.types.beta.threads.thread_message import ThreadMessage # Potentially unused directly
from openai.types.beta.tool_definition import ToolDefinition
# from openai.types.beta.assistant_tool_choice_option import AssistantToolChoiceOption # Potentially unused
from openai.types.shared_params import FunctionDefinition
from openai.types.beta.threads.runs.tool_output_param import ToolOutputParam


from mcp_agent.llm.augmented_llm import AugmentedLLM, MessageRole, ModelT # Added ModelT
from mcp_agent.llm.provider_types import Provider
from mcp_agent.core.request_params import RequestParams
from mcp_agent.core.tool import Tool
from mcp_agent.core.message import (
    PromptMessage,
    PromptMessageMultipart,
    TextContent, # Added
    ImageContent, # Added
    ToolCall,
    ToolResult,
)
from mcp_agent.core.tool_input import CallToolRequest, CallToolResult


AsyncOpenAIClientT = TypeVar("AsyncOpenAIClientT", bound=AsyncOpenAI)
OpenAIClientT = TypeVar("OpenAIClientT", bound=OpenAI)

logger = logging.getLogger(__name__)


class OpenAIResponsesConverter:
    @staticmethod
    async def convert_prompt_message_multipart_to_thread_message_param(
        message: PromptMessageMultipart, async_client: AsyncOpenAI # Added async_client for file uploads
    ) -> MessageCreateParams:
        content_params: List[MessageContentPartParam] = []
        for part in message.content:
            if isinstance(part, TextContent):
                content_params.append(TextContentBlockParam(type="text", text=part.text))
            elif isinstance(part, ImageContent):
                if part.media_type == "url" and part.url:
                    content_params.append(
                        ImageURLContentBlockParam(
                            type="image_url", image_url={"url": part.url}
                        )
                    )
                elif part.media_type == "file" and part.data:
                    try:
                        # Upload the file to OpenAI
                        file_object = await async_client.files.create(
                            file=(BytesIO(part.data), "image.png"), # Provide a generic name or derive from part if possible
                            purpose="vision",
                        )
                        content_params.append(
                            ImageFileContentBlockParam(
                                type="image_file", image_file={"file_id": file_object.id}
                            )
                        )
                        logger.info(f"Uploaded image data, received file_id: {file_object.id}")
                    except Exception as e:
                        logger.error(f"Failed to upload image content: {e}")
                        # Optionally, append a text block indicating failure or skip
                        content_params.append(TextContentBlockParam(type="text", text="[Image upload failed]"))
                # else: logger.warning(f"Unsupported ImageContent part: {part}") #TODO: Re-enable if needed

        # Ensure role is one of "user" or "assistant". Tool role is handled differently.
        role = "user" if message.role == MessageRole.USER else "assistant"
        if message.role == MessageRole.TOOL: # Tool messages are handled by submitting tool outputs, not by adding to thread directly
            logger.warning("Attempted to convert ToolMessage to ThreadMessageParam. This is usually not directly added.")
            # Return empty or raise error, as tool responses are submitted via runs.submit_tool_outputs
            # For now, let's make it an empty user message if this path is hit.
            return MessageCreateParams(role="user", content="[Internal: Tool response placeholder]")


        return MessageCreateParams(role=role, content=content_params if content_params else " ") # OAI requires content

    @staticmethod
    def convert_fastagent_tools_to_openai_tool_definitions(
        tools: List[Tool],
    ) -> List[ToolDefinition]:
        oai_tools: List[ToolDefinition] = []
        for tool in tools:
            try:
                # Pydantic v2 .model_json_schema(), v1 .schema()
                parameters = tool.input_schema.model_json_schema() if hasattr(tool.input_schema, 'model_json_schema') else tool.input_schema.schema()
                
                # Ensure parameters has 'properties' and 'type: object'
                if 'type' not in parameters:
                    parameters['type'] = 'object'
                if 'properties' not in parameters:
                    parameters['properties'] = {}


                oai_tools.append(
                    ToolDefinition(
                        type="function",
                        function=FunctionDefinition(
                            name=tool.name,
                            description=tool.description,
                            parameters=parameters,
                        ),
                    )
                )
            except Exception as e:
                logger.error(f"Error converting tool {tool.name} to OpenAI format: {e}")
        return oai_tools

    @staticmethod
    def convert_thread_messages_to_prompt_messages_multipart(
        messages: List[Message], # These are OAI Message objects
    ) -> List[PromptMessageMultipart]:
        multipart_messages: List[PromptMessageMultipart] = []
        for msg in messages: # msg is openai.types.beta.threads.message.Message
            role = MessageRole.ASSISTANT if msg.role == "assistant" else MessageRole.USER
            
            prompt_content_parts: List[Union[TextContent, ImageContent, ToolCall, ToolResult]] = []

            for content_block in msg.content: # content_block is MessageContentPart (Text, ImageFile)
                if content_block.type == "text":
                    prompt_content_parts.append(TextContent(text=content_block.text.value))
                elif content_block.type == "image_file":
                    # To convert back to ImageContent, we'd need to fetch the file or its URL.
                    # For simplicity, we can represent it as text or a placeholder.
                    # Or, if context allows, fetch and convert to base64 data or URL.
                    # This example just notes the file_id.
                    prompt_content_parts.append(
                        TextContent(text=f"[Image File ID: {content_block.image_file.file_id}]")
                    )
                # TODO: Handle other OAI content types if they arise (e.g. image_url if OAI starts returning those)

            # Check for tool calls (specific to assistant messages)
            # OAI Assistant `tool_calls` are on the `RunStep` object or `Message` object for some older versions.
            # For recent versions, `tool_calls` can be on the message object if the run step involved tool calls that are now part of the message.
            if msg.role == "assistant" and msg.tool_calls:
                for tool_call_obj in msg.tool_calls:
                    if tool_call_obj.type == "function":
                        prompt_content_parts.append(
                            ToolCall(
                                tool_name=tool_call_obj.function.name,
                                tool_arguments=tool_call_obj.function.arguments,
                                tool_call_id=tool_call_obj.id,
                            )
                        )
            
            if not prompt_content_parts and msg.role != "assistant": # Assistant can have empty message if only doing tool calls
                 prompt_content_parts.append(TextContent(text=""))


            multipart_messages.append(
                PromptMessageMultipart(
                    role=role,
                    content=prompt_content_parts,
                    # Map other fields like name, tool_calls if necessary
                    # For OAI, message.id, thread_id, assistant_id, run_id might be relevant metadata.
                    # For now, keeping it simple.
                )
            )
        return multipart_messages


class OpenAIResponsesAgent(AugmentedLLM, Generic[AsyncOpenAIClientT, OpenAIClientT]):
    """
    Provides an interface to OpenAI's Assistants API (beta), leveraging stateful
    Threads and pre-configured Assistants for conversational AI and tool execution.

    This provider allows `fast-agent` to use OpenAI Assistants, which are
    persistent, stateful entities that can be instructed and given tools.
    Interaction occurs within a Thread, which maintains the conversation history.

    Key Configuration:
        - `oai_agent_id` (str): **Required**. The ID of the OpenAI Assistant
          (e.g., "asst_xxxxxxxxxxxx") to be used. This Assistant must be
          created and configured in the OpenAI platform beforehand (or via API).
          It is typically passed via `llm_kwargs` in the `Agent` constructor
          (e.g., `Agent(..., llm_kwargs={"oai_agent_id": "your_asst_id"})`).
        - `oai_thread_id` (Optional[str]): The ID of an existing OpenAI Thread.
          If provided, the agent will attempt to continue the conversation in
          this thread. If `None` (default), a new thread will be created for
          the first interaction (or if message history is not being reused).
          This ID can also be passed via `llm_kwargs`.

    Tool Usage:
        Tools defined within `fast-agent` (e.g., using `@tool_decorator` or
        manual registration with the `default_tool_registry`) are made available
        to the OpenAI Assistant by converting them to OpenAI's `ToolDefinition`
        format during each run. The OpenAI Assistant itself must be pre-configured
        (via the OpenAI platform or API) with corresponding function tool
        definitions (matching names, descriptions, and parameter schemas) to be
        able to correctly request their execution.

    Message Handling:
        User messages (and historical messages, if starting a new thread with history)
        are added to an OpenAI Thread. The Assistant processes these messages within
        the context of the thread's history. The `OpenAIResponsesAgent` manages the
        run lifecycle, including submitting tool outputs if the assistant requests
        tool calls. The final assistant response is then retrieved from the thread.

    Differences from `OpenAIAugmentedLLM`:
        Unlike `OpenAIAugmentedLLM` which primarily uses the stateless Chat Completions
        API (though it can simulate state by resending history), `OpenAIResponsesAgent`
        utilizes the Assistants API. This allows for more complex, stateful interactions
        managed by OpenAI, leveraging Assistants that can have persistent instructions,
        tools, and access to files, all within the context of a persistent Thread.
    """
    def __init__(
        self,
        *args,
        provider: Provider = Provider.OPENAI_RESPONSES,
        oai_thread_id: Optional[str] = None, # Retained from previous, default to None
        **kwargs,
    ):
        super().__init__(*args, provider=provider, **kwargs)

        # Initialize API key and base URL first
        self._configured_api_key = self._api_key()
        self._configured_base_url = self._base_url()

        self.client: OpenAIClientT = OpenAI(api_key=self._configured_api_key, base_url=self._configured_base_url)
        self.async_client: AsyncOpenAIClientT = AsyncOpenAI(api_key=self._configured_api_key, base_url=self._configured_base_url)

        self.beta = self.client.beta 
        self.async_beta = self.async_client.beta

        # Initialize oai_agent_id (Required)
        self.oai_agent_id: Optional[str] = kwargs.get("oai_agent_id")
        if not self.oai_agent_id and self.llm_config and hasattr(self.llm_config, 'config_root'):
            self.oai_agent_id = self.llm_config.config_root.get("oai_agent_id") # Example path
        
        if not self.oai_agent_id:
            raise ValueError("oai_agent_id is required but not found in kwargs or llm_config.config_root")
        
        # Initialize oai_thread_id (Optional, can be created on the fly)
        self.oai_thread_id: Optional[str] = oai_thread_id or kwargs.get("oai_thread_id")
        if not self.oai_thread_id and self.llm_config and hasattr(self.llm_config, 'config_root'):
             self.oai_thread_id = self.llm_config.config_root.get("oai_thread_id")


    def _api_key(self) -> Optional[str]:
        key = None
        if self.llm_config and self.llm_config.credentials:
            key = self.llm_config.credentials.get("openai_responses_api_key")
            if not key:
                key = self.llm_config.credentials.get("api_key") # Fallback
                if key:
                    logger.warning("Using generic 'api_key' for OpenAIResponsesAgent. Consider 'openai_responses_api_key'.")
        if not key:
            logger.error("OpenAI API key not found for OpenAIResponsesAgent.")
        return key

    def _base_url(self) -> Optional[str]:
        url = None
        if self.llm_config and self.llm_config.credentials:
            url = self.llm_config.credentials.get("openai_responses_base_url")
            if not url:
                url = self.llm_config.credentials.get("base_url") # Fallback
        return url # Can be None for default OpenAI URL

    def _initialize_default_params(self, kwargs: dict = None) -> RequestParams:
        if kwargs is None:
            kwargs = {}
        return RequestParams(
            model=kwargs.get("model", self.llm_config.model if self.llm_config else "gpt-4-turbo"),
            max_iterations=kwargs.get("max_iterations", 10), 
            use_history=kwargs.get("use_history", True), # Default to True for Assistants
            temperature=kwargs.get("temperature", self.llm_config.temperature if self.llm_config else 0.7),
            top_p=kwargs.get("top_p", self.llm_config.top_p if self.llm_config else 1.0),
        )

    async def _execute_provider_call(
        self, request_params: RequestParams, provider: Provider
    ) -> Any:
        # Ensure messages are PromptMessageMultipart
        multipart_messages: List[PromptMessageMultipart] = []
        if request_params.messages:
            for msg in request_params.messages:
                if isinstance(msg, PromptMessageMultipart):
                    multipart_messages.append(msg)
                elif isinstance(msg, PromptMessage): # Convert plain PromptMessage
                    # This is a simple conversion; might need more sophistication
                    # based on how PromptMessage content is structured.
                    content_list: List[Union[TextContent, ImageContent]] = []
                    if isinstance(msg.content, str):
                        content_list.append(TextContent(text=msg.content))
                    # else if msg.content is a list of dicts or other complex types, handle here.
                    else: # Assuming msg.content is list of parts for PromptMessage as well for now
                         for part_content in msg.content:
                            if isinstance(part_content, str):
                                content_list.append(TextContent(text=part_content))
                            elif isinstance(part_content, dict) and "text" in part_content: # Basic check
                                content_list.append(TextContent(text=part_content["text"]))
                            # Add more complex conversion if needed

                    multipart_messages.append(
                        PromptMessageMultipart(role=msg.role, content=content_list, name=msg.name)
                    )
                else:
                    logger.warning(f"Skipping unknown message type in request_params.messages: {type(msg)}")
        
        # If multipart_messages is empty and there's a prompt string, create a message from it.
        # This handles cases where request_params.prompt is used instead of request_params.messages.
        if not multipart_messages and request_params.prompt:
            multipart_messages.append(
                PromptMessageMultipart(role=MessageRole.USER, content=[TextContent(text=request_params.prompt)])
            )
        
        if not multipart_messages:
            logger.error("No messages found in request_params for _execute_provider_call")
            raise ValueError("No messages available to send to the assistant.")

        return await self._apply_prompt_provider_specific(
            multipart_messages=multipart_messages,
            request_params=request_params
            # is_template is not used by OAI Assistants in this way
        )

    def _process_response(self, response: Any, provider: Provider) -> PromptMessageMultipart:
        if not isinstance(response, PromptMessageMultipart):
            logger.error(f"Unexpected response type in _process_response: {type(response)}")
            raise TypeError(f"Expected PromptMessageMultipart, got {type(response)}")
        return response

    async def _apply_prompt_provider_specific(
        self, 
        multipart_messages: List[PromptMessageMultipart], 
        request_params: Optional[RequestParams] = None, 
        is_template: bool = False # is_template not typically used by Assistants API
    ) -> PromptMessageMultipart:
        
        if not self.oai_agent_id: # Should have been caught in __init__ but double check
            raise ValueError("OpenAI Agent ID (oai_agent_id) is not set.")

        # Ensure request_params is not None for accessing max_iterations, temp, etc.
        # Use default if not provided.
        current_params = request_params or self._initialize_default_params()

        try:
            # 1. Initialize Thread and Add Messages
            if not self.oai_thread_id:
                logger.info("No existing oai_thread_id, creating a new thread and adding all messages.")
                thread = await self.async_beta.threads.create()
                self.oai_thread_id = thread.id
                logger.info(f"Created new thread with ID: {self.oai_thread_id}")
                # Persist oai_thread_id if context allows (e.g. session state)
                if self.context and hasattr(self.context, 'set_session_value'):
                    self.context.set_session_value('oai_thread_id', self.oai_thread_id)

                for message in multipart_messages:
                    if message.role == MessageRole.TOOL: # Skip adding tool results directly as messages here
                        logger.info(f"Skipping tool message in initial history load for thread {self.oai_thread_id}")
                        continue
                    message_create_params = await OpenAIResponsesConverter.convert_prompt_message_multipart_to_thread_message_param(
                        message, self.async_client
                    )
                    await self.async_beta.threads.messages.create(
                        thread_id=self.oai_thread_id, **message_create_params
                    )
                logger.info(f"Added {len(multipart_messages)} initial messages to new thread {self.oai_thread_id}")

            else: # Existing thread, add only the last user message if new
                logger.info(f"Using existing thread ID: {self.oai_thread_id}")
                if multipart_messages:
                    last_message = multipart_messages[-1]
                    # TODO: Add more sophisticated logic to check if the last message is truly "new"
                    # For now, if it's a user message, we assume it's the one to send.
                    if last_message.role == MessageRole.USER:
                        message_create_params = await OpenAIResponsesConverter.convert_prompt_message_multipart_to_thread_message_param(
                            last_message, self.async_client
                        )
                        await self.async_beta.threads.messages.create(
                            thread_id=self.oai_thread_id, **message_create_params
                        )
                        logger.debug(f"Added last user message to thread {self.oai_thread_id}")
                    elif last_message.role == MessageRole.TOOL:
                        logger.info(f"Last message is a tool result, will be handled by tool submission if applicable. Not adding as a message directly.")
                    # Assistant messages from history are already on the thread.
                else:
                    logger.info("No new messages provided for existing thread.")


            # 2. Prepare Tools
            fa_tools: List[Tool] = await self.aggregator.list_tools()
            oai_tools: List[ToolDefinition] = (
                OpenAIResponsesConverter.convert_fastagent_tools_to_openai_tool_definitions(fa_tools)
            )
            logger.debug(f"Prepared {len(oai_tools)} tools for the assistant run.")

            # 3. Run the Agent
            run_params = {
                "assistant_id": self.oai_agent_id,
                "thread_id": self.oai_thread_id,
                "tools": oai_tools if oai_tools else None,
            }
            if current_params.temperature is not None: run_params["temperature"] = current_params.temperature
            if current_params.top_p is not None: run_params["top_p"] = current_params.top_p
            # Instructions can be passed to override assistant's default
            # if current_params.instruction: run_params["instructions"] = current_params.instruction
            # Model can also be overridden per run
            # if current_params.model: run_params["model"] = current_params.model


            logger.info(f"Creating and polling run for thread {self.oai_thread_id} with assistant {self.oai_agent_id}")
            run = await self.async_beta.threads.runs.create_and_poll(**run_params)
            logger.info(f"Initial run status: {run.status}")

            # 4. Handle Run Steps and Tool Calls
            iteration_count = 0
            # Use max_iterations from current_params
            max_iterations = current_params.max_iterations if current_params.max_iterations is not None else 10


            while run.status == "requires_action" and iteration_count < max_iterations:
                iteration_count += 1
                logger.info(f"Run requires action. Iteration: {iteration_count}/{max_iterations}")
                if run.required_action and run.required_action.type == "submit_tool_outputs":
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    tool_outputs: List[ToolOutputParam] = []

                    for tool_call in tool_calls: # This is openai.types.beta.threads.runs_thread_message_delta.ToolCall
                        function_name = tool_call.function.name
                        function_args_str = tool_call.function.arguments
                        logger.info(f"Tool call requested: {function_name}({function_args_str})")

                        await self.show_tool_call( # This is from AugmentedLLM
                            tool_name=function_name,
                            tool_args=function_args_str, 
                            tool_call_id=tool_call.id
                        )
                        
                        try:
                            arguments_dict = json.loads(function_args_str)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON arguments for tool {function_name}: {function_args_str}")
                            tool_outputs.append(
                                ToolOutputParam(
                                    tool_call_id=tool_call.id,
                                    output=json.dumps({"error": "Invalid JSON arguments"}),
                                )
                            )
                            continue

                        call_request = CallToolRequest(
                            name=function_name,
                            arguments=arguments_dict,
                            tool_call_id=tool_call.id,
                        )
                        
                        tool_result: CallToolResult = await self.call_tool(call_request)
                        result_content_str = tool_result.content_as_string()

                        await self.show_oai_tool_result( # This is a local method
                            tool_result=result_content_str, 
                            tool_call_id=tool_call.id,
                            is_error=tool_result.is_error
                        )

                        tool_outputs.append(
                            ToolOutputParam(
                                tool_call_id=tool_call.id, output=result_content_str
                            )
                        )
                    
                    if tool_outputs:
                        logger.info(f"Submitting {len(tool_outputs)} tool outputs.")
                        try:
                            run = await self.async_beta.threads.runs.submit_tool_outputs_and_poll(
                                run_id=run.id,
                                thread_id=self.oai_thread_id,
                                tool_outputs=tool_outputs,
                            )
                            logger.info(f"Run status after submitting tool outputs: {run.status}")
                        except Exception as e:
                            logger.error(f"Error submitting tool outputs: {e}")
                            # Decide how to handle this: fail the run, or try to continue?
                            # For now, let the exception propagate or raise a specific one.
                            raise Exception(f"Failed to submit tool outputs: {e}") from e
                    else:
                        logger.warning("No tool outputs to submit despite requires_action. Breaking loop.")
                        break 
                else:
                    action_type = run.required_action.type if run.required_action else "None"
                    logger.warning(f"Run requires action but type is not 'submit_tool_outputs': {action_type}. Breaking loop.")
                    break 

            if iteration_count >= max_iterations and run.status == "requires_action":
                logger.warning(f"Max tool call iterations ({max_iterations}) reached.")
                # Potentially cancel the run if it's stuck
                # await self.async_beta.threads.runs.cancel(run_id=run.id, thread_id=self.oai_thread_id)
                raise Exception("Max tool call iterations reached.")

            # 5. Process Final Response
            if run.status == "completed":
                logger.info("Run completed. Fetching messages.")
                # List messages, order descending to get the latest ones first.
                messages_page = await self.async_beta.threads.messages.list(
                    thread_id=self.oai_thread_id, order="desc", limit=20 # Get recent messages
                )
                
                # Filter for the latest assistant message(s) associated with this run.
                assistant_messages_for_run: List[Message] = [
                    m for m in messages_page.data if m.run_id == run.id and m.role == "assistant"
                ]

                if not assistant_messages_for_run:
                    logger.error(f"No assistant messages found for run_id {run.id} after completed run.")
                    # This can happen if the assistant's response was empty or only tool calls that didn't result in a final message.
                    # Create a default empty response.
                    return PromptMessageMultipart(role=MessageRole.ASSISTANT, content=[TextContent(text="")])


                # The latest message from the assistant for this run should be the first in the desc list.
                final_oai_message = assistant_messages_for_run[0]

                converted_messages = (
                    OpenAIResponsesConverter.convert_thread_messages_to_prompt_messages_multipart(
                        [final_oai_message] # Convert just the single, most recent assistant message for this run
                    )
                )
                
                if not converted_messages: # Should not happen if final_oai_message was valid
                    logger.error("Failed to convert final assistant OAI message.")
                    raise Exception("Failed to convert final assistant message.")

                final_assistant_prompt_message = converted_messages[0]
                
                await self.show_assistant_message(final_assistant_prompt_message)
                return final_assistant_prompt_message
            else:
                err_msg = f"OpenAI Assistant run ended with unhandled status: {run.status}."
                if run.last_error:
                    err_msg += f" Error: {run.last_error.message} (Code: {run.last_error.code})"
                logger.error(err_msg)
                raise Exception(err_msg)

        except Exception as e:
            logger.exception(f"Error in OpenAIResponsesAgent _apply_prompt_provider_specific: {e}")
            raise

    def _get_default_provider(self) -> Provider:
        return Provider.OPENAI_RESPONSES

    def _get_provider_model_name(self, provider: Provider, model_name: Optional[str] = None) -> str:
        return model_name or (self.llm_config.model if self.llm_config else "default_assistant_model")

    async def show_oai_tool_result(self, tool_result: str, tool_call_id: str, is_error: bool):
        log_level = logging.ERROR if is_error else logging.INFO
        logger.log(log_level, f"Tool Call ID [{tool_call_id}] Result (is_error={is_error}): {tool_result[:500]}")
        # This method is for internal logging/display, not part of AugmentedLLM interface.
        pass

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT], # ModelT is inherited from AugmentedLLM's Generic type
        request_params: Optional[RequestParams] = None,
    ) -> Tuple[Optional[ModelT], PromptMessageMultipart]:
        """
        Handles a structured call specific to the OpenAI Assistants provider.
        It first gets a regular response, then attempts to parse it into the desired Pydantic model.
        """
        # 1. Call `_apply_prompt_provider_specific` to get the assistant's response.
        assistant_response_message = await self._apply_prompt_provider_specific(
            multipart_messages=multipart_messages, request_params=request_params
        )

        # 2. Call `_structured_from_multipart` to parse the response.
        # This method is inherited from AugmentedLLM.
        parsed_model, raw_message_with_struct_attempt = self._structured_from_multipart(
            message=assistant_response_message, model_class=model
        )

        # 3. Return the tuple from `_structured_from_multipart`.
        return parsed_model, raw_message_with_struct_attempt
