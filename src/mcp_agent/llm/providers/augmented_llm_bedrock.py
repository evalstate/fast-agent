import json
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type

from mcp.types import EmbeddedResource, ImageContent, TextContent
from rich.text import Text

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

if TYPE_CHECKING:
    from mcp import ListToolsResult

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    NoCredentialsError = Exception

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
)

DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"

# Bedrock message format types
BedrockMessage = Dict[str, Any]  # Bedrock message format
BedrockMessageParam = Dict[str, Any]  # Bedrock message parameter format


class BedrockAugmentedLLM(AugmentedLLM[BedrockMessageParam, BedrockMessage]):
    """
    AWS Bedrock implementation of AugmentedLLM using the Converse API.
    Supports all Bedrock models including Nova, Claude, Meta, etc.
    """

    # Bedrock-specific parameter exclusions
    BEDROCK_EXCLUDE_FIELDS = {
        AugmentedLLM.PARAM_MESSAGES,
        AugmentedLLM.PARAM_MODEL,
        AugmentedLLM.PARAM_SYSTEM_PROMPT,
        AugmentedLLM.PARAM_STOP_SEQUENCES,
        AugmentedLLM.PARAM_MAX_TOKENS,
        AugmentedLLM.PARAM_METADATA,
        AugmentedLLM.PARAM_USE_HISTORY,
        AugmentedLLM.PARAM_MAX_ITERATIONS,
        AugmentedLLM.PARAM_PARALLEL_TOOL_CALLS,
        AugmentedLLM.PARAM_TEMPLATE_VARS,
    }

    @classmethod
    def matches_model_pattern(cls, model_name: str) -> bool:
        """Check if a model name matches Bedrock model patterns."""
        # Bedrock model patterns
        bedrock_patterns = [
            r"^amazon\.nova.*",           # Amazon Nova models
            r"^anthropic\.claude.*",      # Anthropic Claude models
            r"^meta\.llama.*",            # Meta Llama models
            r"^mistral\..*",              # Mistral models
            r"^cohere\..*",               # Cohere models
            r"^ai21\..*",                 # AI21 models
            r"^stability\..*",            # Stability AI models
        ]
        
        import re
        return any(re.match(pattern, model_name) for pattern in bedrock_patterns)

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Bedrock LLM with AWS credentials and region."""
        if boto3 is None:
            raise ImportError(
                "boto3 is required for Bedrock support. Install with: pip install boto3"
            )
        
        # Initialize logger
        self.logger = get_logger(__name__)

        # Extract AWS configuration from kwargs first
        self.aws_region = kwargs.pop("region", None)
        self.aws_profile = kwargs.pop("profile", None)
        
        super().__init__(*args, provider=Provider.BEDROCK, **kwargs)
        
        # Use config values if not provided in kwargs (after super().__init__)
        if self.context.config and self.context.config.bedrock:
            if not self.aws_region:
                self.aws_region = self.context.config.bedrock.region
            if not self.aws_profile:
                self.aws_profile = self.context.config.bedrock.profile
        
        # Final fallback to environment variables
        if not self.aws_region:
            self.aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        
        # Initialize AWS clients
        self._bedrock_client = None
        self._bedrock_runtime_client = None

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Bedrock-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with Bedrock-specific settings
        chosen_model = kwargs.get("model", DEFAULT_BEDROCK_MODEL)
        base_params.model = chosen_model

        return base_params

    def _get_bedrock_client(self):
        """Get or create Bedrock client."""
        if self._bedrock_client is None:
            try:
                session = boto3.Session(profile_name=self.aws_profile)
                self._bedrock_client = session.client("bedrock", region_name=self.aws_region)
            except NoCredentialsError as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_client

    def _get_bedrock_runtime_client(self):
        """Get or create Bedrock Runtime client."""
        if self._bedrock_runtime_client is None:
            try:
                session = boto3.Session(profile_name=self.aws_profile)
                self._bedrock_runtime_client = session.client("bedrock-runtime", region_name=self.aws_region)
            except NoCredentialsError as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_runtime_client

    def _supports_streaming_with_tools(self, model: str) -> bool:
        """
        Check if a model supports streaming with tools.
        
        Some models (like AI21 Jamba) support tools but not in streaming mode.
        This method uses regex patterns to identify such models.
        
        Args:
            model: The model name (e.g., "ai21.jamba-1-5-mini-v1:0")
            
        Returns:
            False if the model requires non-streaming for tools, True otherwise
        """
        # Remove any "bedrock." prefix for pattern matching
        clean_model = model.replace("bedrock.", "")
        
        # Models that don't support streaming with tools
        non_streaming_patterns = [
            r"ai21\.jamba",     # All AI21 Jamba models
            r"meta\.llama",     # All Meta Llama models
            r"mistral\.",       # All Mistral models
        ]
        
        for pattern in non_streaming_patterns:
            if re.search(pattern, clean_model, re.IGNORECASE):
                self.logger.debug(f"Model {model} detected as non-streaming for tools (pattern: {pattern})")
                return False
        
        return True

    def _convert_mcp_tools_to_bedrock(self, tools: "ListToolsResult") -> List[Dict[str, Any]]:
        """Convert MCP tools to Bedrock tool format.
        
        Note: Nova models have VERY strict JSON schema requirements:
        - Top level schema must be of type Object
        - ONLY three fields are supported: type, properties, required
        - NO other fields like $schema, description, title, additionalProperties
        - Properties can only have type and description
        - Tools with no parameters should have empty properties object
        """
        bedrock_tools = []
        
        # Create mapping from cleaned names to original names for tool execution
        self.tool_name_mapping = {}
        
        self.logger.debug(f"Converting {len(tools.tools)} MCP tools to Bedrock format")
        
        for tool in tools.tools:
            self.logger.debug(f"Converting MCP tool: {tool.name}")
            
            # Extract and validate the input schema
            input_schema = tool.inputSchema or {}
            
            # Create Nova-compliant schema with ONLY the three allowed fields
            # Always include type and properties (even if empty)
            nova_schema: Dict[str, Any] = {
                "type": "object",
                "properties": {}
            }
            
            # Properties - clean them strictly
            properties: Dict[str, Any] = {}
            if "properties" in input_schema and isinstance(input_schema["properties"], dict):
                for prop_name, prop_def in input_schema["properties"].items():
                    # Only include type and description for each property
                    clean_prop: Dict[str, Any] = {}
                    
                    if isinstance(prop_def, dict):
                        # Only include type (required) and description (optional)
                        clean_prop["type"] = prop_def.get("type", "string")
                        # Nova allows description in properties
                        if "description" in prop_def:
                            clean_prop["description"] = prop_def["description"]
                    else:
                        # Handle simple property definitions
                        clean_prop["type"] = "string"
                    
                    properties[prop_name] = clean_prop
            
            # Always set properties (even if empty for parameterless tools)
            nova_schema["properties"] = properties
                
            # Required fields - only add if present and not empty
            if "required" in input_schema and isinstance(input_schema["required"], list) and input_schema["required"]:
                nova_schema["required"] = input_schema["required"]
            
            # IMPORTANT: Nova tool name compatibility fix
            # Problem: Amazon Nova models fail with "Model produced invalid sequence as part of ToolUse" 
            # when tool names contain hyphens (e.g., "utils-get_current_date_information")
            # Solution: Replace hyphens with underscores for Nova (e.g., "utils_get_current_date_information")
            # Note: Underscores work fine, simple names work fine, but hyphens cause tool calling to fail
            clean_name = tool.name.replace("-", "_")
            
            # Store mapping from cleaned name back to original MCP name
            # This is needed because:
            # 1. Nova receives tools with cleaned names (utils_get_current_date_information)
            # 2. Nova calls tools using cleaned names
            # 3. But MCP server expects original names (utils-get_current_date_information)
            # 4. So we map back: utils_get_current_date_information -> utils-get_current_date_information
            self.tool_name_mapping[clean_name] = tool.name
            
            bedrock_tool = {
                "toolSpec": {
                    "name": clean_name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "inputSchema": {
                        "json": nova_schema
                    }
                }
            }
            
            bedrock_tools.append(bedrock_tool)
            
        self.logger.debug(f"Converted {len(bedrock_tools)} tools for Bedrock")
        return bedrock_tools

    def _convert_messages_to_bedrock(self, messages: List[BedrockMessageParam]) -> List[Dict[str, Any]]:
        """Convert message parameters to Bedrock format."""
        bedrock_messages = []
        for message in messages:
            bedrock_message = {
                "role": message.get("role", "user"),
                "content": []
            }
            
            # Handle different content types
            content = message.get("content", [])
            if isinstance(content, str):
                bedrock_message["content"] = [{"text": content}]
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        bedrock_message["content"].append({"text": item})
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            bedrock_message["content"].append({"text": item.get("text", "")})
                        elif item.get("type") == "tool_use":
                            bedrock_message["content"].append({
                                "toolUse": {
                                    "toolUseId": item.get("id", ""),
                                    "name": item.get("name", ""),
                                    "input": item.get("input", {})
                                }
                            })
                        elif item.get("type") == "tool_result":
                            # Nova-specific tool result format
                            tool_result_content = item.get("content", "")
                            
                            # Try to parse as JSON for structured data
                            try:
                                # If content is already a dict, use it directly
                                if isinstance(tool_result_content, dict):
                                    json_content = tool_result_content
                                else:
                                    # Try to parse as JSON string
                                    parsed_content = json.loads(tool_result_content)
                                    
                                    # Nova requires JSON content to be an object, not a primitive
                                    # If we got a primitive value, wrap it in an object
                                    if isinstance(parsed_content, (str, int, float, bool, list)):
                                        json_content = {"result": parsed_content}
                                    else:
                                        json_content = parsed_content
                                
                                bedrock_message["content"].append({
                                    "toolResult": {
                                        "toolUseId": item.get("tool_call_id", ""),
                                        "content": [{"json": json_content}],
                                        "status": "success"
                                    }
                                })
                            except (json.JSONDecodeError, TypeError):
                                # Fall back to text format for non-JSON content
                                bedrock_message["content"].append({
                                    "toolResult": {
                                        "toolUseId": item.get("tool_call_id", ""),
                                        "content": [{"text": str(tool_result_content)}],
                                        "status": "success"
                                    }
                                })
            
            bedrock_messages.append(bedrock_message)
        
        return bedrock_messages

    async def _process_stream(self, stream_response, model: str) -> BedrockMessage:
        """Process streaming response from Bedrock."""
        estimated_tokens = 0
        response_content = []
        tool_uses = []
        stop_reason = None
        usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            for event in stream_response["stream"]:
                if "messageStart" in event:
                    # Message started
                    continue
                elif "contentBlockStart" in event:
                    # Content block started
                    content_block = event["contentBlockStart"]
                    if "start" in content_block and "toolUse" in content_block["start"]:
                        # Tool use block started
                        tool_use_start = content_block["start"]["toolUse"]
                        self.logger.debug(f"Tool use block started: {tool_use_start}")
                        tool_uses.append({
                            "toolUse": {
                                "toolUseId": tool_use_start.get("toolUseId"),
                                "name": tool_use_start.get("name"),
                                "input": tool_use_start.get("input", {}),
                                "_input_accumulator": ""  # For accumulating streamed input
                            }
                        })
                elif "contentBlockDelta" in event:
                    # Content delta received
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        text = delta["text"]
                        response_content.append(text)
                        # Update streaming progress
                        estimated_tokens = self._update_streaming_progress(text, model, estimated_tokens)
                    elif "toolUse" in delta:
                        # Tool use delta - handle tool call
                        tool_use = delta["toolUse"]
                        self.logger.debug(f"Tool use delta: {tool_use}")
                        if tool_use and tool_uses:
                            # Handle input accumulation for streaming tool arguments
                            if "input" in tool_use:
                                input_data = tool_use["input"]
                                
                                # If input is a dict, merge it directly
                                if isinstance(input_data, dict):
                                    tool_uses[-1]["toolUse"]["input"].update(input_data)
                                # If input is a string, accumulate it for later JSON parsing
                                elif isinstance(input_data, str):
                                    tool_uses[-1]["toolUse"]["_input_accumulator"] += input_data
                                    self.logger.debug(f"Accumulated input: {tool_uses[-1]['toolUse']['_input_accumulator']}")
                                else:
                                    self.logger.debug(f"Tool use input is unexpected type: {type(input_data)}: {input_data}")
                                    # Set the input directly if it's not a dict or string
                                    tool_uses[-1]["toolUse"]["input"] = input_data
                elif "contentBlockStop" in event:
                    # Content block stopped - finalize any accumulated tool input
                    if tool_uses:
                        for tool_use in tool_uses:
                            if "_input_accumulator" in tool_use["toolUse"]:
                                accumulated_input = tool_use["toolUse"]["_input_accumulator"]
                                if accumulated_input:
                                    self.logger.debug(f"Processing accumulated input: {accumulated_input}")
                                    try:
                                        # Try to parse the accumulated input as JSON
                                        parsed_input = json.loads(accumulated_input)
                                        if isinstance(parsed_input, dict):
                                            tool_use["toolUse"]["input"].update(parsed_input)
                                        else:
                                            tool_use["toolUse"]["input"] = parsed_input
                                        self.logger.debug(f"Successfully parsed accumulated input: {parsed_input}")
                                    except json.JSONDecodeError as e:
                                        self.logger.warning(f"Failed to parse accumulated input as JSON: {accumulated_input} - {e}")
                                        # If it's not valid JSON, treat it as a string value
                                        tool_use["toolUse"]["input"] = accumulated_input
                                # Clean up the accumulator
                                del tool_use["toolUse"]["_input_accumulator"]
                    continue
                elif "messageStop" in event:
                    # Message stopped
                    if "stopReason" in event["messageStop"]:
                        stop_reason = event["messageStop"]["stopReason"]
                elif "metadata" in event:
                    # Usage metadata
                    metadata = event["metadata"]
                    if "usage" in metadata:
                        usage = metadata["usage"]
                        actual_tokens = usage.get("outputTokens", 0)
                        if actual_tokens > 0:
                            # Emit final progress with actual token count
                            token_str = str(actual_tokens).rjust(5)
                            data = {
                                "progress_action": ProgressAction.STREAMING,
                                "model": model,
                                "agent_name": self.name,
                                "chat_turn": self.chat_turn(),
                                "details": token_str.strip(),
                            }
                            self.logger.info("Streaming progress", data=data)
        except Exception as e:
            self.logger.error(f"Error processing stream: {e}")
            raise

        # Construct the response message
        full_text = "".join(response_content)
        response = {
            "content": [{"text": full_text}] if full_text else [],
            "stop_reason": stop_reason or "end_turn",
            "usage": {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
            "model": model,
            "role": "assistant",
        }

        # Add tool uses if any
        if tool_uses:
            # Clean up any remaining accumulators before adding to response
            for tool_use in tool_uses:
                if "_input_accumulator" in tool_use["toolUse"]:
                    accumulated_input = tool_use["toolUse"]["_input_accumulator"]
                    if accumulated_input:
                        self.logger.debug(f"Final processing of accumulated input: {accumulated_input}")
                        try:
                            # Try to parse the accumulated input as JSON
                            parsed_input = json.loads(accumulated_input)
                            if isinstance(parsed_input, dict):
                                tool_use["toolUse"]["input"].update(parsed_input)
                            else:
                                tool_use["toolUse"]["input"] = parsed_input
                            self.logger.debug(f"Successfully parsed final accumulated input: {parsed_input}")
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse final accumulated input as JSON: {accumulated_input} - {e}")
                            # If it's not valid JSON, treat it as a string value
                            tool_use["toolUse"]["input"] = accumulated_input
                    # Clean up the accumulator
                    del tool_use["toolUse"]["_input_accumulator"]
            
            response["content"].extend(tool_uses)

        return response

    def _process_non_streaming_response(self, response, model: str) -> BedrockMessage:
        """Process non-streaming response from Bedrock."""
        self.logger.debug(f"Processing non-streaming response: {response}")
        
        # Extract response content
        content = response.get("output", {}).get("message", {}).get("content", [])
        usage = response.get("usage", {})
        stop_reason = response.get("stopReason", "end_turn")
        
        # Show progress for non-streaming (single update)
        if usage.get("outputTokens", 0) > 0:
            token_str = str(usage.get("outputTokens", 0)).rjust(5)
            data = {
                "progress_action": ProgressAction.STREAMING,
                "model": model,
                "agent_name": self.name,
                "chat_turn": self.chat_turn(),
                "details": token_str.strip(),
            }
            self.logger.info("Non-streaming progress", data=data)
        
        # Convert to the same format as streaming response
        processed_response = {
            "content": content,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
            "model": model,
            "role": "assistant",
        }
        
        return processed_response

    async def _bedrock_completion(
        self,
        message_param: BedrockMessageParam,
        request_params: RequestParams | None = None,
    ) -> List[TextContent | ImageContent | EmbeddedResource]:
        """
        Process a query using Bedrock and available tools.
        """
        client = self._get_bedrock_runtime_client()
        
        try:
            messages: List[BedrockMessageParam] = []
            params = self.get_request_params(request_params)
        except (ClientError, BotoCoreError) as e:
            error_msg = str(e)
            if "UnauthorizedOperation" in error_msg or "AccessDenied" in error_msg:
                raise ProviderKeyError(
                    "AWS Bedrock access denied",
                    "Please check your AWS credentials and IAM permissions for Bedrock.",
                ) from e
            else:
                raise ProviderKeyError(
                    "AWS Bedrock error",
                    f"Error accessing Bedrock: {error_msg}",
                ) from e

        # Always include prompt messages, but only include conversation history
        # if use_history is True
        messages.extend(self.history.get(include_completion_history=params.use_history))
        messages.append(message_param)

        # Get available tools
        available_tools = []
        tool_list = None
        try:
            tool_list = await self.aggregator.list_tools()
            self.logger.debug(f"Found {len(tool_list.tools)} MCP tools")
            
            available_tools = self._convert_mcp_tools_to_bedrock(tool_list)
            self.logger.debug(f"Successfully converted {len(available_tools)} tools for Bedrock")
            
        except Exception as e:
            self.logger.error(f"Error fetching or converting MCP tools: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            available_tools = []
            tool_list = None

        responses: List[TextContent | ImageContent | EmbeddedResource] = []
        model = self.default_request_params.model

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=model)

            # Convert messages to Bedrock format
            bedrock_messages = self._convert_messages_to_bedrock(messages)

            # Prepare Bedrock Converse API arguments
            converse_args = {
                "modelId": model,
                "messages": bedrock_messages,
            }

            # Add system prompt if available
            if self.instruction or params.systemPrompt:
                converse_args["system"] = [
                    {"text": self.instruction or params.systemPrompt}
                ]

            # Add tools if available - Nova requires at least one tool if toolConfig is provided
            if available_tools and len(available_tools) > 0:
                converse_args["toolConfig"] = {
                    "tools": available_tools
                }
                self.logger.debug(f"Added {len(available_tools)} tools to Nova request")
            else:
                self.logger.debug("No tools available - omitting toolConfig from Nova request")

            # Add inference configuration
            inference_config = {}
            if params.maxTokens is not None:
                inference_config["maxTokens"] = params.maxTokens
            if params.stopSequences:
                inference_config["stopSequences"] = params.stopSequences
            
            # Nova-specific recommended settings for tool calling
            if model and "nova" in model.lower():
                inference_config["topP"] = 1.0
                inference_config["temperature"] = 1.0
                # Add additionalModelRequestFields for topK
                converse_args["additionalModelRequestFields"] = {
                    "inferenceConfig": {"topK": 1}
                }
            
            if inference_config:
                converse_args["inferenceConfig"] = inference_config

            self.logger.debug(f"Bedrock converse args: {converse_args}")
            
            # Debug: Print the full tool config being sent
            if "toolConfig" in converse_args:
                self.logger.debug(f"Tool config being sent to Bedrock: {json.dumps(converse_args['toolConfig'], indent=2)}")

            try:
                # Choose streaming vs non-streaming based on model capabilities and tool presence
                # Logic: Only use non-streaming when BOTH conditions are true:
                #   1. Tools are available (available_tools is not empty)
                #   2. Model doesn't support streaming with tools
                # Otherwise, always prefer streaming for better UX
                if available_tools and not self._supports_streaming_with_tools(model or DEFAULT_BEDROCK_MODEL):
                    # Use non-streaming API: model requires it for tool calls
                    self.logger.debug(f"Using non-streaming API for {model} with tools (model limitation)")
                    response = client.converse(**converse_args)
                    processed_response = self._process_non_streaming_response(response, model or DEFAULT_BEDROCK_MODEL)
                else:
                    # Use streaming API: either no tools OR model supports streaming with tools
                    streaming_reason = "no tools present" if not available_tools else "model supports streaming with tools"
                    self.logger.debug(f"Using streaming API for {model} ({streaming_reason})")
                    response = client.converse_stream(**converse_args)
                    processed_response = await self._process_stream(response, model or DEFAULT_BEDROCK_MODEL)
            except (ClientError, BotoCoreError) as e:
                error_msg = str(e)
                self.logger.error(f"Bedrock API error: {error_msg}")
                
                # Create error response
                processed_response = {
                    "content": [{"text": f"Error during generation: {error_msg}"}],
                    "stop_reason": "error",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "model": model,
                    "role": "assistant",
                }

            # Track usage
            if processed_response.get("usage"):
                try:
                    usage = processed_response["usage"]
                    turn_usage = TurnUsage(
                        provider=Provider.BEDROCK.value,
                        model=model,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                        cache_creation_input_tokens=0,
                        cache_read_input_tokens=0,
                        raw_usage=usage
                    )
                    self.usage_accumulator.add_turn(turn_usage)
                except Exception as e:
                    self.logger.warning(f"Failed to track usage: {e}")

            self.logger.debug(f"{model} response:", data=processed_response)

            # Convert response to message param and add to messages
            response_message_param = self.convert_message_to_message_param(processed_response)
            messages.append(response_message_param)
            
            # Extract text content for responses
            if processed_response.get("content"):
                for content_item in processed_response["content"]:
                    if content_item.get("text"):
                        responses.append(TextContent(type="text", text=content_item["text"]))

            # Handle different stop reasons
            stop_reason = processed_response.get("stop_reason", "end_turn")
            
            if stop_reason == "end_turn":
                # Extract text for display
                message_text = ""
                for content_item in processed_response.get("content", []):
                    if content_item.get("text"):
                        message_text += content_item["text"]
                
                await self.show_assistant_message(message_text)
                self.logger.debug(f"Iteration {i}: Stopping because stop_reason is 'end_turn'")
                break
            elif stop_reason == "stop_sequence":
                self.logger.debug(f"Iteration {i}: Stopping because stop_reason is 'stop_sequence'")
                break
            elif stop_reason == "max_tokens":
                self.logger.debug(f"Iteration {i}: Stopping because stop_reason is 'max_tokens'")
                if params.maxTokens is not None:
                    message_text = Text(
                        f"the assistant has reached the maximum token limit ({params.maxTokens})",
                        style="dim green italic",
                    )
                else:
                    message_text = Text(
                        "the assistant has reached the maximum token limit",
                        style="dim green italic",
                    )
                await self.show_assistant_message(message_text)
                break
            elif stop_reason == "tool_use":
                # Handle tool use
                message_text = ""
                for content_item in processed_response.get("content", []):
                    if content_item.get("text"):
                        message_text += content_item["text"]
                
                # Collect tool uses
                tool_uses = [
                    content_item for content_item in processed_response.get("content", [])
                    if "toolUse" in content_item
                ]
                
                if tool_uses:
                    if not message_text:
                        message_text = Text(
                            "the assistant requested tool calls",
                            style="dim green italic",
                        )
                    
                    # Process tool calls
                    tool_results = []
                    for tool_idx, tool_use_item in enumerate(tool_uses):
                        self.logger.debug(f"Processing tool use item: {tool_use_item}")
                        tool_use = tool_use_item["toolUse"]
                        self.logger.debug(f"Tool use object: {tool_use}")
                        tool_name = tool_use["name"]
                        tool_args = tool_use["input"]
                        tool_use_id = tool_use["toolUseId"]
                        
                        # Ensure tool_args is a dictionary
                        if not isinstance(tool_args, dict):
                            self.logger.debug(f"Converting tool_args from {type(tool_args)} to dict: {tool_args}")
                            if tool_args == "":
                                tool_args = {}
                            elif isinstance(tool_args, str):
                                try:
                                    # Try to parse as JSON
                                    tool_args = json.loads(tool_args)
                                except json.JSONDecodeError:
                                    self.logger.warning(f"Failed to parse tool_args as JSON: {tool_args}")
                                    tool_args = {}
                            else:
                                tool_args = {}
                        
                        if tool_idx == 0:  # Only show message for first tool use
                            await self.show_assistant_message(message_text, tool_name)
                        
                        # Show tool call with available tools
                        if tool_list and tool_list.tools:
                            self.show_tool_call(tool_list.tools, tool_name, tool_args)
                        else:
                            self.logger.warning(f"Tool list not available for displaying tool call: {tool_name}")
                        
                        # Map the tool name back to original MCP name if needed
                        original_tool_name = getattr(self, 'tool_name_mapping', {}).get(tool_name, tool_name)
                        self.logger.debug(f"Mapping tool name '{tool_name}' to '{original_tool_name}'")
                        
                        # Create tool call request
                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(
                                name=original_tool_name,
                                arguments=tool_args,
                            ),
                        )
                        
                        # Execute tool call
                        tool_result = await self.call_tool(tool_call_request, tool_use_id)
                        tool_results.append(tool_result)
                        
                        # Add tool result to messages
                        tool_result_content = tool_result.content[0].text if tool_result.content else ""
                        
                        # Format tool result for Nova compatibility
                        tool_result_message = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_call_id": tool_use_id,
                                    "content": tool_result_content,
                                }
                            ],
                        }
                        messages.append(tool_result_message)
                    
                    continue  # Continue to next iteration for tool use
                else:
                    # No tool uses but stop_reason was tool_use, treat as end_turn
                    await self.show_assistant_message(message_text)
                    break
            else:
                # Unknown stop reason, continue or break based on content
                message_text = ""
                for content_item in processed_response.get("content", []):
                    if content_item.get("text"):
                        message_text += content_item["text"]
                
                if message_text:
                    await self.show_assistant_message(message_text)
                break

        return responses

    async def generate_messages(
        self,
        message_param: BedrockMessageParam,
        request_params: RequestParams | None = None,
    ) -> PromptMessageMultipart:
        """Generate messages using Bedrock."""
        responses = await self._bedrock_completion(message_param, request_params)
        
        # Convert responses to PromptMessageMultipart
        content_list = []
        for response in responses:
            if isinstance(response, TextContent):
                content_list.append(response)
        
        return PromptMessageMultipart(role="assistant", content=content_list)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageMultipart],
        request_params: RequestParams | None = None,
        is_template: bool = False,
    ) -> PromptMessageMultipart:
        """Apply Bedrock-specific prompt formatting."""
        if not multipart_messages:
            return PromptMessageMultipart(role="user", content=[])
        
        # Check the last message role
        last_message = multipart_messages[-1]
        
        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user" inference is required
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            # Convert each message to Bedrock message parameter format
            bedrock_msg = {
                "role": msg.role,
                "content": []
            }
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    bedrock_msg["content"].append({
                        "type": "text",
                        "text": content_item.text
                    })
            converted.append(bedrock_msg)
        
        # Add messages to history
        self.history.extend(converted, is_prompt=is_template)
        
        if last_message.role == "assistant":
            # For assistant messages: Return the last message (no completion needed)
            return last_message
        
        # Convert the last user message to Bedrock message parameter format
        message_param = {
            "role": last_message.role,
            "content": []
        }
        for content_item in last_message.content:
            if isinstance(content_item, TextContent):
                message_param["content"].append({
                    "type": "text",
                    "text": content_item.text
                })
        
        # Generate response
        return await self.generate_messages(message_param, request_params)

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Apply structured output for Bedrock using prompt engineering (like Anthropic)."""
        request_params = self.get_request_params(request_params)

        # Use simple prompt engineering approach like Anthropic
        multipart_messages[-1].add_text(
            """YOU MUST RESPOND IN THE FOLLOWING FORMAT:
            {schema}
            RESPOND ONLY WITH THE JSON, NO PREAMBLE, CODE FENCES OR 'properties' ARE PERMISSABLE """.format(
                schema=model.model_json_schema()
            )
        )

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    @classmethod
    def convert_message_to_message_param(
        cls, message: BedrockMessage, **kwargs
    ) -> BedrockMessageParam:
        """Convert a Bedrock message to message parameter format."""
        message_param = {
            "role": message.get("role", "assistant"),
            "content": []
        }
        
        for content_item in message.get("content", []):
            if isinstance(content_item, dict):
                if "text" in content_item:
                    message_param["content"].append({
                        "type": "text",
                        "text": content_item["text"]
                    })
                elif "toolUse" in content_item:
                    tool_use = content_item["toolUse"]
                    tool_input = tool_use.get("input", {})
                    
                    # Ensure tool_input is a dictionary
                    if not isinstance(tool_input, dict):
                        if isinstance(tool_input, str):
                            try:
                                tool_input = json.loads(tool_input) if tool_input else {}
                            except json.JSONDecodeError:
                                tool_input = {}
                        else:
                            tool_input = {}
                    
                    message_param["content"].append({
                        "type": "tool_use",
                        "id": tool_use.get("toolUseId", ""),
                        "name": tool_use.get("name", ""),
                        "input": tool_input
                    })
        
        return message_param

    def _api_key(self) -> str:
        """Bedrock doesn't use API keys, returns empty string."""
        return "" 