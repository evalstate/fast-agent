import json
import os
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from rich.text import Text

from fast_agent.event_progress import ProgressAction
from fast_agent.types.llm_stop_reason import LlmStopReason
from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.augmented_llm import AugmentedLLM
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.usage_tracking import TurnUsage
from mcp_agent.logging.logger import get_logger
from mcp_agent.mcp.interfaces import ModelT
from mcp_agent.mcp.prompt_message_multipart import PromptMessageMultipart

# Mapping from Bedrock's snake_case stop reasons to MCP's camelCase
BEDROCK_TO_MCP_STOP_REASON = {
    "end_turn": LlmStopReason.END_TURN.value,
    "stop_sequence": LlmStopReason.STOP_SEQUENCE.value,
    "max_tokens": LlmStopReason.MAX_TOKENS.value,
}

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

try:
    from anthropic.types import ToolParam
except ImportError:
    ToolParam = None


DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"

# Bedrock message format types
BedrockMessage = Dict[str, Any]  # Bedrock message format
BedrockMessageParam = Dict[str, Any]  # Bedrock message parameter format


class ToolSchemaType(Enum):
    """Enum for different tool schema formats used by different model families."""

    DEFAULT = "default"  # Default toolSpec format used by most models (formerly Nova)
    SYSTEM_PROMPT = "system_prompt"  # System prompt-based tool calling format
    ANTHROPIC = "anthropic"  # Native Anthropic tool calling format


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
        AugmentedLLM.PARAM_MCP_METADATA,
    }

    @classmethod
    def matches_model_pattern(cls, model_name: str) -> bool:
        """Check if a model name matches Bedrock model patterns."""
        # Bedrock model patterns
        bedrock_patterns = [
            r"^amazon\.nova.*",  # Amazon Nova models
            r"^anthropic\.claude.*",  # Anthropic Claude models
            r"^meta\.llama.*",  # Meta Llama models
            r"^mistral\..*",  # Mistral models
            r"^cohere\..*",  # Cohere models
            r"^ai21\..*",  # AI21 models
            r"^stability\..*",  # Stability AI models
            r"^openai\..*",  # OpenAI models
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
            # Support both AWS_REGION and AWS_DEFAULT_REGION
            self.aws_region = os.environ.get("AWS_REGION") or os.environ.get(
                "AWS_DEFAULT_REGION", "us-east-1"
            )

        if not self.aws_profile:
            # Support AWS_PROFILE environment variable
            self.aws_profile = os.environ.get("AWS_PROFILE")

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
                self._bedrock_runtime_client = session.client(
                    "bedrock-runtime", region_name=self.aws_region
                )
            except NoCredentialsError as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_runtime_client

    def _get_tool_schema_type(self, model_id: str) -> ToolSchemaType:
        """
        Determine which tool schema format to use based on model family.

        Args:
            model_id: The model ID (e.g., "bedrock.meta.llama3-1-8b-instruct-v1:0")

        Returns:
            ToolSchemaType indicating which format to use
        """
        # Remove any "bedrock." prefix for pattern matching
        clean_model = model_id.replace("bedrock.", "")

        # Anthropic models use native Anthropic format
        if re.search(r"anthropic\.claude", clean_model):
            self.logger.debug(
                f"Model {model_id} detected as Anthropic - using native Anthropic format"
            )
            return ToolSchemaType.ANTHROPIC

        # Scout models use SYSTEM_PROMPT format
        if re.search(r"meta\.llama4-scout", clean_model):
            self.logger.debug(f"Model {model_id} detected as Scout - using SYSTEM_PROMPT format")
            return ToolSchemaType.SYSTEM_PROMPT

        # Other Llama 4 models use default toolConfig format
        if re.search(r"meta\.llama4", clean_model):
            self.logger.debug(
                f"Model {model_id} detected as Llama 4 (non-Scout) - using default toolConfig format"
            )
            return ToolSchemaType.DEFAULT

        # Llama 3.x models use system prompt format
        if re.search(r"meta\.llama3", clean_model):
            self.logger.debug(
                f"Model {model_id} detected as Llama 3.x - using system prompt format"
            )
            return ToolSchemaType.SYSTEM_PROMPT

        # Future: Add other model-specific formats here
        # if re.search(r"mistral\.", clean_model):
        #     return ToolSchemaType.MISTRAL

        # Default to default format for all other models
        self.logger.debug(f"Model {model_id} using default tool format")
        return ToolSchemaType.DEFAULT

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
            r"ai21\.jamba",  # All AI21 Jamba models
            r"meta\.llama",  # All Meta Llama models
            r"mistral\.",  # All Mistral models
            r"amazon\.titan",  # All Amazon Titan models
            r"cohere\.command",  # All Cohere Command models
            r"anthropic\.claude-instant",  # Anthropic Claude Instant models
            r"anthropic\.claude-v2",  # Anthropic Claude v2 models
            r"deepseek\.",  # All DeepSeek models
        ]

        for pattern in non_streaming_patterns:
            if re.search(pattern, clean_model, re.IGNORECASE):
                self.logger.debug(
                    f"Model {model} detected as non-streaming for tools (pattern: {pattern})"
                )
                return False

        return True

    def _supports_tool_use(self, model_id: str) -> bool:
        """
        Determine if a model supports tool use at all.
        Some models don't support tools in any form.
        Based on AWS Bedrock documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html
        """
        # Models that don't support tool use at all
        no_tool_use_patterns = [
            r"ai21\.jamba-instruct",  # AI21 Jamba-Instruct (but not jamba 1.5)
            r"ai21\..*jurassic",  # AI21 Labs Jurassic-2 models
            r"amazon\.titan",  # All Amazon Titan models
            r"anthropic\.claude-v2",  # Anthropic Claude v2 models
            r"anthropic\.claude-instant",  # Anthropic Claude Instant models
            r"cohere\.command(?!-r)",  # Cohere Command (but not Command R/R+)
            r"cohere\.command-light",  # Cohere Command Light
            r"deepseek\.",  # All DeepSeek models
            r"meta\.llama[23](?![-.])",  # Meta Llama 2 and 3 (but not 3.1+, 3.2+, etc.)
            r"meta\.llama3-1-8b",  # Meta Llama 3.1 8b - doesn't support tool calls
            r"meta\.llama3-2-[13]b",  # Meta Llama 3.2 1b and 3b (but not 11b/90b)
            r"meta\.llama3-2-11b",  # Meta Llama 3.2 11b - doesn't support tool calls
            r"mistral\..*-instruct",  # Mistral AI Instruct (but not Mistral Large)
        ]

        for pattern in no_tool_use_patterns:
            if re.search(pattern, model_id):
                self.logger.info(f"Model {model_id} does not support tool use")
                return False

        return True

    def _supports_system_messages(self, model: str) -> bool:
        """
        Check if a model supports system messages.

        Some models (like Titan and Cohere embedding models) don't support system messages.
        This method uses regex patterns to identify such models.

        Args:
            model: The model name (e.g., "amazon.titan-embed-text-v1")

        Returns:
            False if the model doesn't support system messages, True otherwise
        """
        # Remove any "bedrock." prefix for pattern matching
        clean_model = model.replace("bedrock.", "")

        # DEBUG: Print the model names for debugging
        self.logger.info(
            f"DEBUG: Checking system message support for model='{model}', clean_model='{clean_model}'"
        )

        # Models that don't support system messages (reverse logic as suggested)
        no_system_message_patterns = [
            r"amazon\.titan",  # All Amazon Titan models
            r"cohere\.command.*-text",  # Cohere command text models (command-text-v14, command-light-text-v14)
            r"mistral.*mixtral.*8x7b",  # Mistral Mixtral 8x7b models
            r"mistral.mistral-7b-instruct",  # Mistral 7b instruct models
            r"meta\.llama3-2-11b-instruct",  # Specific Meta Llama3 model
        ]

        for pattern in no_system_message_patterns:
            if re.search(pattern, clean_model, re.IGNORECASE):
                self.logger.info(
                    f"DEBUG: Model {model} detected as NOT supporting system messages (pattern: {pattern})"
                )
                return False

        self.logger.info(f"DEBUG: Model {model} detected as supporting system messages")
        return True

    def _convert_tools_nova_format(self, tools: "ListToolsResult") -> List[Dict[str, Any]]:
        """Convert MCP tools to Nova-specific toolSpec format.

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

        self.logger.debug(f"Converting {len(tools.tools)} MCP tools to Nova format")

        for tool in tools.tools:
            self.logger.debug(f"Converting MCP tool: {tool.name}")

            # Extract and validate the input schema
            input_schema = tool.inputSchema or {}

            # Create Nova-compliant schema with ONLY the three allowed fields
            # Always include type and properties (even if empty)
            nova_schema: Dict[str, Any] = {"type": "object", "properties": {}}

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
            if (
                "required" in input_schema
                and isinstance(input_schema["required"], list)
                and input_schema["required"]
            ):
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
                    "inputSchema": {"json": nova_schema},
                }
            }

            bedrock_tools.append(bedrock_tool)

        self.logger.debug(f"Converted {len(bedrock_tools)} tools for Nova format")
        return bedrock_tools

    def _convert_tools_system_prompt_format(self, tools: "ListToolsResult") -> str:
        """Convert MCP tools to system prompt format.

        Uses different formats based on the model:
        - Scout models: Comprehensive system prompt format
        - Other models: Minimal format
        """
        if not tools.tools:
            return ""

        # Create mapping from tool names to original names (no cleaning needed for Llama)
        self.tool_name_mapping = {}

        self.logger.debug(
            f"Converting {len(tools.tools)} MCP tools to Llama native system prompt format"
        )

        # Check if this is a Scout model
        model_id = self.default_request_params.model or DEFAULT_BEDROCK_MODEL
        clean_model = model_id.replace("bedrock.", "")
        is_scout = re.search(r"meta\.llama4-scout", clean_model)

        if is_scout:
            # Use comprehensive system prompt format for Scout models
            prompt_parts = [
                "You are a helpful assistant with access to the following functions. Use them if required:",
                "",
            ]

            # Add each tool definition in JSON format
            for tool in tools.tools:
                self.logger.debug(f"Converting MCP tool: {tool.name}")

                # Use original tool name (no hyphen replacement for Llama)
                tool_name = tool.name

                # Store mapping (identity mapping since no name cleaning)
                self.tool_name_mapping[tool_name] = tool.name

                # Create tool definition in the format Llama expects
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }

                prompt_parts.append(json.dumps(tool_def))

            # Add comprehensive response format instructions for Scout
            prompt_parts.extend(
                [
                    "",
                    "## Rules for Function Calling:",
                    "1. When you need to call a function, use the following format:",
                    "   [function_name(arguments)]",
                    "2. You can call multiple functions in a single response if needed",
                    "3. Always provide the function results in your response to the user",
                    "4. If a function call fails, explain the error and try an alternative approach",
                    "5. Only call functions when necessary to answer the user's question",
                    "",
                    "## Response Rules:",
                    "- Always provide a complete answer to the user's question",
                    "- Include function results in your response",
                    "- Be helpful and informative",
                    "- If you cannot answer without calling a function, call the appropriate function first",
                    "",
                    "## Boundaries:",
                    "- Only call functions that are explicitly provided above",
                    "- Do not make up function names or parameters",
                    "- Follow the exact function signature provided",
                    "- Always validate your function calls before making them",
                ]
            )
        else:
            # Use minimal format for other Llama models
            prompt_parts = [
                "You have the following tools available to help answer the user's request. You can call one or more functions at a time. The functions are described here in JSON-schema format:",
                "",
            ]

            # Add each tool definition in JSON format
            for tool in tools.tools:
                self.logger.debug(f"Converting MCP tool: {tool.name}")

                # Use original tool name (no hyphen replacement for Llama)
                tool_name = tool.name

                # Store mapping (identity mapping since no name cleaning)
                self.tool_name_mapping[tool_name] = tool.name

                # Create tool definition in the format Llama expects
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description or f"Tool: {tool.name}",
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    },
                }

                prompt_parts.append(json.dumps(tool_def))

            # Add the response format instructions based on community best practices
            prompt_parts.extend(
                [
                    "",
                    "To call one or more tools, provide the tool calls on a new line as a JSON-formatted array. Explain your steps in a neutral tone. Then, only call the tools you can for the first step, then end your turn. If you previously received an error, you can try to call the tool again. Give up after 3 errors.",
                    "",
                    "Conform precisely to the single-line format of this example:",
                    "Tool Call:",
                    '[{"name": "SampleTool", "arguments": {"foo": "bar"}},{"name": "SampleTool", "arguments": {"foo": "other"}}]',
                ]
            )

        system_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated Llama native system prompt: {system_prompt}")

        return system_prompt

    def _convert_tools_anthropic_format(self, tools: "ListToolsResult") -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format wrapped in Bedrock toolSpec - preserves raw schema."""
        # No tool name mapping needed for Anthropic (uses original names)
        self.tool_name_mapping = {}

        self.logger.debug(
            f"Converting {len(tools.tools)} MCP tools to Anthropic format with toolSpec wrapper"
        )

        bedrock_tools = []
        for tool in tools.tools:
            self.logger.debug(f"Converting MCP tool: {tool.name}")

            # Store identity mapping (no name cleaning for Anthropic)
            self.tool_name_mapping[tool.name] = tool.name

            # Use raw MCP schema (like native Anthropic provider) - no cleaning
            input_schema = tool.inputSchema or {"type": "object", "properties": {}}

            # Wrap in Bedrock toolSpec format but preserve raw Anthropic schema
            bedrock_tool = {
                "toolSpec": {
                    "name": tool.name,  # Original name, no cleaning
                    "description": tool.description or f"Tool: {tool.name}",
                    "inputSchema": {
                        "json": input_schema  # Raw MCP schema, not cleaned
                    },
                }
            }
            bedrock_tools.append(bedrock_tool)

        self.logger.debug(
            f"Converted {len(bedrock_tools)} tools to Anthropic format with toolSpec wrapper"
        )
        return bedrock_tools

    def _convert_mcp_tools_to_bedrock(
        self, tools: "ListToolsResult"
    ) -> Union[List[Dict[str, Any]], str]:
        """Convert MCP tools to appropriate Bedrock format based on model type."""
        model_id = self.default_request_params.model or DEFAULT_BEDROCK_MODEL
        schema_type = self._get_tool_schema_type(model_id)

        if schema_type == ToolSchemaType.SYSTEM_PROMPT:
            system_prompt = self._convert_tools_system_prompt_format(tools)
            # Store the system prompt for later use in system message
            self._system_prompt_tools = system_prompt
            return system_prompt
        elif schema_type == ToolSchemaType.ANTHROPIC:
            return self._convert_tools_anthropic_format(tools)
        else:
            return self._convert_tools_nova_format(tools)

    def _add_tools_to_request(
        self,
        converse_args: Dict[str, Any],
        available_tools: Union[List[Dict[str, Any]], str],
        model_id: str,
    ) -> None:
        """Add tools to the request in the appropriate format based on model type."""
        schema_type = self._get_tool_schema_type(model_id)

        if schema_type == ToolSchemaType.SYSTEM_PROMPT:
            # System prompt models expect tools in the system prompt, not as API parameters
            # Tools are already handled in the system prompt generation
            self.logger.debug("System prompt tools handled in system prompt")
        elif schema_type == ToolSchemaType.ANTHROPIC:
            # Anthropic models expect toolConfig with tools array (like native provider)
            converse_args["toolConfig"] = {"tools": available_tools}
            self.logger.debug(
                f"Added {len(available_tools)} tools to Anthropic request in toolConfig format"
            )
        else:
            # Nova models expect toolConfig with toolSpec format
            converse_args["toolConfig"] = {"tools": available_tools}
            self.logger.debug(
                f"Added {len(available_tools)} tools to Nova request in toolConfig format"
            )

    def _parse_nova_tool_response(self, processed_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Nova-format tool response (toolUse format)."""
        tool_uses = [
            content_item
            for content_item in processed_response.get("content", [])
            if "toolUse" in content_item
        ]

        parsed_tools = []
        for tool_use_item in tool_uses:
            tool_use = tool_use_item["toolUse"]
            parsed_tools.append(
                {
                    "type": "nova",
                    "name": tool_use["name"],
                    "arguments": tool_use["input"],
                    "id": tool_use["toolUseId"],
                }
            )

        return parsed_tools

    def _parse_system_prompt_tool_response(
        self, processed_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse system prompt tool response format: function calls in text."""
        # Extract text content from the response
        text_content = ""
        for content_item in processed_response.get("content", []):
            if isinstance(content_item, dict) and "text" in content_item:
                text_content += content_item["text"]

        if not text_content:
            return []

        # Look for different tool call formats
        tool_calls = []

        # First try Scout format: [function_name(arguments)]
        scout_pattern = r"\[([^(]+)\(([^)]*)\)\]"
        scout_matches = re.findall(scout_pattern, text_content)
        if scout_matches:
            for i, (func_name, args_str) in enumerate(scout_matches):
                func_name = func_name.strip()
                args_str = args_str.strip()

                # Parse arguments - could be empty, JSON object, or simple values
                arguments = {}
                if args_str:
                    try:
                        # Try to parse as JSON object first
                        if args_str.startswith("{") and args_str.endswith("}"):
                            arguments = json.loads(args_str)
                        else:
                            # For simple values, create a basic structure
                            arguments = {"value": args_str}
                    except json.JSONDecodeError:
                        # If JSON parsing fails, treat as string
                        arguments = {"value": args_str}

                tool_calls.append(
                    {
                        "type": "system_prompt",
                        "name": func_name,
                        "arguments": arguments,
                        "id": f"system_prompt_{func_name}_{i}",
                    }
                )

            if tool_calls:
                return tool_calls

        # Second try: find the "Tool Call:" format
        tool_call_match = re.search(r"Tool Call:\s*(\[.*?\])", text_content, re.DOTALL)
        if tool_call_match:
            json_str = tool_call_match.group(1)
            try:
                parsed_calls = json.loads(json_str)
                if isinstance(parsed_calls, list):
                    for i, call in enumerate(parsed_calls):
                        if isinstance(call, dict) and "name" in call:
                            tool_calls.append(
                                {
                                    "type": "system_prompt",
                                    "name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                    "id": f"system_prompt_{call['name']}_{i}",
                                }
                            )
                    return tool_calls
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse Tool Call JSON array: {json_str} - {e}")

        # Fallback: try to parse any JSON array in the text
        array_match = re.search(r"\[.*?\]", text_content, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                parsed_calls = json.loads(json_str)
                if isinstance(parsed_calls, list):
                    for i, call in enumerate(parsed_calls):
                        if isinstance(call, dict) and "name" in call:
                            tool_calls.append(
                                {
                                    "type": "system_prompt",
                                    "name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                    "id": f"system_prompt_{call['name']}_{i}",
                                }
                            )
                    return tool_calls
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON array: {json_str} - {e}")

        # Fallback: try to parse as single JSON object (backward compatibility)
        try:
            json_match = re.search(r'\{[^}]*"name"[^}]*"arguments"[^}]*\}', text_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                function_call = json.loads(json_str)

                if "name" in function_call:
                    return [
                        {
                            "type": "system_prompt",
                            "name": function_call["name"],
                            "arguments": function_call.get("arguments", {}),
                            "id": f"system_prompt_{function_call['name']}",
                        }
                    ]

        except json.JSONDecodeError as e:
            self.logger.warning(
                f"Failed to parse system prompt tool response as JSON: {text_content} - {e}"
            )

            # Fallback to old custom tag format in case some models still use it
            function_regex = r"<function=([^>]+)>(.*?)</function>"
            match = re.search(function_regex, text_content)

            if match:
                function_name = match.group(1)
                function_args_json = match.group(2)

                try:
                    function_args = json.loads(function_args_json)
                    return [
                        {
                            "type": "system_prompt",
                            "name": function_name,
                            "arguments": function_args,
                            "id": f"system_prompt_{function_name}",
                        }
                    ]
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"Failed to parse fallback custom tag format: {function_args_json}"
                    )

        return []

    def _parse_anthropic_tool_response(
        self, processed_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse Anthropic tool response format (same as native provider)."""
        tool_uses = []

        # Look for toolUse in content items (Bedrock format for Anthropic models)
        for content_item in processed_response.get("content", []):
            if "toolUse" in content_item:
                tool_use = content_item["toolUse"]
                tool_uses.append(
                    {
                        "type": "anthropic",
                        "name": tool_use["name"],
                        "arguments": tool_use["input"],
                        "id": tool_use["toolUseId"],
                    }
                )

        return tool_uses

    def _parse_tool_response(
        self, processed_response: Dict[str, Any], model_id: str
    ) -> List[Dict[str, Any]]:
        """Parse tool response based on model type."""
        schema_type = self._get_tool_schema_type(model_id)

        if schema_type == ToolSchemaType.SYSTEM_PROMPT:
            return self._parse_system_prompt_tool_response(processed_response)
        elif schema_type == ToolSchemaType.ANTHROPIC:
            return self._parse_anthropic_tool_response(processed_response)
        else:
            return self._parse_nova_tool_response(processed_response)

    def _convert_messages_to_bedrock(
        self, messages: List[BedrockMessageParam]
    ) -> List[Dict[str, Any]]:
        """Convert message parameters to Bedrock format."""
        bedrock_messages = []
        for message in messages:
            bedrock_message = {"role": message.get("role", "user"), "content": []}

            content = message.get("content", [])

            if isinstance(content, str):
                bedrock_message["content"].append({"text": content})
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type")
                    if item_type == "text":
                        bedrock_message["content"].append({"text": item.get("text", "")})
                    elif item_type == "tool_use":
                        bedrock_message["content"].append(
                            {
                                "toolUse": {
                                    "toolUseId": item.get("id", ""),
                                    "name": item.get("name", ""),
                                    "input": item.get("input", {}),
                                }
                            }
                        )
                    elif item_type == "tool_result":
                        tool_use_id = item.get("tool_use_id")
                        raw_content = item.get("content", [])
                        status = item.get("status", "success")

                        bedrock_content_list = []
                        if raw_content:
                            for part in raw_content:
                                # FIX: The content parts are dicts, not TextContent objects.
                                if isinstance(part, dict) and "text" in part:
                                    bedrock_content_list.append({"text": part.get("text", "")})

                        # Bedrock requires content for error statuses.
                        if not bedrock_content_list and status == "error":
                            bedrock_content_list.append({"text": "Tool call failed with an error."})

                        bedrock_message["content"].append(
                            {
                                "toolResult": {
                                    "toolUseId": tool_use_id,
                                    "content": bedrock_content_list,
                                    "status": status,
                                }
                            }
                        )

            # Only add the message if it has content
            if bedrock_message["content"]:
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
                        tool_uses.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_use_start.get("toolUseId"),
                                    "name": tool_use_start.get("name"),
                                    "input": tool_use_start.get("input", {}),
                                    "_input_accumulator": "",  # For accumulating streamed input
                                }
                            }
                        )
                elif "contentBlockDelta" in event:
                    # Content delta received
                    delta = event["contentBlockDelta"]["delta"]
                    if "text" in delta:
                        text = delta["text"]
                        response_content.append(text)
                        # Update streaming progress
                        estimated_tokens = self._update_streaming_progress(
                            text, model, estimated_tokens
                        )
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
                                    self.logger.debug(
                                        f"Accumulated input: {tool_uses[-1]['toolUse']['_input_accumulator']}"
                                    )
                                else:
                                    self.logger.debug(
                                        f"Tool use input is unexpected type: {type(input_data)}: {input_data}"
                                    )
                                    # Set the input directly if it's not a dict or string
                                    tool_uses[-1]["toolUse"]["input"] = input_data
                elif "contentBlockStop" in event:
                    # Content block stopped - finalize any accumulated tool input
                    if tool_uses:
                        for tool_use in tool_uses:
                            if "_input_accumulator" in tool_use["toolUse"]:
                                accumulated_input = tool_use["toolUse"]["_input_accumulator"]
                                if accumulated_input:
                                    self.logger.debug(
                                        f"Processing accumulated input: {accumulated_input}"
                                    )
                                    try:
                                        # Try to parse the accumulated input as JSON
                                        parsed_input = json.loads(accumulated_input)
                                        if isinstance(parsed_input, dict):
                                            tool_use["toolUse"]["input"].update(parsed_input)
                                        else:
                                            tool_use["toolUse"]["input"] = parsed_input
                                        self.logger.debug(
                                            f"Successfully parsed accumulated input: {parsed_input}"
                                        )
                                    except json.JSONDecodeError as e:
                                        self.logger.warning(
                                            f"Failed to parse accumulated input as JSON: {accumulated_input} - {e}"
                                        )
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
                        self.logger.debug(
                            f"Final processing of accumulated input: {accumulated_input}"
                        )
                        try:
                            # Try to parse the accumulated input as JSON
                            parsed_input = json.loads(accumulated_input)
                            if isinstance(parsed_input, dict):
                                tool_use["toolUse"]["input"].update(parsed_input)
                            else:
                                tool_use["toolUse"]["input"] = parsed_input
                            self.logger.debug(
                                f"Successfully parsed final accumulated input: {parsed_input}"
                            )
                        except json.JSONDecodeError as e:
                            self.logger.warning(
                                f"Failed to parse final accumulated input as JSON: {accumulated_input} - {e}"
                            )
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
    ) -> List[ContentBlock | CallToolRequestParams]:
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

        # Get available tools - but only if model supports tool use
        available_tools = []
        tool_list = None
        model_to_check = self.default_request_params.model or DEFAULT_BEDROCK_MODEL

        if self._supports_tool_use(model_to_check):
            try:
                tool_list = await self.aggregator.list_tools()
                self.logger.debug(f"Found {len(tool_list.tools)} MCP tools")

                available_tools = self._convert_mcp_tools_to_bedrock(tool_list)
                self.logger.debug(
                    f"Successfully converted {len(available_tools)} tools for Bedrock"
                )

            except Exception as e:
                self.logger.error(f"Error fetching or converting MCP tools: {e}")
                import traceback

                self.logger.debug(f"Traceback: {traceback.format_exc()}")
                available_tools = []
                tool_list = None
        else:
            self.logger.info(
                f"Model {model_to_check} does not support tool use - skipping tool preparation"
            )

        responses: List[ContentBlock] = []
        model = self.default_request_params.model

        for i in range(params.max_iterations):
            self._log_chat_progress(self.chat_turn(), model=model)

            # Process tools BEFORE message conversion for Llama native format
            model_to_check = model or DEFAULT_BEDROCK_MODEL
            schema_type = self._get_tool_schema_type(model_to_check)

            # For Llama native format, we need to store tools before message conversion
            if schema_type == ToolSchemaType.SYSTEM_PROMPT and available_tools:
                has_tools = bool(available_tools) and (
                    (isinstance(available_tools, list) and len(available_tools) > 0)
                    or (isinstance(available_tools, str) and available_tools.strip())
                )

                if has_tools:
                    self._add_tools_to_request({}, available_tools, model_to_check)
                    self.logger.debug("Pre-processed Llama native tools for message injection")

            # Convert messages to Bedrock format
            bedrock_messages = self._convert_messages_to_bedrock(messages)

            # Prepare Bedrock Converse API arguments
            converse_args = {
                "modelId": model,
                "messages": bedrock_messages,
            }

            # Add system prompt if available and supported by the model
            system_text = self.instruction or params.systemPrompt

            # For Llama native format, inject tools into system prompt
            if (
                schema_type == ToolSchemaType.SYSTEM_PROMPT
                and hasattr(self, "_system_prompt_tools")
                and self._system_prompt_tools
            ):
                # Combine system prompt with tools for Llama native format
                if system_text:
                    system_text = f"{system_text}\n\n{self._system_prompt_tools}"
                else:
                    system_text = self._system_prompt_tools
                self.logger.debug("Combined system prompt with system prompt tools")
            elif hasattr(self, "_system_prompt_tools") and self._system_prompt_tools:
                # For other formats, combine system prompt with tools
                if system_text:
                    system_text = f"{system_text}\n\n{self._system_prompt_tools}"
                else:
                    system_text = self._system_prompt_tools
                self.logger.debug("Combined system prompt with tools system prompt")

            self.logger.info(
                f"DEBUG: BEFORE CHECK - model='{model_to_check}', has_system_text={bool(system_text)}"
            )
            self.logger.info(
                f"DEBUG: self.instruction='{self.instruction}', params.systemPrompt='{params.systemPrompt}'"
            )

            supports_system = self._supports_system_messages(model_to_check)
            self.logger.info(f"DEBUG: supports_system={supports_system}")

            if system_text and supports_system:
                converse_args["system"] = [{"text": system_text}]
                self.logger.info(f"DEBUG: Added system prompt to {model_to_check} request")
            elif system_text:
                # For models that don't support system messages, inject system prompt into the first user message
                self.logger.info(
                    f"DEBUG: Injecting system prompt into first user message for {model_to_check} (doesn't support system messages)"
                )
                if bedrock_messages and bedrock_messages[0].get("role") == "user":
                    first_message = bedrock_messages[0]
                    if first_message.get("content") and len(first_message["content"]) > 0:
                        # Prepend system instruction to the first user message
                        original_text = first_message["content"][0].get("text", "")
                        first_message["content"][0]["text"] = (
                            f"System: {system_text}\n\nUser: {original_text}"
                        )
                        self.logger.info("DEBUG: Injected system prompt into first user message")
            else:
                self.logger.info(f"DEBUG: No system text provided for {model_to_check}")

            # Add tools if available - format depends on model type (skip for Llama native as already processed)
            if schema_type != ToolSchemaType.SYSTEM_PROMPT:
                has_tools = bool(available_tools) and (
                    (isinstance(available_tools, list) and len(available_tools) > 0)
                    or (isinstance(available_tools, str) and available_tools.strip())
                )

                if has_tools:
                    self._add_tools_to_request(converse_args, available_tools, model_to_check)
                else:
                    self.logger.debug(
                        "No tools available - omitting tool configuration from request"
                    )

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
                converse_args["additionalModelRequestFields"] = {"inferenceConfig": {"topK": 1}}

            if inference_config:
                converse_args["inferenceConfig"] = inference_config

            self.logger.debug(f"Bedrock converse args: {converse_args}")

            # Debug: Print the actual messages being sent to Bedrock for Llama models
            schema_type = self._get_tool_schema_type(model_to_check)
            if schema_type == ToolSchemaType.SYSTEM_PROMPT:
                self.logger.info("=== SYSTEM PROMPT DEBUG ===")
                self.logger.info("Messages being sent to Bedrock:")
                for i, msg in enumerate(converse_args.get("messages", [])):
                    self.logger.info(f"Message {i} ({msg.get('role', 'unknown')}):")
                    for j, content in enumerate(msg.get("content", [])):
                        if "text" in content:
                            self.logger.info(f"  Content {j}: {content['text'][:500]}...")
                self.logger.info("=== END SYSTEM PROMPT DEBUG ===")

            # Debug: Print the full tool config being sent
            if "toolConfig" in converse_args:
                self.logger.debug(
                    f"Tool config being sent to Bedrock: {json.dumps(converse_args['toolConfig'], indent=2)}"
                )

            try:
                # Choose streaming vs non-streaming based on model capabilities and tool presence
                # Logic: Only use non-streaming when BOTH conditions are true:
                #   1. Tools are available (available_tools is not empty)
                #   2. Model doesn't support streaming with tools
                # Otherwise, always prefer streaming for better UX
                has_tools = bool(available_tools) and (
                    (isinstance(available_tools, list) and len(available_tools) > 0)
                    or (isinstance(available_tools, str) and available_tools.strip())
                )

                if has_tools and not self._supports_streaming_with_tools(
                    model or DEFAULT_BEDROCK_MODEL
                ):
                    # Use non-streaming API: model requires it for tool calls
                    self.logger.debug(
                        f"Using non-streaming API for {model} with tools (model limitation)"
                    )
                    response = client.converse(**converse_args)
                    processed_response = self._process_non_streaming_response(
                        response, model or DEFAULT_BEDROCK_MODEL
                    )
                else:
                    # Use streaming API: either no tools OR model supports streaming with tools
                    streaming_reason = (
                        "no tools present"
                        if not has_tools
                        else "model supports streaming with tools"
                    )
                    self.logger.debug(f"Using streaming API for {model} ({streaming_reason})")
                    response = client.converse_stream(**converse_args)
                    processed_response = await self._process_stream(
                        response, model or DEFAULT_BEDROCK_MODEL
                    )
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
                        raw_usage=usage,
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

            # For Llama native format, check for tool calls even if stop_reason is "end_turn"
            schema_type = self._get_tool_schema_type(model or DEFAULT_BEDROCK_MODEL)
            if schema_type == ToolSchemaType.SYSTEM_PROMPT and stop_reason == "end_turn":
                # Check if there's a tool call in the response
                parsed_tools = self._parse_tool_response(
                    processed_response, model or DEFAULT_BEDROCK_MODEL
                )
                if parsed_tools:
                    # Override stop_reason to handle as tool_use
                    stop_reason = "tool_use"
                    self.logger.debug(
                        "Detected system prompt tool call, overriding stop_reason to 'tool_use'"
                    )

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
            elif stop_reason in ["tool_use", "tool_calls"]:
                # Handle tool use/calls - format depends on model type
                message_text = ""
                for content_item in processed_response.get("content", []):
                    if content_item.get("text"):
                        message_text += content_item["text"]

                # Parse tool calls using model-specific method
                self.logger.info(f"DEBUG: About to parse tool response: {processed_response}")
                parsed_tools = self._parse_tool_response(
                    processed_response, model or DEFAULT_BEDROCK_MODEL
                )
                self.logger.info(f"DEBUG: Parsed tools: {parsed_tools}")

                if parsed_tools:
                    # We will comment out showing the assistant's intermediate message
                    # to make the output less chatty, as requested by the user.
                    # if not message_text:
                    #     message_text = Text(
                    #         "the assistant requested tool calls",
                    #         style="dim green italic",
                    #     )
                    #
                    # await self.show_assistant_message(message_text)

                    # Process tool calls and collect results
                    tool_results_for_batch = []
                    for tool_idx, parsed_tool in enumerate(parsed_tools):
                        # The original name is needed to call the tool, which is in tool_name_mapping.
                        tool_name_from_model = parsed_tool["name"]
                        tool_name = self.tool_name_mapping.get(
                            tool_name_from_model, tool_name_from_model
                        )

                        tool_args = parsed_tool["arguments"]
                        tool_use_id = parsed_tool["id"]

                        self.show_tool_call(tool_list.tools, tool_name, tool_args)

                        tool_call_request = CallToolRequest(
                            method="tools/call",
                            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
                        )

                        # Call the tool and get the result
                        result = await self.call_tool(
                            request=tool_call_request, tool_call_id=tool_use_id
                        )
                        # We will also comment out showing the raw tool result to reduce verbosity.
                        # self.show_tool_result(result)

                        # Add each result to our collection
                        tool_results_for_batch.append((tool_use_id, result, tool_name))
                        responses.extend(result.content)

                    # After processing all tool calls for a turn, clear the intermediate
                    # responses. This ensures that the final returned value only contains
                    # the model's last message, not the reasoning or raw tool output.
                    responses.clear()

                    # Now, create the message with tool results based on the model's schema type.
                    schema_type = self._get_tool_schema_type(model or DEFAULT_BEDROCK_MODEL)

                    if schema_type == ToolSchemaType.SYSTEM_PROMPT:
                        # For system prompt models (like Llama), format results as a simple text message.
                        # The model expects to see the results in a human-readable format to continue.
                        tool_result_parts = []
                        for _, tool_result, tool_name in tool_results_for_batch:
                            result_text = "".join(
                                [
                                    part.text
                                    for part in tool_result.content
                                    if isinstance(part, TextContent)
                                ]
                            )

                            # Create a representation of the tool's output.
                            # Using a JSON-like string is a robust way to present this.
                            result_payload = {
                                "tool_name": tool_name,
                                "status": "error" if tool_result.isError else "success",
                                "result": result_text,
                            }
                            tool_result_parts.append(json.dumps(result_payload))

                        if tool_result_parts:
                            # Combine all tool results into a single text block.
                            full_result_text = f"Tool Results:\n{', '.join(tool_result_parts)}"
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "text", "text": full_result_text}],
                                }
                            )
                    else:
                        # For native tool-using models (Anthropic, Nova), use the structured 'tool_result' format.
                        tool_result_blocks = []
                        for tool_id, tool_result, _ in tool_results_for_batch:
                            # Convert tool result content into a list of content blocks
                            # This mimics the native Anthropic provider's approach.
                            result_content_blocks = []
                            if tool_result.content:
                                for part in tool_result.content:
                                    if isinstance(part, TextContent):
                                        result_content_blocks.append({"text": part.text})
                                    # Note: This can be extended to handle other content types like images
                                    # For now, we are focusing on making text-based tools work correctly.

                            # If there's no content, provide a default message.
                            if not result_content_blocks:
                                result_content_blocks.append(
                                    {"text": "[No content in tool result]"}
                                )

                            # This is the format Bedrock expects for tool results in the Converse API
                            tool_result_blocks.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": result_content_blocks,
                                    "status": "error" if tool_result.isError else "success",
                                }
                            )

                        if tool_result_blocks:
                            # Append a single user message with all the tool results for this turn
                            messages.append(
                                {
                                    "role": "user",
                                    "content": tool_result_blocks,
                                }
                            )

                    continue
                else:
                    # No tool uses but stop_reason was tool_use/tool_calls, treat as end_turn
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

        # Update history
        if params.use_history:
            # Get current prompt messages
            prompt_messages = self.history.get(include_completion_history=False)

            # Calculate new conversation messages (excluding prompts)
            new_messages = messages[len(prompt_messages) :]

            # Remove system prompt from new messages if it was added
            if (self.instruction or params.systemPrompt) and new_messages:
                # System prompt is not added to messages list in Bedrock, so no need to remove it
                pass

            self.history.set(new_messages)

        # Strip leading whitespace from the *last* non-empty text block of the final response
        # to ensure the output is clean.
        if responses:
            for item in reversed(responses):
                if isinstance(item, TextContent) and item.text:
                    item.text = item.text.lstrip()
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
            bedrock_msg = {"role": msg.role, "content": []}
            for content_item in msg.content:
                if isinstance(content_item, TextContent):
                    bedrock_msg["content"].append({"type": "text", "text": content_item.text})
            converted.append(bedrock_msg)

        # Add messages to history
        self.history.extend(converted, is_prompt=is_template)

        if last_message.role == "assistant":
            # For assistant messages: Return the last message (no completion needed)
            return last_message

        # Convert the last user message to Bedrock message parameter format
        message_param = {"role": last_message.role, "content": []}
        for content_item in last_message.content:
            if isinstance(content_item, TextContent):
                message_param["content"].append({"type": "text", "text": content_item.text})

        # Generate response
        return await self.generate_messages(message_param, request_params)

    def _generate_simplified_schema(self, model: Type[ModelT]) -> str:
        """Generates a simplified, human-readable schema with inline enum constraints."""

        def get_field_type_representation(field_type: Any) -> Any:
            """Get a string representation for a field type."""
            # Handle Optional types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                non_none_types = [t for t in field_type.__args__ if t is not type(None)]
                if non_none_types:
                    field_type = non_none_types[0]

            # Handle basic types
            if field_type is str:
                return "string"
            elif field_type is int:
                return "integer"
            elif field_type is float:
                return "float"
            elif field_type is bool:
                return "boolean"

            # Handle Enum types
            elif hasattr(field_type, "__bases__") and any(
                issubclass(base, Enum) for base in field_type.__bases__ if isinstance(base, type)
            ):
                enum_values = [f'"{e.value}"' for e in field_type]
                return f"string (must be one of: {', '.join(enum_values)})"

            # Handle List types
            elif (
                hasattr(field_type, "__origin__")
                and hasattr(field_type, "__args__")
                and field_type.__origin__ is list
            ):
                item_type_repr = "any"
                if field_type.__args__:
                    item_type_repr = get_field_type_representation(field_type.__args__[0])
                return [item_type_repr]

            # Handle nested Pydantic models
            elif hasattr(field_type, "__bases__") and any(
                hasattr(base, "model_fields") for base in field_type.__bases__
            ):
                nested_schema = _generate_schema_dict(field_type)
                return nested_schema

            # Default fallback
            else:
                return "any"

        def _generate_schema_dict(model_class: Type) -> Dict[str, Any]:
            """Recursively generate the schema as a dictionary."""
            schema_dict = {}
            if hasattr(model_class, "model_fields"):
                for field_name, field_info in model_class.model_fields.items():
                    schema_dict[field_name] = get_field_type_representation(field_info.annotation)
            return schema_dict

        schema = _generate_schema_dict(model)
        return json.dumps(schema, indent=2)

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: List[PromptMessageMultipart],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Apply structured output for Bedrock using prompt engineering with a simplified schema."""
        request_params = self.get_request_params(request_params)

        # Generate a simplified, human-readable schema
        simplified_schema = self._generate_simplified_schema(model)

        # Build the new simplified prompt
        prompt_parts = [
            "You are a JSON generator. Respond with JSON that strictly follows the provided schema. Do not add any commentary or explanation.",
            "",
            "JSON Schema:",
            simplified_schema,
            "",
            "IMPORTANT RULES:",
            "- You MUST respond with only raw JSON data. No other text, commentary, or markdown is allowed.",
            "- All field names and enum values are case-sensitive and must match the schema exactly.",
            "- Do not add any extra fields to the JSON response. Only include the fields specified in the schema.",
            "- Valid JSON requires double quotes for all field names and string values. Other types (int, float, boolean, etc.) should not be quoted.",
            "",
            "Now, generate the valid JSON response for the following request:",
        ]

        # Add the new prompt to the last user message
        multipart_messages[-1].add_text("\n".join(prompt_parts))

        self.logger.info(f"DEBUG: Prompt messages: {multipart_messages[-1].content}")

        result: PromptMessageMultipart = await self._apply_prompt_provider_specific(
            multipart_messages, request_params
        )
        return self._structured_from_multipart(result, model)

    def _clean_json_response(self, text: str) -> str:
        """Clean up JSON response by removing text before first { and after last }."""
        if not text:
            return text

        # Find the first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        # If we found both braces, extract just the JSON part
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            return text[first_brace : last_brace + 1]

        # Otherwise return the original text
        return text

    def _structured_from_multipart(
        self, message: PromptMessageMultipart, model: Type[ModelT]
    ) -> Tuple[ModelT | None, PromptMessageMultipart]:
        """Override to apply JSON cleaning before parsing."""
        # Get the text from the multipart message
        text = message.all_text()

        # Clean the JSON response to remove extra text
        cleaned_text = self._clean_json_response(text)

        # If we cleaned the text, create a new multipart with the cleaned text
        if cleaned_text != text:
            from mcp.types import TextContent

            cleaned_multipart = PromptMessageMultipart(
                role=message.role, content=[TextContent(type="text", text=cleaned_text)]
            )
        else:
            cleaned_multipart = message

        # Use the parent class method with the cleaned multipart
        return super()._structured_from_multipart(cleaned_multipart, model)

    @classmethod
    def convert_message_to_message_param(
        cls, message: BedrockMessage, **kwargs
    ) -> BedrockMessageParam:
        """Convert a Bedrock message to message parameter format."""
        message_param = {"role": message.get("role", "assistant"), "content": []}

        for content_item in message.get("content", []):
            if isinstance(content_item, dict):
                if "text" in content_item:
                    message_param["content"].append({"type": "text", "text": content_item["text"]})
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

                    message_param["content"].append(
                        {
                            "type": "tool_use",
                            "id": tool_use.get("toolUseId", ""),
                            "name": tool_use.get("name", ""),
                            "input": tool_input,
                        }
                    )

        return message_param

    def _api_key(self) -> str:
        """Bedrock doesn't use API keys, returns empty string."""
        return ""
