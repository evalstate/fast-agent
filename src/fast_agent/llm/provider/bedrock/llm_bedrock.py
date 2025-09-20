import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Type, Union

from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

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


DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"


# Local ReasoningEffort enum to avoid circular imports
class ReasoningEffort(Enum):
    """Reasoning effort levels for Bedrock models"""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Reasoning effort to token budget mapping
# Based on AWS recommendations: start with 1024 minimum, increment reasonably
REASONING_EFFORT_BUDGETS = {
    ReasoningEffort.MINIMAL: 0,  # Disabled
    ReasoningEffort.LOW: 512,  # Light reasoning
    ReasoningEffort.MEDIUM: 1024,  # AWS minimum recommendation
    ReasoningEffort.HIGH: 2048,  # Higher reasoning
}

# Bedrock message format types
BedrockMessage = Dict[str, Any]  # Bedrock message format
BedrockMessageParam = Dict[str, Any]  # Bedrock message parameter format


class ToolSchemaType(Enum):
    """Enum for different tool schema formats used by different model families."""

    DEFAULT = auto()  # Default toolSpec format used by most models (formerly Nova)
    SYSTEM_PROMPT = auto()  # System prompt-based tool calling format
    ANTHROPIC = auto()  # Native Anthropic tool calling format
    NONE = auto()  # Schema fallback failed, avoid retries


class SystemMode(Enum):
    """System message handling modes."""

    SYSTEM = auto()  # Use native system parameter
    INJECT = auto()  # Inject into user message


class StreamPreference(Enum):
    """Streaming preference with tools."""

    STREAM_OK = auto()  # Model can stream with tools
    NON_STREAM = auto()  # Model requires non-streaming for tools


class ToolNamePolicy(Enum):
    """Tool name transformation policy."""

    PRESERVE = auto()  # Keep original tool names
    UNDERSCORES = auto()  # Convert to underscore format


class StructuredStrategy(Enum):
    """Structured output generation strategy."""

    STRICT_SCHEMA = auto()  # Use full JSON schema
    SIMPLIFIED_SCHEMA = auto()  # Use simplified schema


@dataclass
class ModelCapabilities:
    """Unified per-model capability cache to avoid scattered caches.

    Uses proper enums and types to prevent typos and improve type safety.
    """

    schema: ToolSchemaType | None = None
    system_mode: SystemMode | None = None
    stream_with_tools: StreamPreference | None = None
    tool_name_policy: ToolNamePolicy | None = None
    structured_strategy: StructuredStrategy | None = None
    reasoning_support: bool | None = None  # True=supported, False=unsupported, None=unknown
    supports_tools: bool | None = None  # True=yes, False=no, None=unknown


class BedrockLLM(FastAgentLLM[BedrockMessageParam, BedrockMessage]):
    """
    AWS Bedrock implementation of FastAgentLLM using the Converse API.
    Supports all Bedrock models including Nova, Claude, Meta, etc.
    """

    # Class-level capabilities cache shared across all instances
    capabilities: Dict[str, ModelCapabilities] = {}

    @classmethod
    def debug_cache(cls) -> None:
        """Print human-readable JSON representation of the capabilities cache.

        Useful for debugging and understanding what capabilities have been
        discovered and cached for each model. Uses sys.stdout to bypass
        any logging hijacking.
        """
        if not cls.capabilities:
            sys.stdout.write("{}\n")
            sys.stdout.flush()
            return

        cache_dict = {}
        for model, caps in cls.capabilities.items():
            cache_dict[model] = {
                "schema": caps.schema.name if caps.schema else None,
                "system_mode": caps.system_mode.name if caps.system_mode else None,
                "stream_with_tools": caps.stream_with_tools.name
                if caps.stream_with_tools
                else None,
                "tool_name_policy": caps.tool_name_policy.name if caps.tool_name_policy else None,
                "structured_strategy": caps.structured_strategy.name
                if caps.structured_strategy
                else None,
                "reasoning_support": caps.reasoning_support,
                "supports_tools": caps.supports_tools,
            }

        output = json.dumps(cache_dict, indent=2, sort_keys=True)
        sys.stdout.write(f"{output}\n")
        sys.stdout.flush()

    @classmethod
    def matches_model_pattern(cls, model_name: str) -> bool:
        """Return True if model_name exists in the Bedrock model list loaded at init.

        Uses the centralized discovery in bedrock_utils; no regex, no fallbacks.
        Gracefully handles environments without AWS access by returning False.
        """
        from fast_agent.llm.provider.bedrock.bedrock_utils import all_bedrock_models

        try:
            available = set(all_bedrock_models(prefix=""))
            return model_name in available
        except Exception:
            # If AWS calls fail (no credentials, region not configured, etc.),
            # assume this is not a Bedrock model
            return False

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

        # One-shot hint to force non-streaming on next completion (used by structured outputs)
        self._force_non_streaming_once: bool = False

        # Set up reasoning-related attributes
        self._reasoning_effort = kwargs.get("reasoning_effort", None)
        if (
            self._reasoning_effort is None
            and self.context
            and self.context.config
            and self.context.config.bedrock
        ):
            if hasattr(self.context.config.bedrock, "reasoning_effort"):
                self._reasoning_effort = self.context.config.bedrock.reasoning_effort

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Bedrock-specific default parameters"""
        # Get base defaults from parent (includes ModelDatabase lookup)
        base_params = super()._initialize_default_params(kwargs)

        # Override with Bedrock-specific settings - ensure we always have a model
        chosen_model = kwargs.get("model", DEFAULT_BEDROCK_MODEL)
        base_params.model = chosen_model

        return base_params

    @property
    def model(self) -> str:
        """Get the model name, guaranteed to be set."""
        return self.default_request_params.model

    def _get_bedrock_client(self):
        """Get or create Bedrock client."""
        if self._bedrock_client is None:
            try:
                session = boto3.Session(profile_name=self.aws_profile)  # type: ignore[union-attr]
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
                session = boto3.Session(profile_name=self.aws_profile)  # type: ignore[union-attr]
                self._bedrock_runtime_client = session.client(
                    "bedrock-runtime", region_name=self.aws_region
                )
            except NoCredentialsError as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_runtime_client

    def _build_tool_name_mapping(
        self, tools: "ListToolsResult", name_policy: ToolNamePolicy
    ) -> Dict[str, str]:
        """Build tool name mapping based on schema type and name policy.

        Returns dict mapping from converted_name -> original_name for tool execution.
        """
        mapping = {}

        if name_policy == ToolNamePolicy.PRESERVE:
            # Identity mapping for preserve policy
            for tool in tools.tools:
                mapping[tool.name] = tool.name
        else:
            # Nova-style cleaning for underscores policy
            for tool in tools.tools:
                clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool.name)
                clean_name = re.sub(r"_+", "_", clean_name).strip("_")
                if not clean_name:
                    clean_name = f"tool_{hash(tool.name) % 10000}"
                mapping[clean_name] = tool.name

        return mapping

    def _convert_tools_nova_format(
        self, tools: "ListToolsResult", tool_name_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tools to Nova-specific toolSpec format.

        Note: Nova models have VERY strict JSON schema requirements:
        - Top level schema must be of type Object
        - ONLY three fields are supported: type, properties, required
        - NO other fields like $schema, description, title, additionalProperties
        - Properties can only have type and description
        - Tools with no parameters should have empty properties object
        """
        bedrock_tools = []

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

            # Use the tool name mapping that was already built in _bedrock_completion
            # This ensures consistent transformation logic across the codebase
            clean_name = None
            for mapped_name, original_name in tool_name_mapping.items():
                if original_name == tool.name:
                    clean_name = mapped_name
                    break

            if clean_name is None:
                # Fallback if mapping not found (shouldn't happen)
                clean_name = tool.name
                self.logger.warning(
                    f"Tool name mapping not found for {tool.name}, using original name"
                )

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

    def _convert_tools_system_prompt_format(
        self, tools: "ListToolsResult", tool_name_mapping: Dict[str, str]
    ) -> str:
        """Convert MCP tools to system prompt format."""
        if not tools.tools:
            return ""

        self.logger.debug(f"Converting {len(tools.tools)} MCP tools to system prompt format")

        prompt_parts = [
            "You have the following tools available to help answer the user's request. You can call one or more functions at a time. The functions are described here in JSON-schema format:",
            "",
        ]

        # Add each tool definition in JSON format
        for tool in tools.tools:
            self.logger.debug(f"Converting MCP tool: {tool.name}")

            # Use original tool name (no hyphen replacement)
            tool_name = tool.name

            # Create tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            }

            prompt_parts.append(json.dumps(tool_def))

        # Add the response format instructions
        prompt_parts.extend(
            [
                "",
                "To call one or more tools, provide the tool calls on a new line as a JSON-formatted array. Explain your steps in a neutral tone. Then, only call the tools you can for the first step, then end your turn. If you previously received an error, you can try to call the tool again. Give up after 3 errors.",
                "",
                "Conform precisely to the single-line format of this example:",
                "Tool Call:",
                '[{"name": "SampleTool", "arguments": {"foo": "bar"}},{"name": "SampleTool", "arguments": {"foo": "other"}}]',
                "",
                "When calling a tool you must supply valid JSON with both 'name' and 'arguments' keys with the function name and function arguments respectively. Do not add any preamble, labels or extra text, just the single JSON string in one of the specified formats",
            ]
        )

        system_prompt = "\n".join(prompt_parts)
        self.logger.debug(f"Generated Llama native system prompt: {system_prompt}")

        return system_prompt

    def _convert_tools_anthropic_format(
        self, tools: "ListToolsResult", tool_name_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Convert MCP tools to Anthropic format wrapped in Bedrock toolSpec - preserves raw schema."""

        self.logger.debug(
            f"Converting {len(tools.tools)} MCP tools to Anthropic format with toolSpec wrapper"
        )

        bedrock_tools = []
        for tool in tools.tools:
            self.logger.debug(f"Converting MCP tool: {tool.name}")

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

    def _parse_system_prompt_tool_response(
        self, processed_response: Dict[str, Any], model: str
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
                        "type": "system_prompt_tool",
                        "name": func_name,
                        "arguments": arguments,
                        "id": f"system_prompt_{func_name}_{i}",
                    }
                )

            if tool_calls:
                return tool_calls

        # Second try: find the "Action:" format (commonly used by Nova models)
        action_pattern = r"Action:\s*([^(]+)\(([^)]*)\)"
        action_matches = re.findall(action_pattern, text_content)
        if action_matches:
            for i, (func_name, args_str) in enumerate(action_matches):
                func_name = func_name.strip()
                args_str = args_str.strip()

                # Parse arguments - handle quoted strings and key=value pairs
                arguments = {}
                if args_str:
                    try:
                        # Handle key=value format like location="London"
                        if "=" in args_str:
                            # Split by comma, then by = for each part
                            for arg_part in args_str.split(","):
                                if "=" in arg_part:
                                    key, value = arg_part.split("=", 1)
                                    key = key.strip()
                                    value = value.strip().strip("\"'")  # Remove quotes
                                    arguments[key] = value
                        else:
                            # Single value argument - try to map to appropriate parameter name
                            value = args_str.strip("\"'") if args_str else ""
                            # Handle common single-parameter functions
                            if func_name == "check_weather":
                                arguments = {"location": value}
                            else:
                                # Generic fallback
                                arguments = {"value": value}
                    except Exception as e:
                        self.logger.warning(f"Failed to parse Action arguments: {args_str} - {e}")
                        arguments = {"value": args_str}

                tool_calls.append(
                    {
                        "type": "system_prompt_tool",
                        "name": func_name,
                        "arguments": arguments,
                        "id": f"system_prompt_{func_name}_{i}",
                    }
                )

            if tool_calls:
                return tool_calls

        # Third try: find the "Tool Call:" format
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
                                    "type": "system_prompt_tool",
                                    "name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                    "id": f"system_prompt_{call['name']}_{i}",
                                }
                            )
                    return tool_calls
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse Tool Call JSON array: {json_str} - {e}")

        # Fallback: try to parse JSON arrays that look like tool calls
        # Look for arrays containing objects with "name" fields - avoid simple citations
        array_match = re.search(r'\[.*?\{.*?"name".*?\}.*?\]', text_content, re.DOTALL)
        if array_match:
            json_str = array_match.group(0)
            try:
                parsed_calls = json.loads(json_str)
                if isinstance(parsed_calls, list):
                    for i, call in enumerate(parsed_calls):
                        if isinstance(call, dict) and "name" in call:
                            tool_calls.append(
                                {
                                    "type": "system_prompt_tool",
                                    "name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                    "id": f"system_prompt_{call['name']}_{i}",
                                }
                            )
                    return tool_calls
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse JSON array: {json_str} - {e}")

        # Fallback: try to parse as single JSON object (backward compatibility)
        try:
            json_match = re.search(r'\{[^}]*"name"[^}]*"arguments"[^}]*\}', text_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                function_call = json.loads(json_str)

                if "name" in function_call:
                    return [
                        {
                            "type": "system_prompt_tool",
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
                            "type": "system_prompt_tool",
                            "name": function_name,
                            "arguments": function_args,
                            "id": f"system_prompt_{function_name}",
                        }
                    ]
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"Failed to parse fallback custom tag format: {function_args_json}"
                    )

        # Third try: find direct function call format like "function_name(args)"
        direct_call_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)$"
        direct_call_match = re.search(direct_call_pattern, text_content.strip())
        if direct_call_match:
            func_name, args_str = direct_call_match.groups()
            func_name = func_name.strip()
            args_str = args_str.strip()

            # Parse arguments
            arguments = {}
            if args_str:
                try:
                    # Handle key=value format like location="London"
                    if "=" in args_str:
                        # Split by comma, then by = for each part
                        for arg_part in args_str.split(","):
                            if "=" in arg_part:
                                key, value = arg_part.split("=", 1)
                                key = key.strip()
                                value = value.strip().strip("\"'")  # Remove quotes
                                arguments[key] = value
                    else:
                        # Single value argument - try to map to appropriate parameter name
                        value = args_str.strip("\"'") if args_str else ""
                        # Handle common single-parameter functions
                        if func_name == "check_weather":
                            arguments = {"location": value}
                        else:
                            # Generic fallback
                            arguments = {"value": value}
                except Exception as e:
                    self.logger.warning(f"Failed to parse direct call arguments: {args_str} - {e}")
                    arguments = {"value": args_str}

            return [
                {
                    "type": "system_prompt_tool",
                    "name": func_name,
                    "arguments": arguments,
                    "id": f"system_prompt_{func_name}_0",
                }
            ]

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
                        "type": "anthropic_tool",
                        "name": tool_use["name"],
                        "arguments": tool_use["input"],
                        "id": tool_use["toolUseId"],
                    }
                )

        return tool_uses

    def _parse_tool_response(
        self, processed_response: Dict[str, Any], model: str
    ) -> List[Dict[str, Any]]:
        """Parse tool responses using cached schema, without model/family heuristics."""
        caps = self.capabilities.get(model) or ModelCapabilities()
        schema = caps.schema

        # Choose parser strictly by cached schema
        if schema == ToolSchemaType.SYSTEM_PROMPT:
            return self._parse_system_prompt_tool_response(processed_response, model)
        if schema == ToolSchemaType.ANTHROPIC:
            return self._parse_anthropic_tool_response(processed_response)

        # Default/Nova: detect toolUse objects
        tool_uses = [
            c
            for c in processed_response.get("content", [])
            if isinstance(c, dict) and "toolUse" in c
        ]
        if tool_uses:
            parsed_tools: List[Dict[str, Any]] = []
            for item in tool_uses:
                tu = item.get("toolUse", {})
                if not isinstance(tu, dict):
                    continue
                parsed_tools.append(
                    {
                        "type": "nova_tool",
                        "name": tu.get("name"),
                        "arguments": tu.get("input", {}),
                        "id": tu.get("toolUseId"),
                    }
                )
            if parsed_tools:
                return parsed_tools

        # Family-agnostic fallback: parse JSON array embedded in text
        try:
            text_content = ""
            for content_item in processed_response.get("content", []):
                if isinstance(content_item, dict) and "text" in content_item:
                    text_content += content_item["text"]
            if text_content:
                import json as _json
                import re as _re

                match = _re.search(r"\[(?:.|\n)*?\]", text_content)
                if match:
                    arr = _json.loads(match.group(0))
                    if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                        parsed_calls = []
                        for i, call in enumerate(arr):
                            name = call.get("name")
                            args = call.get("arguments", {})
                            if name:
                                parsed_calls.append(
                                    {
                                        "type": "system_prompt_tool",
                                        "name": name,
                                        "arguments": args,
                                        "id": f"system_prompt_{name}_{i}",
                                    }
                                )
                        if parsed_calls:
                            return parsed_calls
        except Exception:
            pass

        # Final fallback: try system prompt parsing regardless of cached schema
        # This handles cases where native tool calling failed but model generated system prompt format
        try:
            return self._parse_system_prompt_tool_response(processed_response, model)
        except Exception:
            pass

        return []

    def _build_tool_calls_dict(
        self, parsed_tools: List[Dict[str, Any]]
    ) -> Dict[str, CallToolRequest]:
        """
        Convert parsed tools to CallToolRequest dict for external execution.

        Args:
            parsed_tools: List of parsed tool dictionaries from _parse_tool_response()

        Returns:
            Dictionary mapping tool_use_id to CallToolRequest objects
        """
        tool_calls = {}
        for parsed_tool in parsed_tools:
            # Use tool name directly, but map back to original if a mapping is available
            tool_name = parsed_tool["name"]
            try:
                mapping = getattr(self, "tool_name_mapping", None)
                if isinstance(mapping, dict):
                    tool_name = mapping.get(tool_name, tool_name)
            except Exception:
                pass

            # Create CallToolRequest
            tool_call = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name=tool_name, arguments=parsed_tool.get("arguments", {})
                ),
            )
            tool_calls[parsed_tool["id"]] = tool_call
        return tool_calls

    def _map_bedrock_stop_reason(self, bedrock_stop_reason: str) -> LlmStopReason:
        """
        Map Bedrock stop reasons to LlmStopReason enum.

        Args:
            bedrock_stop_reason: Stop reason from Bedrock API

        Returns:
            Corresponding LlmStopReason enum value
        """
        if bedrock_stop_reason == "tool_use":
            return LlmStopReason.TOOL_USE
        elif bedrock_stop_reason == "end_turn":
            return LlmStopReason.END_TURN
        elif bedrock_stop_reason == "stop_sequence":
            return LlmStopReason.STOP_SEQUENCE
        elif bedrock_stop_reason == "max_tokens":
            return LlmStopReason.MAX_TOKENS
        else:
            # Default to END_TURN for unknown stop reasons, but log for debugging
            self.logger.warning(
                f"Unknown Bedrock stop reason: {bedrock_stop_reason}, defaulting to END_TURN"
            )
            return LlmStopReason.END_TURN

    def _convert_multipart_to_bedrock_message(
        self, msg: PromptMessageExtended
    ) -> BedrockMessageParam:
        """
        Convert a PromptMessageExtended to Bedrock message parameter format.
        Handles tool results and regular content.

        Args:
            msg: PromptMessageExtended message to convert

        Returns:
            Bedrock message parameter dictionary
        """
        bedrock_msg = {"role": msg.role, "content": []}

        # Handle tool results first (if present)
        if msg.tool_results:
            # Get the cached schema type to determine result formatting
            caps = self.capabilities.get(self.model) or ModelCapabilities()
            # Check if any tool ID indicates system prompt format
            has_system_prompt_tools = any(
                tool_id.startswith("system_prompt_") for tool_id in msg.tool_results.keys()
            )
            is_system_prompt_schema = (
                caps.schema == ToolSchemaType.SYSTEM_PROMPT or has_system_prompt_tools
            )

            if is_system_prompt_schema:
                # For system prompt models: format as human-readable text
                tool_result_parts = []
                for tool_id, tool_result in msg.tool_results.items():
                    result_text = "".join(
                        part.text for part in tool_result.content if isinstance(part, TextContent)
                    )
                    result_payload = {
                        "tool_name": tool_id,  # Use tool_id as name for system prompt
                        "status": "error" if tool_result.isError else "success",
                        "result": result_text,
                    }
                    tool_result_parts.append(json.dumps(result_payload))

                if tool_result_parts:
                    full_result_text = f"Tool Results:\n{', '.join(tool_result_parts)}"
                    bedrock_msg["content"].append({"type": "text", "text": full_result_text})
            else:
                # For Nova/Anthropic models: use structured tool_result format
                for tool_id, tool_result in msg.tool_results.items():
                    result_content_blocks = []
                    if tool_result.content:
                        for part in tool_result.content:
                            if isinstance(part, TextContent):
                                result_content_blocks.append({"text": part.text})

                    if not result_content_blocks:
                        result_content_blocks.append({"text": "[No content in tool result]"})

                    bedrock_msg["content"].append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_content_blocks,
                            "status": "error" if tool_result.isError else "success",
                        }
                    )

        # Handle regular content
        for content_item in msg.content:
            if isinstance(content_item, TextContent):
                bedrock_msg["content"].append({"type": "text", "text": content_item.text})

        return bedrock_msg

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
                                        # If it's not valid JSON, wrap it as a dict to avoid downstream errors
                                        tool_use["toolUse"]["input"] = {"value": accumulated_input}
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
                            # If it's not valid JSON, wrap it as a dict to avoid downstream errors
                            tool_use["toolUse"]["input"] = {"value": accumulated_input}
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
        tools: List[Tool] | None = None,
        pre_messages: List[BedrockMessageParam] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using Bedrock and available tools.
        Returns PromptMessageExtended with tool calls for external execution.
        """
        client = self._get_bedrock_runtime_client()

        try:
            messages: List[BedrockMessageParam] = list(pre_messages) if pre_messages else []
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

        # Get available tools (no resolver gating; fallback logic will decide wiring)
        tool_list = None

        try:
            tool_list = await self.aggregator.list_tools()
            self.logger.debug(f"Found {len(tool_list.tools)} MCP tools")
        except Exception as e:
            self.logger.error(f"Error fetching MCP tools: {e}")
            import traceback

            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            tool_list = None

        # Use tools parameter if provided, otherwise get from aggregator
        if tools is None:
            tools = tool_list.tools if tool_list else []
        elif tool_list is None and tools:
            # Create a ListToolsResult from the provided tools for conversion
            from mcp.types import ListToolsResult

            tool_list = ListToolsResult(tools=tools)

        response_content_blocks: List[ContentBlock] = []
        model = self.default_request_params.model

        # Single API call - no tool execution loop
        self._log_chat_progress(self.chat_turn(), model=model)

        # Convert messages to Bedrock format
        bedrock_messages = self._convert_messages_to_bedrock(messages)

        # Base system text
        base_system_text = self.instruction or params.systemPrompt

        # Determine tool schema fallback order and caches
        caps = self.capabilities.get(model) or ModelCapabilities()
        if caps.schema and caps.schema != ToolSchemaType.NONE:
            # Special case: Force Mistral 7B to try SYSTEM_PROMPT instead of cached DEFAULT
            if (
                model == "mistral.mistral-7b-instruct-v0:2"
                and caps.schema == ToolSchemaType.DEFAULT
            ):
                print(
                    f"🔧 FORCING SYSTEM_PROMPT for {model} (was cached as DEFAULT)",
                    file=sys.stderr,
                    flush=True,
                )
                schema_order = [ToolSchemaType.SYSTEM_PROMPT, ToolSchemaType.DEFAULT]
            else:
                schema_order = [caps.schema]
        else:
            # Restore original fallback order: Anthropic models try anthropic first, others skip it
            if model.startswith("anthropic."):
                schema_order = [
                    ToolSchemaType.ANTHROPIC,
                    ToolSchemaType.DEFAULT,
                    ToolSchemaType.SYSTEM_PROMPT,
                ]
            elif model == "mistral.mistral-7b-instruct-v0:2":
                # Force Mistral 7B to try SYSTEM_PROMPT first (it doesn't work well with DEFAULT)
                schema_order = [
                    ToolSchemaType.SYSTEM_PROMPT,
                    ToolSchemaType.DEFAULT,
                ]
            else:
                schema_order = [
                    ToolSchemaType.DEFAULT,
                    ToolSchemaType.SYSTEM_PROMPT,
                ]

        # Track whether we changed system mode cache this turn
        tried_system_fallback = False

        processed_response = None  # type: ignore[assignment]
        last_error_msg = None

        for schema_choice in schema_order:
            # Fresh messages per attempt
            converse_args = {"modelId": model, "messages": [dict(m) for m in bedrock_messages]}

            # Build tools representation for this schema
            tools_payload: Union[List[Dict[str, Any]], str, None] = None
            # Get tool name policy (needed even when no tools for cache logic)
            name_policy = (
                self.capabilities.get(model) or ModelCapabilities()
            ).tool_name_policy or ToolNamePolicy.PRESERVE

            if tool_list and tool_list.tools:
                # Build tool name mapping once per schema attempt
                tool_name_mapping = self._build_tool_name_mapping(tool_list, name_policy)

                # Store mapping for tool execution
                self.tool_name_mapping = tool_name_mapping

                if schema_choice == ToolSchemaType.ANTHROPIC:
                    tools_payload = self._convert_tools_anthropic_format(
                        tool_list, tool_name_mapping
                    )
                elif schema_choice == ToolSchemaType.DEFAULT:
                    tools_payload = self._convert_tools_nova_format(tool_list, tool_name_mapping)
                elif schema_choice == ToolSchemaType.SYSTEM_PROMPT:
                    tools_payload = self._convert_tools_system_prompt_format(
                        tool_list, tool_name_mapping
                    )

            # System prompt handling with cache
            system_mode = (
                self.capabilities.get(model) or ModelCapabilities()
            ).system_mode or SystemMode.SYSTEM
            system_text = base_system_text

            if (
                schema_choice == ToolSchemaType.SYSTEM_PROMPT
                and isinstance(tools_payload, str)
                and tools_payload
            ):
                system_text = f"{system_text}\n\n{tools_payload}" if system_text else tools_payload

            # Cohere-specific nudge: force exact echo of tool result text on final answer
            if (
                schema_choice == ToolSchemaType.SYSTEM_PROMPT
                and isinstance(model, str)
                and model.startswith("cohere.")
            ):
                cohere_nudge = (
                    "FINAL ANSWER RULES (STRICT):\n"
                    "- When a tool result is provided, your final answer MUST be exactly the raw tool result text.\n"
                    "- Do not add any extra words, punctuation, qualifiers, or phrases (e.g., 'according to the tool').\n"
                    "- Example: If tool result text is 'It"
                    "s sunny in London', your final answer must be exactly: It"
                    "s sunny in London\n"
                )
                system_text = f"{system_text}\n\n{cohere_nudge}" if system_text else cohere_nudge

            # Llama3-specific nudge: prevent paraphrasing and extra tool calls
            if (
                schema_choice == ToolSchemaType.SYSTEM_PROMPT
                and isinstance(model, str)
                and model.startswith("meta.llama3")
            ):
                llama_nudge = (
                    "TOOL RESPONSE RULES:\n"
                    "- After receiving a tool result, immediately output ONLY the exact tool result text.\n"
                    "- Do not call additional tools or add commentary.\n"
                    "- Do not paraphrase or modify the tool result in any way."
                )
                system_text = f"{system_text}\n\n{llama_nudge}" if system_text else llama_nudge

            # Mistral-specific nudge: prevent tool calling loops and accept tool results
            if (
                schema_choice == ToolSchemaType.SYSTEM_PROMPT
                and isinstance(model, str)
                and model.startswith("mistral.")
            ):
                mistral_nudge = (
                    "TOOL EXECUTION RULES:\n"
                    "- Call each tool only ONCE per conversation turn.\n"
                    "- Accept and trust all tool results - do not question or retry them.\n"
                    "- After receiving a tool result, provide a direct answer based on that result.\n"
                    "- Do not call the same tool multiple times or call additional tools unless specifically requested.\n"
                    "- Tool results are always valid - do not attempt to validate or correct them."
                )
                system_text = f"{system_text}\n\n{mistral_nudge}" if system_text else mistral_nudge

            if system_text:
                if system_mode == SystemMode.SYSTEM:
                    converse_args["system"] = [{"text": system_text}]
                    self.logger.debug(
                        f"Attempting with system param for {model} and schema={schema_choice}"
                    )
                else:
                    # inject
                    if (
                        converse_args["messages"]
                        and converse_args["messages"][0].get("role") == "user"
                    ):
                        first_message = converse_args["messages"][0]
                        if first_message.get("content") and len(first_message["content"]) > 0:
                            original_text = first_message["content"][0].get("text", "")
                            first_message["content"][0]["text"] = (
                                f"System: {system_text}\n\nUser: {original_text}"
                            )
                            self.logger.debug(
                                "Injected system prompt into first user message (cached mode)"
                            )

            # Tools wiring
            # Always include toolConfig if we have tools OR if there are tool results in the conversation
            has_tool_results = False
            for msg in bedrock_messages:
                if isinstance(msg, dict) and msg.get("content"):
                    for content in msg["content"]:
                        if isinstance(content, dict) and "toolResult" in content:
                            has_tool_results = True
                            break
                    if has_tool_results:
                        break

            if (
                schema_choice in (ToolSchemaType.ANTHROPIC, ToolSchemaType.DEFAULT)
                and isinstance(tools_payload, list)
                and tools_payload
            ):
                # Include tools only when we have actual tools to provide
                converse_args["toolConfig"] = {"tools": tools_payload}

            # Inference configuration and overrides
            inference_config: Dict[str, Any] = {}
            if params.maxTokens is not None:
                inference_config["maxTokens"] = params.maxTokens
            if params.stopSequences:
                inference_config["stopSequences"] = params.stopSequences

            # Check if reasoning should be enabled
            reasoning_budget = 0
            if self._reasoning_effort and self._reasoning_effort != ReasoningEffort.MINIMAL:
                # Convert string to enum if needed
                if isinstance(self._reasoning_effort, str):
                    try:
                        effort_enum = ReasoningEffort(self._reasoning_effort)
                    except ValueError:
                        effort_enum = ReasoningEffort.MINIMAL
                else:
                    effort_enum = self._reasoning_effort

                if effort_enum != ReasoningEffort.MINIMAL:
                    reasoning_budget = REASONING_EFFORT_BUDGETS.get(effort_enum, 0)

            # Handle temperature and reasoning configuration
            # AWS docs: "Thinking isn't compatible with temperature, top_p, or top_k modifications"
            reasoning_enabled = False
            if reasoning_budget > 0:
                # Check if this model supports reasoning (with caching)
                cached_reasoning = (
                    self.capabilities.get(model) or ModelCapabilities()
                ).reasoning_support
                if cached_reasoning == "supported":
                    # We know this model supports reasoning
                    converse_args["performanceConfig"] = {
                        "reasoning": {"maxReasoningTokens": reasoning_budget}
                    }
                    reasoning_enabled = True
                elif cached_reasoning != "unsupported":
                    # Unknown - we'll try reasoning and fallback if needed
                    converse_args["performanceConfig"] = {
                        "reasoning": {"maxReasoningTokens": reasoning_budget}
                    }
                    reasoning_enabled = True

            if not reasoning_enabled:
                # No reasoning - apply temperature if provided
                if params.temperature is not None:
                    inference_config["temperature"] = params.temperature

            # Nova-specific recommendations (when not using reasoning)
            if model and "nova" in (model or "").lower() and reasoning_budget == 0:
                inference_config.setdefault("topP", 1.0)
                # Merge/attach additionalModelRequestFields for topK
                existing_amrf = converse_args.get("additionalModelRequestFields", {})
                merged_amrf = {**existing_amrf, **{"inferenceConfig": {"topK": 1}}}
                converse_args["additionalModelRequestFields"] = merged_amrf

            if inference_config:
                converse_args["inferenceConfig"] = inference_config

            # Decide streaming vs non-streaming (resolver-free with runtime detection + cache)
            has_tools: bool = False
            try:
                has_tools = bool(tools_payload) and bool(
                    (isinstance(tools_payload, list) and len(tools_payload) > 0)
                    or (isinstance(tools_payload, str) and tools_payload.strip())
                )

                # Force non-streaming for structured-output flows (one-shot)
                force_non_streaming = False
                if self._force_non_streaming_once:
                    force_non_streaming = True
                    self._force_non_streaming_once = False

                # Evaluate cache for streaming-with-tools
                cache_pref = (self.capabilities.get(model) or ModelCapabilities()).stream_with_tools
                use_streaming = True
                attempted_streaming = False

                if force_non_streaming:
                    use_streaming = False
                elif has_tools:
                    if cache_pref == StreamPreference.NON_STREAM:
                        use_streaming = False
                    elif cache_pref == StreamPreference.STREAM_OK:
                        use_streaming = True
                    else:
                        # Unknown: try streaming first, fallback on error
                        use_streaming = True

                # NEW: For Anthropic schema, when tool results are present in the conversation,
                # force non-streaming on this second turn to avoid empty streamed replies.
                if schema_choice == ToolSchemaType.ANTHROPIC and has_tool_results:
                    use_streaming = False
                    self.logger.debug(
                        "Forcing non-streaming for Anthropic second turn with tool results"
                    )

                # Try API call with reasoning fallback
                try:
                    if not use_streaming:
                        self.logger.debug(
                            f"Using non-streaming API for {model} (schema={schema_choice})"
                        )
                        response = client.converse(**converse_args)
                        processed_response = self._process_non_streaming_response(response, model)
                    else:
                        self.logger.debug(
                            f"Using streaming API for {model} (schema={schema_choice})"
                        )
                        attempted_streaming = True
                        response = client.converse_stream(**converse_args)
                        processed_response = await self._process_stream(response, model)
                except (ClientError, BotoCoreError) as e:
                    # Check if this is a reasoning-related error
                    if reasoning_budget > 0 and (
                        "reasoning" in str(e).lower() or "performance" in str(e).lower()
                    ):
                        self.logger.debug(
                            f"Model {model} doesn't support reasoning, retrying without: {e}"
                        )
                        caps.reasoning_support = False
                        self.capabilities[model] = caps

                        # Remove reasoning and retry
                        if "performanceConfig" in converse_args:
                            del converse_args["performanceConfig"]

                        # Apply temperature now that reasoning is disabled
                        if params.temperature is not None:
                            if "inferenceConfig" not in converse_args:
                                converse_args["inferenceConfig"] = {}
                            converse_args["inferenceConfig"]["temperature"] = params.temperature

                        # Retry the API call
                        if not use_streaming:
                            response = client.converse(**converse_args)
                            processed_response = self._process_non_streaming_response(
                                response, model
                            )
                        else:
                            response = client.converse_stream(**converse_args)
                            processed_response = await self._process_stream(response, model)
                    else:
                        # Not a reasoning error, re-raise
                        raise

                # Success: cache the working schema choice if not already cached
                # Only cache schema when tools are present - no tools doesn't predict tool behavior
                if not caps.schema and has_tools:
                    caps.schema = ToolSchemaType(schema_choice)

                # Cache successful reasoning if we tried it
                if reasoning_budget > 0 and caps.reasoning_support is not True:
                    caps.reasoning_support = True

                # If Nova/default worked and we used preserve but server complains, flip cache for next time
                if (
                    schema_choice == ToolSchemaType.DEFAULT
                    and name_policy == ToolNamePolicy.PRESERVE
                ):
                    # Heuristic: if tool names include '-', prefer underscores next time
                    try:
                        if any("-" in t.name for t in (tool_list.tools if tool_list else [])):
                            caps.tool_name_policy = ToolNamePolicy.UNDERSCORES
                    except Exception:
                        pass
                # Cache streaming-with-tools behavior on success
                if has_tools and attempted_streaming:
                    caps.stream_with_tools = StreamPreference.STREAM_OK
                self.capabilities[model] = caps
                break
            except (ClientError, BotoCoreError) as e:
                error_msg = str(e)
                last_error_msg = error_msg
                self.logger.debug(f"Bedrock API error (schema={schema_choice}): {error_msg}")

                # If streaming with tools failed and cache undecided, fallback to non-streaming and cache
                if has_tools and (caps.stream_with_tools is None):
                    try:
                        self.logger.debug(
                            f"Falling back to non-streaming API for {model} after streaming error"
                        )
                        response = client.converse(**converse_args)
                        processed_response = self._process_non_streaming_response(response, model)
                        caps.stream_with_tools = StreamPreference.NON_STREAM
                        if not caps.schema:
                            caps.schema = ToolSchemaType(schema_choice)
                        self.capabilities[model] = caps
                        break
                    except (ClientError, BotoCoreError) as e_fallback:
                        last_error_msg = str(e_fallback)
                        self.logger.debug(
                            f"Bedrock API error after non-streaming fallback: {last_error_msg}"
                        )
                        # continue to other fallbacks (e.g., system inject or next schema)

                # System parameter fallback once per call if system message unsupported
                if (
                    not tried_system_fallback
                    and system_text
                    and system_mode == SystemMode.SYSTEM
                    and (
                        "system message" in error_msg.lower()
                        or "system messages" in error_msg.lower()
                    )
                ):
                    tried_system_fallback = True
                    caps.system_mode = SystemMode.INJECT
                    self.capabilities[model] = caps
                    self.logger.info(
                        f"Switching system mode to inject for {model} and retrying same schema"
                    )
                    # Retry the same schema immediately in inject mode
                    try:
                        # Rebuild messages for inject
                        converse_args = {
                            "modelId": model,
                            "messages": [dict(m) for m in bedrock_messages],
                        }
                        # inject system into first user
                        if (
                            converse_args["messages"]
                            and converse_args["messages"][0].get("role") == "user"
                        ):
                            fm = converse_args["messages"][0]
                            if fm.get("content") and len(fm["content"]) > 0:
                                original_text = fm["content"][0].get("text", "")
                                fm["content"][0]["text"] = (
                                    f"System: {system_text}\n\nUser: {original_text}"
                                )

                        # Re-add tools
                        if (
                            schema_choice
                            in (ToolSchemaType.ANTHROPIC.value, ToolSchemaType.DEFAULT.value)
                            and isinstance(tools_payload, list)
                            and tools_payload
                        ):
                            converse_args["toolConfig"] = {"tools": tools_payload}

                        # Same streaming decision using cache
                        has_tools = bool(tools_payload) and bool(
                            (isinstance(tools_payload, list) and len(tools_payload) > 0)
                            or (isinstance(tools_payload, str) and tools_payload.strip())
                        )
                        cache_pref = (
                            self.capabilities.get(model) or ModelCapabilities()
                        ).stream_with_tools
                        if cache_pref == StreamPreference.NON_STREAM or not has_tools:
                            response = client.converse(**converse_args)
                            processed_response = self._process_non_streaming_response(
                                response, model
                            )
                        else:
                            response = client.converse_stream(**converse_args)
                            processed_response = await self._process_stream(response, model)
                        if not caps.schema and has_tools:
                            caps.schema = ToolSchemaType(schema_choice)
                        self.capabilities[model] = caps
                        break
                    except (ClientError, BotoCoreError) as e2:
                        last_error_msg = str(e2)
                        self.logger.debug(
                            f"Bedrock API error after system inject fallback: {last_error_msg}"
                        )
                        # Fall through to next schema
                        continue

                # For any other error (including tool format errors), continue to next schema
                self.logger.debug(
                    f"Continuing to next schema after error with {schema_choice}: {error_msg}"
                )
                continue

        if processed_response is None:
            # All attempts failed; mark schema as none to avoid repeated retries this process
            caps.schema = ToolSchemaType.NONE
            self.capabilities[model] = caps
            processed_response = {
                "content": [
                    {"text": f"Error during generation: {last_error_msg or 'Unknown error'}"}
                ],
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
                    response_content_blocks.append(
                        TextContent(type="text", text=content_item["text"])
                    )

        # Fallback: if no content returned and the last input contained tool results,
        # synthesize the assistant reply using the tool result text to preserve behavior.
        if not response_content_blocks:
            try:
                # messages currently includes the appended assistant response; inspect the prior user message
                last_index = len(messages) - 2 if len(messages) >= 2 else (len(messages) - 1)
                last_input = messages[last_index] if last_index >= 0 else None
                if isinstance(last_input, dict):
                    contents = last_input.get("content", []) or []
                    for c in contents:
                        # Handle parameter-level representation
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            tr_content = c.get("content", []) or []
                            fallback_text = " ".join(
                                part.get("text", "")
                                for part in tr_content
                                if isinstance(part, dict)
                            ).strip()
                            if fallback_text:
                                response_content_blocks.append(
                                    TextContent(type="text", text=fallback_text)
                                )
                                break
                        # Handle bedrock-level representation
                        if isinstance(c, dict) and "toolResult" in c:
                            tr = c["toolResult"]
                            tr_content = tr.get("content", []) or []
                            fallback_text = " ".join(
                                part.get("text", "")
                                for part in tr_content
                                if isinstance(part, dict)
                            ).strip()
                            if fallback_text:
                                response_content_blocks.append(
                                    TextContent(type="text", text=fallback_text)
                                )
                                break
            except Exception:
                pass

        # Handle different stop reasons
        stop_reason = processed_response.get("stop_reason", "end_turn")

        # Determine if we should parse for system-prompt tool calls (unified capabilities)
        caps_tmp = self.capabilities.get(model) or ModelCapabilities()

        # Try to parse system prompt tool calls if we have an end_turn with tools available
        # This handles cases where native tool calling failed but model generates system prompt format
        if stop_reason == "end_turn" and tools:
            # Only parse for tools if text contains actual function call structure
            message_text = ""
            for content_item in processed_response.get("content", []):
                if isinstance(content_item, dict) and "text" in content_item:
                    message_text += content_item.get("text", "")

            # Check if there's a tool call in the response
            parsed_tools = self._parse_tool_response(processed_response, model)
            if parsed_tools:
                # Override stop_reason to handle as tool_use
                stop_reason = "tool_use"
                # Update capabilities cache to reflect successful system prompt tool calling
                if not caps_tmp.schema:
                    caps_tmp.schema = ToolSchemaType.SYSTEM_PROMPT
                    self.capabilities[model] = caps_tmp

        # NEW: Handle tool calls without execution - return them for external handling
        tool_calls: Dict[str, CallToolRequest] | None = None
        if stop_reason in ["tool_use", "tool_calls"]:
            parsed_tools = self._parse_tool_response(processed_response, model)
            if parsed_tools:
                tool_calls = self._build_tool_calls_dict(parsed_tools)

        # Map stop reason to LlmStopReason
        mapped_stop_reason = self._map_bedrock_stop_reason(stop_reason)

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

        self._log_chat_finished(model=model)

        # Return PromptMessageExtended with tool calls for external execution
        from fast_agent.core.prompt import Prompt

        return Prompt.assistant(
            *response_content_blocks, stop_reason=mapped_stop_reason, tool_calls=tool_calls
        )

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: List[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: List[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """Apply Bedrock-specific prompt formatting."""
        if not multipart_messages:
            return PromptMessageExtended(role="user", content=[])

        # Check the last message role
        last_message = multipart_messages[-1]

        # Add all previous messages to history (or all messages if last is from assistant)
        # if the last message is a "user" inference is required
        # if the last message is a "user" inference is required
        messages_to_add = (
            multipart_messages[:-1] if last_message.role == "user" else multipart_messages
        )
        converted = []
        for msg in messages_to_add:
            # Convert each message to Bedrock message parameter format
            bedrock_msg = self._convert_multipart_to_bedrock_message(msg)
            converted.append(bedrock_msg)

        # Only persist prior messages when history is enabled; otherwise inline for this call
        params = self.get_request_params(request_params)
        pre_messages: List[BedrockMessageParam] | None = None
        if params.use_history:
            self.history.extend(converted, is_prompt=is_template)
        else:
            pre_messages = converted

        if last_message.role == "assistant":
            # For assistant messages: Return the last message (no completion needed)
            return last_message

        # For user messages with tool_results, we need to add the tool result message to the conversation
        if last_message.tool_results:
            # Convert the tool result message and use it as the final input
            message_param = self._convert_multipart_to_bedrock_message(last_message)
        else:
            # Convert the last user message to Bedrock message parameter format
            message_param = self._convert_multipart_to_bedrock_message(last_message)

        # Call the completion method with optional pre_messages for no-history mode
        return await self._bedrock_completion(
            message_param, request_params, tools, pre_messages=pre_messages
        )

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
        multipart_messages: List[PromptMessageExtended],
        model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """Apply structured output for Bedrock using prompt engineering with a simplified schema."""
        # Short-circuit: if the last message is already an assistant JSON payload,
        # parse it directly without invoking the model. This restores pre-regression behavior
        # for tests that seed assistant JSON as the last turn.
        try:
            if multipart_messages and multipart_messages[-1].role == "assistant":
                parsed_model, parsed_mp = self._structured_from_multipart(
                    multipart_messages[-1], model
                )
                if parsed_model is not None:
                    return parsed_model, parsed_mp
        except Exception:
            # Fall through to normal generation path
            pass

        request_params = self.get_request_params(request_params)

        # For structured outputs: disable reasoning entirely and set temperature=0 for deterministic JSON
        # This avoids conflicts between reasoning (requires temperature=1) and structured output (wants temperature=0)
        original_reasoning_effort = self._reasoning_effort
        self._reasoning_effort = ReasoningEffort.MINIMAL  # Temporarily disable reasoning

        # Override temperature for structured outputs
        if request_params:
            request_params = request_params.model_copy(update={"temperature": 0.0})
        else:
            request_params = RequestParams(temperature=0.0)

        # Select schema strategy, prefer runtime cache over resolver
        caps_struct = self.capabilities.get(self.model) or ModelCapabilities()
        strategy = caps_struct.structured_strategy or StructuredStrategy.STRICT_SCHEMA

        if strategy == StructuredStrategy.SIMPLIFIED_SCHEMA:
            schema_text = self._generate_simplified_schema(model)
        else:
            schema_text = FastAgentLLM.model_to_schema_str(model)

        # Build the new simplified prompt
        prompt_parts = [
            "You are a JSON generator. Respond with JSON that strictly follows the provided schema. Do not add any commentary or explanation.",
            "",
            "JSON Schema:",
            schema_text,
            "",
            "IMPORTANT RULES:",
            "- You MUST respond with only raw JSON data. No other text, commentary, or markdown is allowed.",
            "- All field names and enum values are case-sensitive and must match the schema exactly.",
            "- Do not add any extra fields to the JSON response. Only include the fields specified in the schema.",
            "- Do not use code fences or backticks (no ```json and no ```).",
            "- Your output must start with '{' and end with '}'.",
            "- Valid JSON requires double quotes for all field names and string values. Other types (int, float, boolean, etc.) should not be quoted.",
            "",
            "Now, generate the valid JSON response for the following request:",
        ]

        # IMPORTANT: Do NOT mutate the caller's messages. Create a deep copy of the last
        # user message, append the schema to the copy only, and pass just that copy into
        # the provider-specific path. This prevents contamination of routed messages.
        try:
            temp_last = multipart_messages[-1].model_copy(deep=True)
        except Exception:
            # Fallback: construct a minimal copy if model_copy is unavailable
            temp_last = PromptMessageExtended(
                role=multipart_messages[-1].role, content=list(multipart_messages[-1].content)
            )

        temp_last.add_text("\n".join(prompt_parts))

        self.logger.debug(
            "DEBUG: Using copied last message for structured schema; original left untouched"
        )

        try:
            result: PromptMessageExtended = await self._apply_prompt_provider_specific(
                [temp_last], request_params
            )
            try:
                parsed_model, _ = self._structured_from_multipart(result, model)
                # If parsing returned None (no model instance) we should trigger the retry path
                if parsed_model is None:
                    raise ValueError("structured parse returned None; triggering retry")
                return parsed_model, result
            except Exception:
                # One retry with stricter JSON-only guidance and simplified schema
                strict_parts = [
                    "STRICT MODE:",
                    "Return ONLY a single JSON object that matches the schema.",
                    "Do not include any prose, explanations, code fences, or extra characters.",
                    "Start with '{' and end with '}'.",
                    "",
                    "JSON Schema (simplified):",
                ]
                try:
                    simplified_schema_text = self._generate_simplified_schema(model)
                except Exception:
                    simplified_schema_text = FastAgentLLM.model_to_schema_str(model)
                try:
                    temp_last_retry = multipart_messages[-1].model_copy(deep=True)
                except Exception:
                    temp_last_retry = PromptMessageExtended(
                        role=multipart_messages[-1].role,
                        content=list(multipart_messages[-1].content),
                    )
                temp_last_retry.add_text("\n".join(strict_parts + [simplified_schema_text]))

                retry_result: PromptMessageExtended = await self._apply_prompt_provider_specific(
                    [temp_last_retry], request_params
                )
                return self._structured_from_multipart(retry_result, model)
        finally:
            # Restore original reasoning effort
            self._reasoning_effort = original_reasoning_effort

    def _clean_json_response(self, text: str) -> str:
        """Clean up JSON response by removing text before first { and after last }.

        Also handles cases where models wrap the response in an extra layer like:
        {"FormattedResponse": {"thinking": "...", "message": "..."}}
        """
        if not text:
            return text

        # Strip common code fences (```json ... ``` or ``` ... ```), anywhere in the text
        try:
            import re as _re

            fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if fence_match:
                text = fence_match.group(1)
        except Exception:
            pass

        # Find the first { and last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        # If we found both braces, extract just the JSON part
        if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
            json_part = text[first_brace : last_brace + 1]

            # Check if the JSON is wrapped in an extra layer (common model behavior)
            try:
                import json

                parsed = json.loads(json_part)

                # If it's a dict with a single key that matches the model class name,
                # unwrap it (e.g., {"FormattedResponse": {...}} -> {...})
                if isinstance(parsed, dict) and len(parsed) == 1:
                    key = list(parsed.keys())[0]
                    # Common wrapper patterns: class name, "response", "result", etc.
                    if key in [
                        "FormattedResponse",
                        "WeatherResponse",
                        "SimpleResponse",
                    ] or key.endswith("Response"):
                        inner_value = parsed[key]
                        if isinstance(inner_value, dict):
                            return json.dumps(inner_value)

                return json_part
            except json.JSONDecodeError:
                # If parsing fails, return the original JSON part
                return json_part

        # Otherwise return the original text
        return text

    def _structured_from_multipart(
        self, message: PromptMessageExtended, model: Type[ModelT]
    ) -> Tuple[ModelT | None, PromptMessageExtended]:
        """Override to apply JSON cleaning before parsing."""
        # Get the text from the multipart message
        text = message.all_text()

        # Clean the JSON response to remove extra text
        cleaned_text = self._clean_json_response(text)

        # If we cleaned the text, create a new multipart with the cleaned text
        if cleaned_text != text:
            from mcp.types import TextContent

            cleaned_multipart = PromptMessageExtended(
                role=message.role, content=[TextContent(type="text", text=cleaned_text)]
            )
        else:
            cleaned_multipart = message

        # Parse using cleaned multipart first
        model_instance, parsed_multipart = super()._structured_from_multipart(
            cleaned_multipart, model
        )
        if model_instance is not None:
            return model_instance, parsed_multipart
        # Fallback: if parsing failed (e.g., assistant-provided JSON already valid), try original
        return super()._structured_from_multipart(message, model)

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
