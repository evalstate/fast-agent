import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from types import UnionType
from typing import TYPE_CHECKING, Any, ClassVar, Union, cast, get_args, get_origin

from mcp import Tool
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    TextContent,
)
from pydantic import BaseModel

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import ModelT
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider.bedrock.multipart_converter_bedrock import BedrockConverter
from fast_agent.llm.provider.reasoning_config import reasoning_setting_from_config
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import (
    ReasoningEffortInput,
    ReasoningEffortSetting,
    ReasoningEffortSpec,
    parse_reasoning_setting,
    validate_reasoning_setting,
)
from fast_agent.llm.usage_tracking import TurnUsage
from fast_agent.mcp.helpers.content_helpers import (
    canonicalize_tool_result_content_for_llm,
    tool_result_text_for_llm,
)
from fast_agent.types import PromptMessageExtended, RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.text import casefold_text, strip_casefold
from fast_agent.utils.type_narrowing import is_str_object_dict

# Mapping from Bedrock's snake_case stop reasons to MCP's camelCase.
BEDROCK_STOP_REASON_MAP: dict[str, LlmStopReason] = {
    "tool_use": LlmStopReason.TOOL_USE,
    "end_turn": LlmStopReason.END_TURN,
    "stop_sequence": LlmStopReason.STOP_SEQUENCE,
    "max_tokens": LlmStopReason.MAX_TOKENS,
}
BEDROCK_TO_MCP_STOP_REASON = {
    stop_reason: mapped.value
    for stop_reason, mapped in BEDROCK_STOP_REASON_MAP.items()
    if mapped is not LlmStopReason.TOOL_USE
}

if TYPE_CHECKING:
    from mcp import ListToolsResult

_boto3: Any | None
_BOTOCORE_ERRORS: tuple[type[Exception], ...]
_NO_CREDENTIALS_ERROR: type[Exception]
try:
    _boto3 = importlib.import_module("boto3")
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
    )

    _BOTOCORE_ERRORS = (ClientError, BotoCoreError)
    _NO_CREDENTIALS_ERROR = NoCredentialsError
except ImportError:
    _boto3 = None
    _BOTOCORE_ERRORS = (Exception,)
    _NO_CREDENTIALS_ERROR = Exception


DEFAULT_BEDROCK_MODEL = "amazon.nova-lite-v1:0"


def _require_boto3() -> Any:
    if _boto3 is None:
        raise ImportError("boto3 is required for Bedrock support. Install with: pip install boto3")
    return _boto3


# Reasoning effort to token budget mapping
# Based on AWS recommendations: start with 1024 minimum, increment reasonably
REASONING_EFFORT_BUDGETS = {
    "minimal": 0,  # Disabled
    "low": 512,  # Light reasoning
    "medium": 1024,  # AWS minimum recommendation
    "high": 2048,  # Higher reasoning
}

_SIMPLIFIED_SCHEMA_SCALARS = {
    str: "string",
    int: "integer",
    float: "float",
    bool: "boolean",
}


def _bedrock_union_members(field_type: Any) -> tuple[Any, ...] | None:
    origin = get_origin(field_type)
    if origin not in {Union, UnionType}:
        return None
    return tuple(arg for arg in get_args(field_type) if arg is not type(None))


def _bedrock_enum_representation(field_type: type[Enum]) -> str:
    enum_values = [f'"{entry.value}"' for entry in field_type]
    return f"string (must be one of: {', '.join(enum_values)})"


def _bedrock_simplified_schema_dict(model_class: type[BaseModel]) -> dict[str, Any]:
    return {
        field_name: _bedrock_field_type_representation(field_info.annotation)
        for field_name, field_info in model_class.model_fields.items()
    }


def _bedrock_field_type_representation(field_type: Any) -> Any:
    """Return the simplified Bedrock prompt-schema representation for a model field."""
    union_members = _bedrock_union_members(field_type)
    if union_members is not None:
        if len(union_members) == 1:
            representation = _bedrock_field_type_representation(union_members[0])
        else:
            representation = " or ".join(
                str(_bedrock_field_type_representation(member)) for member in union_members
            )
    elif (scalar_name := _SIMPLIFIED_SCHEMA_SCALARS.get(field_type)) is not None:
        representation = scalar_name
    elif isinstance(field_type, type) and issubclass(field_type, Enum):
        representation = _bedrock_enum_representation(field_type)
    elif get_origin(field_type) is list:
        args = get_args(field_type)
        representation = [_bedrock_field_type_representation(args[0]) if args else "any"]
    elif isinstance(field_type, type) and issubclass(field_type, BaseModel):
        representation = _bedrock_simplified_schema_dict(field_type)
    else:
        representation = "any"
    return representation


def _bedrock_structured_json_prompt(schema_text: str) -> str:
    return "\n".join(
        [
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
    )


def _bedrock_structured_retry_prompt(schema_label: str, schema_text: str) -> str:
    return "\n".join(
        [
            "STRICT MODE:",
            "Return ONLY a single JSON object that matches the schema.",
            "Do not include any prose, explanations, code fences, or extra characters.",
            "Start with '{' and end with '}'.",
            "",
            schema_label,
            schema_text,
        ]
    )


def _is_reasoning_performance_error(error: Exception) -> bool:
    detail = casefold_text(str(error))
    return "reasoning" in detail or "performance" in detail


def _mentions_system_message(error_message: str) -> bool:
    detail = casefold_text(error_message)
    return "system message" in detail or "system messages" in detail


BEDROCK_REASONING_SPEC = ReasoningEffortSpec(
    kind="budget",
    min_budget_tokens=0,
    max_budget_tokens=None,
    default=ReasoningEffortSetting(kind="budget", value=REASONING_EFFORT_BUDGETS["medium"]),
)

# Bedrock message format types
BedrockMessage = dict[str, Any]  # Bedrock message format
BedrockMessageParam = dict[str, Any]  # Bedrock message parameter format


@dataclass
class BedrockStreamState:
    estimated_tokens: int = 0
    response_content: list[str] = field(default_factory=list)
    tool_uses: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, Any] = field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0}
    )


def _extract_tool_call_payload(value: object) -> tuple[str, dict[str, object]] | None:
    """Return normalized tool call data from a parsed JSON-like object."""

    if not is_str_object_dict(value):
        return None

    name_value = value.get("name")
    if not isinstance(name_value, str):
        return None

    arguments_value = value.get("arguments", {})
    arguments = arguments_value if is_str_object_dict(arguments_value) else {}
    return name_value, arguments


def _bedrock_native_tool_uses(
    processed_response: dict[str, Any],
) -> list[dict[str, Any]]:
    parsed_tools: list[dict[str, Any]] = []
    for content_item in processed_response.get("content", []):
        if not isinstance(content_item, dict):
            continue
        tool_use = content_item.get("toolUse")
        if not isinstance(tool_use, dict):
            continue
        parsed_tools.append(
            {
                "type": "nova_tool",
                "name": tool_use.get("name"),
                "arguments": tool_use.get("input", {}),
                "id": tool_use.get("toolUseId"),
            }
        )
    return parsed_tools


def _bedrock_response_text(processed_response: dict[str, Any]) -> str:
    return "".join(
        content_item["text"]
        for content_item in processed_response.get("content", [])
        if isinstance(content_item, dict) and isinstance(content_item.get("text"), str)
    )


def _bedrock_json_array_tool_calls(text_content: str) -> list[dict[str, Any]]:
    match = re.search(r"\[(?:.|\n)*?\]", text_content)
    if match is None:
        return []

    try:
        parsed_array = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed_array, list):
        return []

    parsed_calls: list[dict[str, Any]] = []
    for index, call in enumerate(parsed_array):
        tool_call_payload = _extract_tool_call_payload(call)
        if tool_call_payload is None:
            continue
        name_value, arguments = tool_call_payload
        parsed_calls.append(
            {
                "type": "system_prompt_tool",
                "name": name_value,
                "arguments": arguments,
                "id": f"system_prompt_{name_value}_{index}",
            }
        )
    return parsed_calls


def _system_prompt_tool_call(
    name: str,
    arguments: dict[str, Any] | dict[str, object],
    tool_id: str,
) -> dict[str, Any]:
    return {
        "type": "system_prompt_tool",
        "name": name,
        "arguments": arguments,
        "id": tool_id,
    }


def _scout_tool_arguments(args_str: str) -> dict[str, Any]:
    if not args_str:
        return {}
    if args_str.startswith("{") and args_str.endswith("}"):
        try:
            parsed_args = json.loads(args_str)
        except json.JSONDecodeError:
            return {"value": args_str}
        return parsed_args if isinstance(parsed_args, dict) else {"value": args_str}
    return {"value": args_str}


def _scout_tool_calls(text_content: str) -> list[dict[str, Any]]:
    scout_pattern = r"\[([^(]+)\(([^)]*)\)\]"
    return [
        _system_prompt_tool_call(
            raw_func_name.strip(),
            _scout_tool_arguments(raw_args_str.strip()),
            f"system_prompt_{raw_func_name.strip()}_{index}",
        )
        for index, (raw_func_name, raw_args_str) in enumerate(
            re.findall(scout_pattern, text_content)
        )
    ]


def _json_tool_calls_from_pattern(
    text_content: str,
    pattern: str,
) -> list[dict[str, Any]] | None:
    match = re.search(pattern, text_content, re.DOTALL)
    if match is None:
        return None

    parsed_calls = json.loads(match.group(1) if match.lastindex else match.group(0))
    if not isinstance(parsed_calls, list):
        return None

    return [
        _system_prompt_tool_call(
            name_value,
            arguments,
            f"system_prompt_{name_value}_{index}",
        )
        for index, call in enumerate(parsed_calls)
        for name_value, arguments in [_extract_tool_call_payload(call) or (None, None)]
        if name_value is not None and arguments is not None
    ]


def _single_json_tool_call(text_content: str) -> list[dict[str, Any]] | None:
    pattern = r'\{[^}]*"name"[^}]*"arguments"[^}]*\}'
    match = re.search(pattern, text_content, re.DOTALL)
    if match is None:
        return None

    tool_call_payload = _extract_tool_call_payload(json.loads(match.group(0)))
    if tool_call_payload is None:
        return None

    name_value, arguments = tool_call_payload
    return [
        _system_prompt_tool_call(
            name_value,
            arguments,
            f"system_prompt_{name_value}",
        )
    ]


def _custom_tag_tool_call(text_content: str) -> list[dict[str, Any]] | None:
    match = re.search(r"<function=([^>]+)>(.*?)</function>", text_content)
    if match is None:
        return None

    function_name = match.group(1)
    function_args = json.loads(match.group(2))
    return [
        _system_prompt_tool_call(
            function_name,
            function_args if isinstance(function_args, dict) else {},
            f"system_prompt_{function_name}",
        )
    ]


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


@dataclass
class BedrockAttemptConfig:
    converse_args: dict[str, Any]
    tools_payload: Union[list[dict[str, Any]], str, None]
    tool_name_mapping: dict[str, str] | None
    name_policy: ToolNamePolicy
    system_text: str | None
    system_mode: SystemMode
    has_tool_results: bool
    has_tool_use: bool
    reasoning_budget: int


@dataclass
class BedrockFallbackResult:
    processed_response: dict[str, Any] | None = None
    last_error_msg: str | None = None
    handled: bool = False
    attempted: bool = False


class BedrockLLM(FastAgentLLM[BedrockMessageParam, BedrockMessage]):
    """
    AWS Bedrock implementation of FastAgentLLM using the Converse API.
    Supports all Bedrock models including Nova, Claude, Meta, etc.
    """

    # Class-level capabilities cache shared across all instances
    capabilities: ClassVar[dict[str, ModelCapabilities]] = {}

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
        _require_boto3()

        # Initialize logger
        self.logger = get_logger(__name__)

        # Extract AWS configuration from kwargs first
        self.aws_region = kwargs.pop("region", None)
        self.aws_profile = kwargs.pop("profile", None)
        kwargs.pop("provider", None)
        if args and isinstance(args[0], Provider):
            args = args[1:]

        super().__init__(Provider.BEDROCK, *args, **kwargs)

        self._resolve_aws_configuration()
        self._bedrock_client = None
        self._bedrock_runtime_client = None

        # One-shot hint to force non-streaming on next completion (used by structured outputs)
        self._force_non_streaming_once: bool = False

        self._configure_reasoning_from_kwargs(kwargs)

    def _resolve_aws_configuration(self) -> None:
        bedrock_config = self.context.config.bedrock if self.context.config else None
        if bedrock_config is not None:
            self.aws_region = self.aws_region or bedrock_config.region
            self.aws_profile = self.aws_profile or bedrock_config.profile

        self.aws_region = self.aws_region or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION",
            "us-east-1",
        )
        self.aws_profile = self.aws_profile or os.environ.get("AWS_PROFILE")

    def _configure_reasoning_from_kwargs(self, kwargs: dict) -> None:
        raw_setting = kwargs.get("reasoning_effort")
        if raw_setting is None and self.context.config and self.context.config.bedrock:
            raw_setting, warn_deprecated_reasoning_effort = reasoning_setting_from_config(
                self.context.config.bedrock
            )
            if warn_deprecated_reasoning_effort:
                self.logger.warning(
                    "Bedrock config 'reasoning_effort' is deprecated; use 'reasoning'."
                )

        self._apply_reasoning_setting(raw_setting)
        if self._reasoning_effort_spec is None:
            self._reasoning_effort_spec = BEDROCK_REASONING_SPEC

    def _apply_reasoning_setting(self, raw_setting: ReasoningEffortInput) -> None:
        setting = parse_reasoning_setting(raw_setting)
        if setting is not None:
            try:
                self.set_reasoning_effort(setting)
            except ValueError as exc:
                self.logger.warning(f"Invalid reasoning setting: {exc}")

    def _initialize_default_params(self, kwargs: dict) -> RequestParams:
        """Initialize Bedrock-specific default parameters"""
        return self._initialize_default_params_with_model_fallback(kwargs, DEFAULT_BEDROCK_MODEL)

    @property
    def model(self) -> str:
        """Get the model name, guaranteed to be set."""
        return self.default_request_params.model or DEFAULT_BEDROCK_MODEL

    def _resolve_reasoning_budget(self) -> int:
        setting = self.reasoning_effort
        if setting is None:
            return 0
        if setting.kind == "toggle":
            return 0 if not setting.value else REASONING_EFFORT_BUDGETS["medium"]
        if setting.kind == "effort":
            return REASONING_EFFORT_BUDGETS.get(str(setting.value), 0)
        if setting.kind == "budget":
            return max(0, int(setting.value))
        return 0

    def set_reasoning_effort(self, setting: ReasoningEffortSetting | None) -> None:
        if setting is None:
            self._reasoning_effort = None
            return

        spec = self._reasoning_effort_spec or BEDROCK_REASONING_SPEC
        if setting.kind == "effort":
            budget = REASONING_EFFORT_BUDGETS.get(str(setting.value), 0)
            setting = ReasoningEffortSetting(kind="budget", value=budget)

        self._reasoning_effort = validate_reasoning_setting(setting, spec)

    def _get_bedrock_client(self):
        """Get or create Bedrock client."""
        if self._bedrock_client is None:
            try:
                boto3 = _require_boto3()
                session = boto3.Session(profile_name=self.aws_profile)
                self._bedrock_client = session.client("bedrock", region_name=self.aws_region)
            except _NO_CREDENTIALS_ERROR as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_client

    def _get_bedrock_runtime_client(self):
        """Get or create Bedrock Runtime client."""
        if self._bedrock_runtime_client is None:
            try:
                boto3 = _require_boto3()
                session = boto3.Session(profile_name=self.aws_profile)
                self._bedrock_runtime_client = session.client(
                    "bedrock-runtime", region_name=self.aws_region
                )
            except _NO_CREDENTIALS_ERROR as e:
                raise ProviderKeyError(
                    "AWS credentials not found",
                    "Please configure AWS credentials using AWS CLI, environment variables, or IAM roles.",
                ) from e
        return self._bedrock_runtime_client

    def _convert_extended_messages_to_provider(
        self, messages: list[PromptMessageExtended]
    ) -> list[BedrockMessageParam]:
        """
        Convert PromptMessageExtended list to Bedrock BedrockMessageParam format.
        This is called fresh on every API call from _convert_to_provider_format().

        Args:
            messages: List of PromptMessageExtended objects

        Returns:
            List of Bedrock BedrockMessageParam objects
        """
        converted: list[BedrockMessageParam] = []
        for msg in messages:
            bedrock_msg = BedrockConverter.convert_to_bedrock(msg)
            converted.append(bedrock_msg)
        return converted

    def _build_tool_name_mapping(
        self, tools: "ListToolsResult", name_policy: ToolNamePolicy
    ) -> dict[str, str]:
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

    @staticmethod
    def _resolve_tool_use_name(
        tool_use_id: str,
        tool_list: "ListToolsResult | None",
        tool_name_mapping: dict[str, str] | None,
    ) -> str:
        tool_name = "unknown_tool"
        if tool_list and tool_list.tools:
            # Try to match by checking if any tool name appears in the tool_use_id
            for tool in tool_list.tools:
                if tool.name in tool_use_id or tool_use_id.endswith(f"_{tool.name}"):
                    tool_name = tool.name
                    break
            # If no match, use first tool as fallback
            if tool_name == "unknown_tool":
                tool_name = tool_list.tools[0].name

        if tool_name_mapping:
            for mapped_name, original_name in tool_name_mapping.items():
                if original_name == tool_name:
                    return mapped_name

        return tool_name

    def _convert_tools_nova_format(
        self, tools: "ListToolsResult", tool_name_mapping: dict[str, str]
    ) -> list[dict[str, Any]]:
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
            nova_schema: dict[str, Any] = {"type": "object", "properties": {}}

            # Properties - clean them strictly
            properties: dict[str, Any] = {}
            if "properties" in input_schema and isinstance(input_schema["properties"], dict):
                for prop_name, prop_def in input_schema["properties"].items():
                    # Only include type and description for each property
                    clean_prop: dict[str, Any] = {}

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
        self, tools: "ListToolsResult", tool_name_mapping: dict[str, str]
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
        self, tools: "ListToolsResult", tool_name_mapping: dict[str, str]
    ) -> list[dict[str, Any]]:
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

    def _parse_tool_arguments(self, func_name: str, args_str: str) -> dict[str, Any]:
        """Parse tool call arguments from key=value or single-value format.

        Args:
            func_name: The function name (used for special case handling)
            args_str: The raw argument string to parse

        Returns:
            Dictionary of parsed arguments
        """
        arguments: dict[str, Any] = {}
        if not args_str:
            return arguments
        try:
            if "=" in args_str:
                # Split by comma, then by = for each part
                for arg_part in args_str.split(","):
                    if "=" in arg_part:
                        key, value = arg_part.split("=", 1)
                        arguments[key.strip()] = value.strip().strip("\"'")
            else:
                # Single value argument - try to map to appropriate parameter name
                value = args_str.strip("\"'")
                # Handle common single-parameter functions
                arguments = {"location": value} if func_name == "check_weather" else {"value": value}
        except Exception as e:
            self.logger.warning(f"Failed to parse tool arguments: {args_str} - {e}")
        return arguments

    def _action_tool_calls(self, text_content: str) -> list[dict[str, Any]]:
        action_pattern = r"Action:\s*([^(]+)\(([^)]*)\)"
        return [
            _system_prompt_tool_call(
                func_name,
                self._parse_tool_arguments(func_name, raw_args_str.strip()),
                f"system_prompt_{func_name}_{index}",
            )
            for index, (raw_func_name, raw_args_str) in enumerate(
                re.findall(action_pattern, text_content)
            )
            for func_name in [raw_func_name.strip()]
        ]

    def _logged_json_tool_calls_from_pattern(
        self,
        text_content: str,
        pattern: str,
        warning_prefix: str,
    ) -> list[dict[str, Any]] | None:
        try:
            return _json_tool_calls_from_pattern(text_content, pattern)
        except json.JSONDecodeError as exc:
            self.logger.warning(f"{warning_prefix}: {text_content} - {exc}")
            return None

    def _logged_single_json_tool_call(
        self,
        text_content: str,
    ) -> list[dict[str, Any]] | None:
        try:
            return _single_json_tool_call(text_content)
        except json.JSONDecodeError as exc:
            self.logger.warning(
                f"Failed to parse system prompt tool response as JSON: {text_content} - {exc}"
            )
            return None

    def _logged_custom_tag_tool_call(
        self,
        text_content: str,
    ) -> list[dict[str, Any]] | None:
        try:
            return _custom_tag_tool_call(text_content)
        except json.JSONDecodeError:
            self.logger.warning(f"Failed to parse fallback custom tag format: {text_content}")
            return None

    def _direct_tool_call(self, text_content: str) -> list[dict[str, Any]]:
        direct_call_pattern = r"^([a-zA-Z_][a-zA-Z0-9_]*)\(([^)]*)\)$"
        direct_call_match = re.search(direct_call_pattern, text_content.strip())
        if direct_call_match is None:
            return []

        func_name, args_str = direct_call_match.groups()
        func_name = func_name.strip()
        return [
            _system_prompt_tool_call(
                func_name,
                self._parse_tool_arguments(func_name, args_str.strip()),
                f"system_prompt_{func_name}_0",
            )
        ]

    def _parse_system_prompt_tool_response(
        self, processed_response: dict[str, Any], model: str
    ) -> list[dict[str, Any]]:
        """Parse system prompt tool response format: function calls in text."""
        text_content = _bedrock_response_text(processed_response)
        if not text_content:
            return []

        scout_calls = _scout_tool_calls(text_content)
        if scout_calls:
            return scout_calls

        # Second try: find the "Action:" format (commonly used by Nova models)
        action_calls = self._action_tool_calls(text_content)
        if action_calls:
            return action_calls

        parsed_calls: list[dict[str, Any]] | None = None
        json_patterns = (
            (r"Tool Call:\s*(\[.*?\])", "Failed to parse Tool Call JSON array"),
            (r'\[.*?\{.*?"name".*?\}.*?\]', "Failed to parse JSON array"),
        )
        for pattern, error_message in json_patterns:
            parsed_calls = self._logged_json_tool_calls_from_pattern(
                text_content,
                pattern,
                error_message,
            )
            if parsed_calls is not None:
                break

        if parsed_calls is None:
            parsed_calls = self._logged_single_json_tool_call(text_content)
        if parsed_calls is None:
            parsed_calls = self._logged_custom_tag_tool_call(text_content)
        if parsed_calls is None:
            parsed_calls = self._direct_tool_call(text_content)
        return parsed_calls

    def _parse_anthropic_tool_response(
        self, processed_response: dict[str, Any]
    ) -> list[dict[str, Any]]:
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
        self, processed_response: dict[str, Any], model: str
    ) -> list[dict[str, Any]]:
        """Parse tool responses using cached schema, without model/family heuristics."""
        caps = self.capabilities.get(model) or ModelCapabilities()
        schema = caps.schema

        # Choose parser strictly by cached schema
        if schema == ToolSchemaType.SYSTEM_PROMPT:
            return self._parse_system_prompt_tool_response(processed_response, model)
        if schema == ToolSchemaType.ANTHROPIC:
            return self._parse_anthropic_tool_response(processed_response)

        native_tool_uses = _bedrock_native_tool_uses(processed_response)
        if native_tool_uses:
            return native_tool_uses

        # Family-agnostic fallback: parse JSON array embedded in text
        parsed_calls = _bedrock_json_array_tool_calls(
            _bedrock_response_text(processed_response)
        )
        if parsed_calls:
            return parsed_calls

        # Final fallback: try system prompt parsing regardless of cached schema
        # This handles cases where native tool calling failed but model generated system prompt format
        try:
            return self._parse_system_prompt_tool_response(processed_response, model)
        except Exception:
            pass

        return []

    def _build_tool_calls_dict(
        self, parsed_tools: list[dict[str, Any]]
    ) -> dict[str, CallToolRequest]:
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
        mapped = BEDROCK_STOP_REASON_MAP.get(bedrock_stop_reason)
        if mapped is not None:
            return mapped
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
        content_blocks: list[dict[str, Any]] = []
        bedrock_msg = {"role": msg.role, "content": content_blocks}

        # Handle tool results first (if present)
        if msg.tool_results:
            # Get the cached schema type to determine result formatting
            caps = self.capabilities.get(self.model) or ModelCapabilities()
            # Check if any tool ID indicates system prompt format
            has_system_prompt_tools = any(
                tool_id.startswith("system_prompt_") for tool_id in msg.tool_results
            )
            is_system_prompt_schema = (
                caps.schema == ToolSchemaType.SYSTEM_PROMPT or has_system_prompt_tools
            )

            if is_system_prompt_schema:
                # For system prompt models: format as human-readable text
                tool_result_parts = []
                for tool_id, tool_result in msg.tool_results.items():
                    result_text = tool_result_text_for_llm(
                        tool_result,
                        logger=self.logger,
                        source="bedrock",
                    )
                    result_payload = {
                        "tool_name": tool_id,  # Use tool_id as name for system prompt
                        "status": "error" if tool_result.isError else "success",
                        "result": result_text,
                    }
                    tool_result_parts.append(json.dumps(result_payload))

                if tool_result_parts:
                    full_result_text = f"Tool Results:\n{', '.join(tool_result_parts)}"
                    content_blocks.append({"type": "text", "text": full_result_text})
            else:
                # For Nova/Anthropic models: use structured tool_result format
                for tool_id, tool_result in msg.tool_results.items():
                    result_content_blocks = [
                        {"text": part.text}
                        for part in canonicalize_tool_result_content_for_llm(
                            tool_result,
                            logger=self.logger,
                            source="bedrock",
                        )
                        if isinstance(part, TextContent)
                    ]

                    if not result_content_blocks:
                        result_content_blocks.append({"text": "[No content in tool result]"})

                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_content_blocks,
                            "status": "error" if tool_result.isError else "success",
                        }
                    )

        # Handle regular content
        content_blocks.extend(
            {"type": "text", "text": content_item.text}
            for content_item in msg.content
            if isinstance(content_item, TextContent)
        )

        return bedrock_msg

    def _convert_messages_to_bedrock(
        self, messages: list[BedrockMessageParam]
    ) -> list[dict[str, Any]]:
        """Convert message parameters to Bedrock format."""
        bedrock_messages = []
        for message in messages:
            bedrock_message = {
                "role": message.get("role", "user"),
                "content": self._bedrock_content_blocks(message.get("content", [])),
            }

            # Only add the message if it has content
            if bedrock_message["content"]:
                bedrock_messages.append(bedrock_message)

        return bedrock_messages

    def _bedrock_content_blocks(self, content: object) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"text": content}]
        if not isinstance(content, list):
            return []

        text_blocks: list[dict[str, Any]] = []
        tool_use_blocks: list[dict[str, Any]] = []
        tool_result_blocks: list[dict[str, Any]] = []
        other_blocks: list[dict[str, Any]] = []

        for item in content:
            if not isinstance(item, dict):
                continue
            typed_item = cast("dict[str, Any]", item)
            item_type = typed_item.get("type")
            if item_type == "text":
                text_blocks.append({"text": typed_item.get("text", "")})
            elif item_type == "tool_use":
                tool_use_blocks.append(self._bedrock_tool_use_block(typed_item))
            elif item_type == "tool_result":
                tool_result_blocks.append(self._bedrock_tool_result_block(typed_item))
            else:
                other_blocks.append(typed_item)

        return text_blocks + tool_use_blocks + tool_result_blocks + other_blocks

    @staticmethod
    def _bedrock_tool_use_block(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "toolUse": {
                "toolUseId": item.get("id", ""),
                "name": item.get("name", ""),
                "input": item.get("input", {}),
            }
        }

    @staticmethod
    def _bedrock_tool_result_block(item: dict[str, Any]) -> dict[str, Any]:
        status = item.get("status", "success")
        content = BedrockLLM._bedrock_tool_result_content(item.get("content", []), status)
        return {
            "toolResult": {
                "toolUseId": item.get("tool_use_id"),
                "content": content,
                "status": status,
            }
        }

    @staticmethod
    def _bedrock_tool_result_content(raw_content: object, status: object) -> list[dict[str, str]]:
        content = []
        if isinstance(raw_content, list):
            for part in raw_content:
                if not isinstance(part, dict) or "text" not in part:
                    continue
                typed_part = cast("dict[str, Any]", part)
                content.append({"text": typed_part.get("text", "")})
        if not content and status == "error":
            content.append({"text": "Tool call failed with an error."})
        return content

    def _start_stream_tool_use(
        self,
        content_block: dict[str, Any],
        state: BedrockStreamState,
    ) -> None:
        start = content_block.get("start")
        if not isinstance(start, dict) or "toolUse" not in start:
            return

        tool_use_start = start["toolUse"]
        if not isinstance(tool_use_start, dict):
            return

        self.logger.debug(f"Tool use block started: {tool_use_start}")
        state.tool_uses.append(
            {
                "toolUse": {
                    "toolUseId": tool_use_start.get("toolUseId"),
                    "name": tool_use_start.get("name"),
                    "input": tool_use_start.get("input", {}),
                    "_input_accumulator": "",
                }
            }
        )

    def _handle_stream_text_delta(
        self,
        text: str,
        model: str,
        state: BedrockStreamState,
    ) -> None:
        state.response_content.append(text)
        state.estimated_tokens = self._update_streaming_progress(
            text,
            model,
            state.estimated_tokens,
        )

    def _handle_stream_tool_delta(
        self,
        tool_use: object,
        state: BedrockStreamState,
    ) -> None:
        self.logger.debug(f"Tool use delta: {tool_use}")
        if not isinstance(tool_use, dict) or not state.tool_uses or "input" not in tool_use:
            return

        tool_use_delta = cast("dict[str, Any]", tool_use)
        input_data = tool_use_delta["input"]
        active_tool = state.tool_uses[-1]["toolUse"]
        if isinstance(input_data, dict):
            active_tool["input"].update(input_data)
        elif isinstance(input_data, str):
            active_tool["_input_accumulator"] += input_data
            self.logger.debug(f"Accumulated input: {active_tool['_input_accumulator']}")
        else:
            self.logger.debug(
                f"Tool use input is unexpected type: {type(input_data)}: {input_data}"
            )
            active_tool["input"] = input_data

    def _handle_stream_delta(
        self,
        event: dict[str, Any],
        model: str,
        state: BedrockStreamState,
    ) -> None:
        delta = event["contentBlockDelta"]["delta"]
        if "text" in delta:
            self._handle_stream_text_delta(delta["text"], model, state)
        elif "toolUse" in delta:
            self._handle_stream_tool_delta(delta["toolUse"], state)

    def _apply_accumulated_tool_input(
        self,
        tool_use: dict[str, Any],
        *,
        final: bool = False,
    ) -> None:
        if "_input_accumulator" not in tool_use:
            return

        accumulated_input = tool_use["_input_accumulator"]
        if accumulated_input:
            label = "final accumulated" if final else "accumulated"
            self.logger.debug(f"Processing {label} input: {accumulated_input}")
            try:
                parsed_input = json.loads(accumulated_input)
                if isinstance(parsed_input, dict):
                    tool_use["input"].update(parsed_input)
                else:
                    tool_use["input"] = parsed_input
                self.logger.debug(f"Successfully parsed {label} input: {parsed_input}")
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    f"Failed to parse {label} input as JSON: {accumulated_input} - {exc}"
                )
                tool_use["input"] = {"value": accumulated_input}

        del tool_use["_input_accumulator"]

    def _finalize_stream_tool_inputs(
        self,
        state: BedrockStreamState,
        *,
        final: bool = False,
    ) -> None:
        for tool_use in state.tool_uses:
            self._apply_accumulated_tool_input(tool_use["toolUse"], final=final)

    def _handle_stream_metadata(
        self,
        event: dict[str, Any],
        model: str,
        state: BedrockStreamState,
    ) -> None:
        metadata = event["metadata"]
        if "usage" not in metadata:
            return

        state.usage = metadata["usage"]
        actual_tokens = state.usage.get("outputTokens", 0)
        if actual_tokens <= 0:
            return

        token_str = str(actual_tokens).rjust(5)
        data = {
            "progress_action": ProgressAction.STREAMING,
            "model": model,
            "agent_name": self.name,
            "chat_turn": self.chat_turn(),
            "details": token_str.strip(),
        }
        self.logger.info("Streaming progress", data=data)

    def _bedrock_stream_response(
        self,
        model: str,
        state: BedrockStreamState,
    ) -> BedrockMessage:
        full_text = "".join(state.response_content)
        response_content_items: list[dict[str, Any]] = (
            [{"text": full_text}] if full_text else []
        )

        if state.tool_uses:
            self._finalize_stream_tool_inputs(state, final=True)
            response_content_items.extend(state.tool_uses)

        return {
            "content": response_content_items,
            "stop_reason": state.stop_reason or "end_turn",
            "usage": {
                "input_tokens": state.usage.get("inputTokens", 0),
                "output_tokens": state.usage.get("outputTokens", 0),
            },
            "model": model,
            "role": "assistant",
        }

    async def _process_stream(
        self,
        stream_response,
        model: str,
    ) -> BedrockMessage:
        """Process streaming response from Bedrock."""
        state = BedrockStreamState()

        try:
            # Cancellation is handled via asyncio.Task.cancel() which raises CancelledError
            for event in stream_response["stream"]:
                if "messageStart" in event:
                    continue
                if "contentBlockStart" in event:
                    self._start_stream_tool_use(event["contentBlockStart"], state)
                elif "contentBlockDelta" in event:
                    self._handle_stream_delta(event, model, state)
                elif "contentBlockStop" in event:
                    self._finalize_stream_tool_inputs(state)
                elif "messageStop" in event:
                    state.stop_reason = event["messageStop"].get("stopReason")
                elif "metadata" in event:
                    self._handle_stream_metadata(event, model, state)
        except Exception as e:
            self.logger.error(f"Error processing stream: {e}")
            raise

        return self._bedrock_stream_response(model, state)

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
        return {
            "content": content,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
            },
            "model": model,
            "role": "assistant",
        }

    @staticmethod
    def _bedrock_tool_list(tools: list[Tool] | None) -> "ListToolsResult | None":
        if not tools:
            return None

        from mcp.types import ListToolsResult

        return ListToolsResult(tools=tools)

    @staticmethod
    def _bedrock_schema_order(model: str, caps: ModelCapabilities) -> list[ToolSchemaType]:
        if caps.schema and caps.schema != ToolSchemaType.NONE:
            if (
                model == "mistral.mistral-7b-instruct-v0:2"
                and caps.schema == ToolSchemaType.DEFAULT
            ):
                print(
                    f"🔧 FORCING SYSTEM_PROMPT for {model} (was cached as DEFAULT)",
                    file=sys.stderr,
                    flush=True,
                )
                return [ToolSchemaType.SYSTEM_PROMPT, ToolSchemaType.DEFAULT]
            return [caps.schema]

        if model.startswith("anthropic."):
            return [
                ToolSchemaType.ANTHROPIC,
                ToolSchemaType.DEFAULT,
                ToolSchemaType.SYSTEM_PROMPT,
            ]

        if model == "mistral.mistral-7b-instruct-v0:2":
            return [
                ToolSchemaType.SYSTEM_PROMPT,
                ToolSchemaType.DEFAULT,
            ]

        return [
            ToolSchemaType.DEFAULT,
            ToolSchemaType.SYSTEM_PROMPT,
        ]

    def _track_bedrock_usage(
        self,
        model: str,
        processed_response: dict[str, Any],
    ) -> None:
        usage = processed_response.get("usage")
        if not isinstance(usage, dict):
            return

        try:
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            turn_usage = TurnUsage(
                provider=Provider.BEDROCK,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                raw_usage=usage,
            )
            self.usage_accumulator.add_turn(turn_usage)
        except Exception as exc:
            self.logger.warning(f"Failed to track usage: {exc}")

    @staticmethod
    def _bedrock_text_content_blocks(
        processed_response: dict[str, Any],
    ) -> list[ContentBlock]:
        content_items = processed_response.get("content")
        if not isinstance(content_items, list):
            return []

        return [
            TextContent(type="text", text=content_item["text"])
            for content_item in content_items
            if isinstance(content_item, dict) and content_item.get("text")
        ]

    @staticmethod
    def _tool_result_text(content: object) -> str:
        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            typed_part = cast("dict[str, Any]", part)
            text_parts.append(str(typed_part.get("text", "")))
        return " ".join(text_parts).strip()

    def _fallback_tool_result_content_blocks(
        self,
        messages: list[BedrockMessageParam],
    ) -> list[ContentBlock]:
        last_index = len(messages) - 2 if len(messages) >= 2 else len(messages) - 1
        last_input = messages[last_index] if last_index >= 0 else None
        if not isinstance(last_input, dict):
            return []

        contents = last_input.get("content", []) or []
        for content_item in contents:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "tool_result":
                fallback_text = self._tool_result_text(content_item.get("content", []))
            elif "toolResult" in content_item:
                tool_result = content_item["toolResult"]
                fallback_text = self._tool_result_text(tool_result.get("content", []))
            else:
                fallback_text = ""

            if fallback_text:
                return [TextContent(type="text", text=fallback_text)]

        return []

    def _bedrock_tool_calls_for_stop_reason(
        self,
        processed_response: dict[str, Any],
        model: str,
        tools: list[Tool] | None,
    ) -> tuple[str, dict[str, CallToolRequest] | None]:
        stop_reason_value = processed_response.get("stop_reason", "end_turn")
        stop_reason = stop_reason_value if isinstance(stop_reason_value, str) else "end_turn"
        caps_tmp = self.capabilities.get(model) or ModelCapabilities()

        if stop_reason == "end_turn" and tools:
            parsed_tools = self._parse_tool_response(processed_response, model)
            if parsed_tools:
                stop_reason = "tool_use"
                if not caps_tmp.schema:
                    caps_tmp.schema = ToolSchemaType.SYSTEM_PROMPT
                    self.capabilities[model] = caps_tmp

        if stop_reason not in ["tool_use", "tool_calls"]:
            return stop_reason, None

        parsed_tools = self._parse_tool_response(processed_response, model)
        if not parsed_tools:
            return stop_reason, None
        return stop_reason, self._build_tool_calls_dict(parsed_tools)

    def _bedrock_assistant_response(
        self,
        processed_response: dict[str, Any],
        messages: list[BedrockMessageParam],
        model: str,
        tools: list[Tool] | None,
    ) -> PromptMessageExtended:
        self._track_bedrock_usage(model, processed_response)
        self.logger.debug(f"{model} response:", data=processed_response)

        response_message_param = self.convert_message_to_message_param(processed_response)
        messages.append(response_message_param)

        response_content_blocks = self._bedrock_text_content_blocks(processed_response)
        if not response_content_blocks:
            response_content_blocks = self._fallback_tool_result_content_blocks(messages)

        stop_reason, tool_calls = self._bedrock_tool_calls_for_stop_reason(
            processed_response,
            model,
            tools,
        )
        mapped_stop_reason = self._map_bedrock_stop_reason(stop_reason)

        self.history.set(messages)
        self._log_chat_finished(model=model)

        from fast_agent.core.prompt import Prompt

        return Prompt.assistant(
            *response_content_blocks,
            stop_reason=mapped_stop_reason,
            tool_calls=tool_calls,
        )

    @staticmethod
    def _bedrock_tool_block_flags(messages: list[BedrockMessageParam]) -> tuple[bool, bool]:
        has_tool_results = False
        has_tool_use = False
        for msg in messages:
            if not isinstance(msg, dict) or not msg.get("content"):
                continue
            for content in msg["content"]:
                if not isinstance(content, dict):
                    continue
                has_tool_results = has_tool_results or "toolResult" in content
                has_tool_use = has_tool_use or "toolUse" in content
                if has_tool_results and has_tool_use:
                    return has_tool_results, has_tool_use
        return has_tool_results, has_tool_use

    @staticmethod
    def _tool_results_by_message(
        messages: list[BedrockMessageParam],
    ) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user" or not msg.get("content"):
                continue
            for content in msg["content"]:
                if not isinstance(content, dict) or "toolResult" not in content:
                    continue
                tool_result = content["toolResult"]
                tool_use_id = tool_result.get("toolUseId") or tool_result.get("tool_use_id")
                if tool_use_id:
                    grouped.setdefault(msg_idx, []).append(
                        {
                            "tool_use_id": tool_use_id,
                            "tool_result": tool_result,
                        }
                    )
        return grouped

    def _reconstruct_missing_tool_use_messages(
        self,
        messages: list[BedrockMessageParam],
        tool_list: "ListToolsResult | None",
        tool_name_mapping: dict[str, str] | None,
    ) -> None:
        tool_results_by_msg = self._tool_results_by_message(messages)
        for msg_idx in sorted(tool_results_by_msg.keys(), reverse=True):
            tool_use_blocks = [
                {
                    "toolUse": {
                        "toolUseId": result_info["tool_use_id"],
                        "name": self._resolve_tool_use_name(
                            result_info["tool_use_id"],
                            tool_list,
                            tool_name_mapping,
                        ),
                        "input": {},
                    }
                }
                for result_info in tool_results_by_msg[msg_idx]
            ]

            assistant_msg = {
                "role": "assistant",
                "content": tool_use_blocks,
            }
            messages.insert(msg_idx, assistant_msg)
            self.logger.debug(
                f"Inserted reconstructed assistant message with {len(tool_use_blocks)} toolUse blocks before message {msg_idx}"
            )

    @staticmethod
    def _tool_use_ids(message: BedrockMessageParam) -> list[Any]:
        if not isinstance(message, dict):
            return []
        return [
            content["toolUse"].get("toolUseId") or content["toolUse"].get("tool_use_id")
            for content in message.get("content", [])
            if isinstance(content, dict) and "toolUse" in content
        ]

    @staticmethod
    def _existing_tool_result_ids(message: BedrockMessageParam | None) -> set[Any]:
        if not isinstance(message, dict) or message.get("role") != "user":
            return set()
        return {
            content["toolResult"].get("toolUseId") or content["toolResult"].get("tool_use_id")
            for content in message.get("content", [])
            if isinstance(content, dict) and "toolResult" in content
        }

    @staticmethod
    def _placeholder_tool_results(tool_use_ids: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": "Tool was interrupted."}],
                }
            }
            for tool_use_id in tool_use_ids
        ]

    def _inject_orphaned_tool_results(self, messages: list[BedrockMessageParam]) -> None:
        for msg_idx, msg in enumerate(list(messages)):
            if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("content"):
                continue

            tool_use_ids = self._tool_use_ids(msg)
            if not tool_use_ids:
                continue

            next_msg = messages[msg_idx + 1] if msg_idx + 1 < len(messages) else None
            missing_ids = [
                tool_use_id
                for tool_use_id in tool_use_ids
                if tool_use_id not in self._existing_tool_result_ids(next_msg)
            ]
            if not missing_ids:
                continue

            self.logger.warning(
                f"Detected {len(missing_ids)} orphaned toolUse blocks without toolResult - "
                "injecting placeholder toolResult messages"
            )
            placeholder_content = self._placeholder_tool_results(missing_ids)
            if isinstance(next_msg, dict) and next_msg.get("role") == "user":
                next_msg["content"].extend(placeholder_content)
            else:
                messages.insert(msg_idx + 1, {"role": "user", "content": placeholder_content})

    @staticmethod
    def _noop_tool_spec() -> dict[str, Any]:
        return {
            "toolSpec": {
                "name": "noop",
                "description": "This is a placeholder tool that should be ignored.",
                "inputSchema": {"json": {"type": "object", "properties": {}}},
            }
        }

    def _apply_bedrock_tool_config(
        self,
        converse_args: dict[str, Any],
        schema_choice: ToolSchemaType,
        tools_payload: Union[list[dict[str, Any]], str, None],
        *,
        has_tool_results: bool,
        has_tool_use: bool,
    ) -> None:
        needs_noop = has_tool_results or has_tool_use
        if schema_choice in (ToolSchemaType.ANTHROPIC, ToolSchemaType.DEFAULT):
            if isinstance(tools_payload, list) and tools_payload:
                converse_args["toolConfig"] = {"tools": tools_payload}
            elif needs_noop:
                converse_args["toolConfig"] = {"tools": [self._noop_tool_spec()]}
        elif needs_noop:
            converse_args["toolConfig"] = {"tools": [self._noop_tool_spec()]}

    @staticmethod
    def _bedrock_has_tools_payload(
        tools_payload: Union[list[dict[str, Any]], str, None],
    ) -> bool:
        if isinstance(tools_payload, list):
            return bool(tools_payload)
        if isinstance(tools_payload, str):
            return bool(tools_payload.strip())
        return False

    def _inject_system_text_into_first_user(
        self,
        converse_args: dict[str, Any],
        system_text: str,
        *,
        cached_mode: bool = False,
    ) -> None:
        messages = converse_args["messages"]
        if not messages or messages[0].get("role") != "user":
            return

        first_message = messages[0]
        if not first_message.get("content") or len(first_message["content"]) <= 0:
            return

        original_text = first_message["content"][0].get("text", "")
        first_message["content"][0]["text"] = f"System: {system_text}\n\nUser: {original_text}"
        if cached_mode:
            self.logger.debug("Injected system prompt into first user message (cached mode)")

    def _bedrock_tools_payload(
        self,
        model: str,
        schema_choice: ToolSchemaType,
        tool_list: "ListToolsResult | None",
    ) -> tuple[Union[list[dict[str, Any]], str, None], dict[str, str] | None, ToolNamePolicy]:
        name_policy = (
            self.capabilities.get(model) or ModelCapabilities()
        ).tool_name_policy or ToolNamePolicy.PRESERVE
        if not tool_list or not tool_list.tools:
            return None, None, name_policy

        tool_name_mapping = self._build_tool_name_mapping(tool_list, name_policy)
        self.tool_name_mapping = tool_name_mapping

        if schema_choice == ToolSchemaType.ANTHROPIC:
            return (
                self._convert_tools_anthropic_format(tool_list, tool_name_mapping),
                tool_name_mapping,
                name_policy,
            )
        if schema_choice == ToolSchemaType.DEFAULT:
            return (
                self._convert_tools_nova_format(tool_list, tool_name_mapping),
                tool_name_mapping,
                name_policy,
            )
        if schema_choice == ToolSchemaType.SYSTEM_PROMPT:
            return (
                self._convert_tools_system_prompt_format(tool_list, tool_name_mapping),
                tool_name_mapping,
                name_policy,
            )
        return None, tool_name_mapping, name_policy

    @staticmethod
    def _append_system_text(base_text: str | None, addition: str) -> str:
        return f"{base_text}\n\n{addition}" if base_text else addition

    def _bedrock_system_text_for_attempt(
        self,
        base_system_text: str | None,
        model: str,
        schema_choice: ToolSchemaType,
        tools_payload: Union[list[dict[str, Any]], str, None],
    ) -> str | None:
        system_text = base_system_text
        if (
            schema_choice == ToolSchemaType.SYSTEM_PROMPT
            and isinstance(tools_payload, str)
            and tools_payload
        ):
            system_text = self._append_system_text(system_text, tools_payload)

        if schema_choice != ToolSchemaType.SYSTEM_PROMPT:
            return system_text

        if model.startswith("cohere."):
            return self._append_system_text(
                system_text,
                "FINAL ANSWER RULES (STRICT):\n"
                "- When a tool result is provided, your final answer MUST be exactly the raw tool result text.\n"
                "- Do not add any extra words, punctuation, qualifiers, or phrases (e.g., 'according to the tool').\n"
                "- Example: If tool result text is 'Its sunny in London', your final answer must be exactly: Its sunny in London\n",
            )

        if model.startswith("meta.llama3"):
            return self._append_system_text(
                system_text,
                "TOOL RESPONSE RULES:\n"
                "- After receiving a tool result, immediately output ONLY the exact tool result text.\n"
                "- Do not call additional tools or add commentary.\n"
                "- Do not paraphrase or modify the tool result in any way.",
            )

        if model.startswith("mistral."):
            return self._append_system_text(
                system_text,
                "TOOL EXECUTION RULES:\n"
                "- Call each tool only ONCE per conversation turn.\n"
                "- Accept and trust all tool results - do not question or retry them.\n"
                "- After receiving a tool result, provide a direct answer based on that result.\n"
                "- Do not call the same tool multiple times or call additional tools unless specifically requested.\n"
                "- Tool results are always valid - do not attempt to validate or correct them.",
            )

        return system_text

    def _apply_bedrock_system_text(
        self,
        converse_args: dict[str, Any],
        model: str,
        schema_choice: ToolSchemaType,
        system_text: str | None,
        system_mode: SystemMode,
    ) -> None:
        if not system_text:
            return
        if system_mode == SystemMode.SYSTEM:
            converse_args["system"] = [{"text": system_text}]
            self.logger.debug(
                f"Attempting with system param for {model} and schema={schema_choice}"
            )
            return
        self._inject_system_text_into_first_user(
            converse_args,
            system_text,
            cached_mode=True,
        )

    def _apply_bedrock_inference_config(
        self,
        converse_args: dict[str, Any],
        params: RequestParams,
        model: str,
        caps: ModelCapabilities,
    ) -> int:
        inference_config: dict[str, Any] = {}
        if params.maxTokens is not None:
            inference_config["maxTokens"] = params.maxTokens
        if params.stopSequences:
            inference_config["stopSequences"] = params.stopSequences

        reasoning_budget = self._resolve_reasoning_budget()
        reasoning_enabled = False
        if reasoning_budget > 0:
            cached_reasoning = caps.reasoning_support
            if cached_reasoning is not False:
                converse_args["performanceConfig"] = {
                    "reasoning": {"maxReasoningTokens": reasoning_budget}
                }
                reasoning_enabled = True

        if not reasoning_enabled and params.temperature is not None:
            inference_config["temperature"] = params.temperature

        if model and "nova" in strip_casefold(model) and reasoning_budget == 0:
            inference_config.setdefault("topP", 1.0)
            existing_amrf = converse_args.get("additionalModelRequestFields", {})
            converse_args["additionalModelRequestFields"] = {
                **existing_amrf,
                "inferenceConfig": {"topK": 1},
            }

        if inference_config:
            converse_args["inferenceConfig"] = inference_config

        return reasoning_budget

    def _bedrock_use_streaming(
        self,
        model: str,
        schema_choice: ToolSchemaType,
        *,
        has_tools: bool,
        has_tool_results: bool,
        force_non_streaming: bool,
    ) -> bool:
        if force_non_streaming:
            return False

        cache_pref = (self.capabilities.get(model) or ModelCapabilities()).stream_with_tools
        if has_tools and cache_pref == StreamPreference.NON_STREAM:
            return False

        if schema_choice == ToolSchemaType.ANTHROPIC and has_tool_results:
            self.logger.debug("Forcing non-streaming for Anthropic second turn with tool results")
            return False

        return True

    async def _invoke_bedrock_api(
        self,
        client: Any,
        converse_args: dict[str, Any],
        model: str,
        schema_choice: ToolSchemaType,
        *,
        use_streaming: bool,
    ) -> BedrockMessage:
        if not use_streaming:
            self.logger.debug(f"Using non-streaming API for {model} (schema={schema_choice})")
            response = client.converse(**converse_args)
            return self._process_non_streaming_response(response, model)

        self.logger.debug(f"Using streaming API for {model} (schema={schema_choice})")
        response = client.converse_stream(**converse_args)
        return await self._process_stream(response, model)

    async def _invoke_bedrock_with_reasoning_fallback(
        self,
        client: Any,
        converse_args: dict[str, Any],
        model: str,
        schema_choice: ToolSchemaType,
        params: RequestParams,
        caps: ModelCapabilities,
        *,
        reasoning_budget: int,
        use_streaming: bool,
    ) -> BedrockMessage:
        try:
            return await self._invoke_bedrock_api(
                client,
                converse_args,
                model,
                schema_choice,
                use_streaming=use_streaming,
            )
        except _BOTOCORE_ERRORS as exc:
            if reasoning_budget <= 0 or not _is_reasoning_performance_error(exc):
                raise

            self.logger.debug(f"Model {model} doesn't support reasoning, retrying without: {exc}")
            caps.reasoning_support = False
            self.capabilities[model] = caps
            converse_args.pop("performanceConfig", None)

            if params.temperature is not None:
                retry_inference_config = converse_args.get("inferenceConfig")
                if not isinstance(retry_inference_config, dict):
                    retry_inference_config = {}
                    converse_args["inferenceConfig"] = retry_inference_config
                retry_inference_config["temperature"] = params.temperature

            return await self._invoke_bedrock_api(
                client,
                converse_args,
                model,
                schema_choice,
                use_streaming=use_streaming,
            )

    def _cache_successful_bedrock_attempt(
        self,
        model: str,
        caps: ModelCapabilities,
        schema_choice: ToolSchemaType,
        tool_list: "ListToolsResult | None",
        name_policy: ToolNamePolicy,
        *,
        has_tools: bool,
        attempted_streaming: bool,
        reasoning_budget: int,
    ) -> None:
        if not caps.schema and has_tools:
            caps.schema = schema_choice

        if reasoning_budget > 0 and caps.reasoning_support is not True:
            caps.reasoning_support = True

        if schema_choice == ToolSchemaType.DEFAULT and name_policy == ToolNamePolicy.PRESERVE:
            try:
                if any("-" in tool.name for tool in (tool_list.tools if tool_list else [])):
                    caps.tool_name_policy = ToolNamePolicy.UNDERSCORES
            except Exception:
                pass

        if has_tools and attempted_streaming:
            caps.stream_with_tools = StreamPreference.STREAM_OK
        self.capabilities[model] = caps

    def _prepare_bedrock_attempt(
        self,
        bedrock_messages: list[BedrockMessageParam],
        params: RequestParams,
        model: str,
        base_system_text: str | None,
        caps: ModelCapabilities,
        tool_list: "ListToolsResult | None",
        schema_choice: ToolSchemaType,
    ) -> BedrockAttemptConfig:
        converse_args: dict[str, Any] = {
            "modelId": model,
            "messages": [dict(message) for message in bedrock_messages],
        }
        tools_payload, tool_name_mapping, name_policy = self._bedrock_tools_payload(
            model,
            schema_choice,
            tool_list,
        )
        system_mode = (
            self.capabilities.get(model) or ModelCapabilities()
        ).system_mode or SystemMode.SYSTEM
        system_text = self._bedrock_system_text_for_attempt(
            base_system_text,
            model,
            schema_choice,
            tools_payload,
        )
        self._apply_bedrock_system_text(
            converse_args,
            model,
            schema_choice,
            system_text,
            system_mode,
        )

        has_tool_results, has_tool_use = self._bedrock_tool_block_flags(bedrock_messages)
        if has_tool_results and not has_tool_use:
            self.logger.warning(
                "Detected tool results without corresponding tool use blocks - "
                "reconstructing missing assistant message with tool calls"
            )
            self._reconstruct_missing_tool_use_messages(
                converse_args["messages"],
                tool_list,
                tool_name_mapping,
            )
        if has_tool_use:
            self._inject_orphaned_tool_results(converse_args["messages"])

        self._apply_bedrock_tool_config(
            converse_args,
            schema_choice,
            tools_payload,
            has_tool_results=has_tool_results,
            has_tool_use=has_tool_use,
        )
        reasoning_budget = self._apply_bedrock_inference_config(
            converse_args,
            params,
            model,
            caps,
        )
        return BedrockAttemptConfig(
            converse_args=converse_args,
            tools_payload=tools_payload,
            tool_name_mapping=tool_name_mapping,
            name_policy=name_policy,
            system_text=system_text,
            system_mode=system_mode,
            has_tool_results=has_tool_results,
            has_tool_use=has_tool_use,
            reasoning_budget=reasoning_budget,
        )

    def _try_non_streaming_fallback(
        self,
        client: Any,
        attempt: BedrockAttemptConfig,
        model: str,
        caps: ModelCapabilities,
        schema_choice: ToolSchemaType,
        *,
        has_tools: bool,
    ) -> BedrockFallbackResult:
        if not has_tools or caps.stream_with_tools is not None:
            return BedrockFallbackResult()

        try:
            self.logger.debug(
                f"Falling back to non-streaming API for {model} after streaming error"
            )
            response = client.converse(**attempt.converse_args)
            processed_response = self._process_non_streaming_response(response, model)
            caps.stream_with_tools = StreamPreference.NON_STREAM
            if not caps.schema:
                caps.schema = schema_choice
            self.capabilities[model] = caps
            return BedrockFallbackResult(
                processed_response=processed_response,
                handled=True,
                attempted=True,
            )
        except _BOTOCORE_ERRORS as exc:
            last_error_msg = str(exc)
            self.logger.debug(
                f"Bedrock API error after non-streaming fallback: {last_error_msg}"
            )
            return BedrockFallbackResult(last_error_msg=last_error_msg, attempted=True)

    async def _try_system_inject_fallback(
        self,
        client: Any,
        bedrock_messages: list[BedrockMessageParam],
        model: str,
        caps: ModelCapabilities,
        schema_choice: ToolSchemaType,
        attempt: BedrockAttemptConfig,
        *,
        error_msg: str,
        tried_system_fallback: bool,
    ) -> BedrockFallbackResult:
        if (
            tried_system_fallback
            or not attempt.system_text
            or attempt.system_mode != SystemMode.SYSTEM
            or not _mentions_system_message(error_msg)
        ):
            return BedrockFallbackResult()

        caps.system_mode = SystemMode.INJECT
        self.capabilities[model] = caps
        self.logger.info(f"Switching system mode to inject for {model} and retrying same schema")

        try:
            converse_args: dict[str, Any] = {
                "modelId": model,
                "messages": [dict(message) for message in bedrock_messages],
            }
            self._inject_system_text_into_first_user(converse_args, attempt.system_text)
            self._apply_bedrock_tool_config(
                converse_args,
                schema_choice,
                attempt.tools_payload,
                has_tool_results=attempt.has_tool_results,
                has_tool_use=attempt.has_tool_use,
            )
            if attempt.has_tool_use:
                self._inject_orphaned_tool_results(converse_args["messages"])

            has_tools = self._bedrock_has_tools_payload(attempt.tools_payload)
            cache_pref = (self.capabilities.get(model) or ModelCapabilities()).stream_with_tools
            if cache_pref == StreamPreference.NON_STREAM or not has_tools:
                response = client.converse(**converse_args)
                processed_response = self._process_non_streaming_response(response, model)
            else:
                response = client.converse_stream(**converse_args)
                processed_response = await self._process_stream(response, model)

            if not caps.schema and has_tools:
                caps.schema = schema_choice
            self.capabilities[model] = caps
            return BedrockFallbackResult(
                processed_response=processed_response,
                handled=True,
                attempted=True,
            )
        except _BOTOCORE_ERRORS as exc:
            last_error_msg = str(exc)
            self.logger.debug(
                f"Bedrock API error after system inject fallback: {last_error_msg}"
            )
            return BedrockFallbackResult(last_error_msg=last_error_msg, attempted=True)

    def _bedrock_completion_inputs(
        self,
        message_param: BedrockMessageParam,
        request_params: RequestParams | None,
        pre_messages: list[BedrockMessageParam] | None,
        history: list[PromptMessageExtended] | None,
    ) -> tuple[list[BedrockMessageParam], RequestParams]:
        try:
            messages: list[BedrockMessageParam] = list(pre_messages) if pre_messages else []
            params = self.get_request_params(request_params)
        except _BOTOCORE_ERRORS as exc:
            error_msg = str(exc)
            if "UnauthorizedOperation" in error_msg or "AccessDenied" in error_msg:
                raise ProviderKeyError(
                    "AWS Bedrock access denied",
                    "Please check your AWS credentials and IAM permissions for Bedrock.",
                ) from exc
            raise ProviderKeyError(
                "AWS Bedrock error",
                f"Error accessing Bedrock: {error_msg}",
            ) from exc

        if history:
            messages.extend(self._convert_to_provider_format(history))
        else:
            messages.append(message_param)
        return messages, params

    def _bedrock_error_response(
        self,
        model: str,
        caps: ModelCapabilities,
        last_error_msg: str | None,
    ) -> BedrockMessage:
        caps.schema = ToolSchemaType.NONE
        self.capabilities[model] = caps
        return {
            "content": [
                {"text": f"Error during generation: {last_error_msg or 'Unknown error'}"}
            ],
            "stop_reason": "error",
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "model": model,
            "role": "assistant",
        }

    async def _bedrock_completion(
        self,
        message_param: BedrockMessageParam,
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        pre_messages: list[BedrockMessageParam] | None = None,
        history: list[PromptMessageExtended] | None = None,
    ) -> PromptMessageExtended:
        """
        Process a query using Bedrock and available tools.
        Returns PromptMessageExtended with tool calls for external execution.
        """
        client = self._get_bedrock_runtime_client()
        messages, params = self._bedrock_completion_inputs(
            message_param,
            request_params,
            pre_messages,
            history,
        )

        tool_list = self._bedrock_tool_list(tools)
        model = self.default_request_params.model or DEFAULT_BEDROCK_MODEL

        # Single API call - no tool execution loop
        self._log_chat_progress(self.chat_turn(), model=model)

        # Convert messages to Bedrock format
        bedrock_messages = self._convert_messages_to_bedrock(messages)

        # Base system text
        base_system_text = self.instruction or params.systemPrompt

        # Determine tool schema fallback order and caches
        caps = self.capabilities.get(model) or ModelCapabilities()
        schema_order = self._bedrock_schema_order(model, caps)

        # Track whether we changed system mode cache this turn
        tried_system_fallback = False

        processed_response: dict[str, Any] | None = None
        last_error_msg = None

        for schema_choice in schema_order:
            attempt = self._prepare_bedrock_attempt(
                bedrock_messages,
                params,
                model,
                base_system_text,
                caps,
                tool_list,
                schema_choice,
            )

            # Decide streaming vs non-streaming (resolver-free with runtime detection + cache)
            try:
                has_tools = self._bedrock_has_tools_payload(attempt.tools_payload)

                # Force non-streaming for structured-output flows (one-shot)
                force_non_streaming = False
                if self._force_non_streaming_once:
                    force_non_streaming = True
                    self._force_non_streaming_once = False

                use_streaming = self._bedrock_use_streaming(
                    model,
                    schema_choice,
                    has_tools=has_tools,
                    has_tool_results=attempt.has_tool_results,
                    force_non_streaming=force_non_streaming,
                )
                attempted_streaming = use_streaming

                # Try API call with reasoning fallback
                processed_response = await self._invoke_bedrock_with_reasoning_fallback(
                    client,
                    attempt.converse_args,
                    model,
                    schema_choice,
                    params,
                    caps,
                    reasoning_budget=attempt.reasoning_budget,
                    use_streaming=use_streaming,
                )

                # Success: cache the working schema choice if not already cached
                self._cache_successful_bedrock_attempt(
                    model,
                    caps,
                    schema_choice,
                    tool_list,
                    attempt.name_policy,
                    has_tools=has_tools,
                    attempted_streaming=attempted_streaming,
                    reasoning_budget=attempt.reasoning_budget,
                )
                break
            except _BOTOCORE_ERRORS as e:
                error_msg = str(e)
                last_error_msg = error_msg
                self.logger.debug(f"Bedrock API error (schema={schema_choice}): {error_msg}")

                # If streaming with tools failed and cache undecided, fallback to non-streaming and cache
                fallback_result = self._try_non_streaming_fallback(
                    client,
                    attempt,
                    model,
                    caps,
                    schema_choice,
                    has_tools=has_tools,
                )
                if fallback_result.last_error_msg is not None:
                    last_error_msg = fallback_result.last_error_msg
                if fallback_result.handled:
                    processed_response = fallback_result.processed_response
                    break

                # System parameter fallback once per call if system message unsupported
                system_fallback_result = await self._try_system_inject_fallback(
                    client,
                    bedrock_messages,
                    model,
                    caps,
                    schema_choice,
                    attempt,
                    error_msg=error_msg,
                    tried_system_fallback=tried_system_fallback,
                )
                if system_fallback_result.last_error_msg is not None:
                    last_error_msg = system_fallback_result.last_error_msg
                if system_fallback_result.attempted:
                    tried_system_fallback = True
                if system_fallback_result.handled:
                    processed_response = system_fallback_result.processed_response
                    break

                # For any other error (including tool format errors), continue to next schema
                self.logger.debug(
                    f"Continuing to next schema after error with {schema_choice}: {error_msg}"
                )
                continue

        if processed_response is None:
            processed_response = self._bedrock_error_response(model, caps, last_error_msg)

        return self._bedrock_assistant_response(processed_response, messages, model, tools)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """
        Provider-specific prompt application.
        Templates are handled by the agent; messages already include them.
        """
        if not multipart_messages:
            return PromptMessageExtended(role="user", content=[])

        # Check the last message role
        last_message = multipart_messages[-1]

        if last_message.role == "assistant":
            # For assistant messages: Return the last message (no completion needed)
            return last_message

        effective_params = self.get_request_params(request_params)
        if effective_params.structured_schema:
            _, response = await self._apply_prompt_provider_specific_structured_schema(
                multipart_messages,
                effective_params.structured_schema,
                effective_params,
                tools,
            )
            return response

        # Convert the last user message to Bedrock message parameter format
        message_param = BedrockConverter.convert_to_bedrock(last_message)

        # Call the completion method
        # No need to pass pre_messages - conversion happens in _bedrock_completion
        # via _convert_to_provider_format()
        return await self._bedrock_completion(
            message_param,
            effective_params,
            tools,
            pre_messages=None,
            history=multipart_messages,
        )

    def _generate_simplified_schema(self, model: type[ModelT]) -> str:
        """Generates a simplified, human-readable schema with inline enum constraints."""
        schema = _bedrock_simplified_schema_dict(model)
        return json.dumps(schema, indent=2)

    def _structured_model_schema_text(self, model: type[ModelT]) -> str:
        caps_struct = self.capabilities.get(self.model) or ModelCapabilities()
        strategy = caps_struct.structured_strategy or StructuredStrategy.STRICT_SCHEMA
        if strategy == StructuredStrategy.SIMPLIFIED_SCHEMA:
            return self._generate_simplified_schema(model)
        return FastAgentLLM.model_to_schema_str(model)

    def _structured_raw_schema_text(self, schema: dict[str, Any]) -> str:
        caps_struct = self.capabilities.get(self.model) or ModelCapabilities()
        strategy = caps_struct.structured_strategy or StructuredStrategy.STRICT_SCHEMA
        if strategy == StructuredStrategy.SIMPLIFIED_SCHEMA:
            return json.dumps(schema, indent=2)
        return self.schema_to_schema_str(schema)

    def _structured_json_params(
        self, request_params: RequestParams | None
    ) -> RequestParams:
        return self.get_request_params(request_params).model_copy(
            update={"structured_schema": None, "temperature": 0.0}
        )

    @staticmethod
    def _copy_message_with_text(
        message: PromptMessageExtended,
        text: str,
    ) -> PromptMessageExtended:
        try:
            copied = message.model_copy(deep=True)
        except Exception:
            copied = PromptMessageExtended(
                role=message.role,
                content=list(message.content),
            )
        copied.add_text(text)
        return copied

    def _structured_messages_with_prompt(
        self,
        multipart_messages: list[PromptMessageExtended],
        prompt_text: str,
    ) -> list[PromptMessageExtended]:
        temp_last = self._copy_message_with_text(multipart_messages[-1], prompt_text)
        return [*multipart_messages[:-1], temp_last]

    def _existing_assistant_structured_model(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: type[ModelT],
    ) -> tuple[ModelT, PromptMessageExtended] | None:
        try:
            if not multipart_messages or multipart_messages[-1].role != "assistant":
                return None
            parsed_model, parsed_mp = self._structured_from_multipart(
                multipart_messages[-1], model
            )
            if parsed_model is None:
                return None
            return parsed_model, parsed_mp
        except Exception:
            return None

    def _parse_required_structured_model(
        self,
        result: PromptMessageExtended,
        model: type[ModelT],
    ) -> tuple[ModelT, PromptMessageExtended]:
        parsed_model, parsed_message = self._structured_from_multipart(result, model)
        if parsed_model is None:
            raise ValueError("structured parse returned None; triggering retry")
        return parsed_model, parsed_message

    def _existing_assistant_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
    ) -> tuple[Any, PromptMessageExtended] | None:
        try:
            if not multipart_messages or multipart_messages[-1].role != "assistant":
                return None
            parsed_data, parsed_mp = self._structured_schema_from_multipart(
                multipart_messages[-1],
                schema,
            )
            if parsed_data is None:
                return None
            return parsed_data, parsed_mp
        except Exception:
            return None

    def _parse_required_structured_schema(
        self,
        result: PromptMessageExtended,
        schema: dict[str, Any],
    ) -> tuple[Any, PromptMessageExtended]:
        parsed, parsed_message = self._structured_schema_from_multipart(result, schema)
        if parsed is None:
            raise ValueError("structured parse returned None; triggering retry")
        return parsed, parsed_message

    async def _apply_prompt_provider_specific_structured(
        self,
        multipart_messages: list[PromptMessageExtended],
        model: type[ModelT],
        request_params: RequestParams | None = None,
    ) -> tuple[ModelT | None, PromptMessageExtended]:
        """Apply structured output for Bedrock using prompt engineering with a simplified schema."""
        existing = self._existing_assistant_structured_model(multipart_messages, model)
        if existing is not None:
            return existing

        request_params = self._structured_json_params(request_params)

        # For structured outputs: disable reasoning entirely and set temperature=0 for deterministic JSON
        # This avoids conflicts between reasoning (requires temperature=1) and structured output (wants temperature=0)
        original_reasoning_effort = self.reasoning_effort
        self.set_reasoning_effort(ReasoningEffortSetting(kind="toggle", value=False))

        schema_text = self._structured_model_schema_text(model)
        structured_messages = self._structured_messages_with_prompt(
            multipart_messages,
            _bedrock_structured_json_prompt(schema_text),
        )

        self.logger.debug(
            "DEBUG: Using copied last message for structured schema; original left untouched"
        )

        try:
            result: PromptMessageExtended = await self._apply_prompt_provider_specific(
                structured_messages, request_params
            )
            try:
                parsed_model, _ = self._parse_required_structured_model(result, model)
                return parsed_model, result
            except Exception:
                # One retry with stricter JSON-only guidance and simplified schema
                try:
                    simplified_schema_text = self._generate_simplified_schema(model)
                except Exception:
                    simplified_schema_text = FastAgentLLM.model_to_schema_str(model)
                retry_messages = self._structured_messages_with_prompt(
                    multipart_messages,
                    _bedrock_structured_retry_prompt(
                        "JSON Schema (simplified):",
                        simplified_schema_text,
                    ),
                )
                retry_result: PromptMessageExtended = await self._apply_prompt_provider_specific(
                    retry_messages, request_params
                )
                return self._structured_from_multipart(retry_result, model)
        finally:
            # Restore original reasoning effort
            self.set_reasoning_effort(original_reasoning_effort)

    async def _apply_prompt_provider_specific_structured_schema(
        self,
        multipart_messages: list[PromptMessageExtended],
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        existing = self._existing_assistant_structured_schema(multipart_messages, schema)
        if existing is not None:
            return existing

        request_params = self._structured_json_params(request_params)

        original_reasoning_effort = self.reasoning_effort
        self.set_reasoning_effort(ReasoningEffortSetting(kind="toggle", value=False))

        structured_messages = self._structured_messages_with_prompt(
            multipart_messages,
            _bedrock_structured_json_prompt(self._structured_raw_schema_text(schema)),
        )

        try:
            result = await self._apply_prompt_provider_specific(
                structured_messages,
                request_params,
                tools,
            )
            if result.tool_calls:
                return None, result
            parsed, _ = self._parse_required_structured_schema(result, schema)
            return parsed, result
        except Exception:
            retry_messages = self._structured_messages_with_prompt(
                multipart_messages,
                _bedrock_structured_retry_prompt("JSON Schema:", json.dumps(schema, indent=2)),
            )
            retry_result = await self._apply_prompt_provider_specific(
                retry_messages,
                request_params,
                tools,
            )
            if retry_result.tool_calls:
                return None, retry_result
            return self._structured_schema_from_multipart(retry_result, schema)
        finally:
            self.set_reasoning_effort(original_reasoning_effort)

    def _prepare_structured_request(
        self,
        messages: list[PromptMessageExtended],
        request_params: RequestParams,
        tools: list[Tool] | None = None,
    ) -> tuple[list[PromptMessageExtended], RequestParams]:
        if not self._should_defer_structured_schema_for_tools(messages, request_params, tools):
            return messages, request_params
        return messages, request_params.model_copy(update={"structured_schema": None})

    def _clean_json_response(self, text: str) -> str:
        """Clean up JSON response by removing text before first { and after last }.

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
            return text[first_brace : last_brace + 1]

        # Otherwise return the original text
        return text

    @staticmethod
    def _unwrap_structured_response_wrapper(
        text: str,
        model: type[ModelT],
    ) -> str | None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or len(parsed) != 1:
            return None

        wrapper_key, inner_value = next(iter(parsed.items()))
        if wrapper_key != model.__name__ or not isinstance(inner_value, dict):
            return None
        return json.dumps(inner_value)

    def _structured_from_multipart(
        self, message: PromptMessageExtended, model: type[ModelT]
    ) -> tuple[ModelT | None, PromptMessageExtended]:
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
        unwrapped_text = self._unwrap_structured_response_wrapper(cleaned_text, model)
        if unwrapped_text is not None:
            from mcp.types import TextContent

            unwrapped_multipart = PromptMessageExtended(
                role=message.role,
                content=[TextContent(type="text", text=unwrapped_text)],
            )
            model_instance, parsed_multipart = super()._structured_from_multipart(
                unwrapped_multipart, model
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

    def _provider_api_key(self) -> str:
        """Bedrock doesn't use API keys, returns empty string."""
        return ""
