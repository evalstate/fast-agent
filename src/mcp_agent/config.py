"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from mcp import Implementation
from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerAuthSettings(BaseModel):
    """Represents authentication configuration for a server."""

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPSamplingSettings(BaseModel):
    model: str = "haiku"

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPElicitationSettings(BaseModel):
    mode: Literal["forms", "auto_cancel", "none"] = "none"
    """Elicitation mode: 'forms' (default UI), 'auto_cancel', 'none' (no capability)"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPRootSettings(BaseModel):
    """Represents a root directory configuration for an MCP server."""

    uri: str
    """The URI identifying the root. Must start with file://"""

    name: Optional[str] = None
    """Optional name for the root."""

    server_uri_alias: Optional[str] = None
    """Optional URI alias for presentation to the server"""

    @field_validator("uri", "server_uri_alias")
    @classmethod
    def validate_uri(cls, v: str) -> str:
        """Validate that the URI starts with file:// (required by specification 2024-11-05)"""
        if v and not v.startswith("file://"):
            raise ValueError("Root URI must start with file://")
        return v

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPServerSettings(BaseModel):
    """
    Represents the configuration for an individual server.
    """

    name: str | None = None
    """The name of the server."""

    description: str | None = None
    """The description of the server."""

    transport: Literal["stdio", "sse", "http"] = "stdio"
    """The transport mechanism."""

    command: str | None = None
    """The command to execute the server (e.g. npx)."""

    args: List[str] | None = None
    """The arguments for the server command."""

    read_timeout_seconds: int | None = None
    """The timeout in seconds for the session."""

    read_transport_sse_timeout_seconds: int = 300
    """The timeout in seconds for the server connection."""

    url: str | None = None
    """The URL for the server (e.g. for SSE transport)."""

    headers: Dict[str, str] | None = None
    """Headers dictionary for SSE connections"""

    auth: MCPServerAuthSettings | None = None
    """The authentication configuration for the server."""

    roots: Optional[List[MCPRootSettings]] = None
    """Root directories this server has access to."""

    env: Dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    sampling: MCPSamplingSettings | None = None
    """Sampling settings for this Client/Server pair"""

    elicitation: MCPElicitationSettings | None = None
    """Elicitation settings for this Client/Server pair"""

    cwd: str | None = None
    """Working directory for the executed server command."""

    implementation: Implementation | None = None


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""

    servers: Dict[str, MCPServerSettings] = {}
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AnthropicSettings(BaseModel):
    """
    Settings for using Anthropic models in the fast-agent application.
    """

    api_key: str | None = None

    base_url: str | None = None

    cache_mode: Literal["off", "prompt", "auto"] = "auto"
    """
    Controls how caching is applied for Anthropic models when prompt_caching is enabled globally.
    - "off": No caching, even if global prompt_caching is true.
    - "prompt": Caches tools+system prompt (1 block) and template content. Useful for large, static prompts.
    - "auto": Currently same as "prompt" - caches tools+system prompt (1 block) and template content.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAISettings(BaseModel):
    """
    Settings for using OpenAI models in the fast-agent application.
    """

    api_key: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class DeepSeekSettings(BaseModel):
    """
    Settings for using OpenAI models in the fast-agent application.
    """

    api_key: str | None = None
    # reasoning_effort: Literal["low", "medium", "high"] = "medium"

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GoogleSettings(BaseModel):
    """
    Settings for using OpenAI models in the fast-agent application.
    """

    api_key: str | None = None
    # reasoning_effort: Literal["low", "medium", "high"] = "medium"

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class XAISettings(BaseModel):
    """
    Settings for using xAI Grok models in the fast-agent application.
    """

    api_key: str | None = None
    base_url: str | None = "https://api.x.ai/v1"

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GenericSettings(BaseModel):
    """
    Settings for using OpenAI models in the fast-agent application.
    """

    api_key: str | None = None

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenRouterSettings(BaseModel):
    """
    Settings for using OpenRouter models via its OpenAI-compatible API.
    """

    api_key: str | None = None

    base_url: str | None = None  # Optional override, defaults handled in provider

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AzureSettings(BaseModel):
    """
    Settings for using Azure OpenAI Service in the fast-agent application.
    """

    api_key: str | None = None
    resource_name: str | None = None
    azure_deployment: str | None = None
    api_version: str | None = None
    base_url: str | None = None  # Optional, can be constructed from resource_name

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GroqSettings(BaseModel):
    """
    Settings for using xAI Grok models in the fast-agent application.
    """

    api_key: str | None = None
    base_url: str | None = "https://api.groq.com/openai/v1"

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenTelemetrySettings(BaseModel):
    """
    OTEL settings for the fast-agent application.
    """

    enabled: bool = False

    service_name: str = "fast-agent"

    otlp_endpoint: str = "http://localhost:4318/v1/traces"
    """OTLP endpoint for OpenTelemetry tracing"""

    console_debug: bool = False
    """Log spans to console"""

    sample_rate: float = 1.0
    """Sample rate for tracing (1.0 = sample everything)"""


class TensorZeroSettings(BaseModel):
    """
    Settings for using TensorZero via its OpenAI-compatible API.
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class BedrockSettings(BaseModel):
    """
    Settings for using AWS Bedrock models in the fast-agent application.
    """

    region: str | None = None
    """AWS region for Bedrock service"""

    profile: str | None = None
    """AWS profile to use for authentication"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class HuggingFaceSettings(BaseModel):
    """
    Settings for HuggingFace authentication (used for MCP connections).
    """

    api_key: Optional[str] = None
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class LoggerSettings(BaseModel):
    """
    Logger settings for the fast-agent application.
    """

    type: Literal["none", "console", "file", "http"] = "file"

    level: Literal["debug", "info", "warning", "error"] = "warning"
    """Minimum logging level"""

    progress_display: bool = True
    """Enable or disable the progress display"""

    path: str = "fastagent.jsonl"
    """Path to log file, if logger 'type' is 'file'."""

    batch_size: int = 100
    """Number of events to accumulate before processing"""

    flush_interval: float = 2.0
    """How often to flush events in seconds"""

    max_queue_size: int = 2048
    """Maximum queue size for event processing"""

    # HTTP transport settings
    http_endpoint: str | None = None
    """HTTP endpoint for event transport"""

    http_headers: dict[str, str] | None = None
    """HTTP headers for event transport"""

    http_timeout: float = 5.0
    """HTTP timeout seconds for event transport"""

    show_chat: bool = True
    """Show chat User/Assistant on the console"""
    show_tools: bool = True
    """Show MCP Sever tool calls on the console"""
    truncate_tools: bool = True
    """Truncate display of long tool calls"""
    enable_markup: bool = True
    """Enable markup in console output. Disable for outputs that may conflict with rich console formatting"""
    use_legacy_display: bool = False
    """Use the legacy console display instead of the new style display"""


def find_fastagent_config_files(start_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Find FastAgent configuration files with standardized behavior.

    Returns:
        Tuple of (config_path, secrets_path) where either can be None if not found.

    Strategy:
    1. Find config file recursively from start_path upward
    2. Prefer secrets file in same directory as config file
    3. If no secrets file next to config, search recursively from start_path
    """
    config_path = None
    secrets_path = None

    # First, find the config file with recursive search
    current = start_path.resolve()
    while current != current.parent:
        potential_config = current / "fastagent.config.yaml"
        if potential_config.exists():
            config_path = potential_config
            break
        current = current.parent

    # If config file found, prefer secrets file in the same directory
    if config_path:
        potential_secrets = config_path.parent / "fastagent.secrets.yaml"
        if potential_secrets.exists():
            secrets_path = potential_secrets
        else:
            # If no secrets file next to config, do recursive search from start
            current = start_path.resolve()
            while current != current.parent:
                potential_secrets = current / "fastagent.secrets.yaml"
                if potential_secrets.exists():
                    secrets_path = potential_secrets
                    break
                current = current.parent
    else:
        # No config file found, just search for secrets file
        current = start_path.resolve()
        while current != current.parent:
            potential_secrets = current / "fastagent.secrets.yaml"
            if potential_secrets.exists():
                secrets_path = potential_secrets
                break
            current = current.parent

    return config_path, secrets_path


class Settings(BaseSettings):
    """
    Settings class for the fast-agent application.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
        nested_model_default_partial_update=True,
    )  # Customize the behavior of settings here

    mcp: MCPSettings | None = MCPSettings()
    """MCP config, such as MCP servers"""

    execution_engine: Literal["asyncio"] = "asyncio"
    """Execution engine for the fast-agent application"""

    default_model: str | None = "haiku"
    """
    Default model for agents. Format is provider.model_name.<reasoning_effort>, for example openai.o3-mini.low
    Aliases are provided for common models e.g. sonnet, haiku, gpt-4.1, o3-mini etc.
    """

    auto_sampling: bool = True
    """Enable automatic sampling model selection if not explicitly configured"""

    anthropic: AnthropicSettings | None = None
    """Settings for using Anthropic models in the fast-agent application"""

    otel: OpenTelemetrySettings | None = OpenTelemetrySettings()
    """OpenTelemetry logging settings for the fast-agent application"""

    openai: OpenAISettings | None = None
    """Settings for using OpenAI models in the fast-agent application"""

    deepseek: DeepSeekSettings | None = None
    """Settings for using DeepSeek models in the fast-agent application"""

    google: GoogleSettings | None = None
    """Settings for using DeepSeek models in the fast-agent application"""

    xai: XAISettings | None = None
    """Settings for using xAI Grok models in the fast-agent application"""

    openrouter: OpenRouterSettings | None = None
    """Settings for using OpenRouter models in the fast-agent application"""

    generic: GenericSettings | None = None
    """Settings for using Generic models in the fast-agent application"""

    tensorzero: Optional[TensorZeroSettings] = None
    """Settings for using TensorZero inference gateway"""

    azure: AzureSettings | None = None
    """Settings for using Azure OpenAI Service in the fast-agent application"""

    aliyun: OpenAISettings | None = None
    """Settings for using Aliyun OpenAI Service in the fast-agent application"""

    bedrock: BedrockSettings | None = None
    """Settings for using AWS Bedrock models in the fast-agent application"""

    huggingface: HuggingFaceSettings | None = None
    """Settings for HuggingFace authentication (used for MCP connections)"""

    groq: GroqSettings | None = None
    """Settings for using the Groq provider in the fast-agent application"""

    logger: LoggerSettings | None = LoggerSettings()
    """Logger settings for the fast-agent application"""

    @classmethod
    def find_config(cls) -> Path | None:
        """Find the config file in the current directory or parent directories."""
        current_dir = Path.cwd()

        # Check current directory and parent directories
        while current_dir != current_dir.parent:
            for filename in [
                "fastagent.config.yaml",
            ]:
                config_path = current_dir / filename
                if config_path.exists():
                    return config_path
            current_dir = current_dir.parent

        return None


# Global settings object
_settings: Settings | None = None


def get_settings(config_path: str | None = None) -> Settings:
    """Get settings instance, automatically loading from config file if available."""

    def resolve_env_vars(config_item: Any) -> Any:
        """Recursively resolve environment variables in config data."""
        if isinstance(config_item, dict):
            return {k: resolve_env_vars(v) for k, v in config_item.items()}
        elif isinstance(config_item, list):
            return [resolve_env_vars(i) for i in config_item]
        elif isinstance(config_item, str):
            # Regex to find ${ENV_VAR} or ${ENV_VAR:default_value}
            pattern = re.compile(r"\$\{([^}]+)\}")

            def replace_match(match: re.Match) -> str:
                var_name_with_default = match.group(1)
                if ":" in var_name_with_default:
                    var_name, default_value = var_name_with_default.split(":", 1)
                    return os.getenv(var_name, default_value)
                else:
                    var_name = var_name_with_default
                    env_value = os.getenv(var_name)
                    if env_value is None:
                        # Optionally, raise an error or return the placeholder if the env var is not set
                        # For now, returning the placeholder to avoid breaking if not set and no default
                        # print(f"Warning: Environment variable {var_name} not set and no default provided.")
                        return match.group(0)
                    return env_value

            # Replace all occurrences
            resolved_value = pattern.sub(replace_match, config_item)
            return resolved_value
        return config_item

    def deep_merge(base: dict, update: dict) -> dict:
        """Recursively merge two dictionaries, preserving nested structures."""
        merged = base.copy()
        for key, value in update.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    global _settings

    # If we have a specific config path, always reload settings
    # This ensures each test gets its own config
    if config_path:
        # Reset for the new path
        _settings = None
    elif _settings:
        # Use cached settings only for no specific path
        return _settings

    # Handle config path - convert string to Path if needed
    if config_path:
        config_file = Path(config_path)
        # If it's a relative path and doesn't exist, try finding it
        if not config_file.is_absolute() and not config_file.exists():
            # Try resolving against current directory first
            resolved_path = Path.cwd() / config_file.name
            if resolved_path.exists():
                config_file = resolved_path

        # When config path is explicitly provided, find secrets using standardized logic
        secrets_file = None
        if config_file.exists():
            _, secrets_file = find_fastagent_config_files(config_file.parent)
    else:
        # Use standardized discovery for both config and secrets
        config_file, secrets_file = find_fastagent_config_files(Path.cwd())

    merged_settings = {}

    import yaml  # pylint: disable=C0415

    # Load main config if it exists
    if config_file and config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_settings = yaml.safe_load(f) or {}
            # Resolve environment variables in the loaded YAML settings
            resolved_yaml_settings = resolve_env_vars(yaml_settings)
            merged_settings = resolved_yaml_settings
    elif config_file and not config_file.exists():
        print(f"Warning: Specified config file does not exist: {config_file}")

    # Load secrets file if found (regardless of whether config file exists)
    if secrets_file and secrets_file.exists():
        with open(secrets_file, "r", encoding="utf-8") as f:
            yaml_secrets = yaml.safe_load(f) or {}
            # Resolve environment variables in the loaded secrets YAML
            resolved_secrets_yaml = resolve_env_vars(yaml_secrets)
            merged_settings = deep_merge(merged_settings, resolved_secrets_yaml)

    _settings = Settings(**merged_settings)
    return _settings
