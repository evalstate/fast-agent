"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerAuthSettings(BaseModel):
    """Represents authentication configuration for a server."""

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPSamplingSettings(BaseModel):
    model: str = "haiku"

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


class MCPToolCallConfig(BaseModel):
    """Configuration for a tool call requiring confirmation."""
    
    name: str
    """The name of the tool."""
    
    seek_confirm: bool = False
    """Whether to seek confirmation before executing the tool."""
    
    time_to_confirm: int = 120000
    """Timeout in milliseconds for the confirmation (default: 2 minutes)."""
    
    default: Literal["confirm", "reject"] = "reject"
    """Default action when timeout occurs."""
    
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPServerSettings(BaseModel):
    """
    Represents the configuration for an individual server.
    """

    # TODO: saqadri - server name should be something a server can provide itself during initialization
    name: str | None = None
    """The name of the server."""

    # TODO: saqadri - server description should be something a server can provide itself during initialization
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

    cwd: str | None = None
    """Working directory for the executed server command."""
    
    tool_calls: Optional[List[MCPToolCallConfig]] = None
    """Configuration for tool calls that require confirmation."""


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

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAISettings(BaseModel):
    """
    Settings for using OpenAI models in the fast-agent application.
    """

    api_key: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] = "medium"

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
    
    # PubSub settings
    pubsub_enabled: bool = False
    """Enable pub/sub for agent communication"""


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
    
    pubsub_enabled: bool = False
    """Enable pub/sub for agent communication"""

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

    openrouter: OpenRouterSettings | None = None
    """Settings for using OpenRouter models in the fast-agent application"""

    generic: GenericSettings | None = None
    """Settings for using Generic models in the fast-agent application"""

    tensorzero: Optional[TensorZeroSettings] = None
    """Settings for using TensorZero inference gateway"""

    azure: AzureSettings | None = None
    """Settings for using Azure OpenAI Service in the fast-agent application"""

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


def get_settings(config_path: str | None = None, json_config: dict | None = None) -> Settings:
    """
    Get settings instance, automatically loading from config file if available or using provided JSON config.
    
    Args:
        config_path (str | None): Path to YAML config file
        json_config (dict | None): JSON configuration dict to use instead of loading from file
    
    Returns:
        Settings: The loaded settings
    """

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

    # If direct JSON config is provided, create settings from it
    if json_config is not None:
        # We don't cache settings when direct JSON is provided to ensure 
        # each call with different JSON gets its own config
        return Settings(**json_config)

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
    else:
        config_file = Settings.find_config()

    merged_settings = {}

    if config_file:
        if not config_file.exists():
            print(f"Warning: Specified config file does not exist: {config_file}")
        else:
            import yaml  # pylint: disable=C0415

            # Load main config
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_settings = yaml.safe_load(f) or {}
                merged_settings = yaml_settings
            # Look for secrets files recursively up the directory tree
            # but stop after finding the first one
            current_dir = config_file.parent
            found_secrets = False
            # Start with the absolute path of the config file's directory
            current_dir = config_file.parent.resolve()

            while current_dir != current_dir.parent and not found_secrets:
                for secrets_filename in [
                    "fastagent.secrets.yaml",
                ]:
                    secrets_file = current_dir / secrets_filename
                    if secrets_file.exists():
                        with open(secrets_file, "r", encoding="utf-8") as f:
                            yaml_secrets = yaml.safe_load(f) or {}
                            merged_settings = deep_merge(merged_settings, yaml_secrets)
                            found_secrets = True
                            break
                if not found_secrets:
                    # Get the absolute path of the parent directory
                    current_dir = current_dir.parent.resolve()

            _settings = Settings(**merged_settings)
            return _settings
    else:
        pass

    _settings = Settings()
    return _settings
