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
        if not v.startswith("file://"):
            raise ValueError("Root URI must start with file://")
        return v

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

    transport: Literal["stdio", "sse"] = "stdio"
    """The transport mechanism."""

    command: str | None = None
    """The command to execute the server (e.g. npx)."""

    args: List[str] | None = None
    """The arguments for the server command."""

    read_timeout_seconds: int | None = None
    """The timeout in seconds for the server connection."""

    url: str | None = None
    """The URL for the server (e.g. for SSE transport)."""

    auth: MCPServerAuthSettings | None = None
    """The authentication configuration for the server."""

    roots: Optional[List[MCPRootSettings]] = None
    """Root directories this server has access to."""

    env: Dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    env: Dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    sampling: MCPSamplingSettings | None = None
    """Sampling settings for this Client/Server pair"""


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""

    servers: Dict[str, MCPServerSettings] = {}
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AnthropicSettings(BaseModel):
    """
    Settings for using Anthropic models in the MCP Agent application.
    """

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class CohereSettings(BaseModel):
    """
    Settings for using Cohere models in the MCP Agent application.
    """

    api_key: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAISettings(BaseModel):
    """
    Settings for using OpenAI models in the MCP Agent application.
    """

    api_key: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] = "medium"

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

class DeepSeekSettings(BaseModel):
    """
    Settings for using OpenAI models in the MCP Agent application.
    """

    api_key: str | None = None
    #reasoning_effort: Literal["low", "medium", "high"] = "medium"

    base_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

class TemporalSettings(BaseModel):
    """
    Temporal settings for the MCP Agent application.
    """

    host: str
    namespace: str = "default"
    task_queue: str
    api_key: str | None = None


class UsageTelemetrySettings(BaseModel):
    """
    Settings for usage telemetry in the MCP Agent application.
    Anonymized usage metrics are sent to a telemetry server to help improve the product.
    """

    enabled: bool = True
    """Enable usage telemetry in the MCP Agent application."""

    enable_detailed_telemetry: bool = False
    """If enabled, detailed telemetry data, including prompts and agents, will be sent to the telemetry server."""


class OpenTelemetrySettings(BaseModel):
    """
    OTEL settings for the MCP Agent application.
    """

    enabled: bool = True

    service_name: str = "mcp-agent"
    service_instance_id: str | None = None
    service_version: str | None = None

    otlp_endpoint: str | None = None
    """OTLP endpoint for OpenTelemetry tracing"""

    console_debug: bool = False
    """Log spans to console"""

    sample_rate: float = 1.0
    """Sample rate for tracing (1.0 = sample everything)"""


class LoggerSettings(BaseModel):
    """
    Logger settings for the MCP Agent application.
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


class Settings(BaseSettings):
    """
    Settings class for the MCP Agent application.
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

    execution_engine: Literal["asyncio", "temporal"] = "asyncio"
    """Execution engine for the MCP Agent application"""

    default_model: str | None = "haiku"
    """
    Default model for agents. Format is provider.model_name.<reasoning_effort>, for example openai.o3-mini.low
    Aliases are provided for common models e.g. sonnet, haiku, gpt-4o, o3-mini etc.
    """
    temporal: TemporalSettings | None = None
    """Settings for Temporal workflow orchestration"""

    anthropic: AnthropicSettings | None = None
    """Settings for using Anthropic models in the MCP Agent application"""

    cohere: CohereSettings | None = None
    """Settings for using Cohere models in the MCP Agent application"""

    openai: OpenAISettings | None = None
    """Settings for using OpenAI models in the MCP Agent application"""

    deepseek: DeepSeekSettings | None = None
    """Settings for using DeepSeek models in the MCP Agent application"""

    otel: OpenTelemetrySettings | None = OpenTelemetrySettings()
    """OpenTelemetry logging settings for the MCP Agent application"""

    logger: LoggerSettings | None = LoggerSettings()
    """Logger settings for the MCP Agent application"""

    usage_telemetry: UsageTelemetrySettings | None = UsageTelemetrySettings()
    """Usage tracking settings for the MCP Agent application"""

    @classmethod
    def find_config(cls) -> Path | None:
        """Find the config file in the current directory or parent directories."""
        current_dir = Path.cwd()

        # Check current directory and parent directories
        while current_dir != current_dir.parent:
            for filename in [
                "mcp-agent.config.yaml",
                "mcp_agent.config.yaml",
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
    if _settings:
        return _settings

    config_file = Path(config_path) if config_path else Settings.find_config()
    merged_settings = {}

    if config_file:
        if not config_file.exists():
            pass
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
            while current_dir != current_dir.parent and not found_secrets:
                for secrets_filename in [
                    "mcp-agent.secrets.yaml",
                    "mcp_agent.secrets.yaml",
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
                    current_dir = current_dir.parent

            _settings = Settings(**merged_settings)
            return _settings
    else:
        pass

    _settings = Settings()
    return _settings
