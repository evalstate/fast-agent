"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Importing the MCP Implementation type eagerly pulls in the full MCP server
# stack (uvicorn, Starlette, etc.) which slows down startup. We only need the
# type for annotations, so avoid the runtime import.
if TYPE_CHECKING:
    from mcp import Implementation
else:  # pragma: no cover - used only to satisfy type checkers
    Implementation = Any
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.structured_output_mode import StructuredOutputMode
from fast_agent.llm.text_verbosity import TextVerbosityLevel


class MCPServerAuthSettings(BaseModel):
    """Represents authentication configuration for a server.

    Minimal OAuth v2.1 support with sensible defaults.
    """

    # Enable OAuth for SSE/HTTP transports. If None is provided for the auth block,
    # the system will assume OAuth is enabled by default.
    oauth: bool = True

    # Local callback server configuration
    redirect_port: int = 3030
    redirect_path: str = "/callback"

    # Optional scope override. If set to a list, values are space-joined.
    scope: str | list[str] | None = None

    # Token persistence: use OS keychain via 'keyring' by default; fallback to 'memory'.
    persist: Literal["keyring", "memory"] = "keyring"

    # Client ID Metadata Document (CIMD) URL.
    # When provided and the server advertises client_id_metadata_document_supported=true,
    # this URL will be used as the client_id instead of performing dynamic client registration.
    # Must be a valid HTTPS URL with a non-root pathname (e.g., https://example.com/client.json).
    # See: https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
    client_metadata_url: str | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("client_metadata_url", mode="after")
    @classmethod
    def _validate_client_metadata_url(cls, v: str | None) -> str | None:
        """Validate that client_metadata_url is a valid HTTPS URL with a non-root path."""
        if v is None:
            return None
        from urllib.parse import urlparse

        try:
            parsed = urlparse(v)
            if parsed.scheme != "https":
                raise ValueError("client_metadata_url must use HTTPS scheme")
            if parsed.path in ("", "/"):
                raise ValueError("client_metadata_url must have a non-root pathname")
            return v
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Invalid client_metadata_url: {e}")


class MCPSamplingSettings(BaseModel):
    model: str = "gpt-5-mini?reasoning=low"

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPElicitationSettings(BaseModel):
    mode: Literal["forms", "auto-cancel", "none"] = "none"
    """Elicitation mode: 'forms' (default UI), 'auto-cancel', 'none' (no capability)"""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class MCPTimelineSettings(BaseModel):
    """Configuration for MCP activity timeline display."""

    steps: int = 20
    """Number of timeline buckets to render."""

    step_seconds: int = 30
    """Duration of each timeline bucket in seconds."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @staticmethod
    def _parse_duration(value: str) -> int:
        """Parse simple duration strings like '30s', '2m', '1h' into seconds."""
        pattern = re.compile(r"^\s*(\d+)\s*([smhd]?)\s*$", re.IGNORECASE)
        match = pattern.match(value)
        if not match:
            raise ValueError("Expected duration in seconds (e.g. 30, '45s', '2m').")
        amount = int(match.group(1))
        unit = match.group(2).lower()
        multiplier = {
            "": 1,
            "s": 1,
            "m": 60,
            "h": 3600,
            "d": 86400,
        }.get(unit)
        if multiplier is None:
            raise ValueError("Duration unit must be one of s, m, h, or d.")
        return amount * multiplier

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, value: Any) -> int:
        if isinstance(value, str):
            if not value.strip().isdigit():
                raise ValueError("Timeline steps must be a positive integer.")
            value = int(value.strip())
        elif isinstance(value, float):
            value = int(value)
        if not isinstance(value, int):
            raise TypeError("Timeline steps must be an integer.")
        if value <= 0:
            raise ValueError("Timeline steps must be greater than zero.")
        return value


class SkillsSettings(BaseModel):
    """Configuration for the skills directory override."""

    directories: list[str] | None = None
    marketplace_url: str | None = None
    marketplace_urls: list[str] | None = None

    model_config = ConfigDict(extra="ignore")


class CardsSettings(BaseModel):
    """Configuration for card pack registry selection."""

    marketplace_url: str | None = None
    marketplace_urls: list[str] | None = None

    model_config = ConfigDict(extra="ignore")


class ShellSettings(BaseModel):
    """Configuration for shell execution behavior."""

    timeout_seconds: int = Field(
        default=90,
        description="Maximum seconds to wait for command output before terminating",
    )
    warning_interval_seconds: int = Field(
        default=30,
        description="Show timeout warnings every N seconds",
    )
    interactive_use_pty: bool = Field(
        default=True,
        description="Use a PTY for interactive prompt shell commands",
    )
    output_display_lines: int | None = Field(
        default=5,
        description="Maximum output lines to display (None = no limit)",
    )
    show_bash: bool = Field(
        default=True,
        description="Show shell command output on the console",
    )
    output_byte_limit: int | None = Field(
        default=None,
        description="Override model-based output byte limit (None = auto)",
    )
    missing_cwd_policy: Literal["ask", "create", "warn", "error"] = Field(
        default="warn",
        description="Policy when an agent shell cwd is missing or invalid",
    )

    model_config = ConfigDict(extra="ignore")

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _coerce_timeout(cls, value: Any) -> int:
        """Support duration strings like '90s', '2m', '1h'"""
        if isinstance(value, str):
            return MCPTimelineSettings._parse_duration(value)
        return int(value)

    @field_validator("warning_interval_seconds", mode="before")
    @classmethod
    def _coerce_warning_interval(cls, value: Any) -> int:
        """Support duration strings like '30s', '1m'"""
        if isinstance(value, str):
            return MCPTimelineSettings._parse_duration(value)
        return int(value)

    @field_validator("output_display_lines", mode="before")
    @classmethod
    def _coerce_output_display_lines(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            if not stripped.isdigit():
                raise ValueError("output_display_lines must be a non-negative integer.")
            value = int(stripped)
        else:
            value = int(value)
        if value < 0:
            raise ValueError("output_display_lines must be a non-negative integer.")
        return value


class MCPRootSettings(BaseModel):
    """Represents a root directory configuration for an MCP server."""

    uri: str
    """The URI identifying the root. Must start with file://"""

    name: str | None = None
    """Optional name for the root."""

    server_uri_alias: str | None = None
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

    args: list[str] | None = None
    """The arguments for the server command."""

    read_timeout_seconds: int | None = None
    """The timeout in seconds for the session."""

    ping_interval_seconds: int = 30
    """Interval for MCP ping requests. Set <=0 to disable pinging."""

    max_missed_pings: int = 3
    """Number of consecutive missed ping responses before treating the connection as failed."""

    http_timeout_seconds: int | None = None
    """Overall HTTP timeout (seconds) for StreamableHTTP transport. Defaults to MCP SDK."""

    http_read_timeout_seconds: int | None = None
    """HTTP read timeout (seconds) for StreamableHTTP transport. Defaults to MCP SDK."""

    read_transport_sse_timeout_seconds: int = 300
    """The timeout in seconds for the server connection."""

    url: str | None = None
    """The URL for the server (e.g. for SSE/SHTTP transport)."""

    headers: dict[str, str] | None = None
    """Headers dictionary for HTTP connections"""

    auth: MCPServerAuthSettings | None = None
    """The authentication configuration for the server."""

    roots: list[MCPRootSettings] | None = None
    """Root directories this server has access to."""

    env: dict[str, str] | None = None
    """Environment variables to pass to the server process."""

    sampling: MCPSamplingSettings | None = None
    """Sampling settings for this Client/Server pair"""

    elicitation: MCPElicitationSettings | None = None
    """Elicitation settings for this Client/Server pair"""

    cwd: str | None = None
    """Working directory for the executed server command."""

    load_on_start: bool = True
    """Whether to connect to this server automatically when the agent starts."""

    include_instructions: bool = True
    """Whether to include this server's instructions in the system prompt (default: True)."""

    reconnect_on_disconnect: bool = True
    """Whether to automatically reconnect when the server session is terminated (e.g., 404).

    When enabled, if a remote StreamableHTTP server returns a 404 indicating the session
    has been terminated (e.g., due to server restart), the client will automatically
    attempt to re-initialize the connection and retry the operation.
    """

    implementation: Implementation | None = None

    experimental_session_advertise: bool = False
    """Advertise experimental session capability in client initialize payload."""

    experimental_session_advertise_version: int = 2
    """Version used when advertising experimental session capability."""

    @field_validator("experimental_session_advertise_version", mode="after")
    @classmethod
    def _validate_experimental_session_advertise_version(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("experimental_session_advertise_version must be greater than zero.")
        return value

    @field_validator("max_missed_pings", mode="before")
    @classmethod
    def _coerce_max_missed_pings(cls, value: Any) -> int:
        if isinstance(value, str):
            value = int(value.strip())
        value = int(value)
        if value <= 0:
            raise ValueError("max_missed_pings must be greater than zero.")
        return value

    @model_validator(mode="before")
    @classmethod
    def validate_transport_inference(cls, values):
        """Automatically infer transport type based on url/command presence."""
        import warnings

        if isinstance(values, dict):
            # Check if transport was explicitly provided in the input
            transport_explicit = "transport" in values
            url = values.get("url")
            command = values.get("command")

            # Only infer if transport was not explicitly set
            if not transport_explicit:
                # Check if we have both url and command specified
                has_url = url is not None and str(url).strip()
                has_command = command is not None and str(command).strip()

                if has_url and has_command:
                    warnings.warn(
                        f"MCP Server config has both 'url' ({url}) and 'command' ({command}) specified. "
                        "Preferring HTTP transport and ignoring command.",
                        UserWarning,
                        stacklevel=4,
                    )
                    values["transport"] = "http"
                    values["command"] = None  # Clear command to avoid confusion
                elif has_url and not has_command:
                    values["transport"] = "http"
                elif has_command and not has_url:
                    # Keep default "stdio" for command-based servers
                    values["transport"] = "stdio"
                # If neither url nor command is specified, keep default "stdio"

        return values


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""

    servers: dict[str, MCPServerSettings] = {}
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @classmethod
    def _normalize_target_list_entries(
        cls,
        raw_targets: Any,
    ) -> dict[str, dict[str, Any]]:
        if raw_targets is None:
            return {}

        if not isinstance(raw_targets, list):
            raise ValueError("`mcp.targets` must be a list")

        from fast_agent.mcp.connect_targets import resolve_target_entry

        normalized_targets: dict[str, dict[str, Any]] = {}
        for index, raw_entry in enumerate(raw_targets):
            entry: dict[str, Any]
            if isinstance(raw_entry, str):
                entry = {"target": raw_entry}
            elif isinstance(raw_entry, dict):
                entry = raw_entry
            else:
                raise ValueError(f"`mcp.targets[{index}]` must be a string or mapping")

            target_value = entry.get("target")
            source_path = f"mcp.targets[{index}].target"
            if not isinstance(target_value, str) or not target_value.strip():
                raise ValueError(f"`{source_path}` must be a non-empty string")

            name_value = entry.get("name")
            if name_value is not None and (not isinstance(name_value, str) or not name_value.strip()):
                raise ValueError(f"`mcp.targets[{index}].name` must be a non-empty string")

            overrides = {key: value for key, value in entry.items() if key != "target"}
            resolved_name, resolved_settings = resolve_target_entry(
                target=target_value,
                default_name=name_value if isinstance(name_value, str) else None,
                overrides=overrides,
                source_path=source_path,
            )

            resolved_payload = resolved_settings.model_dump(mode="python")
            existing_payload = normalized_targets.get(resolved_name)
            if existing_payload is not None and existing_payload != resolved_payload:
                raise ValueError(
                    " ".join(
                        [
                            f"`mcp.targets[{index}]` resolves to duplicate server name '{resolved_name}'",
                            "with different settings.",
                            "Set an explicit unique `name`.",
                        ]
                    )
                )

            normalized_targets[resolved_name] = resolved_payload

        return normalized_targets

    @classmethod
    def _normalize_server_map_entries(
        cls,
        raw_servers: Any,
    ) -> dict[Any, Any] | None:
        if raw_servers is None:
            return {}
        if not isinstance(raw_servers, dict):
            return None

        from fast_agent.mcp.connect_targets import resolve_target_entry

        normalized_servers: dict[Any, Any] = {}
        for server_key, raw_entry in raw_servers.items():
            if not isinstance(raw_entry, dict) or "target" not in raw_entry:
                normalized_servers[server_key] = raw_entry
                continue

            source_name = str(server_key)
            source_path = f"mcp.servers.{source_name}.target"
            target_value = raw_entry.get("target")
            if not isinstance(target_value, str) or not target_value.strip():
                raise ValueError(f"`{source_path}` must be a non-empty string")

            overrides = {key: value for key, value in raw_entry.items() if key != "target"}
            _resolved_name, resolved_settings = resolve_target_entry(
                target=target_value,
                default_name=source_name,
                overrides=overrides,
                source_path=source_path,
            )
            normalized_servers[server_key] = resolved_settings.model_dump(mode="python")

        return normalized_servers

    @model_validator(mode="before")
    @classmethod
    def _normalize_server_targets(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        raw_servers = values.get("servers")
        raw_targets = values.get("targets")

        normalized_targets = cls._normalize_target_list_entries(raw_targets)
        normalized_servers = cls._normalize_server_map_entries(raw_servers)

        if normalized_servers is None:
            return values

        merged_servers: dict[Any, Any] = dict(normalized_targets)
        merged_servers.update(normalized_servers)

        normalized_values = dict(values)
        normalized_values["servers"] = merged_servers
        normalized_values.pop("targets", None)
        return normalized_values


_DOMAIN_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _validate_domain_list(domains: list[str] | None) -> list[str] | None:
    if domains is None:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_domain in domains:
        domain = (raw_domain or "").strip().lower()
        if not domain:
            raise ValueError("Domain entries must be non-empty strings.")
        if "://" in domain:
            raise ValueError("Domain entries must not include URL schemes.")
        if any(char in domain for char in ("/", "?", "#", "@", " ")):
            raise ValueError("Domain entries must be hostnames without paths or credentials.")

        wildcard = False
        if domain.startswith("*."):
            wildcard = True
            domain = domain[2:]

        labels = [label for label in domain.split(".") if label]
        if len(labels) < 2:
            raise ValueError("Domain entries must include a TLD (e.g., example.com).")
        if not all(_DOMAIN_LABEL_RE.match(label) for label in labels):
            raise ValueError(f"Invalid domain entry: '{raw_domain}'.")

        normalized_domain = f"*.{domain}" if wildcard else domain
        if normalized_domain not in seen:
            seen.add(normalized_domain)
            normalized.append(normalized_domain)

    return normalized


class AnthropicUserLocationSettings(BaseModel):
    """Approximate user location for Anthropic web search tool requests."""

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None


class AnthropicWebSearchSettings(BaseModel):
    """Anthropic built-in web_search server tool settings."""

    enabled: bool = False
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    user_location: AnthropicUserLocationSettings | None = None

    @field_validator("max_uses")
    @classmethod
    def _validate_max_uses(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_uses must be greater than zero when provided.")
        return value

    @field_validator("allowed_domains", "blocked_domains")
    @classmethod
    def _validate_domains(cls, value: list[str] | None) -> list[str] | None:
        return _validate_domain_list(value)

    @model_validator(mode="after")
    def _validate_domain_xor(self) -> "AnthropicWebSearchSettings":
        if self.allowed_domains and self.blocked_domains:
            raise ValueError("allowed_domains and blocked_domains are mutually exclusive.")
        return self


class AnthropicWebFetchSettings(BaseModel):
    """Anthropic built-in web_fetch server tool settings."""

    enabled: bool = False
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations_enabled: bool = False
    max_content_tokens: int | None = None

    @field_validator("max_uses", "max_content_tokens")
    @classmethod
    def _validate_positive_limits(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("Tool limits must be greater than zero when provided.")
        return value

    @field_validator("allowed_domains", "blocked_domains")
    @classmethod
    def _validate_domains(cls, value: list[str] | None) -> list[str] | None:
        return _validate_domain_list(value)

    @model_validator(mode="after")
    def _validate_domain_xor(self) -> "AnthropicWebFetchSettings":
        if self.allowed_domains and self.blocked_domains:
            raise ValueError("allowed_domains and blocked_domains are mutually exclusive.")
        return self


class AnthropicSettings(BaseModel):
    """Settings for using Anthropic models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="Anthropic API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    default_model: str | None = Field(
        default=None,
        description="Default model when Anthropic provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None, description="Custom headers to pass with every request"
    )
    cache_mode: Literal["off", "prompt", "auto"] = Field(
        default="auto",
        description="Caching mode: off (disabled), prompt (cache tools+system), auto (same as prompt)",
    )
    cache_ttl: Literal["5m", "1h"] = Field(
        default="5m",
        description="Cache TTL: 5m (standard) or 1h (extended, additional cost)",
    )
    reasoning: ReasoningEffortSetting | str | int | bool | None = Field(
        default=None,
        description=(
            "Reasoning setting. Supports effort strings (for adaptive models), budget tokens "
            "(int), or toggle (bool). Use 0 or false to disable."
        ),
    )
    structured_output_mode: StructuredOutputMode | Literal["auto"] = Field(
        default="auto",
        description="Structured output mode: auto, json, or tool_use",
    )
    web_search: AnthropicWebSearchSettings = Field(default_factory=AnthropicWebSearchSettings)
    web_fetch: AnthropicWebFetchSettings = Field(default_factory=AnthropicWebFetchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAIUserLocationSettings(BaseModel):
    """Approximate user location for OpenAI web search tool requests."""

    type: Literal["approximate"] = "approximate"
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None


class OpenAIWebSearchSettings(BaseModel):
    """OpenAI Responses web_search tool settings."""

    enabled: bool = False
    tool_type: Literal["web_search", "web_search_preview"] = "web_search"
    search_context_size: Literal["low", "medium", "high"] | None = None
    allowed_domains: list[str] | None = None
    user_location: OpenAIUserLocationSettings | None = None
    external_web_access: bool | None = None

    @field_validator("allowed_domains")
    @classmethod
    def _validate_allowed_domains(cls, value: list[str] | None) -> list[str] | None:
        normalized = _validate_domain_list(value)
        if normalized is not None and len(normalized) > 100:
            raise ValueError("allowed_domains supports at most 100 domains.")
        return normalized


class OpenAISettings(BaseModel):
    """Settings for using OpenAI models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    default_model: str | None = Field(
        default=None,
        description="Default model when OpenAI provider is selected without an explicit model",
    )
    reasoning: ReasoningEffortSetting | str | int | bool | None = Field(
        default=None,
        description="Unified reasoning setting (effort level or budget)",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="medium",
        description="Default reasoning effort: minimal, low, medium, high",
    )
    text_verbosity: TextVerbosityLevel = Field(
        default="medium",
        description="Text verbosity level: low, medium, high",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )
    transport: Literal["sse", "websocket", "auto"] = Field(
        default="sse",
        description="Responses transport mode: sse (default), websocket, or auto fallback.",
    )
    web_search: OpenAIWebSearchSettings = Field(default_factory=OpenAIWebSearchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenResponsesSettings(BaseModel):
    """Settings for using Open Responses models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="Open Responses API key")
    base_url: str | None = Field(default=None, description="Open Responses endpoint URL")
    default_model: str | None = Field(
        default=None,
        description=(
            "Default model when Open Responses provider is selected without an explicit model"
        ),
    )
    reasoning: ReasoningEffortSetting | str | int | bool | None = Field(
        default=None,
        description="Unified reasoning setting (effort level or budget)",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="medium",
        description="Default reasoning effort: minimal, low, medium, high",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )
    transport: Literal["sse", "websocket", "auto"] = Field(
        default="sse",
        description="Responses transport mode: sse (default), websocket, or auto fallback.",
    )
    web_search: OpenAIWebSearchSettings = Field(default_factory=OpenAIWebSearchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class CodexResponsesSettings(BaseModel):
    """Settings for using Codex Responses via ChatGPT OAuth tokens."""

    api_key: str | None = Field(default=None, description="Codex Responses API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    default_model: str | None = Field(
        default=None,
        description=(
            "Default model when Codex Responses provider is selected without an explicit model"
        ),
    )
    text_verbosity: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Text verbosity level: low, medium, high",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )
    transport: Literal["sse", "websocket", "auto"] = Field(
        default="sse",
        description="Responses transport mode: sse (default), websocket, or auto fallback.",
    )
    web_search: OpenAIWebSearchSettings = Field(default_factory=OpenAIWebSearchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class DeepSeekSettings(BaseModel):
    """Settings for using DeepSeek models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="DeepSeek API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    default_model: str | None = Field(
        default=None,
        description="Default model when DeepSeek provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GoogleSettings(BaseModel):
    """Settings for using Google models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="Google API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    default_model: str | None = Field(
        default=None,
        description="Default model when Google provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class XAISettings(BaseModel):
    """Settings for using xAI Grok models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="xAI API key")
    base_url: str | None = Field(
        default="https://api.x.ai/v1",
        description="xAI API endpoint (default: https://api.x.ai/v1)",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model when xAI provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GenericSettings(BaseModel):
    """Settings for using generic OpenAI-compatible models (e.g., Ollama)."""

    api_key: str | None = Field(default=None, description="API key (default: 'ollama' for Ollama)")
    base_url: str | None = Field(
        default=None,
        description="API endpoint (default: http://localhost:11434/v1 for Ollama)",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model when generic provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenRouterSettings(BaseModel):
    """Settings for using OpenRouter models via its OpenAI-compatible API."""

    api_key: str | None = Field(default=None, description="OpenRouter API key")
    base_url: str | None = Field(
        default=None,
        description="Override API endpoint (default: https://openrouter.ai/api/v1)",
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Default model when OpenRouter provider is selected without an explicit model"
        ),
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class AzureSettings(BaseModel):
    """Settings for using Azure OpenAI Service in the fast-agent application."""

    api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    resource_name: str | None = Field(
        default=None,
        description="Azure resource name (do not use with base_url)",
    )
    azure_deployment: str | None = Field(
        default=None,
        description="Azure deployment name (required)",
    )
    api_version: str | None = Field(default=None, description="API version (e.g., 2023-05-15)")
    base_url: str | None = Field(
        default=None,
        description="Full endpoint URL (do not use with resource_name)",
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Default deployment/model when Azure provider is selected without an explicit model"
        ),
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class GroqSettings(BaseModel):
    """Settings for using Groq models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="Groq API key")
    base_url: str | None = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq API endpoint",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model when Groq provider is selected without an explicit model",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenTelemetrySettings(BaseModel):
    """OpenTelemetry settings for the fast-agent application."""

    enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing")
    service_name: str = Field(default="fast-agent", description="OTEL service name")
    otlp_endpoint: str = Field(
        default="http://localhost:4318/v1/traces",
        description="OTLP endpoint for tracing",
    )
    console_debug: bool = Field(default=False, description="Log spans to console")
    sample_rate: float = Field(
        default=1.0,
        description="Sample rate for tracing (1.0 = sample everything)",
    )


class TensorZeroSettings(BaseModel):
    """Settings for using TensorZero LLM gateway."""

    base_url: str | None = Field(
        default=None,
        description="TensorZero endpoint (default: http://localhost:3000)",
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Default function name when TensorZero provider is selected without an explicit model"
        ),
    )
    api_key: str | None = Field(default=None, description="TensorZero API key (if required)")
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class BedrockSettings(BaseModel):
    """Settings for using AWS Bedrock models in the fast-agent application."""

    region: str | None = Field(default=None, description="AWS region for Bedrock (e.g., us-east-1)")
    profile: str | None = Field(
        default=None,
        description="AWS profile for authentication (default: 'default')",
    )
    default_model: str | None = Field(
        default=None,
        description="Default model when Bedrock provider is selected without an explicit model",
    )
    reasoning: ReasoningEffortSetting | str | int | bool | None = Field(
        default=None,
        description="Unified reasoning setting (effort level or budget)",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="minimal",
        description="Default reasoning effort: minimal, low, medium, high",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class HuggingFaceSettings(BaseModel):
    """Settings for HuggingFace Inference Providers."""

    api_key: str | None = Field(default=None, description="HuggingFace token (HF_TOKEN)")
    base_url: str | None = Field(
        default=None,
        description="Override router endpoint (default: https://router.huggingface.co/v1)",
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Default model when HuggingFace provider is selected without an explicit model"
        ),
    )
    default_provider: str | None = Field(
        default=None,
        description="Default inference provider (e.g., groq, fireworks-ai, cerebras)",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )

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

    enable_prompt_marks: bool = True
    """Emit OSC 133 prompt marks for terminals that support scrollbar markers."""
    streaming: Literal["markdown", "plain", "none"] = "markdown"
    """Streaming renderer for assistant responses"""

    message_style: Literal["classic", "a3"] = "a3"
    """Chat message layout style for console output."""


def find_fastagent_config_files(start_path: Path) -> tuple[Path | None, Path | None]:
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


def resolve_config_search_root(
    start_path: Path,
    *,
    env_dir: str | Path | None = None,
) -> Path:
    """Resolve the base path for discovering config and secrets files.

    If env_dir is provided (or ENVIRONMENT_DIR is set), search from there instead
    of the current working directory.
    """
    base = start_path.resolve()
    override = env_dir if env_dir is not None else os.getenv("ENVIRONMENT_DIR")
    if not override:
        return base

    root = Path(override).expanduser()
    if not root.is_absolute():
        root = (base / root).resolve()
    return root


def resolve_env_vars(config_item: Any) -> Any:
    """Recursively resolve environment variables in config data."""
    if isinstance(config_item, dict):
        return {k: resolve_env_vars(v) for k, v in config_item.items()}
    if isinstance(config_item, list):
        return [resolve_env_vars(i) for i in config_item]
    if isinstance(config_item, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replace_match(match: re.Match[str]) -> str:
            var_name_with_default = match.group(1)
            if ":" in var_name_with_default:
                var_name, default_value = var_name_with_default.split(":", 1)
                return os.getenv(var_name, default_value)

            var_name = var_name_with_default
            env_value = os.getenv(var_name)
            if env_value is None:
                return match.group(0)
            return env_value

        return pattern.sub(replace_match, config_item)

    return config_item


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, preserving nested structures."""
    merged = base.copy()
    for key, value in update.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            existing = merged[key]
            if isinstance(existing, dict):
                merged[key] = deep_merge(existing, value)
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    import yaml  # pylint: disable=C0415

    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        return {}
    return resolve_env_vars(payload)


def find_project_config_file(start_path: Path) -> Path | None:
    """Find project-level ``fastagent.config.yaml`` from ``start_path`` upward."""
    current = start_path.resolve()
    while current != current.parent:
        candidate = current / "fastagent.config.yaml"
        if candidate.exists():
            return candidate
        current = current.parent
    return None


def resolve_environment_config_file(
    start_path: Path,
    *,
    env_dir: str | Path | None = None,
) -> Path:
    """Return the env overlay config path: ``<env>/fastagent.config.yaml``."""
    base = start_path.resolve()
    override = env_dir if env_dir is not None else os.getenv("ENVIRONMENT_DIR")

    if override:
        env_root = Path(override).expanduser()
        if not env_root.is_absolute():
            env_root = (base / env_root).resolve()
    else:
        env_root = (base / DEFAULT_ENVIRONMENT_DIR).resolve()

    return env_root / "fastagent.config.yaml"


def load_layered_model_settings(
    *,
    start_path: Path,
    env_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load layered model settings from project + env config.

    Precedence: project config < env config.
    ``model_aliases`` uses deep-merge semantics, while ``default_model`` uses
    scalar replacement semantics.
    """
    layered: dict[str, Any] = {}

    for config_path in (
        find_project_config_file(start_path),
        resolve_environment_config_file(start_path, env_dir=env_dir),
    ):
        payload = load_yaml_mapping(config_path)
        if not payload:
            continue

        if "default_model" in payload:
            layered["default_model"] = payload["default_model"]

        if "model_aliases" in payload:
            aliases_payload = payload["model_aliases"]
            existing_aliases = layered.get("model_aliases")
            if isinstance(existing_aliases, dict) and isinstance(aliases_payload, dict):
                layered["model_aliases"] = deep_merge(existing_aliases, aliases_payload)
            else:
                layered["model_aliases"] = aliases_payload

    return layered


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

    environment_dir: str | None = None
    """Base directory for fast-agent runtime data (defaults to .fast-agent)."""

    default_model: str | None = None
    """
    Default model for agents. Format is provider.model_name.<reasoning_effort> or provider.model?reasoning=<value>,
    for example openai.o3-mini.low or openai.o3-mini?reasoning=high.
    Aliases are provided for common models e.g. sonnet, haiku, gpt-4.1, o3-mini etc.
    If not set, falls back to FAST_AGENT_MODEL env var, then to "gpt-5-mini.low".
    """

    model_aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    """Model aliases grouped by namespace (e.g. $system.default)."""

    auto_sampling: bool = True
    """Enable automatic sampling model selection if not explicitly configured"""

    session_history: bool = True
    """Persist session history in the environment sessions folder (default: True)."""

    session_history_window: int = 20
    """Maximum number of sessions to keep in the rolling window (default: 20)."""

    anthropic: AnthropicSettings | None = None
    """Settings for using Anthropic models in the fast-agent application"""

    otel: OpenTelemetrySettings | None = OpenTelemetrySettings()
    """OpenTelemetry logging settings for the fast-agent application"""

    openai: OpenAISettings | None = None
    """Settings for using OpenAI models in the fast-agent application"""

    responses: OpenAISettings | None = None
    """Settings for using OpenAI Responses models in the fast-agent application"""

    openresponses: OpenResponsesSettings | None = None
    """Settings for using Open Responses models in the fast-agent application"""

    codexresponses: CodexResponsesSettings | None = None
    """Settings for using Codex Responses models in the fast-agent application"""

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

    tensorzero: TensorZeroSettings | None = None
    """Settings for using TensorZero inference gateway"""

    azure: AzureSettings | None = None
    """Settings for using Azure OpenAI Service in the fast-agent application"""

    aliyun: OpenAISettings | None = None
    """Settings for using Aliyun OpenAI Service in the fast-agent application"""

    bedrock: BedrockSettings | None = None
    """Settings for using AWS Bedrock models in the fast-agent application"""

    hf: HuggingFaceSettings | None = None
    """Settings for HuggingFace authentication (used for MCP connections)"""

    groq: GroqSettings | None = None
    """Settings for using the Groq provider in the fast-agent application"""

    logger: LoggerSettings = LoggerSettings()
    """Logger settings for the fast-agent application"""

    # MCP UI integration mode for handling ui:// embedded resources from MCP tool results
    mcp_ui_mode: Literal["disabled", "enabled", "auto"] = "enabled"
    """Controls handling of MCP UI embedded resources:
    - "disabled": Do not process ui:// resources
    - "enabled": Always extract ui:// resources into message channels (default)
    - "auto": Extract and automatically open ui:// resources.
    """

    # Output directory for MCP-UI generated HTML files (relative to CWD if not absolute)
    mcp_ui_output_dir: str = ".fast-agent/ui"
    """Directory where MCP-UI HTML files are written. Relative paths are resolved from CWD."""

    mcp_timeline: MCPTimelineSettings = MCPTimelineSettings()
    """Display settings for MCP activity timelines."""

    skills: SkillsSettings = SkillsSettings()
    """Local skills discovery and selection settings."""

    cards: CardsSettings = CardsSettings()
    """Card pack registry selection settings."""

    shell_execution: ShellSettings = ShellSettings()
    """Shell execution timeout and warning settings."""

    llm_retries: int = 1
    """
    Number of times to retry transient LLM API errors.
    Defaults to 1; can be overridden via config or FAST_AGENT_RETRIES env.
    """

    _config_file: str | None = PrivateAttr(default=None)
    _secrets_file: str | None = PrivateAttr(default=None)

    @field_validator("model_aliases")
    @classmethod
    def _validate_model_aliases(
        cls,
        value: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate model alias namespace/key names and normalize values."""
        valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
        normalized: dict[str, dict[str, str]] = {}

        for namespace, entries in value.items():
            if not valid_name.fullmatch(namespace):
                raise ValueError(
                    "model_aliases namespace names must match [A-Za-z_][A-Za-z0-9_-]*"
                )

            normalized_entries: dict[str, str] = {}
            for key, model in entries.items():
                if not valid_name.fullmatch(key):
                    raise ValueError(
                        "model_aliases keys must match [A-Za-z_][A-Za-z0-9_-]*"
                    )

                model_value = model.strip()
                if not model_value:
                    raise ValueError(
                        f"model_aliases.{namespace}.{key} must be a non-empty model string"
                    )
                normalized_entries[key] = model_value

            normalized[namespace] = normalized_entries

        return normalized

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


def get_settings(config_path: str | os.PathLike[str] | None = None) -> Settings:
    """Get settings instance, automatically loading from config file if available."""

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
    config_file: Path | None
    secrets_file: Path | None
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
        search_root = resolve_config_search_root(Path.cwd())
        config_file, secrets_file = find_fastagent_config_files(search_root)

    merged_settings: dict[str, Any] = {}

    # Load main config if it exists
    if config_file and config_file.exists():
        merged_settings = load_yaml_mapping(config_file)
    elif config_file and not config_file.exists():
        print(f"Warning: Specified config file does not exist: {config_file}")

    # Model settings layering is deterministic and independent of whichever
    # single config file happened to be discovered first.
    if config_path is None:
        layered_model_settings = load_layered_model_settings(
            start_path=Path.cwd(),
            env_dir=os.getenv("ENVIRONMENT_DIR"),
        )
        merged_settings = deep_merge(merged_settings, layered_model_settings)

    # Load secrets file if found (regardless of whether config file exists)
    if secrets_file and secrets_file.exists():
        merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_file))

    legacy_keys: list[str] = []
    anthropic_settings = merged_settings.get("anthropic")
    if isinstance(anthropic_settings, dict):
        for key in ("thinking_enabled", "thinking_budget_tokens"):
            if key in anthropic_settings:
                legacy_keys.append(key)
    legacy_env = [
        key
        for key in ("ANTHROPIC__THINKING_ENABLED", "ANTHROPIC__THINKING_BUDGET_TOKENS")
        if os.getenv(key) is not None
    ]
    if legacy_keys or legacy_env:
        warnings.warn(
            "Anthropic config keys 'thinking_enabled'/'thinking_budget_tokens' are deprecated and "
            "ignored. Use 'anthropic.reasoning' instead.",
            UserWarning,
            stacklevel=3,
        )

    _settings = Settings(**merged_settings)
    _settings._config_file = str(config_file) if config_file else None
    _settings._secrets_file = str(secrets_file) if secrets_file else None
    return _settings


def update_global_settings(settings: Settings) -> None:
    """Update the global settings instance.

    This is used to propagate CLI overrides (like --skills-dir) into the
    global settings so that functions like resolve_skill_directories()
    work correctly without needing to pass settings around explicitly.
    """
    global _settings
    _settings = settings
