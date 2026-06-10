"""
Reading settings from environment variables and providing a settings object
for the application configuration.
"""

import os
import re
import warnings
from dataclasses import dataclass
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

from fast_agent.command_actions import PluginCommandActionSpec, parse_plugin_command_action_specs
from fast_agent.core.exceptions import ConfigFileError
from fast_agent.home import (
    ConfigDiscoveryResult,
    discover_config_files,
    find_config_in_directory,
    resolve_fast_agent_home,
)
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.llm.structured_output_mode import StructuredOutputMode
from fast_agent.llm.task_budget import parse_task_budget_tokens, validate_task_budget_tokens
from fast_agent.llm.text_verbosity import TextVerbosityLevel
from fast_agent.mcp.provider_management import (
    normalize_access_token,
    normalize_client_managed_url_server,
    normalize_connector_id,
    normalize_provider_managed_url_server,
    validate_provider_managed_server_settings,
)
from fast_agent.mcp.ui_modes import McpUIMode
from fast_agent.types.streaming import StreamingMode
from fast_agent.utils.action_normalization import (
    FALSE_ACTION_ALIASES,
    TRUE_ACTION_ALIASES,
    normalize_action_token,
    on_off_label,
)
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.numeric import int_or_none
from fast_agent.utils.text import strip_casefold, strip_str_to_none
from fast_agent.utils.transports import McpClientTransport
from fast_agent.utils.type_narrowing import is_str_object_dict

type TerminalImageSize = int | Literal["auto"] | str | None
type ShellWriteTextFileMode = Literal["auto", "on", "off", "apply_patch"]

SHELL_WRITE_TEXT_FILE_MODES: tuple[ShellWriteTextFileMode, ...] = (
    "auto",
    "on",
    "off",
    "apply_patch",
)
SHELL_WRITE_TEXT_FILE_MODE_HELP = "|".join(SHELL_WRITE_TEXT_FILE_MODES)
_SHELL_WRITE_TEXT_FILE_MODE_ALIASES: dict[str, ShellWriteTextFileMode] = {
    **{mode: mode for mode in SHELL_WRITE_TEXT_FILE_MODES},
    **{alias: "on" for alias in TRUE_ACTION_ALIASES},
    **{alias: "off" for alias in FALSE_ACTION_ALIASES},
}


def _normalized_config_token(value: str) -> tuple[str, str]:
    stripped = value.strip()
    return stripped, normalize_action_token(stripped)


def normalize_shell_write_text_file_mode(value: Any) -> ShellWriteTextFileMode | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return on_off_label(value)

    if isinstance(value, str):
        _, normalized = _normalized_config_token(value)
        return _SHELL_WRITE_TEXT_FILE_MODE_ALIASES.get(normalized)

    return None


def _has_nonblank(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _reject_bool_integer_field(value: Any, *, field_name: str) -> Any:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")
    return value


def _reject_bool_number_field(value: Any, *, field_name: str) -> Any:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number.")
    return value


class MCPServerAuthSettings(BaseModel):
    """Represents authentication configuration for a server.

    Minimal OAuth v2.1 support with sensible defaults.
    """

    # Enable OAuth for SSE/HTTP transports when the auth block is present.
    # If the auth block is omitted entirely, fast-agent starts unauthenticated
    # and escalates to OAuth on a 401 challenge.
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

    @field_validator("redirect_port", mode="before")
    @classmethod
    def _reject_bool_redirect_port(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="redirect_port")

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
            raise ValueError(f"Invalid client_metadata_url: {e}") from e


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
        unit = strip_casefold(match.group(2))
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
        if isinstance(value, bool):
            raise TypeError("Timeline steps must be an integer.")
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


class PluginsSettings(BaseModel):
    """Configuration for command plugin discovery and marketplace selection."""

    enabled: list[str] = Field(default_factory=list)
    marketplace_url: str | None = None
    marketplace_urls: list[str] | None = None
    config: dict[str, dict[str, Any]] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


class ShellSettings(BaseModel):
    """Configuration for shell execution behavior."""

    timeout_seconds: int = Field(
        default=90,
        description="Maximum seconds without command output before terminating",
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
        description=(
            "Maximum shell output lines to display "
            "(head/tail with an ellipsis when truncated; None = no limit)"
        ),
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
    prefer_local_shell: bool = Field(
        default=False,
        description=(
            "In ACP mode, keep the local fast-agent shell runtime instead of replacing it "
            "with the ACP client's terminal runtime when the client advertises terminal support"
        ),
    )
    enable_read_text_file: bool = Field(
        default=True,
        description=(
            "Expose a local read_text_file tool (ACP-compatible signature) "
            "when shell runtime is enabled"
        ),
    )
    enable_attach_media: Literal["auto", "on", "off"] = Field(
        default="auto",
        description=(
            "Expose attach_media when shell runtime is enabled. 'auto' exposes it only "
            "for models with non-text attachment support; 'on' exposes it with call-time "
            "validation; 'off' disables it."
        ),
    )
    write_text_file_mode: ShellWriteTextFileMode | None = Field(
        default=None,
        description=(
            "Control which local file edit tool is exposed when shell runtime is enabled "
            "('auto' uses apply_patch for Codex and GPT-5.2+ models, and exposes "
            "write_text_file plus edit_file otherwise; 'on' always exposes write_text_file "
            "plus edit_file; 'apply_patch' always exposes apply_patch; 'off' disables "
            "local file edit tools)"
        ),
    )
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce_deprecated_attach_resource(cls, data: Any) -> Any:
        if isinstance(data, dict) and "enable_attach_media" not in data:
            old_value = data.get("enable_attach_resource")
            if old_value is not None:
                data = dict(data)
                data["enable_attach_media"] = old_value
        return data

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def _coerce_timeout(cls, value: Any) -> int:
        """Support duration strings like '90s', '2m', '1h'"""
        _reject_bool_integer_field(value, field_name="timeout_seconds")
        if isinstance(value, str):
            return MCPTimelineSettings._parse_duration(value)
        return int(value)

    @field_validator("warning_interval_seconds", mode="before")
    @classmethod
    def _coerce_warning_interval(cls, value: Any) -> int:
        """Support duration strings like '30s', '1m'"""
        _reject_bool_integer_field(value, field_name="warning_interval_seconds")
        if isinstance(value, str):
            return MCPTimelineSettings._parse_duration(value)
        return int(value)

    @field_validator("output_display_lines", mode="before")
    @classmethod
    def _coerce_output_display_lines(cls, value: Any) -> int | None:
        if value is None:
            return None
        _reject_bool_integer_field(value, field_name="output_display_lines")
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

    @field_validator("output_byte_limit", mode="before")
    @classmethod
    def _coerce_output_byte_limit(cls, value: Any) -> int | None:
        if value is None:
            return None
        _reject_bool_integer_field(value, field_name="output_byte_limit")
        if isinstance(value, str):
            return int(value.strip())
        return int(value)

    @field_validator("write_text_file_mode", mode="before")
    @classmethod
    def _coerce_write_text_file_mode(cls, value: Any) -> ShellWriteTextFileMode | None:
        normalized = normalize_shell_write_text_file_mode(value)
        if normalized is not None or value is None:
            return normalized
        raise ValueError("write_text_file_mode must be one of: auto, on, off, apply_patch")


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

    management: Literal["client", "provider"] = "client"
    """Whether fast-agent connects locally or delegates MCP execution to the provider."""

    transport: McpClientTransport = "stdio"
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

    connector_id: str | None = None
    """OpenAI Responses provider-managed connector identifier."""

    headers: dict[str, str] | None = None
    """Headers dictionary for HTTP connections"""

    access_token: str | None = None
    """Provider-neutral bearer token for local URL servers or provider-managed MCP."""

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

    defer_loading: bool = False
    """Provider-managed OpenAI Responses hint to defer remote tool loading."""

    @field_validator(
        "read_timeout_seconds",
        "ping_interval_seconds",
        "http_timeout_seconds",
        "http_read_timeout_seconds",
        "read_transport_sse_timeout_seconds",
        mode="before",
    )
    @classmethod
    def _reject_bool_timeout_value(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="MCP timeout fields")

    @field_validator("max_missed_pings", mode="before")
    @classmethod
    def _coerce_max_missed_pings(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError("max_missed_pings must be an integer.")
        if isinstance(value, str):
            value = int(value.strip())
        value = int(value)
        if value <= 0:
            raise ValueError("max_missed_pings must be greater than zero.")
        return value

    @field_validator("access_token", mode="before")
    @classmethod
    def _normalize_access_token(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("access_token must be a string")
        return normalize_access_token(value)

    @field_validator("connector_id", mode="before")
    @classmethod
    def _normalize_connector_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("connector_id must be a string")
        return normalize_connector_id(value)

    @model_validator(mode="before")
    @classmethod
    def validate_transport_inference(cls, values):
        """Automatically infer transport type based on url/command presence."""
        if not isinstance(values, dict) or "transport" in values:
            return values

        url = values.get("url")
        command = values.get("command")
        has_url = _has_nonblank(url)
        has_command = _has_nonblank(command)

        if has_url:
            if has_command:
                warnings.warn(
                    f"MCP Server config has both 'url' ({url}) and 'command' ({command}) specified. "
                    "Preferring HTTP transport and ignoring command.",
                    UserWarning,
                    stacklevel=4,
                )
                values["command"] = None
            values["transport"] = "http"
        elif has_command:
            values["transport"] = "stdio"

        return values

    @model_validator(mode="after")
    def _normalize_management_specific_settings(self) -> "MCPServerSettings":
        if self.management == "provider":
            validation = validate_provider_managed_server_settings(self)
            if not validation.has_exactly_one_source():
                raise ValueError(
                    "Provider-managed MCP servers require exactly one of url or connector_id"
                )

            if validation.invalid_fields:
                invalid_list = ", ".join(validation.invalid_fields)
                raise ValueError(
                    f"Provider-managed MCP servers have unsupported settings: {invalid_list}"
                )
            if validation.missing_connector_access_token:
                raise ValueError("Provider-managed connectors require access_token")
            if validation.has_url:
                provider_url = self.url
                if provider_url is None:
                    raise ValueError("Provider-managed URL servers require url")
                self.url = normalize_provider_managed_url_server(
                    transport=self.transport,
                    url=provider_url,
                )
            return self

        if self.connector_id is not None:
            raise ValueError("connector_id is only supported for provider-managed MCP servers")

        if self.access_token is not None and not self.url:
            raise ValueError("access_token requires a URL-based MCP server")

        if self.url:
            normalized_server = normalize_client_managed_url_server(
                transport=self.transport,
                url=self.url,
                headers=self.headers,
                access_token=self.access_token,
            )
            self.url = normalized_server.url
            self.headers = normalized_server.headers
        return self


class MCPSettings(BaseModel):
    """Configuration for all MCP servers."""

    servers: dict[str, MCPServerSettings] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @staticmethod
    def _serialize_resolved_target_settings(
        settings: MCPServerSettings,
    ) -> dict[str, Any]:
        """Serialize shorthand target settings back to an idempotent raw payload."""
        payload = settings.model_dump(mode="python")

        # resolve_target_entry() already normalizes access_token into Authorization
        # for client-managed URL servers. Strip that synthesized header here so the
        # final MCPServerSettings validation only applies the normalization once.
        if settings.management != "provider" and settings.access_token is not None:
            headers = payload.get("headers")
            if isinstance(headers, dict):
                expected_authorization = f"Bearer {settings.access_token}"
                filtered_headers = {
                    key: value
                    for key, value in headers.items()
                    if not (
                        isinstance(key, str)
                        and strip_casefold(key) == "authorization"
                        and value == expected_authorization
                    )
                }
                payload["headers"] = filtered_headers or None

        return payload

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
            if isinstance(raw_entry, str):
                entry: dict[str, object] = {"target": raw_entry}
            elif is_str_object_dict(raw_entry):
                entry = raw_entry
            else:
                raise ValueError(f"`mcp.targets[{index}]` must be a string or mapping")

            target_value = strip_str_to_none(entry.get("target"))
            source_path = f"mcp.targets[{index}].target"
            if target_value is None:
                raise ValueError(f"`{source_path}` must be a non-empty string")

            raw_name = entry.get("name")
            name_value = strip_str_to_none(raw_name)
            if raw_name is not None and name_value is None:
                raise ValueError(f"`mcp.targets[{index}].name` must be a non-empty string")

            overrides = {key: value for key, value in entry.items() if key != "target"}
            resolved_entry = resolve_target_entry(
                target=target_value,
                default_name=name_value,
                overrides=overrides,
                source_path=source_path,
            )

            resolved_payload = cls._serialize_resolved_target_settings(resolved_entry.settings)
            existing_payload = normalized_targets.get(resolved_entry.server_name)
            if existing_payload is not None and existing_payload != resolved_payload:
                raise ValueError(
                    " ".join(
                        [
                            f"`mcp.targets[{index}]` resolves to duplicate server name '{resolved_entry.server_name}'",
                            "with different settings.",
                            "Set an explicit unique `name`.",
                        ]
                    )
                )

            normalized_targets[resolved_entry.server_name] = resolved_payload

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
            target_value = strip_str_to_none(raw_entry.get("target"))
            if target_value is None:
                raise ValueError(f"`{source_path}` must be a non-empty string")

            overrides = {key: value for key, value in raw_entry.items() if key != "target"}
            resolved_entry = resolve_target_entry(
                target=target_value,
                default_name=source_name,
                overrides=overrides,
                source_path=source_path,
            )
            normalized_servers[server_key] = cls._serialize_resolved_target_settings(
                resolved_entry.settings
            )

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
    for raw_domain in domains:
        domain_value = strip_str_to_none(raw_domain)
        if domain_value is None:
            raise ValueError("Domain entries must be non-empty strings.")
        domain = strip_casefold(domain_value)
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
        normalized.append(normalized_domain)

    return unique_preserve_order(normalized)


def _validate_domain_list_limit(
    domains: list[str] | None,
    *,
    max_count: int,
    error_message: str,
) -> list[str] | None:
    normalized = _validate_domain_list(domains)
    if normalized is not None and len(normalized) > max_count:
        raise ValueError(error_message)
    return normalized


def _validate_mutually_exclusive_domain_filters(
    first: list[str] | None,
    second: list[str] | None,
    *,
    error_message: str,
) -> None:
    if first and second:
        raise ValueError(error_message)


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

    @field_validator("max_uses", mode="before")
    @classmethod
    def _reject_bool_max_uses(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="max_uses")

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
        _validate_mutually_exclusive_domain_filters(
            self.allowed_domains,
            self.blocked_domains,
            error_message="allowed_domains and blocked_domains are mutually exclusive.",
        )
        return self


class AnthropicWebFetchSettings(BaseModel):
    """Anthropic built-in web_fetch server tool settings."""

    enabled: bool = False
    max_uses: int | None = None
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None
    citations_enabled: bool = False
    max_content_tokens: int | None = None

    @field_validator("max_uses", mode="before")
    @classmethod
    def _reject_bool_max_uses(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="max_uses")

    @field_validator("max_content_tokens", mode="before")
    @classmethod
    def _reject_bool_max_content_tokens(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="max_content_tokens")

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
        _validate_mutually_exclusive_domain_filters(
            self.allowed_domains,
            self.blocked_domains,
            error_message="allowed_domains and blocked_domains are mutually exclusive.",
        )
        return self


class AnthropicVertexSettings(BaseModel):
    """Anthropic-on-Vertex configuration."""

    enabled: bool = False
    project_id: str | None = None
    location: str | None = None
    base_url: str | None = None


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
    task_budget: int | str | None = Field(
        default=None,
        description=(
            "Anthropic task budget for agentic loops. Supports raw token counts or shorthand "
            "like 20k/128k. Use off/default to disable."
        ),
    )
    structured_output_mode: StructuredOutputMode | Literal["auto"] = Field(
        default="auto",
        description="Structured output mode: auto, json, or tool_use",
    )
    vertex_ai: AnthropicVertexSettings = Field(default_factory=AnthropicVertexSettings)
    web_search: AnthropicWebSearchSettings = Field(default_factory=AnthropicWebSearchSettings)
    web_fetch: AnthropicWebFetchSettings = Field(default_factory=AnthropicWebFetchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @field_validator("task_budget", mode="before")
    @classmethod
    def _coerce_task_budget(cls, value: Any) -> int | None | Any:
        if value is None:
            return None
        if isinstance(value, int):
            return validate_task_budget_tokens(value)
        if isinstance(value, str):
            parsed = parse_task_budget_tokens(value)
            return validate_task_budget_tokens(parsed)
        return value


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
        return _validate_domain_list_limit(
            value,
            max_count=100,
            error_message="allowed_domains supports at most 100 domains.",
        )


class XAIWebSearchSettings(OpenAIWebSearchSettings):
    """xAI Responses web_search tool settings."""

    tool_type: Literal["web_search"] = "web_search"
    excluded_domains: list[str] | None = Field(
        default=None,
        description="Domains to exclude from xAI web search results (maximum 5).",
    )
    enable_image_understanding: bool | None = Field(
        default=None,
        description="Enable xAI image understanding for images encountered during web search.",
    )

    @field_validator("allowed_domains")
    @classmethod
    def _validate_xai_allowed_domains(cls, value: list[str] | None) -> list[str] | None:
        return _validate_domain_list_limit(
            value,
            max_count=5,
            error_message="xAI allowed_domains supports at most 5 domains.",
        )

    @field_validator("excluded_domains")
    @classmethod
    def _validate_excluded_domains(cls, value: list[str] | None) -> list[str] | None:
        return _validate_domain_list_limit(
            value,
            max_count=5,
            error_message="xAI excluded_domains supports at most 5 domains.",
        )

    @model_validator(mode="after")
    def _validate_domain_filters(self) -> "XAIWebSearchSettings":
        _validate_mutually_exclusive_domain_filters(
            self.allowed_domains,
            self.excluded_domains,
            error_message="xAI web_search cannot set both allowed_domains and excluded_domains.",
        )
        return self


class ResponsesProviderSettingsBase(BaseModel):
    """Shared settings for Responses-family providers."""

    default_model: str | None = Field(
        default=None,
        description="Default model when the provider is selected without an explicit model",
    )
    text_verbosity: TextVerbosityLevel = Field(
        default="medium",
        description="Text verbosity level: low, medium, high",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers for all API requests",
    )
    transport: Literal["sse", "websocket", "auto"] | None = Field(
        default=None,
        description=(
            "Responses transport mode override: sse, websocket, or auto. "
            "When unset, OpenAI Responses and Codex Responses prefer websocket "
            "with automatic SSE fallback."
        ),
    )
    service_tier: Literal["fast", "flex"] | None = Field(
        default=None,
        description="Responses service tier: fast (priority) or flex.",
    )
    web_search: OpenAIWebSearchSettings = Field(default_factory=OpenAIWebSearchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class OpenAISettings(ResponsesProviderSettingsBase):
    """Settings for using OpenAI models in the fast-agent application."""

    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    reasoning: ReasoningEffortSetting | str | int | bool | None = Field(
        default=None,
        description="Unified reasoning setting (effort level or budget)",
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = Field(
        default="medium",
        description="Default reasoning effort: minimal, low, medium, high",
    )


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
    service_tier: Literal["fast", "flex"] | None = Field(
        default=None,
        description="Responses service tier: fast (priority) or flex.",
    )
    web_search: OpenAIWebSearchSettings = Field(default_factory=OpenAIWebSearchSettings)

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class CodexResponsesSettings(ResponsesProviderSettingsBase):
    """Settings for using Codex Responses via ChatGPT OAuth tokens."""

    api_key: str | None = Field(default=None, description="Codex Responses API key")
    base_url: str | None = Field(default=None, description="Override API endpoint")
    service_tier: Literal["fast"] | None = Field(
        default=None,
        description="Codex Responses service tier: fast (priority) or unset (standard).",
    )


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
    transport: Literal["sse", "websocket", "auto"] | None = Field(
        default=None,
        description="Responses transport mode override: sse, websocket, or auto fallback.",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class XAISettings(BaseModel):
    """Settings for using xAI Grok models via the Responses API."""

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
    web_search: XAIWebSearchSettings = Field(default_factory=XAIWebSearchSettings)
    x_search: bool = Field(default=False, description="Enable xAI X Search remote tool.")

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

    @field_validator("sample_rate", mode="before")
    @classmethod
    def _reject_bool_sample_rate(cls, value: Any) -> Any:
        return _reject_bool_number_field(value, field_name="sample_rate")


class LiteLLMSettings(BaseModel):
    """Settings for the LiteLLM provider (embedded SDK or proxy mode)."""

    api_key: str | None = Field(
        default=None,
        description=(
            "Optional LiteLLM proxy API key. Leave unset to let LiteLLM resolve "
            "credentials from per-provider env vars (ANTHROPIC_API_KEY, "
            "OPENAI_API_KEY, etc.) at call time."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Optional LiteLLM proxy base URL (e.g. http://localhost:4000). "
            "When set, every call routes through the proxy."
        ),
    )
    default_model: str | None = Field(
        default=None,
        description=(
            "Default LiteLLM model spec when the LiteLLM provider is selected "
            "without an explicit model (e.g. 'anthropic/claude-sonnet-4-5')."
        ),
    )
    drop_params: bool = Field(
        default=True,
        description=(
            "Forward `drop_params=True` to litellm.acompletion so unsupported "
            "kwargs are stripped per backing provider rather than raising."
        ),
    )
    extra_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional kwargs forwarded verbatim to litellm.acompletion. Useful "
            "for routing-specific options like `metadata`, `tags`, `caching`."
        ),
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Custom headers forwarded as `extra_headers` to LiteLLM.",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


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


class TerminalImageSettings(BaseModel):
    """Terminal image rendering settings for chat/tool output."""

    enabled: bool = True
    """Render image content in the terminal when supported."""

    backend: Literal[
        "auto",
        "textual-image",
        "kitty",
        "sixel",
        "halfcell",
        "unicode",
        "none",
    ] = "auto"
    """Terminal image backend to use."""

    width: TerminalImageSize = "80%"
    """Image render width: cells, percentage (e.g. '80%'), 'auto', or null."""

    height: TerminalImageSize = "auto"
    """Image render height: cells, percentage (e.g. '40%'), 'auto', or null."""

    render_assistant: bool = True
    """Render images in final assistant messages."""

    @field_validator("width", "height", mode="before")
    @classmethod
    def _validate_image_size(cls, value: Any) -> TerminalImageSize:
        if value is None:
            return None
        if (int_value := int_or_none(value)) is not None:
            value = int_value
            if value < 0:
                raise ValueError("terminal image size must be non-negative")
            return value
        if isinstance(value, str):
            stripped, normalized = _normalized_config_token(value)
            if normalized in {"", "none", "null"}:
                return None
            if normalized == "auto":
                return "auto"
            if stripped.endswith("%") and stripped[:-1].isdecimal():
                return stripped
            if stripped.isdecimal():
                return int(stripped)
        raise ValueError("terminal image size must be an integer, percentage, 'auto', or null")


class TUISettings(BaseModel):
    """Interactive TUI settings."""

    completion_menu_reserved_lines: int = Field(
        default=6,
        ge=0,
        description="Prompt-toolkit lines reserved below the input for completion menus.",
    )

    model_config = ConfigDict(extra="ignore")


class LoggerSettings(BaseModel):
    """
    Logger settings for the fast-agent application.
    """

    type: Literal["none", "console", "file", "http"] = "file"

    level: Literal["debug", "info", "warning", "error"] = "warning"
    """Minimum logging level"""

    progress_display: bool = True
    """Enable or disable the progress display"""

    path: str = "fast-agent-log.jsonl"
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

    @field_validator("batch_size", "max_queue_size", mode="before")
    @classmethod
    def _reject_bool_integer_controls(cls, value: Any) -> Any:
        return _reject_bool_integer_field(value, field_name="logger integer controls")

    @field_validator("flush_interval", "http_timeout", mode="before")
    @classmethod
    def _reject_bool_number_controls(cls, value: Any) -> Any:
        return _reject_bool_number_field(value, field_name="logger numeric controls")

    show_chat: bool = True
    """Show chat User/Assistant on the console"""
    stream_reprint_banner: bool = True
    """Show a bright banner before reprinted final streamed assistant responses"""
    show_tools: bool = True
    """Show MCP Sever tool calls on the console"""
    truncate_tools: bool = True
    """Truncate display of long tool calls"""
    enable_markup: bool = True
    """Enable markup in console output. Disable for outputs that may conflict with rich console formatting"""

    enable_prompt_marks: bool = True
    """Emit OSC 133 prompt marks for terminals that support scrollbar markers."""
    streaming: StreamingMode = "markdown"
    """Streaming renderer for assistant responses"""
    theme_file: str | None = None
    """Optional Rich theme file for console styles. Relative paths resolve from fast-agent.yaml."""
    code_theme: str = "native"
    """Pygments / Rich syntax theme for fenced code blocks and markdown code rendering."""
    render_fences_with_syntax: bool = True
    """Render assistant markdown code fences with Rich Syntax instead of markdown fence blocks"""
    code_word_wrap: bool = True
    """Wrap Syntax-rendered code blocks instead of cropping at the viewport edge"""
    terminal_images: TerminalImageSettings = Field(default_factory=TerminalImageSettings)
    """Render image content in capable terminals."""
    apply_patch_preview_max_lines: int | None = Field(
        default=120,
        description=(
            "Maximum lines to show in apply_patch previews before appending "
            "'(+N more lines)' (0/None = no limit)"
        ),
    )
    """Maximum lines to show in apply_patch previews before truncation"""

    _theme_file_config_path: str | None = PrivateAttr(default=None)

    @field_validator("apply_patch_preview_max_lines", mode="before")
    @classmethod
    def _coerce_apply_patch_preview_max_lines(cls, value: Any) -> int | None:
        if value is None:
            return None
        _reject_bool_integer_field(value, field_name="apply_patch_preview_max_lines")
        if isinstance(value, str):
            stripped, normalized = _normalized_config_token(value)
            if normalized in {"", "none", "null", "all", "unlimited"}:
                return None
            value = int(stripped)
        else:
            value = int(value)
        if value == 0:
            return None
        if value < 0:
            raise ValueError("apply_patch_preview_max_lines must be non-negative.")
        return value


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

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigFileError(f"Failed to parse YAML file: {path}", str(exc)) from exc
    if not isinstance(payload, dict):
        return {}
    return resolve_env_vars(payload)


def load_implicit_settings(
    *,
    start_path: Path,
    env_dir: str | Path | None = None,
    noenv: bool = False,
) -> tuple[dict[str, Any], ConfigDiscoveryResult]:
    """Load settings from the discovered config file."""
    home = resolve_fast_agent_home(cwd=start_path, cli_override=env_dir, noenv=noenv)
    discovery = discover_config_files(cwd=start_path, home=home)
    merged: dict[str, Any] = {}
    if discovery.config_path and discovery.config_path.exists():
        merged = load_yaml_mapping(discovery.config_path)
    return merged, discovery


def load_selected_settings(
    *,
    start_path: Path,
    env_dir: str | Path | None = None,
    noenv: bool = False,
) -> tuple[dict[str, Any], ConfigDiscoveryResult]:
    """Load first-found config/secrets settings with home then cwd precedence."""
    return load_implicit_settings(start_path=start_path, env_dir=env_dir, noenv=noenv)


@dataclass(frozen=True)
class _LoadedSettingsSources:
    merged_settings: dict[str, Any]
    discovery: ConfigDiscoveryResult
    config_file: Path | None
    secrets_file: Path | None
    config_sources: list[tuple[Path, dict[str, Any]]]


def _plugin_settings_mapping(settings: dict[str, Any]) -> dict[str, Any] | None:
    plugins = settings.get("plugins")
    if isinstance(plugins, dict):
        return plugins
    return None


def _enabled_plugin_names(*plugin_sources: dict[str, Any]) -> list[str]:
    enabled: list[str] = []
    for source in plugin_sources:
        raw_enabled = source.get("enabled")
        if not isinstance(raw_enabled, list):
            continue
        for item in raw_enabled:
            if not isinstance(item, str):
                continue
            plugin_name = item.strip()
            if plugin_name and plugin_name not in enabled:
                enabled.append(plugin_name)
    return enabled


def _is_active_config(config_path: Path, active_config_file: Path | None) -> bool:
    return active_config_file is not None and config_path.resolve() == active_config_file.resolve()


def _merge_home_plugin_settings(
    settings: dict[str, Any],
    *,
    global_plugin_home: Path | None,
    active_config_file: Path | None,
) -> dict[str, Any]:
    """Merge only global plugin selections into the active settings.

    General config discovery intentionally picks one main config file. Plugins are
    different: global plugin installs should augment project-local plugin
    selections instead of replacing them.
    """
    if global_plugin_home is None:
        return settings
    home_config = find_config_in_directory(global_plugin_home)
    if home_config is None:
        return settings
    if _is_active_config(home_config, active_config_file):
        return settings

    home_settings = load_yaml_mapping(home_config)
    home_plugins = _plugin_settings_mapping(home_settings)
    if home_plugins is None:
        return settings

    merged = dict(settings)
    active_plugins = _plugin_settings_mapping(merged) or {}

    plugin_settings = deep_merge(home_plugins, active_plugins)
    enabled = _enabled_plugin_names(home_plugins, active_plugins)
    if enabled:
        plugin_settings["enabled"] = enabled

    merged["plugins"] = plugin_settings
    return merged


def _expand_user_path(path: Path, *, home: Path) -> Path:
    text = str(path)
    if text == "~":
        return home
    if text.startswith("~/"):
        return home / text[2:]
    return path


def resolve_global_plugin_home_path(
    *,
    fast_agent_home: str | None,
    home: Path,
    cwd: Path,
    noenv: bool = False,
) -> Path | None:
    if noenv:
        return None

    if fast_agent_home:
        path = _expand_user_path(Path(fast_agent_home), home=home)
        if not path.is_absolute():
            path = cwd / path
        return path.resolve()

    return (home / ".fast-agent").resolve()


def _resolve_global_plugin_home(*, noenv: bool) -> Path | None:
    try:
        home = Path.home()
    except RuntimeError:
        return None
    return resolve_global_plugin_home_path(
        fast_agent_home=os.getenv("FAST_AGENT_HOME"),
        home=home,
        cwd=Path.cwd(),
        noenv=noenv,
    )


def load_layered_model_settings(
    *,
    start_path: Path,
    env_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load layered model settings from project + env config.

    Precedence: project config < env config.
    ``model_references`` uses deep-merge semantics, while ``default_model`` uses
    scalar replacement semantics.
    """
    layered: dict[str, Any] = {}

    home = resolve_fast_agent_home(cwd=start_path, cli_override=env_dir)
    project_config = find_config_in_directory(start_path)
    env_config = find_config_in_directory(home.path) if home is not None else None

    config_paths: list[Path] = []
    for config_path in (project_config, env_config):
        if config_path is not None and config_path not in config_paths:
            config_paths.append(config_path)

    for config_path in config_paths:
        model_settings = _model_settings_from_mapping(load_yaml_mapping(config_path))
        layered = _merge_model_layer(layered, model_settings)

    return layered


def _model_settings_from_mapping(settings: dict[str, Any]) -> dict[str, Any]:
    model_settings: dict[str, Any] = {}
    if "default_model" in settings:
        model_settings["default_model"] = settings["default_model"]
    if "model_references" in settings:
        model_settings["model_references"] = settings["model_references"]
    return model_settings


def _merge_model_layer(base: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if "default_model" in layer:
        merged["default_model"] = layer["default_model"]

    if "model_references" in layer:
        base_references = merged.get("model_references")
        layer_references = layer["model_references"]
        if isinstance(base_references, dict) and isinstance(layer_references, dict):
            merged["model_references"] = deep_merge(base_references, layer_references)
        else:
            merged["model_references"] = layer_references

    return merged


def _lookup_nested_mapping_value(
    mapping: dict[str, Any], path: tuple[str, ...]
) -> tuple[bool, Any]:
    """Return whether a nested mapping path exists plus its value."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


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

    mcp: MCPSettings | None = Field(default_factory=MCPSettings)
    """MCP config, such as MCP servers"""

    execution_engine: Literal["asyncio"] = "asyncio"
    """Execution engine for the fast-agent application"""

    environment_dir: str | None = None
    """Base directory for fast-agent runtime data (defaults to .fast-agent)."""

    default_model: str | None = None
    """
    Default model for agents. Format is provider.model?reasoning=<value>,
    for example openai.o3-mini?reasoning=high.
    Built-in model presets are provided for common models e.g. sonnet, haiku, gpt-4.1, o3-mini etc.
    If not set, falls back to FAST_AGENT_MODEL env var, then to "gpt-5-mini?reasoning=low".
    """

    model_references: dict[str, dict[str, str]] = Field(default_factory=dict)
    """Model references grouped by namespace (e.g. $system.default)."""

    model_source: str | None = None
    """Where the default model was resolved from for the current run, if noteworthy."""

    cli_model_override: str | None = None
    """Model override supplied by the CLI for the current run, if any."""

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

    litellm: LiteLLMSettings | None = None
    """Settings for the LiteLLM provider (routes via the LiteLLM SDK)"""

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

    logger: LoggerSettings = Field(default_factory=LoggerSettings)
    """Logger settings for the fast-agent application"""

    # MCP UI integration mode for handling ui:// embedded resources from MCP tool results
    mcp_ui_mode: McpUIMode = "enabled"
    """Controls handling of MCP UI embedded resources:
    - "disabled": Do not process ui:// resources
    - "enabled": Always extract ui:// resources into message channels (default)
    - "auto": Extract and automatically open ui:// resources.
    """

    # Output directory for MCP-UI generated HTML files (relative to CWD if not absolute)
    mcp_ui_output_dir: str = ".fast-agent/ui"
    """Directory where MCP-UI HTML files are written. Relative paths are resolved from CWD."""

    mcp_timeline: MCPTimelineSettings = Field(default_factory=MCPTimelineSettings)
    """Display settings for MCP activity timelines."""

    skills: SkillsSettings = Field(default_factory=SkillsSettings)
    """Local skills discovery and selection settings."""

    cards: CardsSettings = Field(default_factory=CardsSettings)
    """Card pack registry selection settings."""

    plugins: PluginsSettings = Field(default_factory=PluginsSettings)
    """Command plugin selection and marketplace settings."""

    tui: TUISettings = Field(default_factory=TUISettings)
    """Interactive TUI settings."""

    commands: dict[str, PluginCommandActionSpec] | None = None
    """Global plugin command actions loaded from fast-agent.yaml."""

    shell_execution: ShellSettings = Field(default_factory=ShellSettings)
    """Shell execution timeout and warning settings."""

    llm_retries: int = 2
    """
    Number of times to retry transient LLM API errors.
    Defaults to 2; can be overridden via config or FAST_AGENT_RETRIES env.
    """

    _config_file: str | None = PrivateAttr(default=None)
    _secrets_file: str | None = PrivateAttr(default=None)
    _fast_agent_home: str | None = PrivateAttr(default=None)
    _fast_agent_home_source: str | None = PrivateAttr(default=None)
    _fast_agent_global_plugin_home: str | None = PrivateAttr(default=None)
    _fast_agent_noenv: bool = PrivateAttr(default=False)
    _fast_agent_settings_source: Literal["manual", "discovered"] = PrivateAttr(default="manual")

    @field_validator("commands", mode="before")
    @classmethod
    def _validate_plugin_commands(cls, value: Any) -> dict[str, PluginCommandActionSpec] | None:
        return parse_plugin_command_action_specs(value, source="fast-agent.yaml")

    @field_validator("model_references")
    @classmethod
    def _validate_model_references(
        cls,
        value: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Validate model reference namespace/key names and normalize values."""
        valid_name = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
        normalized: dict[str, dict[str, str]] = {}

        for namespace, entries in value.items():
            if not valid_name.fullmatch(namespace):
                raise ValueError(
                    "model_references namespace names must match [A-Za-z_][A-Za-z0-9_-]*"
                )

            normalized_entries: dict[str, str] = {}
            for key, model in entries.items():
                if not valid_name.fullmatch(key):
                    raise ValueError("model_references keys must match [A-Za-z_][A-Za-z0-9_-]*")

                model_value = model.strip()
                if not model_value:
                    raise ValueError(
                        f"model_references.{namespace}.{key} must be a non-empty model string"
                    )
                normalized_entries[key] = model_value

            normalized[namespace] = normalized_entries

        return normalized

    @classmethod
    def find_config(cls) -> Path | None:
        """Find the preferred config file in the current directory."""
        config_path = Path.cwd() / "fast-agent.yaml"
        return config_path if config_path.exists() else None


# Global settings object
_settings: Settings | None = None


def _cached_settings_match_environment_request(
    settings: Settings,
    *,
    env_dir: str | Path | None,
    noenv: bool,
) -> bool:
    if settings._fast_agent_settings_source == "manual":
        return not noenv and env_dir is None

    if noenv:
        return settings._fast_agent_noenv

    if settings._fast_agent_noenv:
        return False

    requested_global_home = _resolve_global_plugin_home(noenv=False)
    if settings._fast_agent_global_plugin_home != (
        str(requested_global_home) if requested_global_home is not None else None
    ):
        return False

    if env_dir is None and settings._fast_agent_home is None:
        return True

    requested_home = resolve_fast_agent_home(
        cwd=Path.cwd(),
        cli_override=env_dir,
        noenv=False,
    )
    return (
        requested_home is not None
        and settings._fast_agent_home == str(requested_home.path)
        and (env_dir is None or settings._fast_agent_home_source == "cli")
    )


def _env_dir_override(env_dir: str | os.PathLike[str] | None) -> str | Path | None:
    if env_dir is None or isinstance(env_dir, str):
        return env_dir
    return Path(env_dir)


def _resolve_explicit_config_path(config_path: str | os.PathLike[str]) -> Path:
    config_file = Path(config_path)
    if config_file.is_absolute() or config_file.exists():
        return config_file

    resolved_path = Path.cwd() / config_file.name
    if resolved_path.exists():
        return resolved_path
    return config_file


def _load_explicit_settings_sources(
    config_path: str | os.PathLike[str],
    *,
    env_dir: str | Path | None,
    noenv: bool,
) -> _LoadedSettingsSources:
    config_file = _resolve_explicit_config_path(config_path)
    discovery = discover_config_files(
        cwd=Path.cwd(),
        home=resolve_fast_agent_home(
            cwd=Path.cwd(),
            cli_override=env_dir,
            noenv=noenv,
        ),
        explicit_config_path=config_file,
    )
    secrets_file = discovery.secrets_path if config_file.exists() else None
    merged_settings: dict[str, Any] = {}
    config_sources: list[tuple[Path, dict[str, Any]]] = []
    if config_file.exists():
        merged_settings = load_yaml_mapping(config_file)
        config_sources.append((config_file, merged_settings))
    else:
        print(f"Warning: Specified config file does not exist: {config_file}")

    return _LoadedSettingsSources(
        merged_settings=merged_settings,
        discovery=discovery,
        config_file=config_file,
        secrets_file=secrets_file,
        config_sources=config_sources,
    )


def _load_implicit_settings_sources(
    *,
    env_dir: str | Path | None,
    noenv: bool,
) -> _LoadedSettingsSources:
    merged_settings, discovery = load_implicit_settings(
        start_path=Path.cwd(),
        env_dir=env_dir,
        noenv=noenv,
    )
    config_file = discovery.config_path
    config_sources: list[tuple[Path, dict[str, Any]]] = []
    if config_file and config_file.exists():
        config_sources.append((config_file, load_yaml_mapping(config_file)))

    return _LoadedSettingsSources(
        merged_settings=merged_settings,
        discovery=discovery,
        config_file=config_file,
        secrets_file=discovery.secrets_path,
        config_sources=config_sources,
    )


def _load_settings_sources(
    config_path: str | os.PathLike[str] | None,
    *,
    env_dir: str | Path | None,
    noenv: bool,
) -> _LoadedSettingsSources:
    if config_path:
        return _load_explicit_settings_sources(
            config_path,
            env_dir=env_dir,
            noenv=noenv,
        )
    return _load_implicit_settings_sources(env_dir=env_dir, noenv=noenv)


def _warn_for_legacy_anthropic_reasoning_config(merged_settings: dict[str, Any]) -> None:
    legacy_keys: list[str] = []
    anthropic_settings = merged_settings.get("anthropic")
    if isinstance(anthropic_settings, dict):
        legacy_keys.extend(
            key
            for key in ("thinking_enabled", "thinking_budget_tokens")
            if key in anthropic_settings
        )
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


def _set_theme_file_config_path(
    settings: Settings,
    config_sources: list[tuple[Path, dict[str, Any]]],
) -> None:
    current_theme_file = settings.logger.theme_file
    if current_theme_file is None:
        return
    for source_path, source_mapping in reversed(config_sources):
        found, source_value = _lookup_nested_mapping_value(
            source_mapping,
            ("logger", "theme_file"),
        )
        if found and source_value == current_theme_file:
            settings.logger._theme_file_config_path = str(source_path)
            break


def _settings_from_sources(
    sources: _LoadedSettingsSources,
    *,
    global_plugin_home: Path | None,
    noenv: bool,
) -> Settings:
    settings = Settings(**sources.merged_settings)
    settings._config_file = str(sources.config_file) if sources.config_file else None
    settings._secrets_file = str(sources.secrets_file) if sources.secrets_file else None
    settings._fast_agent_home = str(sources.discovery.home.path) if sources.discovery.home else None
    settings._fast_agent_home_source = (
        sources.discovery.home.source if sources.discovery.home else None
    )
    settings._fast_agent_global_plugin_home = (
        str(global_plugin_home) if global_plugin_home is not None else None
    )
    settings._fast_agent_noenv = noenv
    settings._fast_agent_settings_source = "discovered"
    _set_theme_file_config_path(settings, sources.config_sources)
    settings.commands = _merge_enabled_plugin_commands(settings)
    return settings


def get_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env_dir: str | os.PathLike[str] | None = None,
    noenv: bool = False,
) -> Settings:
    """Get settings instance, automatically loading from config file if available."""

    global _settings

    env_dir_override = _env_dir_override(env_dir)

    if config_path:
        _settings = None
    elif _settings and _cached_settings_match_environment_request(
        _settings,
        env_dir=env_dir_override,
        noenv=noenv,
    ):
        return _settings

    sources = _load_settings_sources(config_path, env_dir=env_dir_override, noenv=noenv)
    global_plugin_home = _resolve_global_plugin_home(noenv=noenv)
    merged_settings = _merge_home_plugin_settings(
        sources.merged_settings,
        global_plugin_home=global_plugin_home,
        active_config_file=sources.config_file,
    )

    if sources.secrets_file and sources.secrets_file.exists():
        merged_settings = deep_merge(merged_settings, load_yaml_mapping(sources.secrets_file))

    _warn_for_legacy_anthropic_reasoning_config(merged_settings)
    _settings = _settings_from_sources(
        _LoadedSettingsSources(
            merged_settings=merged_settings,
            discovery=sources.discovery,
            config_file=sources.config_file,
            secrets_file=sources.secrets_file,
            config_sources=sources.config_sources,
        ),
        global_plugin_home=global_plugin_home,
        noenv=noenv,
    )
    return _settings


def _merge_enabled_plugin_commands(settings: Settings) -> dict[str, PluginCommandActionSpec] | None:
    inline_commands = settings.commands or {}
    enabled_sources = _enabled_plugin_sources(settings)
    if not enabled_sources.home and not enabled_sources.project:
        return inline_commands or None

    from fast_agent.paths import resolve_environment_paths
    from fast_agent.plugins.operations import load_enabled_plugin_commands

    plugin_commands: dict[str, PluginCommandActionSpec] = {}
    if enabled_sources.home and settings._fast_agent_global_plugin_home:
        plugin_commands.update(
            _load_enabled_plugin_commands_from_root(
                destination_root=Path(settings._fast_agent_global_plugin_home) / "plugins",
                enabled=enabled_sources.home,
                scope="global",
                load_enabled_plugin_commands=load_enabled_plugin_commands,
            )
        )

    if enabled_sources.project:
        plugin_commands.update(
            _load_enabled_plugin_commands_from_root(
                destination_root=resolve_environment_paths(settings).plugins,
                enabled=enabled_sources.project,
                scope="project",
                load_enabled_plugin_commands=load_enabled_plugin_commands,
            )
        )

    merged = dict(plugin_commands)
    merged.update(inline_commands)
    return merged or None


def _load_enabled_plugin_commands_from_root(
    *,
    destination_root: Path,
    enabled: list[str],
    scope: str,
    load_enabled_plugin_commands,
) -> dict[str, PluginCommandActionSpec]:
    try:
        return load_enabled_plugin_commands(
            destination_root=destination_root,
            enabled=enabled,
        )
    except Exception as exc:
        warnings.warn(
            f"Failed to load enabled fast-agent plugins from {scope}: {exc}",
            UserWarning,
            stacklevel=3,
        )
        return {}


@dataclass(frozen=True, slots=True)
class _EnabledPluginSources:
    home: list[str]
    project: list[str]


def _enabled_plugin_sources(settings: Settings) -> _EnabledPluginSources:
    """Return enabled plugins grouped by FAST_AGENT_HOME and active project config."""
    enabled = list(settings.plugins.enabled)
    if not enabled:
        return _EnabledPluginSources(home=[], project=[])

    global_config = _global_plugin_config_for_plugin_merge(settings)
    if global_config is None:
        return _EnabledPluginSources(home=[], project=enabled)

    home_enabled = _enabled_plugins_from_config(global_config)
    active_config = Path(settings._config_file) if settings._config_file else None
    project_enabled: list[str] = []
    if active_config is not None and active_config.exists():
        try:
            same_as_global = active_config.resolve() == global_config.resolve()
        except OSError:
            same_as_global = False
        if not same_as_global:
            project_enabled = _enabled_plugins_from_config(active_config)

    known = {*home_enabled, *project_enabled}
    project_enabled.extend(name for name in enabled if name not in known)
    return _EnabledPluginSources(home=home_enabled, project=project_enabled)


def _global_plugin_config_for_plugin_merge(settings: Settings) -> Path | None:
    if not settings._fast_agent_global_plugin_home:
        return None

    home_config = find_config_in_directory(Path(settings._fast_agent_global_plugin_home))
    if home_config is None:
        return None

    active_config = Path(settings._config_file) if settings._config_file else None
    if active_config is not None:
        try:
            if active_config.resolve() == home_config.resolve():
                return None
        except OSError:
            return None

    return home_config


def _enabled_plugins_from_config(config_path: Path) -> list[str]:
    data = load_yaml_mapping(config_path)
    plugins = data.get("plugins")
    if not isinstance(plugins, dict):
        return []

    raw_enabled = plugins.get("enabled")
    if not isinstance(raw_enabled, list):
        return []

    enabled: list[str] = []
    for item in raw_enabled:
        if isinstance(item, str):
            name = item.strip()
            if name and name not in enabled:
                enabled.append(name)
    return enabled


def update_global_settings(settings: Settings) -> None:
    """Update the global settings instance.

    This is used to propagate CLI overrides (like --skills-dir) into the
    global settings so that functions like resolve_skill_directories()
    work correctly without needing to pass settings around explicitly.
    """
    global _settings
    settings._fast_agent_settings_source = "manual"
    _settings = settings
