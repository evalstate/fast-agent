"""Source-preserving MCP server declarations and effective materialization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings

_SOURCE_FIELDS = frozenset({"transport", "url", "command", "args", "connector_id"})
_SECRET_KEYS = frozenset(
    {
        "access_token",
        "authorization",
        "client_secret",
        "password",
        "secret",
        "token",
    }
)
_CREDENTIAL_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "clientsecret",
        "cookie",
        "password",
        "proxyauthorization",
        "secret",
        "setcookie",
        "token",
        "xapikey",
    }
)


def _is_credential_key(key: str) -> bool:
    compact = "".join(character for character in strip_casefold(key) if character.isalnum())
    return compact in _CREDENTIAL_KEYS


def _redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    from fast_agent.mcp.connect_targets import redact_mcp_url

    redacted: dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = strip_casefold(key)
        if normalized_key in _SECRET_KEYS or _is_credential_key(key):
            redacted[key] = "[REDACTED]"
        elif normalized_key == "headers" and isinstance(value, dict):
            redacted[key] = {
                header: "[REDACTED]" if _is_credential_key(header) else header_value
                for header, header_value in value.items()
            }
        elif normalized_key in {"target", "url"} and isinstance(value, str) and "://" in value:
            redacted[key] = redact_mcp_url(value)
        elif normalized_key == "env" and isinstance(value, dict):
            redacted[key] = dict.fromkeys(value, "[REDACTED]")
        elif isinstance(value, dict):
            redacted[key] = _redact_mapping(value)
        elif isinstance(value, list):
            redacted[key] = [
                _redact_mapping(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            redacted[key] = value
    return redacted


class MCPServerDeclaration(BaseModel):
    """Raw declaration retained independently from effective runtime settings."""

    target: str | None = None
    name: str | None = None
    description: str | None = None
    management: Literal["client", "provider"] | None = None
    connector_id: str | None = None
    headers: dict[str, str] | None = None
    access_token: str | None = None
    defer_loading: bool | None = None
    auth: dict[str, Any] | None = None
    protocol_mode: Literal["auto", "modern", "legacy"] | None = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @classmethod
    def from_source(
        cls,
        *,
        name: str,
        source: dict[str, Any],
        source_path: str,
    ) -> "MCPServerDeclaration":
        payload = dict(source)
        nested_name = payload.get("name")
        if nested_name is not None and nested_name != name:
            raise ValueError(
                f"`{source_path}.name` must match its map key '{name}', got {nested_name!r}"
            )
        payload["name"] = name
        if "target" in payload:
            conflicting = sorted(_SOURCE_FIELDS.intersection(payload))
            if conflicting:
                fields = ", ".join(conflicting)
                raise ValueError(
                    f"`{source_path}.target` cannot be combined with source fields: {fields}. "
                    "Remove them or replace `target` with explicit transport settings."
                )
        return cls.model_validate(payload)

    def source_view(self, *, redact: bool = True, include_name: bool = False) -> dict[str, Any]:
        source = self.model_dump(
            mode="python",
            exclude_none=True,
            exclude_unset=True,
        )
        if not include_name:
            source.pop("name", None)
        return _redact_mapping(source) if redact else source

    def materialize(
        self,
        *,
        source_path: str,
        defaults: dict[str, Any] | None = None,
    ) -> "MCPServerSettings":
        from fast_agent.config import MCPServerSettings
        from fast_agent.mcp.connect_targets import resolve_target_entry

        source = self.model_dump(mode="python", exclude_none=True, exclude_unset=True)
        source_fields = set(source)
        target = source.pop("target", None)
        name = self.name
        if target is not None:
            resolved = resolve_target_entry(
                target=target,
                default_name=name,
                overrides=source,
                source_path=f"{source_path}.target",
            )
            settings = resolved.settings
        else:
            settings = MCPServerSettings.model_validate(source)

        if defaults:
            updates = {
                field: value for field, value in defaults.items() if field not in source_fields
            }
            if updates:
                settings = settings.model_copy(update=updates)
        return settings


def effective_server_view(settings: Any, *, redact: bool = True) -> dict[str, Any]:
    payload = settings.model_dump(mode="python", exclude_none=True)
    return _redact_mapping(payload) if redact else payload


__all__ = ["MCPServerDeclaration", "effective_server_view"]
