"""Shared AgentCard ``mcp_connect`` parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from fast_agent.agents.agent_types import (
    MCPConnectSourceForm,
    MCPConnectTarget,
)
from fast_agent.core.agent_card_rules import MCP_CONNECT_ALLOWED_KEYS
from fast_agent.utils.text import strip_str_to_none
from fast_agent.utils.type_narrowing import is_str_object_dict

_PROTOCOL_MODES = frozenset({"auto", "modern", "legacy"})


@dataclass(frozen=True, slots=True)
class ParsedMCPConnect:
    entries: list[MCPConnectTarget]
    field_paths: list[str]
    source_form: MCPConnectSourceForm
    errors: list[str]


def _optional_non_empty_string(
    raw_entry: dict[str, Any],
    field_path: str,
    key: str,
    errors: list[str],
) -> str | None:
    value = raw_entry.get(key)
    if value is None:
        return None
    normalized = strip_str_to_none(value)
    if normalized is None:
        errors.append(f"'{field_path}.{key}' must be a non-empty string")
    return normalized


def _optional_string(
    raw_entry: dict[str, Any],
    field_path: str,
    key: str,
    errors: list[str],
) -> str | None:
    value = raw_entry.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"'{field_path}.{key}' must be a string")
        return None
    return value


def _headers(
    value: Any,
    field_path: str,
    errors: list[str],
) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        errors.append(f"'{field_path}.headers' must be a mapping")
        return None

    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if strip_str_to_none(key) is None:
            errors.append(f"'{field_path}.headers' keys must be non-empty strings")
            return None
        if not isinstance(header_value, str):
            errors.append(f"'{field_path}.headers' values must be strings")
            return None
        headers[key] = header_value
    return headers


def _auth(value: Any, field_path: str, errors: list[str]) -> dict[str, Any] | None:
    if value is None:
        return None
    if not is_str_object_dict(value):
        errors.append(f"'{field_path}.auth' must be a mapping")
        return None
    return value.copy()


def _optional_bool(value: Any, field_path: str, key: str, errors: list[str]) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        errors.append(f"'{field_path}.{key}' must be a boolean")
        return None
    return value


def _protocol_mode(
    value: Any,
    field_path: str,
    errors: list[str],
) -> Literal["auto", "modern", "legacy"] | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PROTOCOL_MODES:
        errors.append(f"'{field_path}.protocol_mode' must be one of auto, modern, legacy")
        return None
    return value


def _management_mode(
    value: Any,
    field_path: str,
    errors: list[str],
) -> Literal["client", "provider"] | None:
    if value is None:
        return None
    normalized = strip_str_to_none(value)
    if normalized == "client":
        return "client"
    if normalized == "provider":
        return "provider"
    errors.append(f"'{field_path}.management' must be one of client, provider")
    return None


def _parse_entry(
    raw_entry: dict[str, Any],
    *,
    field_path: str,
    implicit_name: str | None,
    errors: list[str],
) -> MCPConnectTarget | None:
    error_count = len(errors)
    unknown_keys = set(raw_entry) - MCP_CONNECT_ALLOWED_KEYS
    if unknown_keys:
        unknown_text = ", ".join(sorted(str(key) for key in unknown_keys))
        errors.append(f"'{field_path}' has unsupported keys: {unknown_text}")

    target = _optional_non_empty_string(raw_entry, field_path, "target", errors)
    explicit_name = _optional_non_empty_string(raw_entry, field_path, "name", errors)
    connector_id = _optional_non_empty_string(raw_entry, field_path, "connector_id", errors)
    if implicit_name is not None and explicit_name is not None and explicit_name != implicit_name:
        errors.append(f"'{field_path}.name' must match mapping key '{implicit_name}' when provided")
    name = implicit_name or explicit_name

    if target is None and connector_id is None:
        errors.append(
            f"'{field_path}.target' must be a non-empty string unless connector_id is set"
        )
    elif target is not None and connector_id is not None:
        errors.append(f"'{field_path}' must set exactly one of 'target' or 'connector_id'")
    if connector_id is not None and name is None:
        errors.append(f"'{field_path}.name' must be a non-empty string when connector_id is set")

    entry = MCPConnectTarget(
        target=target,
        name=name,
        description=_optional_string(raw_entry, field_path, "description", errors),
        management=_management_mode(raw_entry.get("management"), field_path, errors),
        connector_id=connector_id,
        headers=_headers(raw_entry.get("headers"), field_path, errors),
        access_token=_optional_string(raw_entry, field_path, "access_token", errors),
        defer_loading=_optional_bool(
            raw_entry.get("defer_loading"), field_path, "defer_loading", errors
        ),
        auth=_auth(raw_entry.get("auth"), field_path, errors),
        protocol_mode=_protocol_mode(raw_entry.get("protocol_mode"), field_path, errors),
    )
    return entry if len(errors) == error_count else None


def parse_mcp_connect_entries(value: Any) -> ParsedMCPConnect:
    """Parse list-compatible or canonical named mapping declarations."""
    if value is None:
        return ParsedMCPConnect(entries=[], field_paths=[], source_form="list", errors=[])

    errors: list[str] = []
    entries: list[MCPConnectTarget] = []
    field_paths: list[str] = []
    if isinstance(value, list):
        source_form: MCPConnectSourceForm = "list"
        raw_entries = [
            (f"mcp_connect[{index}]", None, raw_entry) for index, raw_entry in enumerate(value)
        ]
    elif isinstance(value, dict):
        source_form = "mapping"
        raw_entries = []
        for raw_name, raw_entry in value.items():
            name = strip_str_to_none(raw_name)
            if name is None:
                errors.append("'mcp_connect' keys must be non-empty strings")
                continue
            raw_entries.append((f"mcp_connect.{name}", name, raw_entry))
    else:
        return ParsedMCPConnect(
            entries=[],
            field_paths=[],
            source_form="list",
            errors=["'mcp_connect' must be a mapping or list"],
        )

    for field_path, implicit_name, raw_entry in raw_entries:
        if not is_str_object_dict(raw_entry):
            errors.append(f"'{field_path}' must be a mapping")
            continue
        entry = _parse_entry(
            raw_entry,
            field_path=field_path,
            implicit_name=implicit_name,
            errors=errors,
        )
        if entry is not None:
            entries.append(entry)
            field_paths.append(field_path)

    if source_form == "mapping" and len({entry.name for entry in entries}) != len(entries):
        errors.append("'mcp_connect' mapping keys must be unique after trimming")

    return ParsedMCPConnect(
        entries=entries,
        field_paths=field_paths,
        source_form=source_form,
        errors=errors,
    )


def validate_parsed_mcp_connect_entry(entry: MCPConnectTarget, field_path: str) -> None:
    entry.materialize(source_path=field_path)


def validate_mcp_connect_entries(value: Any, errors: list[str]) -> None:
    parsed = parse_mcp_connect_entries(value)
    errors.extend(parsed.errors)
    for field_path, entry in zip(parsed.field_paths, parsed.entries, strict=True):
        try:
            validate_parsed_mcp_connect_entry(entry, field_path)
        except Exception as exc:
            errors.append(f"Invalid mcp_connect target '{field_path}': {exc}")
