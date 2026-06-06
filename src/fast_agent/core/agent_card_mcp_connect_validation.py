"""Agent card mcp_connect validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fast_agent.config import MCPServerSettings
from fast_agent.core.agent_card_rules import MCP_CONNECT_ALLOWED_KEYS
from fast_agent.mcp.connect_targets import resolve_target_entry
from fast_agent.utils.text import strip_str_to_none
from fast_agent.utils.type_narrowing import is_str_object_dict


@dataclass(frozen=True)
class _McpConnectEntry:
    index: int
    target: str | None
    name: str | None
    connector_id: str | None
    headers: dict[str, str] | None
    auth: dict[str, Any] | None
    management: str | None
    description: str | None
    access_token: str | None
    defer_loading: bool | None


def _optional_non_empty_mcp_string(
    raw_entry: dict[str, Any],
    idx: int,
    key: str,
    errors: list[str],
) -> tuple[str | None, bool]:
    value = raw_entry.get(key)
    if value is None:
        return None, True
    normalized = strip_str_to_none(value)
    if normalized is None:
        errors.append(f"'mcp_connect[{idx}].{key}' must be a non-empty string")
        return None, False
    return normalized, True


def _optional_mcp_string(
    raw_entry: dict[str, Any],
    idx: int,
    key: str,
    errors: list[str],
) -> tuple[str | None, bool]:
    value = raw_entry.get(key)
    if value is None:
        return None, True
    if not isinstance(value, str):
        errors.append(f"'mcp_connect[{idx}].{key}' must be a string")
        return None, False
    return value, True


def _mcp_connect_headers(
    value: Any,
    idx: int,
    errors: list[str],
) -> tuple[dict[str, str] | None, bool]:
    if value is None:
        return None, True
    if not isinstance(value, dict):
        errors.append(f"'mcp_connect[{idx}].headers' must be a mapping")
        return None, False

    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if strip_str_to_none(key) is None:
            errors.append(f"'mcp_connect[{idx}].headers' keys must be non-empty strings")
            return None, False
        if not isinstance(header_value, str):
            errors.append(f"'mcp_connect[{idx}].headers' values must be strings")
            return None, False
        headers[key] = header_value
    return headers, True


def _mcp_connect_auth(
    value: Any,
    idx: int,
    errors: list[str],
) -> tuple[dict[str, Any] | None, bool]:
    if value is None:
        return None, True
    if not is_str_object_dict(value):
        errors.append(f"'mcp_connect[{idx}].auth' must be a mapping")
        return None, False
    return value.copy(), True


def _mcp_connect_defer_loading(
    value: Any,
    idx: int,
    errors: list[str],
) -> tuple[bool | None, bool]:
    if value is None:
        return None, True
    if not isinstance(value, bool):
        errors.append(f"'mcp_connect[{idx}].defer_loading' must be a boolean")
        return None, False
    return value, True


def _parse_mcp_connect_entry(
    raw_entry: dict[str, Any],
    idx: int,
    errors: list[str],
) -> _McpConnectEntry | None:
    unknown_keys = set(raw_entry.keys()) - MCP_CONNECT_ALLOWED_KEYS
    if unknown_keys:
        unknown_text = ", ".join(sorted(str(key) for key in unknown_keys))
        errors.append(f"'mcp_connect[{idx}]' has unsupported keys: {unknown_text}")

    target, target_ok = _optional_non_empty_mcp_string(raw_entry, idx, "target", errors)
    name, name_ok = _optional_non_empty_mcp_string(raw_entry, idx, "name", errors)
    connector_id, connector_id_ok = _optional_non_empty_mcp_string(
        raw_entry, idx, "connector_id", errors
    )
    headers, headers_ok = _mcp_connect_headers(raw_entry.get("headers"), idx, errors)
    auth, auth_ok = _mcp_connect_auth(raw_entry.get("auth"), idx, errors)
    management, management_ok = _optional_non_empty_mcp_string(
        raw_entry, idx, "management", errors
    )
    description, description_ok = _optional_mcp_string(raw_entry, idx, "description", errors)
    access_token, access_token_ok = _optional_mcp_string(raw_entry, idx, "access_token", errors)
    defer_loading, defer_loading_ok = _mcp_connect_defer_loading(
        raw_entry.get("defer_loading"), idx, errors
    )

    if not all(
        [
            target_ok,
            name_ok,
            connector_id_ok,
            headers_ok,
            auth_ok,
            management_ok,
            description_ok,
            access_token_ok,
            defer_loading_ok,
        ]
    ):
        return None

    if target is None and connector_id is None:
        errors.append(
            f"'mcp_connect[{idx}].target' must be a non-empty string unless connector_id is set"
        )
        return None
    if target is not None and connector_id is not None:
        errors.append(f"'mcp_connect[{idx}]' must set exactly one of 'target' or 'connector_id'")
        return None
    if connector_id is not None and name is None:
        errors.append(f"'mcp_connect[{idx}].name' must be a non-empty string when connector_id is set")
        return None

    return _McpConnectEntry(
        index=idx,
        target=target,
        name=name,
        connector_id=connector_id,
        headers=headers,
        auth=auth,
        management=management,
        description=description,
        access_token=access_token,
        defer_loading=defer_loading,
    )


def _validate_mcp_connector_entry(entry: _McpConnectEntry) -> None:
    payload: dict[str, Any] = {
        "name": entry.name,
        "description": entry.description,
        "management": entry.management,
        "connector_id": entry.connector_id,
        "headers": entry.headers,
        "access_token": entry.access_token,
        "auth": entry.auth,
    }
    if entry.defer_loading is not None:
        payload["defer_loading"] = entry.defer_loading
    MCPServerSettings.model_validate(payload)


def _validate_mcp_target_entry(entry: _McpConnectEntry) -> None:
    if entry.target is None:
        raise ValueError("'target' is required")

    overrides: dict[str, Any] = {}
    if entry.description is not None:
        overrides["description"] = entry.description
    if entry.management is not None:
        overrides["management"] = entry.management
    if entry.headers is not None:
        overrides["headers"] = entry.headers
    if entry.access_token is not None:
        overrides["access_token"] = entry.access_token
    if entry.defer_loading is not None:
        overrides["defer_loading"] = entry.defer_loading
    if entry.auth is not None:
        overrides["auth"] = entry.auth

    resolve_target_entry(
        target=entry.target,
        default_name=entry.name,
        overrides=overrides,
        source_path=f"mcp_connect[{entry.index}].target",
    )


def _validate_mcp_connect_entry(raw_entry: dict[str, Any], idx: int, errors: list[str]) -> None:
    entry = _parse_mcp_connect_entry(raw_entry, idx, errors)
    if entry is None:
        return

    try:
        if entry.connector_id is not None:
            _validate_mcp_connector_entry(entry)
        else:
            _validate_mcp_target_entry(entry)
    except Exception as exc:
        errors.append(f"Invalid mcp_connect target at index {idx}: {exc}")


def validate_mcp_connect_entries(value: Any, errors: list[str]) -> None:
    if value is None:
        return

    if not isinstance(value, list):
        errors.append("'mcp_connect' must be a list")
        return

    for idx, raw_entry in enumerate(value):
        if not is_str_object_dict(raw_entry):
            errors.append(f"'mcp_connect[{idx}]' must be a mapping")
            continue
        _validate_mcp_connect_entry(raw_entry, idx, errors)
