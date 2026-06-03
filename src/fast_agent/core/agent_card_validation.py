"""Safe AgentCard validation helpers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from frontmatter import loads as load_frontmatter

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import MCPServerSettings, resolve_env_vars
from fast_agent.core.agent_card_paths import (
    is_agent_card_path,
    is_markdown_agent_card_path,
    is_yaml_agent_card_path,
)
from fast_agent.core.agent_card_rules import (
    ALLOWED_FIELDS_BY_TYPE,
    MCP_CONNECT_ALLOWED_KEYS,
    REQUIRED_FIELDS_BY_TYPE,
    CardType,
    normalize_card_type,
)
from fast_agent.core.exceptions import AgentConfigError, format_fast_agent_error
from fast_agent.core.tool_input_schema import validate_tool_input_schema
from fast_agent.core.validation import (
    collect_dependencies_from_fields,
    find_dependency_cycle,
    get_agent_dependencies,
    get_card_dependency_field_specs,
)
from fast_agent.mcp.connect_targets import resolve_target_entry
from fast_agent.tools.function_tool_config import (
    FunctionToolSpec,
    function_tool_entrypoint,
    parse_function_tool_card_entry,
)
from fast_agent.tools.python_file_loader import parse_callable_file_spec
from fast_agent.utils.text import strip_casefold, strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
from fast_agent.utils.type_narrowing import is_str_object_dict

_FILE_PLACEHOLDER_PATTERN = re.compile(r"\{\{file:([^}]+)\}\}")


@dataclass(frozen=True)
class AgentCardScanResult:
    name: str
    type: str
    path: Path
    errors: list[str]
    dependencies: set[str]
    ignored_reason: str | None = None


@dataclass(frozen=True)
class LoadedAgentIssue:
    name: str
    source: str
    message: str


@dataclass(frozen=True)
class _ScannedCardDetails:
    result: AgentCardScanResult
    servers: list[str]
    function_tools: list[str | FunctionToolSpec]
    messages: list[str]
    shell_cwd: Path | None


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


def collect_agent_card_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        entry
        for entry in sorted(directory.iterdir())
        if entry.is_file() and is_agent_card_path(entry)
    ]




def collect_agent_card_names(sources: Iterable[str]) -> set[str]:
    """Collect AgentCard names from local files or directories.

    URL sources are ignored. Any card parsing errors are treated as best-effort
    and skipped, mirroring validation behavior elsewhere.
    """
    names: set[str] = set()
    for source in sources:
        if source.startswith(("http://", "https://")):
            continue
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            entries = scan_agent_card_directory(source_path)
            for entry in entries:
                if entry.name != "—" and entry.ignored_reason is None:
                    names.add(entry.name)
            continue
        try:
            from fast_agent.core.agent_card_loader import load_agent_cards

            cards = load_agent_cards(source_path)
        except Exception:
            continue
        for card in cards:
            names.add(card.name)

    return names


def scan_agent_card_directory(
    directory: Path,
    *,
    server_names: set[str] | None = None,
    extra_agent_names: set[str] | None = None,
) -> list[AgentCardScanResult]:
    card_files = collect_agent_card_files(directory)
    if not card_files:
        return []
    return _scan_agent_card_files(
        card_files,
        server_names=server_names,
        extra_agent_names=extra_agent_names,
    )


def scan_agent_card_path(
    path: Path,
    *,
    server_names: set[str] | None = None,
    extra_agent_names: set[str] | None = None,
) -> list[AgentCardScanResult]:
    path = path.expanduser()
    if path.is_dir():
        return scan_agent_card_directory(
            path,
            server_names=server_names,
            extra_agent_names=extra_agent_names,
        )
    if not path.exists() or not is_agent_card_path(path):
        return []
    return _scan_agent_card_files(
        [path],
        server_names=server_names,
        extra_agent_names=extra_agent_names,
    )


def _replace_scan_errors(
    entry: AgentCardScanResult,
    errors: list[str],
) -> AgentCardScanResult:
    return AgentCardScanResult(
        name=entry.name,
        type=entry.type,
        path=entry.path,
        errors=errors,
        dependencies=entry.dependencies,
        ignored_reason=entry.ignored_reason,
    )


def _append_scan_error(entry: AgentCardScanResult, error: str) -> AgentCardScanResult:
    return _replace_scan_errors(entry, [*entry.errors, error])


def _empty_scanned_card_details(
    *,
    name: str,
    type_key: str,
    path: Path,
    errors: list[str] | None = None,
    ignored_reason: str | None = None,
) -> _ScannedCardDetails:
    return _ScannedCardDetails(
        result=AgentCardScanResult(
            name=name,
            type=type_key,
            path=path,
            errors=errors or [],
            dependencies=set(),
            ignored_reason=ignored_reason,
        ),
        servers=[],
        function_tools=[],
        messages=[],
        shell_cwd=None,
    )


def _should_ignore_markdown_card(card_path: Path) -> bool:
    return is_markdown_agent_card_path(card_path) and not _markdown_has_frontmatter(card_path)


def _load_scan_card_raw(card_path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        return _load_card_raw(card_path)
    except Exception as exc:
        raise ValueError(str(exc)) from exc


def _instruction_texts(raw_instruction: Any, body: str | None) -> list[str]:
    instruction_texts: list[str] = []
    if (instruction := strip_str_to_none(raw_instruction)) is not None:
        instruction_texts.append(instruction)
    if (body_text := strip_to_none(body)) is not None:
        instruction_texts.append(body_text)
    return instruction_texts


def _validate_instruction_file_placeholders(
    instruction_texts: list[str],
    errors: list[str],
) -> None:
    for instruction_text in instruction_texts:
        for file_path_str in _iter_file_placeholders(instruction_text):
            file_path = Path(file_path_str).expanduser()
            if file_path.is_absolute():
                errors.append(
                    "Instruction file template paths must be relative "
                    f"({{{{file:{file_path_str}}}}})"
                )
                continue
            resolved_path = (Path.cwd() / file_path).resolve()
            if not resolved_path.exists():
                errors.append(
                    "Instruction file not found "
                    f"({{{{file:{file_path_str}}}}} -> {resolved_path})"
                )


def _scan_single_agent_card_file(card_path: Path) -> _ScannedCardDetails:
    errors: list[str] = []
    if _should_ignore_markdown_card(card_path):
        return _empty_scanned_card_details(
            name=card_path.stem.replace(" ", "_"),
            type_key="ignored",
            path=card_path,
            ignored_reason="no frontmatter",
        )

    try:
        raw, body = _load_scan_card_raw(card_path)
    except ValueError as exc:
        return _empty_scanned_card_details(
            name="—",
            type_key="unknown",
            path=card_path,
            errors=[str(exc)],
        )

    name = _normalize_card_name(raw.get("name"), card_path, errors)
    type_key = _normalize_card_type(raw.get("type"), errors)

    schema_version = raw.get("schema_version")
    if schema_version is not None and (
        isinstance(schema_version, bool) or not isinstance(schema_version, int)
    ):
        errors.append("'schema_version' must be an integer")

    unknown_fields = set(raw.keys()) - ALLOWED_FIELDS_BY_TYPE[type_key]
    if unknown_fields:
        unknown_text = ", ".join(sorted(unknown_fields))
        errors.append(f"Unsupported fields for type '{type_key}': {unknown_text}")

    required_fields = REQUIRED_FIELDS_BY_TYPE[type_key]
    errors.extend(
        f"Missing required field '{field}'"
        for field in required_fields
        if raw.get(field) is None
    )

    servers = _ensure_str_list(raw.get("servers"), "servers", errors)
    _validate_mcp_connect_entries(raw.get("mcp_connect"), errors)
    function_tools = _ensure_function_tool_list(raw.get("function_tools"), "function_tools", errors)
    messages = _ensure_str_list(raw.get("messages"), "messages", errors)
    _validate_tool_input_schema(raw.get("tool_input_schema"), errors)
    shell_cwd = _resolve_shell_cwd(raw.get("cwd"), errors)
    dependencies = _card_dependencies(type_key, raw, errors)
    _validate_instruction_file_placeholders(
        _instruction_texts(raw.get("instruction"), body),
        errors,
    )

    return _ScannedCardDetails(
        result=AgentCardScanResult(
            name=name,
            type=type_key,
            path=card_path,
            errors=errors,
            dependencies=dependencies,
            ignored_reason=None,
        ),
        servers=servers,
        function_tools=function_tools,
        messages=messages,
        shell_cwd=shell_cwd,
    )


def _apply_supplemental_scan_checks(
    details: _ScannedCardDetails,
    *,
    server_names: set[str] | None,
) -> AgentCardScanResult:
    entry = details.result
    errors = list(entry.errors)

    if server_names is not None and details.servers:
        missing_servers = sorted(server for server in details.servers if server not in server_names)
        if missing_servers:
            errors.append(f"References missing servers: {', '.join(missing_servers)}")

    if details.function_tools:
        base_path = entry.path.parent
        for spec in details.function_tools:
            entrypoint = function_tool_entrypoint(spec)
            if entrypoint is None:
                continue
            error = _check_function_tool_spec(entrypoint, base_path)
            if error:
                errors.append(error)

    if details.messages:
        _validate_message_files(details.messages, entry.path.parent, errors)

    if details.shell_cwd is not None:
        _validate_shell_cwd(details.shell_cwd, errors)

    return _replace_scan_errors(entry, errors)


def _apply_duplicate_name_errors(entries: list[AgentCardScanResult]) -> list[AgentCardScanResult]:
    name_to_paths: dict[str, list[Path]] = {}
    for entry in entries:
        if entry.name == "—" or entry.ignored_reason is not None:
            continue
        name_to_paths.setdefault(entry.name, []).append(entry.path)

    updated_entries = list(entries)
    for name, paths in name_to_paths.items():
        if len(paths) <= 1:
            continue
        for idx, entry in enumerate(updated_entries):
            if entry.path in paths:
                updated_entries[idx] = _append_scan_error(
                    entry,
                    f"Duplicate agent name '{name}'",
                )
    return updated_entries


def _available_scan_names(
    entries: list[AgentCardScanResult],
    extra_agent_names: set[str] | None,
) -> set[str]:
    available_names = {
        entry.name for entry in entries if entry.name != "—" and entry.ignored_reason is None
    }
    if extra_agent_names:
        available_names |= extra_agent_names
    return available_names


def _apply_missing_dependency_errors(
    entries: list[AgentCardScanResult],
    *,
    available_names: set[str],
) -> list[AgentCardScanResult]:
    updated_entries = list(entries)
    for idx, entry in enumerate(updated_entries):
        missing = sorted(dep for dep in entry.dependencies if dep not in available_names)
        if missing:
            updated_entries[idx] = _append_scan_error(
                entry,
                f"References missing agents: {', '.join(missing)}",
            )
    return updated_entries


def _apply_cycle_errors(
    entries: list[AgentCardScanResult],
    *,
    available_names: set[str],
) -> list[AgentCardScanResult]:
    cycle_candidates = sorted(available_names)
    if not cycle_candidates:
        return entries

    dependencies = {
        entry.name: {dep for dep in entry.dependencies if dep in available_names}
        for entry in entries
        if entry.name in available_names
    }
    cycle = find_dependency_cycle(cycle_candidates, dependencies)
    if not cycle:
        return entries

    cycle_message = f"Circular dependency detected: {' -> '.join(cycle)}"
    cycle_nodes = set(cycle)
    updated_entries = list(entries)
    for idx, entry in enumerate(updated_entries):
        if entry.name in cycle_nodes:
            updated_entries[idx] = _append_scan_error(entry, cycle_message)
    return updated_entries


def _scan_agent_card_files(
    card_files: list[Path],
    *,
    server_names: set[str] | None = None,
    extra_agent_names: set[str] | None = None,
) -> list[AgentCardScanResult]:
    entries = [
        _apply_supplemental_scan_checks(
            _scan_single_agent_card_file(card_path),
            server_names=server_names,
        )
        for card_path in card_files
    ]
    entries = _apply_duplicate_name_errors(entries)
    available_names = _available_scan_names(entries, extra_agent_names)
    entries = _apply_missing_dependency_errors(entries, available_names=available_names)
    return _apply_cycle_errors(entries, available_names=available_names)


def _iter_file_placeholders(text: str) -> Iterable[str]:
    for match in _FILE_PLACEHOLDER_PATTERN.finditer(text or ""):
        value = strip_to_none(match.group(1))
        if value is not None:
            yield value


def find_loaded_agent_issues(
    agents: Mapping[str, dict[str, Any]],
    *,
    extra_agent_names: set[str] | None = None,
    server_names: set[str] | None = None,
) -> tuple[list[LoadedAgentIssue], set[str]]:
    issues: list[LoadedAgentIssue] = []
    removed: set[str] = set()
    available = set(agents.keys()) | (extra_agent_names or set())
    remaining = set(agents.keys())

    while True:
        invalid_names: list[str] = []
        for name in sorted(remaining):
            agent_data = agents[name]
            source_path = str(agent_data.get("source_path") or name)
            missing = sorted(dep for dep in _loaded_agent_dependencies(agent_data) if dep not in available)
            if missing:
                issues.append(
                    LoadedAgentIssue(
                        name=name,
                        source=source_path,
                        message=f"Agent '{name}' references missing components: {', '.join(missing)}",
                    )
                )
                invalid_names.append(name)
                continue

            raw_config = agent_data.get("config")
            config = raw_config if isinstance(raw_config, AgentConfig) else None
            if config is not None and config.servers and server_names is not None:
                missing_servers = sorted(s for s in config.servers if s not in server_names)
                if missing_servers:
                    issues.append(
                        LoadedAgentIssue(
                            name=name,
                            source=source_path,
                            message=(
                                f"Agent '{name}' references missing servers: "
                                f"{', '.join(missing_servers)}"
                            ),
                        )
                    )
                    invalid_names.append(name)
                    continue

            if config is not None and config.function_tools:
                base_path = Path(source_path).expanduser().resolve().parent
                for spec in _iter_function_tool_specs(config.function_tools):
                    error = _check_function_tool_spec(spec, base_path)
                    if error:
                        issues.append(
                            LoadedAgentIssue(
                                name=name,
                                source=source_path,
                                message=error,
                            )
                        )
                        invalid_names.append(name)
                        break

        if not invalid_names:
            break

        invalid_set = set(invalid_names)
        removed |= invalid_set
        remaining -= invalid_set
        available -= invalid_set

    return issues, removed


def _load_card_raw(path: Path) -> tuple[dict[str, Any], str | None]:
    if is_yaml_agent_card_path(path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("AgentCard YAML must be a mapping")
        resolved = resolve_env_vars(data)
        if not isinstance(resolved, dict):
            raise ValueError("AgentCard YAML must be a mapping")
        return resolved, None
    if is_markdown_agent_card_path(path):
        raw_text = path.read_text(encoding="utf-8")
        if raw_text.startswith("\ufeff"):
            raw_text = raw_text.lstrip("\ufeff")
        post = load_frontmatter(raw_text)
        metadata = post.metadata or {}
        if not isinstance(metadata, dict):
            raise ValueError("Frontmatter must be a mapping")
        resolved = resolve_env_vars(dict(metadata))
        if not isinstance(resolved, dict):
            raise ValueError("Frontmatter must be a mapping")
        return resolved, post.content or ""
    raise ValueError("Unsupported AgentCard file extension")


def _markdown_has_frontmatter(path: Path) -> bool:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")
    for line in raw_text.splitlines():
        marker = strip_to_none(line)
        if marker is None:
            continue
        return marker in ("---", "+++")
    return False


def _normalize_card_name(raw_name: Any, path: Path, errors: list[str]) -> str:
    if raw_name is None:
        return path.stem.replace(" ", "_")
    name = strip_str_to_none(raw_name)
    if name is None:
        errors.append("'name' must be a non-empty string")
        return path.stem.replace(" ", "_")
    return name.replace(" ", "_")


def _normalize_card_type(raw_type: Any, errors: list[str]) -> CardType:
    if raw_type is None:
        return "agent"
    if not isinstance(raw_type, str):
        errors.append("'type' must be a string")
        return "agent"
    type_key = normalize_card_type(raw_type)
    if type_key is None:
        errors.append(f"Unsupported agent type '{raw_type}'")
        return "agent"
    return type_key


def _ensure_str_list(value: Any, field: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        entries: list[str] = []
        for entry in value:
            if strip_str_to_none(entry) is None:
                errors.append(f"'{field}' entries must be non-empty strings")
                continue
            entries.append(entry)
        return entries
    errors.append(f"'{field}' must be a string or list of strings")
    return []


def _ensure_str(value: Any, field: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    normalized = strip_str_to_none(value)
    if normalized is None:
        errors.append(f"'{field}' must be a non-empty string")
        return None
    return normalized


def _resolve_shell_cwd(value: Any, errors: list[str]) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append("'cwd' must be a string")
        return None

    cwd_value = strip_str_to_none(value)
    if cwd_value is None:
        errors.append("'cwd' must be a non-empty string")
        return None

    configured = Path(cwd_value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (Path.cwd() / configured).resolve()


def _validate_tool_input_schema(value: Any, errors: list[str]) -> None:
    validation = validate_tool_input_schema(value)
    errors.extend(f"'tool_input_schema' {error}" for error in validation.errors)


def _validate_shell_cwd(cwd: Path, errors: list[str]) -> None:
    if not cwd.exists():
        errors.append(f"Shell cwd does not exist ({cwd})")
        return
    if not cwd.is_dir():
        errors.append(f"Shell cwd is not a directory ({cwd})")


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


def _validate_mcp_connect_entries(value: Any, errors: list[str]) -> None:
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


def _resolve_message_path(message_path_str: str, base_path: Path) -> Path:
    message_path = Path(message_path_str).expanduser()
    if not message_path.is_absolute():
        message_path = (base_path / message_path).resolve()
    return message_path


def _validate_message_files(
    messages: list[str],
    base_path: Path,
    errors: list[str],
) -> None:
    message_paths: list[Path] = []
    for message_path_str in messages:
        message_path = _resolve_message_path(message_path_str, base_path)
        if not message_path.exists():
            errors.append(f"History file not found ({message_path})")
            continue
        message_paths.append(message_path)

    if not message_paths:
        return

    from fast_agent.mcp.prompts.prompt_load import load_prompt

    for message_path in message_paths:
        try:
            load_prompt(message_path)
        except AgentConfigError as exc:
            errors.append(
                " ".join(
                    [
                        f"History file failed to load ({message_path}):",
                        format_fast_agent_error(exc),
                    ]
                )
            )
        except Exception as exc:
            errors.append(f"History file failed to load ({message_path}): {exc}")


def _check_function_tool_spec(spec: str, base_path: Path) -> str | None:
    parsed = _parse_function_tool_spec(spec, base_path)
    if isinstance(parsed, str):
        return parsed

    module_path, func_name = parsed
    tree = _parse_function_tool_module(module_path)
    if isinstance(tree, str):
        return tree

    if _module_defines_function(tree, func_name):
        return None
    return f"Function '{func_name}' not found in {module_path.name}"


def _parse_function_tool_spec(spec: str, base_path: Path) -> tuple[Path, str] | str:
    try:
        parsed = parse_callable_file_spec(
            spec,
            invalid_message="Invalid function tool spec '{spec}'",
        )
    except AgentConfigError as exc:
        return str(exc)

    module_path = Path(parsed.module_path_text)
    if not module_path.is_absolute():
        module_path = (base_path / module_path).resolve()

    if not module_path.exists():
        return f"Function tool module file not found ({module_path})"
    if strip_casefold(module_path.suffix) != ".py":
        return f"Function tool module must be a .py file ({module_path})"
    return module_path, parsed.callable_name


def _parse_function_tool_module(module_path: Path) -> ast.Module | str:
    try:
        module_text = module_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return f"Failed to read tool module ({module_path}): {exc}"
    try:
        return ast.parse(module_text)
    except SyntaxError as exc:
        return f"Failed to parse tool module ({module_path}): {exc}"


def _module_defines_function(tree: ast.Module, func_name: str) -> bool:
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
        for node in tree.body
    )


def _card_dependencies(type_key: str, raw: dict[str, Any], errors: list[str]) -> set[str]:
    dependency_fields = get_card_dependency_field_specs(type_key)
    if not dependency_fields:
        return set()

    normalized_raw: dict[str, Any] = {}
    for field_spec in dependency_fields:
        field_name = field_spec.field_name
        if field_spec.multiple:
            normalized_raw[field_name] = _ensure_str_list(raw.get(field_name), field_name, errors)
            continue
        normalized_raw[field_name] = _ensure_str(raw.get(field_name), field_name, errors)

    return collect_dependencies_from_fields(normalized_raw, dependency_fields)


def _loaded_agent_dependencies(agent_data: dict[str, Any]) -> set[str]:
    return get_agent_dependencies(agent_data)


def _iter_function_tool_specs(tool_specs: Iterable[Any]) -> Iterable[str]:
    for spec in tool_specs:
        entrypoint = function_tool_entrypoint(spec)
        if entrypoint:
            yield entrypoint


def _ensure_function_tool_list(
    raw_value: object,
    field_name: str,
    errors: list[str],
) -> list[str | FunctionToolSpec]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        try:
            return [
                parse_function_tool_card_entry(
                    raw_value,
                    field_path=field_name,
                )
            ]
        except ValueError as exc:
            errors.append(str(exc))
            return []
    if not isinstance(raw_value, list):
        errors.append(f"'{field_name}' must be a string or list")
        return []

    values: list[str | FunctionToolSpec] = []
    for index, entry in enumerate(raw_value):
        try:
            values.append(
                parse_function_tool_card_entry(
                    entry,
                    field_path=f"{field_name}[{index}]",
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
    return values
