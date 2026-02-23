"""Safe AgentCard validation helpers."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from frontmatter import loads as load_frontmatter

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.exceptions import AgentConfigError, format_fast_agent_error
from fast_agent.core.validation import find_dependency_cycle
from fast_agent.mcp.connect_targets import resolve_target_entry

CARD_EXTENSIONS = {".md", ".markdown", ".yaml", ".yml"}

_CARD_REQUIRED_FIELDS = {
    "chain": ("sequence",),
    "parallel": ("fan_out",),
    "evaluator_optimizer": ("generator", "evaluator"),
    "router": ("agents",),
    "orchestrator": ("agents",),
    "iterative_planner": ("agents",),
    "maker": ("worker",),
    "smart": (),
}

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


def collect_agent_card_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        entry
        for entry in sorted(directory.iterdir())
        if entry.is_file() and entry.suffix.lower() in CARD_EXTENSIONS
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
        except Exception:  # noqa: BLE001
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
    if not path.exists() or path.suffix.lower() not in CARD_EXTENSIONS:
        return []
    return _scan_agent_card_files(
        [path],
        server_names=server_names,
        extra_agent_names=extra_agent_names,
    )


def _scan_agent_card_files(
    card_files: list[Path],
    *,
    server_names: set[str] | None = None,
    extra_agent_names: set[str] | None = None,
) -> list[AgentCardScanResult]:
    entries: list[AgentCardScanResult] = []
    name_to_paths: dict[str, list[Path]] = {}
    for card_path in card_files:
        errors: list[str] = []
        if (
            card_path.suffix.lower() in {".md", ".markdown"}
            and not _markdown_has_frontmatter(card_path)
        ):
            entries.append(
                AgentCardScanResult(
                    name=card_path.stem.replace(" ", "_"),
                    type="ignored",
                    path=card_path,
                    errors=[],
                    dependencies=set(),
                    ignored_reason="no frontmatter",
                )
            )
            continue
        try:
            raw, body = _load_card_raw(card_path)
        except Exception as exc:  # noqa: BLE001
            entries.append(
                AgentCardScanResult(
                    name="—",
                    type="unknown",
                    path=card_path,
                    errors=[str(exc)],
                    dependencies=set(),
                    ignored_reason=None,
                )
            )
            continue

        name = _normalize_card_name(raw.get("name"), card_path, errors)
        type_key = _normalize_card_type(raw.get("type"), errors)

        schema_version = raw.get("schema_version")
        if schema_version is not None and not isinstance(schema_version, int):
            errors.append("'schema_version' must be an integer")

        required_fields = _CARD_REQUIRED_FIELDS.get(type_key, ())
        for field in required_fields:
            if raw.get(field) is None:
                errors.append(f"Missing required field '{field}'")

        servers = _ensure_str_list(raw.get("servers"), "servers", errors)
        _validate_mcp_connect_entries(raw.get("mcp_connect"), errors)
        function_tools = _ensure_str_list(raw.get("function_tools"), "function_tools", errors)
        messages = _ensure_str_list(raw.get("messages"), "messages", errors)
        shell_cwd = _resolve_shell_cwd(raw.get("cwd"), errors)
        dependencies = _card_dependencies(type_key, raw, errors)

        instruction_texts: list[str] = []
        raw_instruction = raw.get("instruction")
        if isinstance(raw_instruction, str) and raw_instruction.strip():
            instruction_texts.append(raw_instruction)
        if isinstance(body, str) and body.strip():
            instruction_texts.append(body)

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

        entries.append(
            AgentCardScanResult(
                name=name,
                type=type_key,
                path=card_path,
                errors=errors,
                dependencies=dependencies,
                ignored_reason=None,
            )
        )

        name_to_paths.setdefault(name, []).append(card_path)

        if server_names is not None and servers:
            missing_servers = sorted(s for s in servers if s not in server_names)
            if missing_servers:
                errors.append(f"References missing servers: {', '.join(missing_servers)}")

        if function_tools:
            base_path = card_path.parent
            for spec in function_tools:
                error = _check_function_tool_spec(spec, base_path)
                if error:
                    errors.append(error)

        if messages:
            _validate_message_files(messages, card_path.parent, errors)

        if shell_cwd is not None:
            _validate_shell_cwd(shell_cwd, errors)

        entries[-1] = AgentCardScanResult(
            name=name,
            type=type_key,
            path=card_path,
            errors=errors,
            dependencies=dependencies,
            ignored_reason=None,
        )

    for name, paths in name_to_paths.items():
        if len(paths) <= 1:
            continue
        for idx, entry in enumerate(entries):
            if entry.path in paths:
                entries[idx] = AgentCardScanResult(
                    name=entry.name,
                    type=entry.type,
                    path=entry.path,
                    errors=entry.errors + [f"Duplicate agent name '{name}'"],
                    dependencies=entry.dependencies,
                    ignored_reason=entry.ignored_reason,
                )

    available_names = {
        entry.name
        for entry in entries
        if entry.name != "—" and entry.ignored_reason is None
    }
    if extra_agent_names:
        available_names |= extra_agent_names
    for idx, entry in enumerate(entries):
        missing = sorted(dep for dep in entry.dependencies if dep not in available_names)
        if missing:
            entries[idx] = AgentCardScanResult(
                name=entry.name,
                type=entry.type,
                path=entry.path,
                errors=entry.errors + [f"References missing agents: {', '.join(missing)}"],
                dependencies=entry.dependencies,
                ignored_reason=entry.ignored_reason,
            )

    cycle_candidates = sorted(available_names)
    if cycle_candidates:
        dependencies = {
            entry.name: {dep for dep in entry.dependencies if dep in available_names}
            for entry in entries
            if entry.name in available_names
        }
        cycle = find_dependency_cycle(cycle_candidates, dependencies)
        if cycle:
            cycle_message = f"Circular dependency detected: {' -> '.join(cycle)}"
            cycle_nodes = set(cycle)
            for idx, entry in enumerate(entries):
                if entry.name in cycle_nodes:
                    entries[idx] = AgentCardScanResult(
                        name=entry.name,
                        type=entry.type,
                        path=entry.path,
                        errors=entry.errors + [cycle_message],
                        dependencies=entry.dependencies,
                        ignored_reason=entry.ignored_reason,
                    )

    return entries


def _iter_file_placeholders(text: str) -> Iterable[str]:
    for match in _FILE_PLACEHOLDER_PATTERN.finditer(text or ""):
        value = match.group(1).strip()
        if value:
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

            config = agent_data.get("config")
            if config and getattr(config, "servers", None) and server_names is not None:
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

            if config and getattr(config, "function_tools", None):
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
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("AgentCard YAML must be a mapping")
        return data, None
    if suffix in {".md", ".markdown"}:
        raw_text = path.read_text(encoding="utf-8")
        if raw_text.startswith("\ufeff"):
            raw_text = raw_text.lstrip("\ufeff")
        post = load_frontmatter(raw_text)
        metadata = post.metadata or {}
        if not isinstance(metadata, dict):
            raise ValueError("Frontmatter must be a mapping")
        return dict(metadata), post.content or ""
    raise ValueError("Unsupported AgentCard file extension")


def _markdown_has_frontmatter(path: Path) -> bool:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped in ("---", "+++")
    return False


def _normalize_card_name(raw_name: Any, path: Path, errors: list[str]) -> str:
    if raw_name is None:
        return path.stem.replace(" ", "_")
    if not isinstance(raw_name, str) or not raw_name.strip():
        errors.append("'name' must be a non-empty string")
        return path.stem.replace(" ", "_")
    return raw_name.strip().replace(" ", "_")


def _normalize_card_type(raw_type: Any, errors: list[str]) -> str:
    if raw_type is None:
        return "agent"
    if not isinstance(raw_type, str):
        errors.append("'type' must be a string")
        return "agent"
    type_key = raw_type.strip().lower() or "agent"
    if type_key not in {
        "agent",
        "smart",
        "chain",
        "parallel",
        "evaluator_optimizer",
        "router",
        "orchestrator",
        "iterative_planner",
        "maker",
    }:
        errors.append(f"Unsupported agent type '{raw_type}'")
    return type_key


def _ensure_str_list(value: Any, field: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        entries: list[str] = []
        for entry in value:
            if not isinstance(entry, str) or not entry.strip():
                errors.append(f"'{field}' entries must be non-empty strings")
                continue
            entries.append(entry)
        return entries
    errors.append(f"'{field}' must be a string or list of strings")
    return []


def _ensure_str(value: Any, field: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.append(f"'{field}' must be a non-empty string")
        return None
    return value.strip()


def _resolve_shell_cwd(value: Any, errors: list[str]) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append("'cwd' must be a string")
        return None

    cwd_value = value.strip()
    if not cwd_value:
        errors.append("'cwd' must be a non-empty string")
        return None

    configured = Path(cwd_value).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (Path.cwd() / configured).resolve()


def _validate_shell_cwd(cwd: Path, errors: list[str]) -> None:
    if not cwd.exists():
        errors.append(f"Shell cwd does not exist ({cwd})")
        return
    if not cwd.is_dir():
        errors.append(f"Shell cwd is not a directory ({cwd})")


def _validate_mcp_connect_entries(value: Any, errors: list[str]) -> None:
    if value is None:
        return

    if not isinstance(value, list):
        errors.append("'mcp_connect' must be a list")
        return

    for idx, raw_entry in enumerate(value):
        if not isinstance(raw_entry, dict):
            errors.append(f"'mcp_connect[{idx}]' must be a mapping")
            continue

        unknown_keys = set(raw_entry.keys()) - {"target", "name", "headers", "auth"}
        if unknown_keys:
            unknown_text = ", ".join(sorted(str(key) for key in unknown_keys))
            errors.append(f"'mcp_connect[{idx}]' has unsupported keys: {unknown_text}")

        target_value = raw_entry.get("target")
        if not isinstance(target_value, str) or not target_value.strip():
            errors.append(f"'mcp_connect[{idx}].target' must be a non-empty string")
            continue

        name_value = raw_entry.get("name")
        if name_value is not None and (not isinstance(name_value, str) or not name_value.strip()):
            errors.append(f"'mcp_connect[{idx}].name' must be a non-empty string")
            continue

        headers_value = raw_entry.get("headers")
        resolved_headers: dict[str, str] | None = None
        if headers_value is not None:
            if not isinstance(headers_value, dict):
                errors.append(f"'mcp_connect[{idx}].headers' must be a mapping")
                continue
            resolved_headers = {}
            for key, header_value in headers_value.items():
                if not isinstance(key, str) or not key.strip():
                    errors.append(f"'mcp_connect[{idx}].headers' keys must be non-empty strings")
                    resolved_headers = None
                    break
                if not isinstance(header_value, str):
                    errors.append(f"'mcp_connect[{idx}].headers' values must be strings")
                    resolved_headers = None
                    break
                resolved_headers[key] = header_value
            if headers_value and resolved_headers is None:
                continue

        auth_value = raw_entry.get("auth")
        resolved_auth: dict[str, Any] | None = None
        if auth_value is not None:
            if not isinstance(auth_value, dict):
                errors.append(f"'mcp_connect[{idx}].auth' must be a mapping")
                continue
            resolved_auth = dict(auth_value)

        try:
            overrides: dict[str, Any] = {}
            if resolved_headers is not None:
                overrides["headers"] = resolved_headers
            if resolved_auth is not None:
                overrides["auth"] = resolved_auth
            resolve_target_entry(
                target=target_value.strip(),
                default_name=name_value.strip() if isinstance(name_value, str) else None,
                overrides=overrides,
                source_path=f"mcp_connect[{idx}].target",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as card scan issue
            errors.append(f"Invalid mcp_connect target at index {idx}: {exc}")


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
        except Exception as exc:  # noqa: BLE001
            errors.append(f"History file failed to load ({message_path}): {exc}")


def _check_function_tool_spec(spec: str, base_path: Path) -> str | None:
    if ":" not in spec:
        return f"Invalid function tool spec '{spec}'"
    module_path_str, func_name = spec.rsplit(":", 1)
    module_path = Path(module_path_str)
    if not module_path.is_absolute():
        module_path = (base_path / module_path).resolve()
    if not module_path.exists():
        return f"Function tool module file not found ({module_path})"
    if module_path.suffix.lower() != ".py":
        return f"Function tool module must be a .py file ({module_path})"
    try:
        module_text = module_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        return f"Failed to read tool module ({module_path}): {exc}"
    try:
        tree = ast.parse(module_text)
    except Exception as exc:  # noqa: BLE001
        return f"Failed to parse tool module ({module_path}): {exc}"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return None
    return f"Function '{func_name}' not found in {module_path.name}"


def _card_dependencies(type_key: str, raw: dict[str, Any], errors: list[str]) -> set[str]:
    deps: set[str] = set()
    if type_key in {"agent", "smart"}:
        deps.update(_ensure_str_list(raw.get("agents"), "agents", errors))
    elif type_key == "chain":
        deps.update(_ensure_str_list(raw.get("sequence"), "sequence", errors))
    elif type_key == "parallel":
        deps.update(_ensure_str_list(raw.get("fan_out"), "fan_out", errors))
        fan_in = _ensure_str(raw.get("fan_in"), "fan_in", errors)
        if fan_in:
            deps.add(fan_in)
    elif type_key in {"router", "orchestrator", "iterative_planner"}:
        deps.update(_ensure_str_list(raw.get("agents"), "agents", errors))
    elif type_key == "evaluator_optimizer":
        evaluator = _ensure_str(raw.get("evaluator"), "evaluator", errors)
        generator = _ensure_str(raw.get("generator"), "generator", errors)
        if evaluator:
            deps.add(evaluator)
        if generator:
            deps.add(generator)
    elif type_key == "maker":
        worker = _ensure_str(raw.get("worker"), "worker", errors)
        if worker:
            deps.add(worker)
    return deps


def _loaded_agent_dependencies(agent_data: dict[str, Any]) -> set[str]:
    agent_type = agent_data.get("type")
    if isinstance(agent_type, AgentType):
        agent_type = agent_type.value
    if not isinstance(agent_type, str):
        return set()

    deps: set[str] = set()
    if agent_type in {AgentType.BASIC.value, AgentType.SMART.value}:
        deps.update(agent_data.get("child_agents") or [])
    elif agent_type == AgentType.CHAIN.value:
        deps.update(agent_data.get("sequence") or [])
    elif agent_type == AgentType.PARALLEL.value:
        deps.update(agent_data.get("fan_out") or [])
        fan_in = agent_data.get("fan_in")
        if fan_in:
            deps.add(fan_in)
    elif agent_type in {AgentType.ORCHESTRATOR.value, AgentType.ITERATIVE_PLANNER.value}:
        deps.update(agent_data.get("child_agents") or [])
    elif agent_type == AgentType.ROUTER.value:
        deps.update(agent_data.get("router_agents") or [])
    elif agent_type == AgentType.EVALUATOR_OPTIMIZER.value:
        evaluator = agent_data.get("evaluator")
        generator = agent_data.get("generator")
        if evaluator:
            deps.add(evaluator)
        if generator:
            deps.add(generator)
    elif agent_type == AgentType.MAKER.value:
        worker = agent_data.get("worker")
        if worker:
            deps.add(worker)
    return {dep for dep in deps if isinstance(dep, str)}


def _iter_function_tool_specs(tool_specs: Iterable[Any]) -> Iterable[str]:
    for spec in tool_specs:
        if isinstance(spec, str):
            yield spec
