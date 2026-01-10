"""Safe AgentCard validation helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from frontmatter import loads as load_frontmatter

from fast_agent.agents.agent_types import AgentType

CARD_EXTENSIONS = {".md", ".markdown", ".yaml", ".yml"}

_CARD_REQUIRED_FIELDS = {
    "chain": ("sequence",),
    "parallel": ("fan_out",),
    "evaluator_optimizer": ("generator", "evaluator"),
    "router": ("agents",),
    "orchestrator": ("agents",),
    "iterative_planner": ("agents",),
    "maker": ("worker",),
}


@dataclass(frozen=True)
class AgentCardScanResult:
    name: str
    type: str
    path: Path
    errors: list[str]
    dependencies: set[str]


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


def scan_agent_card_directory(
    directory: Path,
    *,
    server_names: set[str] | None = None,
) -> list[AgentCardScanResult]:
    entries: list[AgentCardScanResult] = []
    card_files = collect_agent_card_files(directory)
    if not card_files:
        return entries

    name_to_paths: dict[str, list[Path]] = {}
    for card_path in card_files:
        errors: list[str] = []
        try:
            raw, _body = _load_card_raw(card_path)
        except Exception as exc:  # noqa: BLE001
            entries.append(
                AgentCardScanResult(
                    name="—",
                    type="unknown",
                    path=card_path,
                    errors=[str(exc)],
                    dependencies=set(),
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
        function_tools = _ensure_str_list(raw.get("function_tools"), "function_tools", errors)
        messages = _ensure_str_list(raw.get("messages"), "messages", errors)
        dependencies = _card_dependencies(type_key, raw, errors)

        entries.append(
            AgentCardScanResult(
                name=name,
                type=type_key,
                path=card_path,
                errors=errors,
                dependencies=dependencies,
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
            for message_path_str in messages:
                message_path = Path(message_path_str).expanduser()
                if not message_path.is_absolute():
                    message_path = (card_path.parent / message_path).resolve()
                if not message_path.exists():
                    errors.append(f"History file not found ({message_path})")

        entries[-1] = AgentCardScanResult(
            name=name,
            type=type_key,
            path=card_path,
            errors=errors,
            dependencies=dependencies,
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
                )

    available_names = {entry.name for entry in entries if entry.name != "—"}
    for idx, entry in enumerate(entries):
        missing = sorted(dep for dep in entry.dependencies if dep not in available_names)
        if missing:
            entries[idx] = AgentCardScanResult(
                name=entry.name,
                type=entry.type,
                path=entry.path,
                errors=entry.errors + [f"References missing agents: {', '.join(missing)}"],
                dependencies=entry.dependencies,
            )

    return entries


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
    if type_key == "agent":
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
    if agent_type == AgentType.BASIC.value:
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
