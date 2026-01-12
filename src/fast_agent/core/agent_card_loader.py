"""AgentCard loader and export helpers for Markdown/YAML card files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import frontmatter
import yaml

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
from fast_agent.core.direct_decorators import _resolve_instruction
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.types import RequestParams

_TYPE_MAP: dict[str, AgentType] = {
    "agent": AgentType.BASIC,
    "chain": AgentType.CHAIN,
    "parallel": AgentType.PARALLEL,
    "evaluator_optimizer": AgentType.EVALUATOR_OPTIMIZER,
    "router": AgentType.ROUTER,
    "orchestrator": AgentType.ORCHESTRATOR,
    "iterative_planner": AgentType.ITERATIVE_PLANNER,
    "maker": AgentType.MAKER,
}

_COMMON_FIELDS = {"type", "name", "instruction", "description", "default", "tool_only", "schema_version"}

_ALLOWED_FIELDS_BY_TYPE: dict[str, set[str]] = {
    "agent": {
        *_COMMON_FIELDS,
        "agents",
        "servers",
        "tools",
        "resources",
        "prompts",
        "skills",
        "model",
        "use_history",
        "request_params",
        "human_input",
        "api_key",
        "history_source",
        "history_merge_target",
        "max_parallel",
        "child_timeout_sec",
        "max_display_instances",
        "function_tools",
        "tool_hooks",
        "messages",
        "shell",
        "cwd",
    },
    "chain": {
        *_COMMON_FIELDS,
        "sequence",
        "cumulative",
    },
    "parallel": {
        *_COMMON_FIELDS,
        "fan_out",
        "fan_in",
        "include_request",
    },
    "evaluator_optimizer": {
        *_COMMON_FIELDS,
        "generator",
        "evaluator",
        "min_rating",
        "max_refinements",
        "refinement_instruction",
        "messages",
    },
    "router": {
        *_COMMON_FIELDS,
        "agents",
        "servers",
        "tools",
        "resources",
        "prompts",
        "model",
        "use_history",
        "request_params",
        "human_input",
        "api_key",
        "messages",
    },
    "orchestrator": {
        *_COMMON_FIELDS,
        "agents",
        "model",
        "use_history",
        "request_params",
        "human_input",
        "api_key",
        "plan_type",
        "plan_iterations",
        "messages",
    },
    "iterative_planner": {
        *_COMMON_FIELDS,
        "agents",
        "model",
        "request_params",
        "api_key",
        "plan_iterations",
        "messages",
    },
    "MAKER": {
        *_COMMON_FIELDS,
        "worker",
        "k",
        "max_samples",
        "match_strategy",
        "red_flag_max_length",
        "messages",
    },
}

_REQUIRED_FIELDS_BY_TYPE: dict[str, set[str]] = {
    "agent": set(),
    "chain": {"sequence"},
    "parallel": {"fan_out"},
    "evaluator_optimizer": {"generator", "evaluator"},
    "router": {"agents"},
    "orchestrator": {"agents"},
    "iterative_planner": {"agents"},
    "MAKER": {"worker"},
}

_HISTORY_DELIMITERS = {"---USER", "---ASSISTANT", "---RESOURCE"}

_AGENT_TYPE_TO_CARD_TYPE: dict[str, str] = {
    AgentType.BASIC.value: "agent",
    AgentType.CHAIN.value: "chain",
    AgentType.PARALLEL.value: "parallel",
    AgentType.EVALUATOR_OPTIMIZER.value: "evaluator_optimizer",
    AgentType.ROUTER.value: "router",
    AgentType.ORCHESTRATOR.value: "orchestrator",
    AgentType.ITERATIVE_PLANNER.value: "iterative_planner",
    AgentType.MAKER.value: "MAKER",
}

_DEFAULT_USE_HISTORY_BY_TYPE: dict[str, bool] = {
    "agent": True,
    "chain": True,
    "parallel": True,
    "evaluator_optimizer": True,
    "router": False,
    "orchestrator": False,
    "iterative_planner": False,
    "MAKER": True,
}


@dataclass(frozen=True)
class LoadedAgentCard:
    name: str
    path: Path
    agent_data: dict[str, Any]
    message_files: list[Path]


def load_agent_cards(path: Path) -> list[LoadedAgentCard]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise AgentConfigError(f"AgentCard path not found: {path}")

    if path.is_dir():
        cards: list[LoadedAgentCard] = []
        for entry in sorted(path.iterdir()):
            if entry.is_dir():
                continue
            if entry.suffix.lower() not in {".md", ".markdown", ".yaml", ".yml"}:
                continue
            if entry.suffix.lower() in {".md", ".markdown"} and not _markdown_has_frontmatter(
                entry
            ):
                continue
            cards.extend(_load_agent_card_file(entry))
        _ensure_unique_names(cards, path)
        return cards

    if path.suffix.lower() not in {".md", ".markdown", ".yaml", ".yml"}:
        raise AgentConfigError(f"Unsupported AgentCard file extension: {path}")
    if path.suffix.lower() in {".md", ".markdown"} and not _markdown_has_frontmatter(path):
        raise AgentConfigError(
            "AgentCard markdown files must include frontmatter",
            f"Missing frontmatter in {path}",
        )

    cards = _load_agent_card_file(path)
    _ensure_unique_names(cards, path)
    return cards


def _ensure_unique_names(cards: Iterable[LoadedAgentCard], path: Path) -> None:
    seen: dict[str, Path] = {}
    for card in cards:
        if card.name in seen:
            raise AgentConfigError(
                f"Duplicate agent name '{card.name}' in {path}",
                f"Conflicts: {seen[card.name]} and {card.path}",
            )
        seen[card.name] = card.path


def _load_agent_card_file(path: Path) -> list[LoadedAgentCard]:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = _load_yaml_card(path)
        return [_build_card_from_data(path, raw, body=None)]
    if suffix in {".md", ".markdown"}:
        metadata, body = _load_markdown_card(path)
        return [_build_card_from_data(path, metadata, body=body)]
    raise AgentConfigError(f"Unsupported AgentCard file: {path}")


def _load_yaml_card(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AgentConfigError(f"Failed to parse YAML in {path}", str(exc)) from exc

    if not isinstance(data, dict):
        raise AgentConfigError(f"AgentCard YAML must be a mapping in {path}")
    return data


def _load_markdown_card(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        if raw_text.startswith("\ufeff"):
            raw_text = raw_text.lstrip("\ufeff")
        post = frontmatter.loads(raw_text)
    except Exception as exc:  # noqa: BLE001
        raise AgentConfigError(f"Failed to parse frontmatter in {path}", str(exc)) from exc

    metadata = post.metadata or {}
    if not isinstance(metadata, dict):
        raise AgentConfigError(f"Frontmatter must be a mapping in {path}")

    body = post.content or ""
    return dict(metadata), body


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


def _build_card_from_data(
    path: Path,
    raw: dict[str, Any],
    *,
    body: str | None,
) -> LoadedAgentCard:
    raw = dict(raw)
    card_type_raw = raw.get("type")
    if card_type_raw is None:
        card_type_key = "agent"
    elif isinstance(card_type_raw, str):
        card_type_key = card_type_raw.strip() or "agent"
    else:
        raise AgentConfigError(f"'type' must be a string in {path}")
    card_type_key_norm = card_type_key.lower()
    if card_type_key_norm == "maker":
        type_key = "MAKER"
    else:
        type_key = card_type_key_norm

    if type_key not in _ALLOWED_FIELDS_BY_TYPE:
        raise AgentConfigError(f"Unsupported agent type '{card_type_raw}' in {path}")

    allowed_fields = _ALLOWED_FIELDS_BY_TYPE[type_key]
    unknown_fields = set(raw.keys()) - allowed_fields
    if unknown_fields:
        unknown_list = ", ".join(sorted(unknown_fields))
        raise AgentConfigError(
            f"Unsupported fields for type '{type_key}' in {path}",
            f"Unknown fields: {unknown_list}",
        )

    schema_version = raw.get("schema_version", 1)
    if not isinstance(schema_version, int):
        raise AgentConfigError(f"'schema_version' must be an integer in {path}")

    name = _resolve_name(raw.get("name"), path)
    instruction = _resolve_instruction_field(raw.get("instruction"), body, path)
    description = _ensure_optional_str(raw.get("description"), "description", path)

    required_fields = _REQUIRED_FIELDS_BY_TYPE[type_key]
    missing = [field for field in required_fields if field not in raw or raw[field] is None]
    if missing:
        missing_list = ", ".join(missing)
        raise AgentConfigError(
            f"Missing required fields for type '{type_key}' in {path}",
            f"Required: {missing_list}",
        )

    message_files = _resolve_message_files(raw.get("messages"), path, type_key)

    agent_type = _TYPE_MAP[type_key.lower()] if type_key != "MAKER" else AgentType.MAKER
    agent_data = _build_agent_data(
        agent_type=agent_type,
        type_key=type_key,
        name=name,
        instruction=instruction,
        description=description,
        raw=raw,
        path=path,
    )
    agent_data["schema_version"] = schema_version
    if message_files:
        agent_data["message_files"] = message_files

    return LoadedAgentCard(
        name=name,
        path=path,
        agent_data=agent_data,
        message_files=message_files,
    )


def _resolve_name(raw_name: Any, path: Path) -> str:
    if raw_name is None:
        return path.stem.replace(" ", "_")
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise AgentConfigError(f"'name' must be a non-empty string in {path}")
    return raw_name.strip().replace(" ", "_")


def _resolve_instruction_field(
    raw_instruction: Any,
    body: str | None,
    path: Path,
) -> str:
    body_instruction = ""
    if body is not None:
        body_instruction = _extract_body_instruction(body, path)

    if raw_instruction is not None and body_instruction:
        raise AgentConfigError(
            "Instruction cannot be provided in both body and 'instruction' field",
            f"Path: {path}",
        )

    if raw_instruction is not None:
        if not isinstance(raw_instruction, str):
            raise AgentConfigError(f"'instruction' must be a string in {path}")
        resolved = _resolve_instruction(raw_instruction.strip())
        if not resolved.strip():
            raise AgentConfigError(f"'instruction' must not be empty in {path}")
        return resolved

    if body_instruction:
        resolved = _resolve_instruction(body_instruction)
        if not resolved.strip():
            raise AgentConfigError(f"Instruction body must not be empty in {path}")
        return resolved

    return DEFAULT_AGENT_INSTRUCTION


def _extract_body_instruction(body: str, path: Path) -> str:
    if not body:
        return ""
    lines = body.splitlines()
    first_non_empty = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_non_empty = idx
            break
    if first_non_empty is None:
        return ""

    if lines[first_non_empty].strip() == "---SYSTEM":
        lines = lines[first_non_empty + 1 :]
    else:
        lines = lines[first_non_empty:]

    if any(line.strip() in _HISTORY_DELIMITERS for line in lines):
        raise AgentConfigError(
            "Inline history blocks are not supported inside AgentCard body",
            f"Path: {path}",
        )

    return "\n".join(lines).strip()


def _resolve_message_files(raw_messages: Any, path: Path, type_key: str) -> list[Path]:
    if raw_messages is None:
        return []
    if not isinstance(raw_messages, (str, list)):
        raise AgentConfigError(f"'messages' must be a string or list in {path}")
    if isinstance(raw_messages, str):
        entries = [raw_messages]
    else:
        entries = raw_messages
    if not entries:
        return []

    message_paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise AgentConfigError(f"'messages' entries must be strings in {path}")
        candidate = Path(entry).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        if not candidate.exists():
            raise AgentConfigError(
                f"History file not found for AgentCard '{type_key}' in {path}",
                f"Missing: {candidate}",
            )
        message_paths.append(candidate)
    return message_paths


def _build_agent_data(
    *,
    agent_type: AgentType,
    type_key: str,
    name: str,
    instruction: str,
    description: str | None,
    raw: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    servers = _ensure_str_list(raw.get("servers", []), "servers", path)
    tools = _ensure_filter_map(raw.get("tools", {}), "tools", path)
    resources = _ensure_filter_map(raw.get("resources", {}), "resources", path)
    prompts = _ensure_filter_map(raw.get("prompts", {}), "prompts", path)

    model = raw.get("model")
    use_history = _default_use_history(type_key, raw.get("use_history"))
    request_params = _ensure_request_params(raw.get("request_params"), path)
    human_input = _ensure_bool(raw.get("human_input"), "human_input", path, default=False)
    default = _ensure_bool(raw.get("default"), "default", path, default=False)
    tool_only = _ensure_bool(raw.get("tool_only"), "tool_only", path, default=False)

    # Validate mutual exclusivity of default and tool_only
    if default and tool_only:
        raise AgentConfigError(
            f"Agent '{name}' cannot have both 'default' and 'tool_only' set to true in {path}",
            "A tool-only agent cannot be the default agent.",
        )

    api_key = raw.get("api_key")

    # Parse function_tools - can be a string or list of strings
    function_tools_raw = raw.get("function_tools")
    function_tools = None
    if function_tools_raw is not None:
        if isinstance(function_tools_raw, str):
            function_tools = [function_tools_raw]
        elif isinstance(function_tools_raw, list):
            function_tools = [str(t) for t in function_tools_raw]

    # Parse shell and cwd for sub-agent shell access
    shell = _ensure_bool(raw.get("shell"), "shell", path, default=False)
    cwd_str = raw.get("cwd")
    cwd: Path | None = None
    if cwd_str is not None:
        if not isinstance(cwd_str, str):
            raise AgentConfigError(f"'cwd' must be a string in {path}")
        cwd = Path(cwd_str).expanduser()

    config = AgentConfig(
        name=name,
        instruction=instruction,
        description=description,
        servers=servers,
        tools=tools,
        resources=resources,
        prompts=prompts,
        skills=raw.get("skills") if raw.get("skills") is not None else SKILLS_DEFAULT,
        model=model,
        use_history=use_history,
        human_input=human_input,
        default=default,
        tool_only=tool_only,
        api_key=api_key,
        function_tools=function_tools,
        shell=shell,
        cwd=cwd,
    )

    if request_params is not None:
        config.default_request_params = request_params
        config.default_request_params.systemPrompt = config.instruction
        config.default_request_params.use_history = config.use_history

    agent_data: dict[str, Any] = {
        "config": config,
        "type": agent_type.value,
        "func": None,
        "source_path": str(path),
        "tool_only": tool_only,
    }

    if type_key == "agent":
        agents = _ensure_str_list(raw.get("agents", []), "agents", path)
        if agents:
            agent_data["child_agents"] = agents
            opts = _agents_as_tools_options(raw, path)
            if opts:
                agent_data["agents_as_tools_options"] = opts
        if "function_tools" in raw:
            agent_data["function_tools"] = raw.get("function_tools")
        if "tool_hooks" in raw:
            agent_data["tool_hooks"] = raw.get("tool_hooks")
    elif type_key == "chain":
        sequence = _ensure_str_list(raw.get("sequence", []), "sequence", path)
        if not sequence:
            raise AgentConfigError(f"'sequence' must include at least one agent in {path}")
        agent_data["sequence"] = sequence
        agent_data["cumulative"] = _ensure_bool(raw.get("cumulative"), "cumulative", path)
    elif type_key == "parallel":
        fan_out = _ensure_str_list(raw.get("fan_out", []), "fan_out", path)
        if not fan_out:
            raise AgentConfigError(f"'fan_out' must include at least one agent in {path}")
        agent_data["fan_out"] = fan_out
        fan_in = raw.get("fan_in")
        if fan_in is not None and not isinstance(fan_in, str):
            raise AgentConfigError(f"'fan_in' must be a string in {path}")
        agent_data["fan_in"] = fan_in
        agent_data["include_request"] = _ensure_bool(
            raw.get("include_request"), "include_request", path, default=True
        )
    elif type_key == "evaluator_optimizer":
        agent_data["generator"] = _ensure_str(raw.get("generator"), "generator", path)
        agent_data["evaluator"] = _ensure_str(raw.get("evaluator"), "evaluator", path)
        agent_data["min_rating"] = _ensure_str(raw.get("min_rating", "GOOD"), "min_rating", path)
        agent_data["max_refinements"] = _ensure_int(
            raw.get("max_refinements", 3), "max_refinements", path
        )
        agent_data["refinement_instruction"] = raw.get("refinement_instruction")
    elif type_key == "router":
        router_agents = _ensure_str_list(raw.get("agents", []), "agents", path)
        if not router_agents:
            raise AgentConfigError(f"'agents' must include at least one agent in {path}")
        agent_data["router_agents"] = router_agents
        agent_data["instruction"] = instruction
    elif type_key in {"orchestrator", "iterative_planner"}:
        child_agents = _ensure_str_list(raw.get("agents", []), "agents", path)
        if not child_agents:
            raise AgentConfigError(f"'agents' must include at least one agent in {path}")
        agent_data["child_agents"] = child_agents
        if type_key == "orchestrator":
            agent_data["plan_type"] = _ensure_str(raw.get("plan_type", "full"), "plan_type", path)
            agent_data["plan_iterations"] = _ensure_int(
                raw.get("plan_iterations", 5), "plan_iterations", path
            )
        else:
            agent_data["plan_iterations"] = _ensure_int(
                raw.get("plan_iterations", -1), "plan_iterations", path
            )
    elif type_key == "MAKER":
        agent_data["worker"] = _ensure_str(raw.get("worker"), "worker", path)
        agent_data["k"] = _ensure_int(raw.get("k", 3), "k", path)
        agent_data["max_samples"] = _ensure_int(raw.get("max_samples", 50), "max_samples", path)
        agent_data["match_strategy"] = _ensure_str(
            raw.get("match_strategy", "exact"), "match_strategy", path
        )
        red_flag = raw.get("red_flag_max_length")
        if red_flag is not None:
            red_flag = _ensure_int(red_flag, "red_flag_max_length", path)
        agent_data["red_flag_max_length"] = red_flag

    return agent_data


def _default_use_history(type_key: str, raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if type_key in {"router", "orchestrator", "iterative_planner"}:
        return False
    return True


def _ensure_bool(value: Any, field: str, path: Path, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise AgentConfigError(f"'{field}' must be a boolean in {path}")


def _ensure_str(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(f"'{field}' must be a non-empty string in {path}")
    return value


def _ensure_optional_str(value: Any, field: str, path: Path) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AgentConfigError(f"'{field}' must be a non-empty string in {path}")
    return value.strip()


def _ensure_str_list(value: Any, field: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentConfigError(f"'{field}' must be a list of strings in {path}")
    result: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            raise AgentConfigError(f"'{field}' entries must be strings in {path}")
        result.append(entry)
    return result


def _ensure_filter_map(value: Any, field: str, path: Path) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AgentConfigError(f"'{field}' must be a mapping in {path}")
    result: dict[str, list[str]] = {}
    for key, entry in value.items():
        if not isinstance(key, str) or not key.strip():
            raise AgentConfigError(f"'{field}' keys must be strings in {path}")
        if not isinstance(entry, list):
            raise AgentConfigError(f"'{field}' values must be lists in {path}")
        for item in entry:
            if not isinstance(item, str) or not item.strip():
                raise AgentConfigError(f"'{field}' values must be strings in {path}")
        result[key] = entry
    return result


def _ensure_request_params(value: Any, path: Path) -> RequestParams | None:
    if value is None:
        return None
    if isinstance(value, RequestParams):
        return value
    if not isinstance(value, dict):
        raise AgentConfigError(f"'request_params' must be a mapping in {path}")
    try:
        return RequestParams(**value)
    except Exception as exc:  # noqa: BLE001
        raise AgentConfigError(f"Invalid request_params in {path}", str(exc)) from exc


def _agents_as_tools_options(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    options: dict[str, Any] = {}
    history_source = raw.get("history_source")
    history_merge_target = raw.get("history_merge_target")
    max_parallel = raw.get("max_parallel")
    child_timeout_sec = raw.get("child_timeout_sec")
    max_display_instances = raw.get("max_display_instances")

    if history_source is not None:
        options["history_source"] = _ensure_optional_str(
            history_source, "history_source", path
        )
    if history_merge_target is not None:
        options["history_merge_target"] = _ensure_optional_str(
            history_merge_target, "history_merge_target", path
        )
    if max_parallel is not None:
        options["max_parallel"] = _ensure_int(max_parallel, "max_parallel", path)
    if child_timeout_sec is not None:
        options["child_timeout_sec"] = _ensure_float(
            child_timeout_sec, "child_timeout_sec", path
        )
    if max_display_instances is not None:
        options["max_display_instances"] = _ensure_int(
            max_display_instances, "max_display_instances", path
        )
    return options


def _ensure_int(value: Any, field: str, path: Path) -> int:
    if not isinstance(value, int):
        raise AgentConfigError(f"'{field}' must be an integer in {path}")
    return value


def _ensure_float(value: Any, field: str, path: Path) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise AgentConfigError(f"'{field}' must be a number in {path}")


def dump_agents_to_dir(
    agents: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    as_yaml: bool = False,
    message_map: dict[str, list[Path]] | None = None,
) -> None:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in sorted(agents.keys()):
        output_path = output_dir / f"{name}.{'yaml' if as_yaml else 'md'}"
        message_paths = message_map.get(name) if message_map else None
        dump_agent_to_path(
            name,
            agents[name],
            output_path,
            as_yaml=as_yaml,
            message_paths=message_paths,
        )


def dump_agent_to_path(
    name: str,
    agent_data: dict[str, Any],
    output_path: Path,
    *,
    as_yaml: bool = False,
    message_paths: list[Path] | None = None,
) -> None:
    payload = dump_agent_to_string(
        name, agent_data, as_yaml=as_yaml, message_paths=message_paths
    )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")


def dump_agent_to_string(
    name: str,
    agent_data: dict[str, Any],
    *,
    as_yaml: bool = False,
    message_paths: list[Path] | None = None,
) -> str:
    """Render an AgentCard to a string."""
    card_dict, instruction = _build_card_dump(name, agent_data, message_paths)
    if as_yaml:
        card_dict = dict(card_dict)
        card_dict["instruction"] = instruction
        payload = yaml.safe_dump(
            card_dict,
            sort_keys=False,
            allow_unicode=False,
        ).rstrip()
        return f"{payload}\n"

    frontmatter = yaml.safe_dump(
        card_dict,
        sort_keys=False,
        allow_unicode=False,
    ).rstrip()
    return f"---\n{frontmatter}\n---\n{instruction.rstrip()}\n"


def _build_card_dump(
    name: str,
    agent_data: dict[str, Any],
    message_paths: list[Path] | None,
) -> tuple[dict[str, Any], str]:
    agent_type_value = agent_data.get("type")
    if not isinstance(agent_type_value, str):
        raise AgentConfigError(f"Agent '{name}' is missing a valid type")
    card_type = _AGENT_TYPE_TO_CARD_TYPE.get(agent_type_value)
    if card_type is None:
        raise AgentConfigError(f"Agent '{name}' has unsupported type '{agent_type_value}'")

    config = agent_data.get("config")
    if not isinstance(config, AgentConfig):
        raise AgentConfigError(f"Agent '{name}' is missing AgentConfig")

    instruction = config.instruction
    if not instruction:
        raise AgentConfigError(f"Agent '{name}' is missing instruction")

    card: dict[str, Any] = {"type": card_type, "name": name}
    schema_version = agent_data.get("schema_version")
    if isinstance(schema_version, int):
        card["schema_version"] = schema_version

    allowed_fields = _ALLOWED_FIELDS_BY_TYPE.get(card_type, set())

    if config.default and "default" in allowed_fields:
        card["default"] = True

    if config.tool_only and "tool_only" in allowed_fields:
        card["tool_only"] = True

    if config.description and "description" in allowed_fields:
        card["description"] = config.description

    if config.model and "model" in allowed_fields:
        card["model"] = config.model

    if "use_history" in allowed_fields:
        default_use_history = _DEFAULT_USE_HISTORY_BY_TYPE.get(card_type, True)
        if config.use_history != default_use_history:
            card["use_history"] = config.use_history

    if config.human_input and "human_input" in allowed_fields:
        card["human_input"] = True

    if config.api_key and "api_key" in allowed_fields:
        card["api_key"] = config.api_key

    if config.servers and "servers" in allowed_fields:
        card["servers"] = list(config.servers)

    if config.tools and "tools" in allowed_fields:
        card["tools"] = config.tools

    if config.resources and "resources" in allowed_fields:
        card["resources"] = config.resources

    if config.prompts and "prompts" in allowed_fields:
        card["prompts"] = config.prompts

    serialized_skills = _serialize_skills(config.skills)
    if serialized_skills is not None and "skills" in allowed_fields:
        card["skills"] = serialized_skills

    request_params_dump = _dump_request_params(config.default_request_params)
    if request_params_dump and "request_params" in allowed_fields:
        card["request_params"] = request_params_dump

    if message_paths and "messages" in allowed_fields:
        card["messages"] = [str(path) for path in message_paths]

    if card_type == "agent":
        child_agents = agent_data.get("child_agents") or []
        if child_agents:
            card["agents"] = list(child_agents)
        opts = agent_data.get("agents_as_tools_options") or {}
        if "history_source" in opts and opts["history_source"] is not None:
            history_source = opts["history_source"]
            card["history_source"] = (
                history_source.value
                if hasattr(history_source, "value")
                else history_source
            )
        if "history_merge_target" in opts and opts["history_merge_target"] is not None:
            history_merge_target = opts["history_merge_target"]
            card["history_merge_target"] = (
                history_merge_target.value
                if hasattr(history_merge_target, "value")
                else history_merge_target
            )
        if "max_parallel" in opts and opts["max_parallel"] is not None:
            card["max_parallel"] = opts["max_parallel"]
        if "child_timeout_sec" in opts and opts["child_timeout_sec"] is not None:
            card["child_timeout_sec"] = opts["child_timeout_sec"]
        if "max_display_instances" in opts and opts["max_display_instances"] is not None:
            card["max_display_instances"] = opts["max_display_instances"]
        function_tools = _serialize_string_list(agent_data.get("function_tools"))
        if function_tools is not None:
            card["function_tools"] = function_tools
        tool_hooks = _serialize_string_list(agent_data.get("tool_hooks"))
        if tool_hooks is not None:
            card["tool_hooks"] = tool_hooks
    elif card_type == "chain":
        card["sequence"] = list(agent_data.get("sequence") or [])
        cumulative = agent_data.get("cumulative", False)
        if cumulative:
            card["cumulative"] = True
    elif card_type == "parallel":
        card["fan_out"] = list(agent_data.get("fan_out") or [])
        fan_in = agent_data.get("fan_in")
        if fan_in:
            card["fan_in"] = fan_in
        include_request = agent_data.get("include_request")
        if include_request is False:
            card["include_request"] = False
    elif card_type == "evaluator_optimizer":
        card["generator"] = agent_data.get("generator")
        card["evaluator"] = agent_data.get("evaluator")
        if "min_rating" in agent_data:
            card["min_rating"] = agent_data.get("min_rating")
        if "max_refinements" in agent_data:
            card["max_refinements"] = agent_data.get("max_refinements")
        if "refinement_instruction" in agent_data:
            card["refinement_instruction"] = agent_data.get("refinement_instruction")
    elif card_type == "router":
        card["agents"] = list(agent_data.get("router_agents") or [])
    elif card_type == "orchestrator":
        card["agents"] = list(agent_data.get("child_agents") or [])
        card["plan_type"] = agent_data.get("plan_type", "full")
        card["plan_iterations"] = agent_data.get("plan_iterations", 5)
    elif card_type == "iterative_planner":
        card["agents"] = list(agent_data.get("child_agents") or [])
        card["plan_iterations"] = agent_data.get("plan_iterations", -1)
    elif card_type == "MAKER":
        card["worker"] = agent_data.get("worker")
        card["k"] = agent_data.get("k", 3)
        card["max_samples"] = agent_data.get("max_samples", 50)
        card["match_strategy"] = agent_data.get("match_strategy", "exact")
        red_flag = agent_data.get("red_flag_max_length")
        if red_flag is not None:
            card["red_flag_max_length"] = red_flag

    return card, instruction


def _dump_request_params(params: RequestParams | None) -> dict[str, Any] | None:
    if params is None:
        return None
    dump = params.model_dump(
        exclude_defaults=True,
        exclude={"messages", "systemPrompt", "use_history", "model"},
    )
    return dump or None


def _serialize_skills(
    skills: Any,
) -> str | list[str] | None:
    if skills is None:
        return None
    if isinstance(skills, Path):
        return str(skills)
    if isinstance(skills, str):
        return skills
    if isinstance(skills, list):
        serialized: list[str] = []
        for item in skills:
            if isinstance(item, Path):
                serialized.append(str(item))
            elif isinstance(item, str):
                serialized.append(item)
        return serialized if serialized else None
    return None


def _serialize_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        return None
    if not value:
        return []
    if all(isinstance(item, str) for item in value):
        return list(value)
    return None
