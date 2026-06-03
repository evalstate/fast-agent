"""AgentCard loader and export helpers for Markdown/YAML card files."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import frontmatter
import yaml

from fast_agent.agents.agent_types import (
    AgentConfig,
    AgentType,
    FunctionToolConfig,
    LifecycleHooksConfig,
    MCPConnectTarget,
)
from fast_agent.command_actions import PluginCommandActionSpec, parse_plugin_command_action_specs
from fast_agent.config import MCPServerSettings, resolve_env_vars
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION, SMART_AGENT_INSTRUCTION
from fast_agent.core.agent_card_paths import (
    is_agent_card_path,
    is_markdown_agent_card_path,
    is_yaml_agent_card_path,
)
from fast_agent.core.agent_card_rules import (
    AGENT_TYPE_TO_CARD_TYPE,
    ALLOWED_FIELDS_BY_TYPE,
    CARD_TYPE_TO_AGENT_TYPE,
    DEFAULT_USE_HISTORY_BY_TYPE,
    MCP_CONNECT_ALLOWED_KEYS,
    REQUIRED_FIELDS_BY_TYPE,
    CardType,
    normalize_card_type,
)
from fast_agent.core.agent_card_types import AgentCardData
from fast_agent.core.direct_decorators import _resolve_instruction
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.tool_input_schema import validate_tool_input_schema
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.tools.function_tool_config import (
    parse_function_tool_card_entry,
    serialize_function_tools,
)
from fast_agent.types import RequestParams
from fast_agent.utils.text import strip_str_to_none, strip_to_none
from fast_agent.utils.type_narrowing import is_str_object_dict

CardTypeSerializer = Callable[[dict[str, Any], AgentCardData, AgentConfig], None]

_HISTORY_DELIMITERS = {"---USER", "---ASSISTANT", "---RESOURCE"}


@dataclass(frozen=True)
class LoadedAgentCard:
    name: str
    path: Path
    agent_data: AgentCardData
    message_files: list[Path]


@dataclass(frozen=True, slots=True)
class _MarkdownCard:
    metadata: dict[str, Any]
    body: str


@dataclass(frozen=True, slots=True)
class _ParsedMCPConnectEntry:
    target: str | None
    name: str | None
    description: str | None
    management: str | None
    connector_id: str | None
    headers: dict[str, str] | None
    access_token: str | None
    defer_loading: bool | None
    auth: dict[str, Any] | None


def load_agent_cards(path: Path) -> list[LoadedAgentCard]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise AgentConfigError(f"AgentCard path not found: {path}")

    if path.is_dir():
        cards: list[LoadedAgentCard] = []
        for entry in sorted(path.iterdir()):
            if entry.is_dir():
                continue
            if not is_agent_card_path(entry):
                continue
            if is_markdown_agent_card_path(entry) and not _markdown_has_frontmatter(entry):
                continue
            cards.extend(_load_agent_card_file(entry))
        _ensure_unique_names(cards, path)
        return cards

    if not is_agent_card_path(path):
        raise AgentConfigError(f"Unsupported AgentCard file extension: {path}")
    if path.name == "SKILL.md":
        raise AgentConfigError(
            "SKILL.md is an Agent Skill manifest, not an AgentCard",
            "Use read_text_file/read_skill to inspect skill instructions, or use /skills to manage skills.",
        )
    if is_markdown_agent_card_path(path) and not _markdown_has_frontmatter(path):
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
    if path.name == "SKILL.md":
        raise AgentConfigError(
            "SKILL.md is an Agent Skill manifest, not an AgentCard",
            "Use read_text_file/read_skill to inspect skill instructions, or use /skills to manage skills.",
        )

    if is_yaml_agent_card_path(path):
        raw = _load_yaml_card(path)
        return [_build_card_from_data(path, raw, body=None)]
    if is_markdown_agent_card_path(path):
        card = _load_markdown_card(path)
        return [_build_card_from_data(path, card.metadata, body=card.body)]
    raise AgentConfigError(f"Unsupported AgentCard file: {path}")


def _load_yaml_card(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AgentConfigError(f"Failed to parse YAML in {path}", str(exc)) from exc

    if not isinstance(data, dict):
        raise AgentConfigError(f"AgentCard YAML must be a mapping in {path}")
    resolved = resolve_env_vars(data)
    if not isinstance(resolved, dict):
        raise AgentConfigError(f"AgentCard YAML must be a mapping in {path}")
    return resolved


def _load_markdown_card(path: Path) -> _MarkdownCard:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentConfigError(f"Failed to parse frontmatter in {path}", str(exc)) from exc

    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")
    try:
        post = frontmatter.loads(raw_text)
    except (ValueError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"Failed to parse frontmatter in {path}", str(exc)) from exc

    metadata = post.metadata or {}
    if not isinstance(metadata, dict):
        raise AgentConfigError(f"Frontmatter must be a mapping in {path}")

    body = post.content or ""
    resolved = resolve_env_vars(dict(metadata))
    if not isinstance(resolved, dict):
        raise AgentConfigError(f"Frontmatter must be a mapping in {path}")
    return _MarkdownCard(metadata=resolved, body=body)


def _markdown_has_frontmatter(path: Path) -> bool:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if raw_text.startswith("\ufeff"):
        raw_text = raw_text.lstrip("\ufeff")
    for line in raw_text.splitlines():
        stripped = strip_to_none(line)
        if stripped is None:
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
    if card_type_raw is not None and not isinstance(card_type_raw, str):
        raise AgentConfigError(f"'type' must be a string in {path}")

    type_key = normalize_card_type(card_type_raw)
    if type_key is None:
        raise AgentConfigError(f"Unsupported agent type '{card_type_raw}' in {path}")

    allowed_fields = ALLOWED_FIELDS_BY_TYPE[type_key]
    unknown_fields = set(raw.keys()) - allowed_fields
    if unknown_fields:
        unknown_list = ", ".join(sorted(unknown_fields))
        raise AgentConfigError(
            f"Unsupported fields for type '{type_key}' in {path}",
            f"Unknown fields: {unknown_list}",
        )

    schema_version = raw.get("schema_version", 1)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise AgentConfigError(f"'schema_version' must be an integer in {path}")

    name = _resolve_name(raw.get("name"), path)
    default_instruction = (
        SMART_AGENT_INSTRUCTION if type_key == "smart" else DEFAULT_AGENT_INSTRUCTION
    )
    instruction = _resolve_instruction_field(
        raw.get("instruction"),
        body,
        path,
        default_instruction=default_instruction,
    )
    description = _ensure_optional_str(raw.get("description"), "description", path)

    required_fields = REQUIRED_FIELDS_BY_TYPE[type_key]
    missing = [field for field in required_fields if field not in raw or raw[field] is None]
    if missing:
        missing_list = ", ".join(missing)
        raise AgentConfigError(
            f"Missing required fields for type '{type_key}' in {path}",
            f"Required: {missing_list}",
        )

    message_files = _resolve_message_files(raw.get("messages"), path, type_key)

    agent_type = CARD_TYPE_TO_AGENT_TYPE[type_key]
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
    name = strip_str_to_none(raw_name)
    if name is None:
        raise AgentConfigError(f"'name' must be a non-empty string in {path}")
    return name.replace(" ", "_")


def _resolve_instruction_field(
    raw_instruction: Any,
    body: str | None,
    path: Path,
    *,
    default_instruction: str = DEFAULT_AGENT_INSTRUCTION,
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
        resolved = _resolve_instruction(strip_to_none(raw_instruction) or "")
        if strip_to_none(resolved) is None:
            raise AgentConfigError(f"'instruction' must not be empty in {path}")
        return resolved

    if body_instruction:
        resolved = _resolve_instruction(body_instruction)
        if strip_to_none(resolved) is None:
            raise AgentConfigError(f"Instruction body must not be empty in {path}")
        return resolved

    return default_instruction


def _extract_body_instruction(body: str, path: Path) -> str:
    if not body:
        return ""
    lines = body.splitlines()
    first_non_empty = None
    for idx, line in enumerate(lines):
        if strip_to_none(line) is not None:
            first_non_empty = idx
            break
    if first_non_empty is None:
        return ""

    if strip_to_none(lines[first_non_empty]) == "---SYSTEM":
        lines = lines[first_non_empty + 1 :]
    else:
        lines = lines[first_non_empty:]

    if any(strip_to_none(line) in _HISTORY_DELIMITERS for line in lines):
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
    entries = [raw_messages] if isinstance(raw_messages, str) else raw_messages
    if not entries:
        return []

    message_paths: list[Path] = []
    for entry in entries:
        if strip_str_to_none(entry) is None:
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


def _ensure_function_tools(value: Any, path: Path) -> list[FunctionToolConfig] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [
            parse_function_tool_card_entry(
                entry,
                field_path=f"function_tools[{index}]",
            )
            for index, entry in enumerate(value)
        ]
    raise AgentConfigError(f"'function_tools' must be a string or list in {path}")


def _ensure_cwd(value: Any, path: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AgentConfigError(f"'cwd' must be a string in {path}")
    return Path(value).expanduser()


def _ensure_hook_map(value: Any, field: str, path: Path) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AgentConfigError(f"'{field}' must be a dict in {path}")
    return {str(k): str(v) for k, v in value.items()}


def _ensure_lifecycle_hooks(value: Any, path: Path) -> "LifecycleHooksConfig":
    lifecycle_hooks = _ensure_hook_map(value, "lifecycle_hooks", path)
    if lifecycle_hooks is None:
        return None

    from fast_agent.hooks.lifecycle_hook_types import VALID_LIFECYCLE_HOOK_TYPES

    invalid_types = set(lifecycle_hooks.keys()) - VALID_LIFECYCLE_HOOK_TYPES
    if invalid_types:
        raise AgentConfigError(
            f"Invalid lifecycle hook types: {invalid_types}",
            f"Valid types are: {sorted(VALID_LIFECYCLE_HOOK_TYPES)}",
        )
    return cast("LifecycleHooksConfig", lifecycle_hooks)


def _ensure_default_flags(raw: dict[str, Any], name: str, path: Path) -> tuple[bool, bool]:
    default = _ensure_bool(raw.get("default"), "default", path, default=False)
    tool_only = _ensure_bool(raw.get("tool_only"), "tool_only", path, default=False)
    if default and tool_only:
        raise AgentConfigError(
            f"Agent '{name}' cannot have both 'default' and 'tool_only' set to true in {path}",
            "A tool-only agent cannot be the default agent.",
        )
    return default, tool_only


def _apply_request_params_defaults(config: AgentConfig, request_params: RequestParams | None) -> None:
    if request_params is None:
        return
    config.default_request_params = request_params
    config.default_request_params.systemPrompt = config.instruction
    config.default_request_params.use_history = config.use_history


def _build_agent_data(
    *,
    agent_type: AgentType,
    type_key: CardType,
    name: str,
    instruction: str,
    description: str | None,
    raw: dict[str, Any],
    path: Path,
) -> AgentCardData:
    servers = _ensure_str_list(raw.get("servers", []), "servers", path)
    mcp_connect = _ensure_mcp_connect_entries(raw.get("mcp_connect"), path)
    tools = _ensure_filter_map(raw.get("tools", {}), "tools", path)
    resources = _ensure_filter_map(raw.get("resources", {}), "resources", path)
    prompts = _ensure_filter_map(raw.get("prompts", {}), "prompts", path)

    model = raw.get("model")
    use_history = _default_use_history(type_key, raw.get("use_history"))
    request_params = _ensure_request_params(raw.get("request_params"), path)
    human_input = _ensure_bool(raw.get("human_input"), "human_input", path, default=False)
    default, tool_only = _ensure_default_flags(raw, name, path)

    api_key = raw.get("api_key")
    tool_input_schema = _ensure_tool_input_schema(raw.get("tool_input_schema"), path)
    function_tools = _ensure_function_tools(raw.get("function_tools"), path)
    shell_default = type_key == "smart"
    shell = _ensure_bool(raw.get("shell"), "shell", path, default=shell_default)
    cwd = _ensure_cwd(raw.get("cwd"), path)
    tool_hooks = _ensure_hook_map(raw.get("tool_hooks"), "tool_hooks", path)
    lifecycle_hooks = _ensure_lifecycle_hooks(raw.get("lifecycle_hooks"), path)

    commands = _ensure_plugin_commands(raw.get("commands"), path)
    trim_tool_history = _ensure_bool(raw.get("trim_tool_history"), "trim_tool_history", path)

    config = AgentConfig(
        name=name,
        instruction=instruction,
        description=description,
        tool_input_schema=tool_input_schema,
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
        tool_hooks=tool_hooks,
        lifecycle_hooks=lifecycle_hooks,
        commands=commands,
        trim_tool_history=trim_tool_history,
        mcp_connect=mcp_connect,
        source_path=path,
    )

    _apply_request_params_defaults(config, request_params)

    agent_data: AgentCardData = {
        "config": config,
        "type": agent_type.value,
        "func": None,
        "source_path": str(path),
        "tool_only": tool_only,
    }

    _apply_type_specific_agent_data(
        type_key=type_key,
        raw=raw,
        path=path,
        agent_data=agent_data,
        instruction=instruction,
    )

    return agent_data


CardTypeDataHandler = Callable[[dict[str, Any], Path, AgentCardData, str], None]


def _apply_type_specific_agent_data(
    *,
    type_key: CardType,
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    handler = _CARD_TYPE_DATA_HANDLERS.get(type_key)
    if handler is not None:
        handler(raw, path, agent_data, instruction)


def _apply_basic_agent_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
    agents = _ensure_str_list(raw.get("agents", []), "agents", path)
    if agents:
        agent_data["child_agents"] = agents
        opts = _agents_as_tools_options(raw, path)
        if opts:
            agent_data["agents_as_tools_options"] = opts
    if "function_tools" in raw:
        agent_data["function_tools"] = raw.get("function_tools")


def _apply_chain_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
    sequence = _ensure_str_list(raw.get("sequence", []), "sequence", path)
    if not sequence:
        raise AgentConfigError(f"'sequence' must include at least one agent in {path}")
    agent_data["sequence"] = sequence
    agent_data["cumulative"] = _ensure_bool(raw.get("cumulative"), "cumulative", path)


def _apply_parallel_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
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


def _apply_evaluator_optimizer_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
    agent_data["generator"] = _ensure_str(raw.get("generator"), "generator", path)
    agent_data["evaluator"] = _ensure_str(raw.get("evaluator"), "evaluator", path)
    agent_data["min_rating"] = _ensure_str(raw.get("min_rating", "GOOD"), "min_rating", path)
    agent_data["max_refinements"] = _ensure_int(
        raw.get("max_refinements", 3), "max_refinements", path
    )
    agent_data["refinement_instruction"] = raw.get("refinement_instruction")


def _apply_router_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    router_agents = _ensure_str_list(raw.get("agents", []), "agents", path)
    if not router_agents:
        raise AgentConfigError(f"'agents' must include at least one agent in {path}")
    agent_data["router_agents"] = router_agents
    agent_data["instruction"] = instruction


def _apply_orchestrator_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
    _apply_planner_child_agents(raw, path, agent_data)
    agent_data["plan_type"] = _ensure_str(raw.get("plan_type", "full"), "plan_type", path)
    agent_data["plan_iterations"] = _ensure_int(
        raw.get("plan_iterations", 5), "plan_iterations", path
    )


def _apply_iterative_planner_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
    _apply_planner_child_agents(raw, path, agent_data)
    agent_data["plan_iterations"] = _ensure_int(
        raw.get("plan_iterations", -1), "plan_iterations", path
    )


def _apply_planner_child_agents(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
) -> None:
    child_agents = _ensure_str_list(raw.get("agents", []), "agents", path)
    if not child_agents:
        raise AgentConfigError(f"'agents' must include at least one agent in {path}")
    agent_data["child_agents"] = child_agents


def _apply_maker_data(
    raw: dict[str, Any],
    path: Path,
    agent_data: AgentCardData,
    instruction: str,
) -> None:
    del instruction
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


_CARD_TYPE_DATA_HANDLERS: dict[CardType, CardTypeDataHandler] = {
    "agent": _apply_basic_agent_data,
    "smart": _apply_basic_agent_data,
    "chain": _apply_chain_data,
    "parallel": _apply_parallel_data,
    "evaluator_optimizer": _apply_evaluator_optimizer_data,
    "router": _apply_router_data,
    "orchestrator": _apply_orchestrator_data,
    "iterative_planner": _apply_iterative_planner_data,
    "MAKER": _apply_maker_data,
}


def _ensure_plugin_commands(raw_commands: Any, path: Path) -> dict[str, PluginCommandActionSpec] | None:
    return parse_plugin_command_action_specs(raw_commands, source=str(path))


def _default_use_history(type_key: str, raw_value: Any) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    return type_key not in {"router", "orchestrator", "iterative_planner"}


def _ensure_bool(value: Any, field: str, path: Path, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise AgentConfigError(f"'{field}' must be a boolean in {path}")


def _ensure_str(value: Any, field: str, path: Path) -> str:
    if strip_str_to_none(value) is None:
        raise AgentConfigError(f"'{field}' must be a non-empty string in {path}")
    return value


def _ensure_optional_str(value: Any, field: str, path: Path) -> str | None:
    if value is None:
        return None
    normalized = strip_str_to_none(value)
    if normalized is None:
        raise AgentConfigError(f"'{field}' must be a non-empty string in {path}")
    return normalized


def _ensure_str_list(value: Any, field: str, path: Path) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentConfigError(f"'{field}' must be a list of strings in {path}")
    result: list[str] = []
    for entry in value:
        if strip_str_to_none(entry) is None:
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
        if strip_str_to_none(key) is None:
            raise AgentConfigError(f"'{field}' keys must be strings in {path}")
        if not isinstance(entry, list):
            raise AgentConfigError(f"'{field}' values must be lists in {path}")
        for item in entry:
            if strip_str_to_none(item) is None:
                raise AgentConfigError(f"'{field}' values must be strings in {path}")
        result[key] = entry
    return result


def _ensure_headers_map(value: Any, field: str, path: Path) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AgentConfigError(f"'{field}' must be a mapping in {path}")

    headers: dict[str, str] = {}
    for key, header_value in value.items():
        if strip_str_to_none(key) is None:
            raise AgentConfigError(f"'{field}' keys must be non-empty strings in {path}")
        if not isinstance(header_value, str):
            raise AgentConfigError(f"'{field}' values must be strings in {path}")
        headers[key] = header_value
    return headers


def _ensure_auth_map(value: Any, field: str, path: Path) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AgentConfigError(f"'{field}' must be a mapping in {path}")
    return dict(value)


def _ensure_optional_bool(value: Any, field: str, path: Path) -> bool | None:
    if value is None:
        return None
    return _ensure_bool(value, field, path)


def _validate_mcp_connect_keys(raw_entry: dict[str, Any], idx: int, path: Path) -> None:
    unknown_keys = set(raw_entry.keys()) - MCP_CONNECT_ALLOWED_KEYS
    if not unknown_keys:
        return
    unknown_text = ", ".join(sorted(str(key) for key in unknown_keys))
    raise AgentConfigError(
        f"'mcp_connect[{idx}]' has unsupported keys in {path}",
        f"Unknown keys: {unknown_text}",
    )


def _mcp_connect_target(raw_entry: dict[str, Any], idx: int, path: Path) -> str | None:
    target_raw = raw_entry.get("target")
    if target_raw is None:
        return None
    target = strip_str_to_none(target_raw)
    if target is None:
        raise AgentConfigError(f"'mcp_connect[{idx}].target' must be a non-empty string in {path}")
    return target


def _mcp_connect_name(raw_entry: dict[str, Any], idx: int, path: Path) -> str | None:
    name = _ensure_optional_str(raw_entry.get("name"), f"mcp_connect[{idx}].name", path)
    if raw_entry.get("connector_id") is not None and name is None:
        raise AgentConfigError(
            f"'mcp_connect[{idx}].name' must be a non-empty string in {path} "
            "when connector_id is set"
        )
    return name


def _ensure_mcp_target_xor_connector(
    target: str | None,
    connector_id: str | None,
    idx: int,
    path: Path,
) -> None:
    if target is None and connector_id is None:
        raise AgentConfigError(
            f"'mcp_connect[{idx}].target' must be a non-empty string in {path} "
            "unless connector_id is set"
        )
    if target is not None and connector_id is not None:
        raise AgentConfigError(
            f"'mcp_connect[{idx}]' must set exactly one of 'target' or 'connector_id' in {path}"
        )


def _parse_mcp_connect_entry(
    raw_entry: dict[str, Any],
    idx: int,
    path: Path,
) -> _ParsedMCPConnectEntry:
    _validate_mcp_connect_keys(raw_entry, idx, path)
    target = _mcp_connect_target(raw_entry, idx, path)
    connector_id = _ensure_optional_str(
        raw_entry.get("connector_id"),
        f"mcp_connect[{idx}].connector_id",
        path,
    )
    _ensure_mcp_target_xor_connector(target, connector_id, idx, path)

    return _ParsedMCPConnectEntry(
        target=target,
        name=_mcp_connect_name(raw_entry, idx, path),
        description=_ensure_optional_str(
            raw_entry.get("description"),
            f"mcp_connect[{idx}].description",
            path,
        ),
        management=_ensure_optional_str(
            raw_entry.get("management"),
            f"mcp_connect[{idx}].management",
            path,
        ),
        connector_id=connector_id,
        headers=_ensure_headers_map(raw_entry.get("headers"), f"mcp_connect[{idx}].headers", path),
        access_token=_ensure_optional_str(
            raw_entry.get("access_token"),
            f"mcp_connect[{idx}].access_token",
            path,
        ),
        defer_loading=_ensure_optional_bool(
            raw_entry.get("defer_loading"),
            f"mcp_connect[{idx}].defer_loading",
            path,
        ),
        auth=_ensure_auth_map(raw_entry.get("auth"), f"mcp_connect[{idx}].auth", path),
    )


def _validate_provider_mcp_connect_entry(
    entry: _ParsedMCPConnectEntry,
    idx: int,
    path: Path,
) -> None:
    if entry.connector_id is None:
        return

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
    try:
        MCPServerSettings.model_validate(payload)
    except Exception as exc:
        raise AgentConfigError(f"Invalid 'mcp_connect[{idx}]' in {path}", str(exc)) from exc


def _mcp_connect_target_from_entry(
    entry: _ParsedMCPConnectEntry,
) -> MCPConnectTarget:
    return MCPConnectTarget(
        target=entry.target,
        name=entry.name,
        description=entry.description,
        management=entry.management,
        connector_id=entry.connector_id,
        headers=entry.headers,
        access_token=entry.access_token,
        defer_loading=entry.defer_loading,
        auth=entry.auth,
    )


def _ensure_mcp_connect_entries(value: Any, path: Path) -> list[MCPConnectTarget]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AgentConfigError(f"'mcp_connect' must be a list in {path}")

    entries: list[MCPConnectTarget] = []
    for idx, raw_entry in enumerate(value):
        if not is_str_object_dict(raw_entry):
            raise AgentConfigError(
                f"'mcp_connect[{idx}]' must be a mapping in {path}",
            )

        entry = _parse_mcp_connect_entry(raw_entry, idx, path)
        _validate_provider_mcp_connect_entry(entry, idx, path)
        entries.append(_mcp_connect_target_from_entry(entry))

    return entries


def _ensure_request_params(value: Any, path: Path) -> RequestParams | None:
    if value is None:
        return None
    if isinstance(value, RequestParams):
        return value
    if not isinstance(value, dict):
        raise AgentConfigError(f"'request_params' must be a mapping in {path}")
    try:
        return RequestParams(**value)
    except Exception as exc:
        raise AgentConfigError(f"Invalid request_params in {path}", str(exc)) from exc


def _ensure_tool_input_schema(value: Any, path: Path) -> dict[str, Any] | None:
    validation = validate_tool_input_schema(value)
    if validation.errors:
        details = "; ".join(validation.errors)
        raise AgentConfigError(
            f"Invalid 'tool_input_schema' in {path}",
            details,
        )

    for warning_message in validation.warnings:
        warnings.warn(
            f"{path}: tool_input_schema {warning_message}",
            UserWarning,
            stacklevel=3,
        )

    return validation.normalized


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
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentConfigError(f"'{field}' must be an integer in {path}")
    return value


def _ensure_float(value: Any, field: str, path: Path) -> float:
    if isinstance(value, bool):
        raise AgentConfigError(f"'{field}' must be a number in {path}")
    if isinstance(value, (int, float)):
        return float(value)
    raise AgentConfigError(f"'{field}' must be a number in {path}")


def dump_agents_to_dir(
    agents: dict[str, AgentCardData],
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
    agent_data: AgentCardData,
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
    agent_data: AgentCardData,
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


def _resolve_dump_card_type(name: str, agent_data: AgentCardData) -> CardType:
    agent_type_value = agent_data.get("type")
    if not isinstance(agent_type_value, str):
        raise AgentConfigError(f"Agent '{name}' is missing a valid type")

    card_type = AGENT_TYPE_TO_CARD_TYPE.get(agent_type_value)
    if card_type is None:
        raise AgentConfigError(f"Agent '{name}' has unsupported type '{agent_type_value}'")
    return card_type


def _resolve_dump_config(name: str, agent_data: AgentCardData) -> AgentConfig:
    config = agent_data.get("config")
    if not isinstance(config, AgentConfig):
        raise AgentConfigError(f"Agent '{name}' is missing AgentConfig")
    return config


def _resolve_dump_instruction(name: str, config: AgentConfig) -> str:
    instruction = config.instruction
    if not instruction:
        raise AgentConfigError(f"Agent '{name}' is missing instruction")
    return instruction


def _set_allowed(
    card: dict[str, Any],
    allowed_fields: set[str],
    field: str,
    value: Any,
    *,
    when: bool,
) -> None:
    if when and field in allowed_fields:
        card[field] = value


def _serialize_use_history(
    card: dict[str, Any],
    card_type: CardType,
    allowed_fields: set[str],
    use_history: bool,
) -> None:
    if "use_history" not in allowed_fields:
        return
    default_use_history = DEFAULT_USE_HISTORY_BY_TYPE[card_type]
    if use_history != default_use_history:
        card["use_history"] = use_history


def _serialize_optional_common_fields(
    card: dict[str, Any],
    allowed_fields: set[str],
    config: AgentConfig,
) -> None:
    _set_allowed(card, allowed_fields, "default", True, when=config.default)
    _set_allowed(card, allowed_fields, "tool_only", True, when=config.tool_only)
    _set_allowed(
        card,
        allowed_fields,
        "description",
        config.description,
        when=bool(config.description),
    )
    _set_allowed(
        card,
        allowed_fields,
        "tool_input_schema",
        config.tool_input_schema,
        when=config.tool_input_schema is not None,
    )
    _set_allowed(card, allowed_fields, "model", config.model, when=bool(config.model))
    _set_allowed(card, allowed_fields, "human_input", True, when=config.human_input)
    _set_allowed(card, allowed_fields, "api_key", config.api_key, when=bool(config.api_key))
    _set_allowed(
        card,
        allowed_fields,
        "servers",
        list(config.servers),
        when=bool(config.servers),
    )
    _set_allowed(
        card,
        allowed_fields,
        "mcp_connect",
        _serialize_mcp_connect_targets(config.mcp_connect),
        when=bool(config.mcp_connect),
    )
    _set_allowed(card, allowed_fields, "tools", config.tools, when=bool(config.tools))
    _set_allowed(
        card,
        allowed_fields,
        "resources",
        config.resources,
        when=bool(config.resources),
    )
    _set_allowed(card, allowed_fields, "prompts", config.prompts, when=bool(config.prompts))


def _serialize_common_card_fields(
    card: dict[str, Any],
    card_type: CardType,
    agent_data: AgentCardData,
    config: AgentConfig,
    message_paths: list[Path] | None,
) -> None:
    schema_version = agent_data.get("schema_version")
    if isinstance(schema_version, int):
        card["schema_version"] = schema_version

    allowed_fields = ALLOWED_FIELDS_BY_TYPE[card_type]

    _serialize_optional_common_fields(card, allowed_fields, config)
    _serialize_use_history(card, card_type, allowed_fields, config.use_history)

    serialized_skills = _serialize_skills(config.skills)
    if serialized_skills is not None and "skills" in allowed_fields:
        card["skills"] = serialized_skills

    request_params_dump = _dump_request_params(config.default_request_params)
    if request_params_dump and "request_params" in allowed_fields:
        card["request_params"] = request_params_dump

    if message_paths and "messages" in allowed_fields:
        card["messages"] = [str(path) for path in message_paths]


def _optional_mcp_connect_fields(entry: MCPConnectTarget) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if entry.target is not None:
        fields["target"] = entry.target
    if entry.name:
        fields["name"] = entry.name
    if entry.description:
        fields["description"] = entry.description
    if entry.management:
        fields["management"] = entry.management
    if entry.connector_id is not None:
        fields["connector_id"] = entry.connector_id
    if entry.headers is not None:
        fields["headers"] = dict(entry.headers)
    if entry.access_token is not None:
        fields["access_token"] = entry.access_token
    if entry.defer_loading is not None:
        fields["defer_loading"] = entry.defer_loading
    if entry.auth is not None:
        fields["auth"] = dict(entry.auth)
    return fields


def _serialize_mcp_connect_targets(targets: list[MCPConnectTarget]) -> list[dict[str, Any]]:
    return [_optional_mcp_connect_fields(entry) for entry in targets]


def _serialize_agent_like_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    config: AgentConfig,
) -> None:
    child_agents = agent_data.get("child_agents") or []
    if child_agents:
        card["agents"] = list(child_agents)

    _serialize_agents_as_tools_options(card, agent_data.get("agents_as_tools_options"))

    function_tools = serialize_function_tools(agent_data.get("function_tools"))
    if function_tools is None:
        function_tools = serialize_function_tools(config.function_tools)
    if function_tools is not None:
        card["function_tools"] = function_tools

    if config.tool_hooks:
        card["tool_hooks"] = config.tool_hooks

    if config.lifecycle_hooks:
        card["lifecycle_hooks"] = config.lifecycle_hooks

    if config.commands:
        card["commands"] = {
            name: {
                key: value
                for key, value in {
                    "description": spec.description,
                    "input_hint": spec.input_hint,
                    "handler": spec.handler,
                    "key": spec.key,
                }.items()
                if value is not None
            }
            for name, spec in config.commands.items()
        }

    if config.trim_tool_history:
        card["trim_tool_history"] = True


def _serialize_agents_as_tools_options(card: dict[str, Any], options: object) -> None:
    if not is_str_object_dict(options):
        return

    history_source = options.get("history_source")
    if history_source is not None:
        card["history_source"] = _enum_value_or_self(history_source)

    history_merge_target = options.get("history_merge_target")
    if history_merge_target is not None:
        card["history_merge_target"] = _enum_value_or_self(history_merge_target)

    max_parallel = options.get("max_parallel")
    if max_parallel is not None:
        card["max_parallel"] = max_parallel

    child_timeout_sec = options.get("child_timeout_sec")
    if child_timeout_sec is not None:
        card["child_timeout_sec"] = child_timeout_sec

    max_display_instances = options.get("max_display_instances")
    if max_display_instances is not None:
        card["max_display_instances"] = max_display_instances


def _enum_value_or_self(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _serialize_chain_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["sequence"] = list(agent_data.get("sequence") or [])
    cumulative = agent_data.get("cumulative", False)
    if cumulative:
        card["cumulative"] = True


def _serialize_parallel_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["fan_out"] = list(agent_data.get("fan_out") or [])
    fan_in = agent_data.get("fan_in")
    if fan_in:
        card["fan_in"] = fan_in

    include_request = agent_data.get("include_request")
    if include_request is False:
        card["include_request"] = False


def _serialize_evaluator_optimizer_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["generator"] = agent_data.get("generator")
    card["evaluator"] = agent_data.get("evaluator")

    if "min_rating" in agent_data:
        card["min_rating"] = agent_data.get("min_rating")
    if "max_refinements" in agent_data:
        card["max_refinements"] = agent_data.get("max_refinements")
    if "refinement_instruction" in agent_data:
        card["refinement_instruction"] = agent_data.get("refinement_instruction")


def _serialize_router_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["agents"] = list(agent_data.get("router_agents") or [])


def _serialize_orchestrator_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["agents"] = list(agent_data.get("child_agents") or [])
    card["plan_type"] = agent_data.get("plan_type", "full")
    card["plan_iterations"] = agent_data.get("plan_iterations", 5)


def _serialize_iterative_planner_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["agents"] = list(agent_data.get("child_agents") or [])
    card["plan_iterations"] = agent_data.get("plan_iterations", -1)


def _serialize_maker_fields(
    card: dict[str, Any],
    agent_data: AgentCardData,
    _config: AgentConfig,
) -> None:
    card["worker"] = agent_data.get("worker")
    card["k"] = agent_data.get("k", 3)
    card["max_samples"] = agent_data.get("max_samples", 50)
    card["match_strategy"] = agent_data.get("match_strategy", "exact")

    red_flag = agent_data.get("red_flag_max_length")
    if red_flag is not None:
        card["red_flag_max_length"] = red_flag


_CARD_SERIALIZERS: dict[CardType, CardTypeSerializer] = {
    "agent": _serialize_agent_like_fields,
    "smart": _serialize_agent_like_fields,
    "chain": _serialize_chain_fields,
    "parallel": _serialize_parallel_fields,
    "evaluator_optimizer": _serialize_evaluator_optimizer_fields,
    "router": _serialize_router_fields,
    "orchestrator": _serialize_orchestrator_fields,
    "iterative_planner": _serialize_iterative_planner_fields,
    "MAKER": _serialize_maker_fields,
}


def _build_card_dump(
    name: str,
    agent_data: AgentCardData,
    message_paths: list[Path] | None,
) -> tuple[dict[str, Any], str]:
    card_type = _resolve_dump_card_type(name, agent_data)
    config = _resolve_dump_config(name, agent_data)
    instruction = _resolve_dump_instruction(name, config)

    card: dict[str, Any] = {"type": card_type, "name": name}
    _serialize_common_card_fields(card, card_type, agent_data, config, message_paths)
    serializer = _CARD_SERIALIZERS[card_type]
    serializer(card, agent_data, config)

    return card, instruction


def _dump_request_params(params: RequestParams | None) -> dict[str, Any] | None:
    if params is None:
        return None
    dump = params.model_dump(
        exclude_defaults=True,
        exclude={"messages", "systemPrompt", "use_history", "model", "batch_context"},
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
        return [item for item in value if isinstance(item, str)]
    return None
