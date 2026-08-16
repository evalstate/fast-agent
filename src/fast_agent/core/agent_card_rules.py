"""Shared AgentCard parsing rules used by loader and validator."""

from __future__ import annotations

from typing import Any, Literal

from fast_agent.agents.agent_types import AgentType
from fast_agent.utils.action_normalization import normalize_action_token

CardType = Literal[
    "agent",
    "chain",
    "parallel",
    "evaluator_optimizer",
    "router",
    "orchestrator",
    "iterative_planner",
    "MAKER",
    "a2a",
]

LEGACY_SMART_TYPE_WARNING = (
    "AgentCard type 'smart' is deprecated in fast-agent 0.10 and will be removed in 0.11. "
    "Treating it as type 'agent' and defaulting subagents and harness_tools to true. "
    "The legacy Smart tool is not restored; use harness_tools for supported commands, "
    "including /mcp and /skills."
)

CARD_TYPE_TO_AGENT_TYPE: dict[CardType, AgentType] = {
    "agent": AgentType.BASIC,
    "chain": AgentType.CHAIN,
    "parallel": AgentType.PARALLEL,
    "evaluator_optimizer": AgentType.EVALUATOR_OPTIMIZER,
    "router": AgentType.ROUTER,
    "orchestrator": AgentType.ORCHESTRATOR,
    "iterative_planner": AgentType.ITERATIVE_PLANNER,
    "MAKER": AgentType.MAKER,
    "a2a": AgentType.A2A,
}

AGENT_TYPE_TO_CARD_TYPE: dict[str, CardType] = {
    agent_type.value: card_type for card_type, agent_type in CARD_TYPE_TO_AGENT_TYPE.items()
}

COMMON_CARD_FIELDS = {
    "type",
    "name",
    "instruction",
    "description",
    "default",
    "tool_only",
    "schema_version",
}

AGENT_CARD_FIELDS = {
    *COMMON_CARD_FIELDS,
    "subagents",
    "subagent_model",
    "harness_tools",
    "agents",
    "servers",
    "tools",
    "resources",
    "prompts",
    "mcp_connect",
    "skills",
    "model",
    "use_history",
    "save_trajectory",
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
    "lifecycle_hooks",
    "commands",
    "trim_tool_history",
    "tool_input_schema",
    "messages",
    "shell",
    "cwd",
    "variables",
}

ALLOWED_FIELDS_BY_TYPE: dict[CardType, set[str]] = {
    "agent": set(AGENT_CARD_FIELDS),
    "chain": {
        *COMMON_CARD_FIELDS,
        "sequence",
        "cumulative",
    },
    "parallel": {
        *COMMON_CARD_FIELDS,
        "fan_out",
        "fan_in",
        "include_request",
    },
    "evaluator_optimizer": {
        *COMMON_CARD_FIELDS,
        "generator",
        "evaluator",
        "min_rating",
        "max_refinements",
        "refinement_instruction",
        "messages",
    },
    "router": {
        *COMMON_CARD_FIELDS,
        "agents",
        "servers",
        "tools",
        "resources",
        "prompts",
        "model",
        "use_history",
        "save_trajectory",
        "request_params",
        "human_input",
        "api_key",
        "messages",
    },
    "orchestrator": {
        *COMMON_CARD_FIELDS,
        "agents",
        "model",
        "use_history",
        "save_trajectory",
        "request_params",
        "human_input",
        "api_key",
        "plan_type",
        "plan_iterations",
        "messages",
    },
    "iterative_planner": {
        *COMMON_CARD_FIELDS,
        "agents",
        "model",
        "request_params",
        "api_key",
        "plan_iterations",
        "messages",
    },
    "MAKER": {
        *COMMON_CARD_FIELDS,
        "worker",
        "k",
        "max_samples",
        "match_strategy",
        "red_flag_max_length",
        "messages",
    },
    "a2a": {
        *COMMON_CARD_FIELDS,
        "url",
        "transport",
        "streaming",
        "polling",
        "accepted_output_modes",
        "headers",
        "auth",
        "relative_card_path",
        "request_timeout_seconds",
    },
}

REQUIRED_FIELDS_BY_TYPE: dict[CardType, set[str]] = {
    "agent": set(),
    "chain": {"sequence"},
    "parallel": {"fan_out"},
    "evaluator_optimizer": {"generator", "evaluator"},
    "router": {"agents"},
    "orchestrator": {"agents"},
    "iterative_planner": {"agents"},
    "MAKER": {"worker"},
    "a2a": {"url"},
}

DEFAULT_USE_HISTORY_BY_TYPE: dict[CardType, bool] = {
    "agent": True,
    "chain": True,
    "parallel": True,
    "evaluator_optimizer": True,
    "router": False,
    "orchestrator": False,
    "iterative_planner": False,
    "MAKER": True,
    "a2a": True,
}

MCP_CONNECT_ALLOWED_KEYS = frozenset(
    {
        "target",
        "name",
        "description",
        "management",
        "connector_id",
        "headers",
        "access_token",
        "defer_loading",
        "auth",
        "protocol_mode",
    }
)


def normalize_card_type(raw_type: str | None) -> CardType | None:
    if raw_type is None:
        return "agent"

    type_key = normalize_action_token(raw_type) or "agent"
    normalized = "MAKER" if type_key == "maker" else type_key
    if normalized not in ALLOWED_FIELDS_BY_TYPE:
        return None

    return normalized


def apply_legacy_smart_defaults(raw: dict[str, Any]) -> bool:
    """Normalize the 0.10 compatibility alias without restoring a Smart agent type."""
    raw_type = raw.get("type")
    if not isinstance(raw_type, str) or normalize_action_token(raw_type) != "smart":
        return False

    raw["type"] = "agent"
    raw.setdefault("subagents", True)
    raw.setdefault("harness_tools", True)
    return True
