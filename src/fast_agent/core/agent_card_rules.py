"""Shared AgentCard parsing rules used by loader and validator."""

from __future__ import annotations

from typing import Literal, cast

from fast_agent.agents.agent_types import AgentType

CardType = Literal[
    "agent",
    "smart",
    "chain",
    "parallel",
    "evaluator_optimizer",
    "router",
    "orchestrator",
    "iterative_planner",
    "MAKER",
]

CARD_TYPE_TO_AGENT_TYPE: dict[CardType, AgentType] = {
    "agent": AgentType.BASIC,
    "smart": AgentType.SMART,
    "chain": AgentType.CHAIN,
    "parallel": AgentType.PARALLEL,
    "evaluator_optimizer": AgentType.EVALUATOR_OPTIMIZER,
    "router": AgentType.ROUTER,
    "orchestrator": AgentType.ORCHESTRATOR,
    "iterative_planner": AgentType.ITERATIVE_PLANNER,
    "MAKER": AgentType.MAKER,
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
    "agents",
    "servers",
    "tools",
    "resources",
    "prompts",
    "mcp_connect",
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
    "lifecycle_hooks",
    "trim_tool_history",
    "tool_input_schema",
    "messages",
    "shell",
    "cwd",
}

ALLOWED_FIELDS_BY_TYPE: dict[CardType, set[str]] = {
    "agent": set(AGENT_CARD_FIELDS),
    "smart": set(AGENT_CARD_FIELDS),
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
}

REQUIRED_FIELDS_BY_TYPE: dict[CardType, set[str]] = {
    "agent": set(),
    "smart": set(),
    "chain": {"sequence"},
    "parallel": {"fan_out"},
    "evaluator_optimizer": {"generator", "evaluator"},
    "router": {"agents"},
    "orchestrator": {"agents"},
    "iterative_planner": {"agents"},
    "MAKER": {"worker"},
}

DEFAULT_USE_HISTORY_BY_TYPE: dict[CardType, bool] = {
    "agent": True,
    "smart": True,
    "chain": True,
    "parallel": True,
    "evaluator_optimizer": True,
    "router": False,
    "orchestrator": False,
    "iterative_planner": False,
    "MAKER": True,
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
    }
)


def normalize_card_type(raw_type: str | None) -> CardType | None:
    if raw_type is None:
        return "agent"

    type_key = raw_type.strip().lower() or "agent"
    normalized = "MAKER" if type_key == "maker" else type_key
    if normalized not in ALLOWED_FIELDS_BY_TYPE:
        return None

    return cast("CardType", normalized)
