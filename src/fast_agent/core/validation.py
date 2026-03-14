"""
Validation utilities for FastAgent configuration and dependencies.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from fast_agent.agents.agent_types import AgentType
from fast_agent.core.agent_card_types import AgentCardData
from fast_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ServerConfigError,
)
from fast_agent.interfaces import LlmAgentProtocol
from fast_agent.llm.fastagent_llm import FastAgentLLM

_BASIC_LIKE_AGENT_TYPE_VALUES = frozenset(
    {
        AgentType.LLM.value,
        AgentType.BASIC.value,
    }
)

_PLANNER_COMPATIBLE_WORKFLOW_TYPES = frozenset(
    {
        AgentType.CHAIN.value,
        AgentType.EVALUATOR_OPTIMIZER.value,
        AgentType.ITERATIVE_PLANNER.value,
        AgentType.MAKER.value,
        AgentType.ORCHESTRATOR.value,
        AgentType.PARALLEL.value,
        AgentType.ROUTER.value,
    }
)


@dataclass(frozen=True)
class _DependencyFieldRule:
    field_name: str
    missing_label: str


@dataclass(frozen=True)
class DependencyFieldSpec:
    field_name: str
    multiple: bool


CompatibilityCheck = Callable[
    [str, Mapping[str, Any], Mapping[str, AgentCardData | dict[str, Any]]],
    None,
]


@dataclass(frozen=True)
class _WorkflowReferenceRule:
    component_label: str
    dependency_fields: tuple[_DependencyFieldRule, ...] = ()
    compatibility_check: CompatibilityCheck | None = None
    combine_missing_fields: bool = False


_WORKFLOW_REFERENCE_RULES: dict[str, _WorkflowReferenceRule] = {
    AgentType.PARALLEL.value: _WorkflowReferenceRule(
        component_label="Parallel workflow",
        dependency_fields=(
            _DependencyFieldRule("fan_in", "fan_in component"),
            _DependencyFieldRule("fan_out", "fan_out components"),
        ),
    ),
    AgentType.ORCHESTRATOR.value: _WorkflowReferenceRule(
        component_label="Orchestrator",
        dependency_fields=(
            _DependencyFieldRule("child_agents", "agents"),
        ),
    ),
    AgentType.ITERATIVE_PLANNER.value: _WorkflowReferenceRule(
        component_label="Iterative planner",
        dependency_fields=(
            _DependencyFieldRule("child_agents", "agents"),
        ),
    ),
    AgentType.ROUTER.value: _WorkflowReferenceRule(
        component_label="Router",
        dependency_fields=(
            _DependencyFieldRule("router_agents", "agents"),
        ),
    ),
    AgentType.EVALUATOR_OPTIMIZER.value: _WorkflowReferenceRule(
        component_label="Evaluator-Optimizer",
        dependency_fields=(
            _DependencyFieldRule("evaluator", "evaluator"),
            _DependencyFieldRule("generator", "generator"),
        ),
        combine_missing_fields=True,
    ),
    AgentType.CHAIN.value: _WorkflowReferenceRule(
        component_label="Chain",
        dependency_fields=(
            _DependencyFieldRule("sequence", "agents"),
        ),
    ),
    AgentType.MAKER.value: _WorkflowReferenceRule(
        component_label="Maker",
        dependency_fields=(
            _DependencyFieldRule("worker", "worker"),
        ),
    ),
}

_AGENT_DEPENDENCY_FIELD_SPECS: dict[str, tuple[DependencyFieldSpec, ...]] = {
    AgentType.CHAIN.value: (DependencyFieldSpec("sequence", multiple=True),),
    AgentType.CUSTOM.value: (DependencyFieldSpec("child_agents", multiple=True),),
    AgentType.EVALUATOR_OPTIMIZER.value: (
        DependencyFieldSpec("evaluator", multiple=False),
        DependencyFieldSpec("generator", multiple=False),
        DependencyFieldSpec("eval_optimizer_agents", multiple=True),
    ),
    AgentType.ITERATIVE_PLANNER.value: (DependencyFieldSpec("child_agents", multiple=True),),
    AgentType.MAKER.value: (DependencyFieldSpec("worker", multiple=False),),
    AgentType.ORCHESTRATOR.value: (DependencyFieldSpec("child_agents", multiple=True),),
    AgentType.SMART.value: (DependencyFieldSpec("child_agents", multiple=True),),
    AgentType.PARALLEL.value: (
        DependencyFieldSpec("fan_out", multiple=True),
        DependencyFieldSpec("fan_in", multiple=False),
        DependencyFieldSpec("parallel_agents", multiple=True),
    ),
    AgentType.ROUTER.value: (DependencyFieldSpec("router_agents", multiple=True),),
}

_CARD_DEPENDENCY_FIELD_SPECS: dict[str, tuple[DependencyFieldSpec, ...]] = {
    "agent": (DependencyFieldSpec("agents", multiple=True),),
    "smart": (DependencyFieldSpec("agents", multiple=True),),
    "chain": (DependencyFieldSpec("sequence", multiple=True),),
    "parallel": (
        DependencyFieldSpec("fan_out", multiple=True),
        DependencyFieldSpec("fan_in", multiple=False),
    ),
    "router": (DependencyFieldSpec("agents", multiple=True),),
    "orchestrator": (DependencyFieldSpec("agents", multiple=True),),
    "iterative_planner": (DependencyFieldSpec("agents", multiple=True),),
    "evaluator_optimizer": (
        DependencyFieldSpec("evaluator", multiple=False),
        DependencyFieldSpec("generator", multiple=False),
    ),
    "MAKER": (DependencyFieldSpec("worker", multiple=False),),
}


def normalize_agent_type_value(agent_type: AgentType | str | None) -> str | None:
    if isinstance(agent_type, AgentType):
        return agent_type.value
    if not isinstance(agent_type, str):
        return None

    normalized = agent_type.strip().lower()
    return normalized or None


def is_basic_like_agent_type(agent_type: AgentType | str | None) -> bool:
    return normalize_agent_type_value(agent_type) in _BASIC_LIKE_AGENT_TYPE_VALUES


def get_agent_dependency_attribute_names(agent_type: AgentType | str | None) -> tuple[str, ...]:
    return tuple(
        field_spec.field_name
        for field_spec in get_agent_dependency_field_specs(agent_type)
    )


def get_agent_dependency_field_specs(
    agent_type: AgentType | str | None,
) -> tuple[DependencyFieldSpec, ...]:
    normalized = normalize_agent_type_value(agent_type)
    if normalized is None:
        return ()
    if is_basic_like_agent_type(normalized):
        return (DependencyFieldSpec("child_agents", multiple=True),)
    return _AGENT_DEPENDENCY_FIELD_SPECS.get(normalized, ())


def get_card_dependency_field_specs(card_type: str | None) -> tuple[DependencyFieldSpec, ...]:
    if not isinstance(card_type, str) or not card_type:
        return ()
    return _CARD_DEPENDENCY_FIELD_SPECS.get(card_type, ())


def get_custom_agent_class_reference(agent_data: Mapping[str, Any]) -> Any:
    return agent_data.get("agent_class") or agent_data.get("cls")


def _validate_custom_agent_reference(name: str, agent_data: Mapping[str, Any]) -> None:
    if get_custom_agent_class_reference(agent_data) is None:
        raise AgentConfigError(
            f"Custom agent '{name}' missing class reference ('agent_class' or 'cls')"
        )


def _iter_dependency_values(
    agent_data: Mapping[str, Any],
    field_name: str,
) -> list[str]:
    dependency_value = agent_data.get(field_name)
    if dependency_value is None:
        return []
    if isinstance(dependency_value, str):
        return [dependency_value] if dependency_value else []
    return [value for value in dependency_value if isinstance(value, str)]


def collect_dependencies_from_fields(
    agent_data: Mapping[str, Any],
    dependency_fields: tuple[DependencyFieldSpec, ...],
) -> set[str]:
    deps: set[str] = set()
    for field_spec in dependency_fields:
        deps.update(_iter_dependency_values(agent_data, field_spec.field_name))
    return deps


def _missing_dependency_values(
    agent_data: Mapping[str, Any],
    field_rule: _DependencyFieldRule,
    available_components: set[str],
) -> list[str]:
    return [
        value
        for value in _iter_dependency_values(agent_data, field_rule.field_name)
        if value not in available_components
    ]


def _raise_missing_dependency_error(
    name: str,
    rule: _WorkflowReferenceRule,
    missing_by_field: list[tuple[_DependencyFieldRule, list[str]]],
) -> None:
    non_empty = [(field_rule, values) for field_rule, values in missing_by_field if values]
    if not non_empty:
        return

    if not rule.combine_missing_fields or len(non_empty) == 1:
        field_rule, values = non_empty[0]
        raise AgentConfigError(
            f"{rule.component_label} '{name}' references non-existent {field_rule.missing_label}: "
            f"{', '.join(values)}"
        )

    missing_parts = [
        f"{field_rule.missing_label}: {value}"
        for field_rule, values in non_empty
        for value in values
    ]
    raise AgentConfigError(
        f"{rule.component_label} '{name}' references non-existent components: "
        f"{', '.join(missing_parts)}"
    )


def _validate_planner_children(
    name: str,
    agent_data: Mapping[str, Any],
    agents: Mapping[str, AgentCardData | dict[str, Any]],
) -> None:
    child_agents = _iter_dependency_values(agent_data, "child_agents")
    for agent_name in child_agents:
        child_data = agents[agent_name]
        child_type = normalize_agent_type_value(child_data.get("type"))
        if is_basic_like_agent_type(child_type) or child_type in {
            AgentType.SMART.value,
            AgentType.CUSTOM.value,
        }:
            continue

        func = child_data["func"]
        if not (
            isinstance(func, FastAgentLLM)
            or child_type in _PLANNER_COMPATIBLE_WORKFLOW_TYPES
            or (isinstance(func, LlmAgentProtocol) and func.llm is not None)
        ):
            raise AgentConfigError(
                f"Agent '{agent_name}' used by '{name}' lacks LLM capability",
                "All agents used by orchestrators or iterative planners must be LLM-capable "
                "(either an AugmentedLLM or implement LlmAgentProtocol)",
            )


_WORKFLOW_REFERENCE_RULES = {
    **_WORKFLOW_REFERENCE_RULES,
    AgentType.ORCHESTRATOR.value: _WorkflowReferenceRule(
        component_label="Orchestrator",
        dependency_fields=(
            _DependencyFieldRule("child_agents", "agents"),
        ),
        compatibility_check=_validate_planner_children,
    ),
    AgentType.ITERATIVE_PLANNER.value: _WorkflowReferenceRule(
        component_label="Iterative planner",
        dependency_fields=(
            _DependencyFieldRule("child_agents", "agents"),
        ),
        compatibility_check=_validate_planner_children,
    ),
}


def validate_server_references(context, agents: Mapping[str, AgentCardData | dict[str, Any]]) -> None:
    """
    Validate that all server references in agent configurations exist in config.
    Raises ServerConfigError if any referenced servers are not defined.

    Args:
        context: Application context
        agents: Dictionary of agent configurations
    """
    if not context.config.mcp or not context.config.mcp.servers:
        available_servers = set()
    else:
        available_servers = set(context.config.mcp.servers.keys())

    # Check each agent's server references
    for name, agent_data in agents.items():
        config = agent_data["config"]
        if config.servers:
            missing = [s for s in config.servers if s not in available_servers]
            if missing:
                raise ServerConfigError(
                    f"Missing server configuration for agent '{name}'",
                    f"The following servers are referenced but not defined in config: {', '.join(missing)}",
                )


def validate_workflow_references(agents: Mapping[str, AgentCardData | dict[str, Any]]) -> None:
    """
    Validate that all workflow references point to valid agents/workflows.
    Also validates that referenced agents have required configuration.
    Raises AgentConfigError if any validation fails.

    Args:
        agents: Dictionary of agent configurations
    """
    available_components = set(agents.keys())

    for name, agent_data in agents.items():
        agent_type = normalize_agent_type_value(agent_data.get("type"))
        if agent_type == AgentType.CUSTOM.value:
            _validate_custom_agent_reference(name, agent_data)

        rule = _WORKFLOW_REFERENCE_RULES.get(agent_type or "")
        if rule is None:
            continue

        missing_by_field = [
            (
                field_rule,
                _missing_dependency_values(agent_data, field_rule, available_components),
            )
            for field_rule in rule.dependency_fields
        ]
        _raise_missing_dependency_error(name, rule, missing_by_field)

        if rule.compatibility_check is not None:
            rule.compatibility_check(name, agent_data, agents)


def get_dependencies(
    name: str,
    agents: Mapping[str, AgentCardData | dict[str, Any]],
    visited: set,
    path: set,
    agent_type: AgentType | None = None,
) -> list[str]:
    """
    Get dependencies for an agent in topological order.
    Works for both Parallel and Chain workflows.

    Args:
        name: Name of the agent
        agents: Dictionary of agent configurations
        visited: Set of already visited agents
        path: Current path for cycle detection
        agent_type: Optional type filter (e.g., only check Parallel or Chain)

    Returns:
        List of agent names in dependency order

    Raises:
        CircularDependencyError: If circular dependency detected
    """
    if name in path:
        path_str = " -> ".join(path)
        raise CircularDependencyError(f"Path: {path_str} -> {name}")

    if name in visited:
        return []

    if name not in agents:
        return []

    config = agents[name]
    config_agent_type = normalize_agent_type_value(config.get("type"))

    # Skip if not the requested type (when filtering)
    if agent_type and config_agent_type != agent_type.value:
        return []

    path.add(name)
    deps = []

    # Handle dependencies based on agent type
    if config_agent_type == AgentType.PARALLEL.value:
        # Get dependencies from fan-out agents
        for fan_out in config["fan_out"]:
            deps.extend(get_dependencies(fan_out, agents, visited, path, agent_type))
    elif config_agent_type == AgentType.CHAIN.value:
        # Get dependencies from sequence agents
        sequence = config.get("sequence", config.get("router_agents", []))
        for agent_name in sequence:
            deps.extend(get_dependencies(agent_name, agents, visited, path, agent_type))

    # Add this agent after its dependencies
    deps.append(name)
    visited.add(name)
    path.remove(name)

    return deps


def get_agent_dependencies(agent_data: AgentCardData | dict[str, Any]) -> set[str]:
    dependency_fields = get_agent_dependency_field_specs(agent_data.get("type"))
    if not dependency_fields:
        return set()
    return collect_dependencies_from_fields(agent_data, dependency_fields)


def find_dependency_cycle(
    agent_names: Sequence[str],
    dependencies: Mapping[str, set[str]],
) -> list[str] | None:
    available = set(agent_names)
    visited: set[str] = set()
    active: set[str] = set()
    stack: list[tuple[str, Iterator[str]]] = []

    for start in agent_names:
        if start in visited:
            continue

        stack.append((start, iter(dependencies.get(start, set()))))
        active.add(start)

        while stack:
            node, dep_iter = stack[-1]
            try:
                dep = next(dep_iter)
            except StopIteration:
                stack.pop()
                active.remove(node)
                visited.add(node)
                continue

            if dep not in available:
                continue

            if dep in active:
                path_nodes = [entry[0] for entry in stack]
                cycle_start = path_nodes.index(dep)
                return path_nodes[cycle_start:] + [dep]

            if dep in visited:
                continue

            stack.append((dep, iter(dependencies.get(dep, set()))))
            active.add(dep)

    return None


def get_dependencies_groups(
    agents_dict: Mapping[str, AgentCardData | dict[str, Any]], allow_cycles: bool = False
) -> list[list[str]]:
    """
    Get dependencies between agents and group them into dependency layers.
    Each layer can be initialized in parallel.

    Args:
        agents_dict: Dictionary of agent configurations
        allow_cycles: Whether to allow cyclic dependencies

    Returns:
        List of lists, where each inner list is a group of agents that can be initialized together

    Raises:
        CircularDependencyError: If circular dependency detected and allow_cycles is False
    """
    # Get all agent names
    agent_names = list(agents_dict.keys())

    # Dictionary to store dependencies for each agent
    dependencies = {
        name: get_agent_dependencies(agent_data) for name, agent_data in agents_dict.items()
    }

    # Check for cycles if not allowed
    if not allow_cycles:
        cycle = find_dependency_cycle(agent_names, dependencies)
        if cycle:
            raise CircularDependencyError(
                f"Circular dependency detected: {' -> '.join(cycle)}"
            )

    # Group agents by dependency level
    result = []
    remaining = set(agent_names)

    while remaining:
        # Find all agents that have no remaining dependencies
        current_level = set()
        for name in remaining:
            if not dependencies[name] & remaining:  # If no dependencies in remaining agents
                current_level.add(name)

        if not current_level:
            if allow_cycles:
                # If cycles are allowed, just add one remaining node to break the cycle
                current_level.add(next(iter(remaining)))
            else:
                # This should not happen if we checked for cycles
                raise CircularDependencyError("Unresolvable dependency cycle detected")

        # Add the current level to the result
        result.append(list(current_level))

        # Remove current level from remaining
        remaining -= current_level

    return result


def validate_provider_keys_post_creation(active_agents: dict[str, Any]) -> None:
    """
    Validate that API keys are available for all created agents with LLMs.

    This runs after agent creation when we have actual agent instances.

    Args:
        active_agents: Dictionary of created agent instances

    Raises:
        ProviderKeyError: If any required API key is missing
    """
    for agent_name, agent in active_agents.items():
        llm = getattr(agent, "_llm", None)
        if llm:
            # This throws a ProviderKeyError if the key is not present
            llm._api_key()
