"""
Validation utilities for FastAgent configuration and dependencies.
"""

from typing import Any, Dict, List

from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.exceptions import (
    AgentConfigError,
    CircularDependencyError,
    ServerConfigError,
)
from mcp_agent.llm.augmented_llm import AugmentedLLM


def validate_server_references(context, agents: Dict[str, Dict[str, Any]]) -> None:
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


def validate_workflow_references(agents: Dict[str, Dict[str, Any]]) -> None:
    """
    Validate that all workflow references point to valid agents/workflows.
    Also validates that referenced agents have required configuration.
    Raises AgentConfigError if any validation fails.

    Args:
        agents: Dictionary of agent configurations
    """
    available_components = set(agents.keys())

    for name, agent_data in agents.items():
        agent_type = agent_data["type"]  # This is a string from config
        
        # Note: Compare string values from config with the Enum's string value
        if agent_type == AgentType.PARALLEL.value:
            # Check fan_in exists
            fan_in = agent_data["fan_in"]
            if fan_in and fan_in not in available_components:
                raise AgentConfigError(
                    f"Parallel workflow '{name}' references non-existent fan_in component: {fan_in}"
                )

            # Check fan_out agents exist
            fan_out = agent_data["fan_out"]
            missing = [a for a in fan_out if a not in available_components]
            if missing:
                raise AgentConfigError(
                    f"Parallel workflow '{name}' references non-existent fan_out components: {', '.join(missing)}"
                )

        elif agent_type == AgentType.ORCHESTRATOR.value:
            # Check all child agents exist and are properly configured
            child_agents = agent_data["child_agents"]
            missing = [a for a in child_agents if a not in available_components]
            if missing:
                raise AgentConfigError(
                    f"Orchestrator '{name}' references non-existent agents: {', '.join(missing)}"
                )

            # Validate child agents have required LLM configuration
            for agent_name in child_agents:
                child_data = agents[agent_name]
                if child_data["type"] == AgentType.BASIC.value:
                    # For basic agents, we'll validate LLM config during creation
                    continue
                # Check if it's a workflow type or has LLM capability
                # Workflows like EvaluatorOptimizer and Parallel are valid for orchestrator
                func = child_data["func"]
                workflow_types = [
                    AgentType.EVALUATOR_OPTIMIZER.value,
                    AgentType.PARALLEL.value,
                    AgentType.ROUTER.value,
                    AgentType.CHAIN.value,
                ]

                if not (
                    isinstance(func, AugmentedLLM)
                    or child_data["type"] in workflow_types
                    or (hasattr(func, "_llm") and func._llm is not None)
                ):
                    raise AgentConfigError(
                        f"Agent '{agent_name}' used by orchestrator '{name}' lacks LLM capability",
                        "All agents used by orchestrators must be LLM-capable (either an AugmentedLLM or have an _llm property)",
                    )

        elif agent_type == AgentType.ROUTER.value:
            # Check all referenced agents exist
            router_agents = agent_data["router_agents"]
            missing = [a for a in router_agents if a not in available_components]
            if missing:
                raise AgentConfigError(
                    f"Router '{name}' references non-existent agents: {', '.join(missing)}"
                )

        elif agent_type == AgentType.EVALUATOR_OPTIMIZER.value:
            # Check both evaluator and optimizer exist
            evaluator = agent_data["evaluator"]
            generator = agent_data["generator"]
            missing = []
            if evaluator not in available_components:
                missing.append(f"evaluator: {evaluator}")
            if generator not in available_components:
                missing.append(f"generator: {generator}")
            if missing:
                raise AgentConfigError(
                    f"Evaluator-Optimizer '{name}' references non-existent components: {', '.join(missing)}"
                )

        elif agent_type == AgentType.CHAIN.value:
            # Check that all agents in the sequence exist
            sequence = agent_data.get("sequence", agent_data.get("agents", []))
            missing = [a for a in sequence if a not in available_components]
            if missing:
                raise AgentConfigError(
                    f"Chain '{name}' references non-existent agents: {', '.join(missing)}"
                )


def get_dependencies(
    name: str,
    agents: Dict[str, Dict[str, Any]],
    visited: set,
    path: set,
    agent_type: AgentType = None,
) -> List[str]:
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

    # Skip if not the requested type (when filtering)
    if agent_type and config["type"] != agent_type.value:
        return []

    path.add(name)
    deps = []

    # Handle dependencies based on agent type
    if config["type"] == AgentType.PARALLEL.value:
        # Get dependencies from fan-out agents
        for fan_out in config["fan_out"]:
            deps.extend(get_dependencies(fan_out, agents, visited, path, agent_type))
    elif config["type"] == AgentType.CHAIN.value:
        # Get dependencies from sequence agents
        sequence = config.get("sequence", config.get("router_agents", []))
        for agent_name in sequence:
            deps.extend(get_dependencies(agent_name, agents, visited, path, agent_type))

    # Add this agent after its dependencies
    deps.append(name)
    visited.add(name)
    path.remove(name)

    return deps


def get_dependencies_groups(
    agents_dict: Dict[str, Dict[str, Any]], allow_cycles: bool = False
) -> List[List[str]]:
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
    dependencies = {name: set() for name in agent_names}

    # Build the dependency graph
    for name, agent_data in agents_dict.items():
        agent_type = agent_data["type"]  # This is a string from config
        
        # Note: Compare string values from config with the Enum's string value
        if agent_type == AgentType.PARALLEL.value:
            # Parallel agents depend on their fan-out and fan-in agents
            dependencies[name].update(agent_data.get("parallel_agents", []))
            # Also add explicit fan_out dependencies if present
            if "fan_out" in agent_data:
                dependencies[name].update(agent_data["fan_out"])
            # Add explicit fan_in dependency if present
            if "fan_in" in agent_data and agent_data["fan_in"]:
                dependencies[name].add(agent_data["fan_in"])
        elif agent_type == AgentType.CHAIN.value:
            # Chain agents depend on the agents in their sequence
            dependencies[name].update(agent_data.get("sequence", []))
        elif agent_type == AgentType.ROUTER.value:
            # Router agents depend on the agents they route to
            dependencies[name].update(agent_data.get("router_agents", []))
        elif agent_type == AgentType.ORCHESTRATOR.value:
            # Orchestrator agents depend on their child agents
            dependencies[name].update(agent_data.get("child_agents", []))
        elif agent_type == AgentType.EVALUATOR_OPTIMIZER.value:
            # Evaluator-Optimizer agents depend on their evaluator and generator agents
            if "evaluator" in agent_data:
                dependencies[name].add(agent_data["evaluator"])
            if "generator" in agent_data:
                dependencies[name].add(agent_data["generator"])
            # For backward compatibility - also check eval_optimizer_agents if present
            dependencies[name].update(agent_data.get("eval_optimizer_agents", []))

    # Check for cycles if not allowed
    if not allow_cycles:
        visited = set()
        path = set()

        def visit(node) -> None:
            if node in path:
                path_str = " -> ".join(path) + " -> " + node
                raise CircularDependencyError(f"Circular dependency detected: {path_str}")
            if node in visited:
                return

            path.add(node)
            for dep in dependencies[node]:
                if dep in agent_names:  # Skip dependencies to non-existent agents
                    visit(dep)
            path.remove(node)
            visited.add(node)

        # Check each node
        for name in agent_names:
            if name not in visited:
                visit(name)

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
