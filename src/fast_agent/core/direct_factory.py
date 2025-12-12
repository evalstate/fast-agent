"""
Direct factory functions for creating agent and workflow instances without proxies.
Implements type-safe factories with improved error handling.
"""

import os
from functools import partial
from typing import Any, Protocol, TypeVar

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.evaluator_optimizer import (
    EvaluatorOptimizerAgent,
    QualityRating,
)
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.core import Core
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.validation import get_dependencies_groups
from fast_agent.event_progress import ProgressAction
from fast_agent.interfaces import (
    AgentProtocol,
    LLMFactoryProtocol,
    ModelFactoryFunctionProtocol,
)
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.ui_agent import McpAgentWithUI
from fast_agent.types import RequestParams

# Type aliases for improved readability and IDE support
AgentDict = dict[str, AgentProtocol]
AgentConfigDict = dict[str, dict[str, Any]]
T = TypeVar("T")  # For generic types


logger = get_logger(__name__)


def _create_agent_with_ui_if_needed(
    agent_class: type,
    config: Any,
    context: Any,
) -> Any:
    """
    Create an agent with UI support if MCP UI mode is enabled.

    Args:
        agent_class: The agent class to potentially enhance with UI
        config: Agent configuration
        context: Application context

    Returns:
        Either a UI-enhanced agent instance or the original agent instance
    """
    # Check UI mode from settings
    settings = context.config if hasattr(context, "config") else None
    ui_mode = getattr(settings, "mcp_ui_mode", "auto") if settings else "auto"

    if ui_mode != "disabled" and agent_class == McpAgent:
        # Use the UI-enhanced agent class instead of the base class
        return McpAgentWithUI(config=config, context=context, ui_mode=ui_mode)
    else:
        # Create the original agent instance
        return agent_class(config=config, context=context)


class AgentCreatorProtocol(Protocol):
    """Protocol for agent creator functions."""

    async def __call__(
        self,
        app_instance: Core,
        agents_dict: AgentConfigDict,
        agent_type: AgentType,
        active_agents: AgentDict | None = None,
        model_factory_func: ModelFactoryFunctionProtocol | None = None,
        **kwargs: Any,
    ) -> AgentDict: ...


HARDCODED_DEFAULT_MODEL = "gpt-5-mini.low"


def get_model_factory(
    context,
    model: str | None = None,
    request_params: RequestParams | None = None,
    default_model: str | None = None,
    cli_model: str | None = None,
) -> LLMFactoryProtocol:
    """
    Get model factory using specified or default model.
    Model string is parsed by ModelFactory to determine provider and reasoning effort.

    Precedence (lowest to highest):
        1. Hardcoded default (gpt-5-mini.low)
        2. FAST_AGENT_MODEL environment variable
        3. Config file default_model
        4. CLI --model argument
        5. Decorator model parameter

    Args:
        context: Application context
        model: Optional model specification string (highest precedence)
        request_params: Optional RequestParams to configure LLM behavior
        default_model: Default model from configuration
        cli_model: Model specified via command line

    Returns:
        ModelFactory instance for the specified or default model
    """
    # Hardcoded default has lowest precedence
    model_spec = HARDCODED_DEFAULT_MODEL

    # Environment variable has next precedence
    env_model = os.getenv("FAST_AGENT_MODEL")
    if env_model:
        model_spec = env_model

    # Config has next precedence
    if default_model or context.config.default_model:
        model_spec = default_model or context.config.default_model

    # Command line override has next precedence
    if cli_model:
        model_spec = cli_model

    # Model from decorator has highest precedence
    if model:
        model_spec = model

    # Update or create request_params with the final model choice
    if request_params:
        request_params = request_params.model_copy(update={"model": model_spec})
    else:
        request_params = RequestParams(model=model_spec)

    # Let model factory handle the model string parsing and setup
    return ModelFactory.create_factory(model_spec)


def get_default_model_source(
    config_default_model: str | None = None,
    cli_model: str | None = None,
) -> str | None:
    """
    Determine the source of the default model selection.
    Returns "environment variable", "config file", or None (if CLI or hardcoded default).

    This is used to display informational messages about where the model
    configuration is coming from. Only shows a message for env var or config file,
    not for explicit CLI usage or the hardcoded system default.
    """
    # CLI model is explicit - no message needed
    if cli_model:
        return None

    # Check if config file has a default model
    if config_default_model:
        return "config file"

    # Check if environment variable is set
    if os.getenv("FAST_AGENT_MODEL"):
        return "environment variable"

    return None


async def create_agents_by_type(
    app_instance: Core,
    agents_dict: AgentConfigDict,
    agent_type: AgentType,
    model_factory_func: ModelFactoryFunctionProtocol,
    active_agents: AgentDict | None = None,
    **kwargs: Any,
) -> AgentDict:
    """
    Generic method to create agents of a specific type without using proxies.

    Args:
        app_instance: The main application instance
        agents_dict: Dictionary of agent configurations
        agent_type: Type of agents to create
        active_agents: Dictionary of already created agents (for dependencies)
        model_factory_func: Function for creating model factories
        **kwargs: Additional type-specific parameters

    Returns:
        Dictionary of initialized agent instances
    """
    if active_agents is None:
        active_agents = {}

    # Create a dictionary to store the initialized agents
    result_agents: AgentDict = {}

    # Get all agents of the specified type
    for name, agent_data in agents_dict.items():
        # Compare type string from config with Enum value
        if agent_data["type"] == agent_type.value:
            # Get common configuration
            config = agent_data["config"]

            # Type-specific initialization based on the Enum type
            # Note: Above we compared string values from config, here we compare Enum objects directly
            if agent_type == AgentType.BASIC:
                # If BASIC agent declares child_agents, build an Agents-as-Tools wrapper
                child_names = agent_data.get("child_agents", []) or []
                if child_names:
                    # Ensure child agents are already created
                    child_agents: list[AgentProtocol] = []
                    for agent_name in child_names:
                        if agent_name not in active_agents:
                            raise AgentConfigError(f"Agent {agent_name} not found")
                        child_agents.append(active_agents[agent_name])

                    # Import here to avoid circulars at module import time
                    from fast_agent.agents.workflow.agents_as_tools_agent import (
                        AgentsAsToolsAgent,
                        AgentsAsToolsOptions,
                    )
                    raw_opts = agent_data.get("agents_as_tools_options") or {}
                    opt_kwargs = {k: v for k, v in raw_opts.items() if v is not None}
                    options = AgentsAsToolsOptions(**opt_kwargs)

                    agent = AgentsAsToolsAgent(
                        config=config,
                        context=app_instance.context,
                        agents=child_agents,  # expose children as tools
                        options=options,
                    )

                    await agent.initialize()

                    # Attach LLM to the agent
                    llm_factory = model_factory_func(model=config.model)
                    await agent.attach_llm(
                        llm_factory,
                        request_params=config.default_request_params,
                        api_key=config.api_key,
                    )
                    result_agents[name] = agent

                    # Log successful agent creation
                    logger.info(
                        f"Loaded {name}",
                        data={
                            "progress_action": ProgressAction.LOADED,
                            "agent_name": name,
                            "target": name,
                        },
                    )
                else:
                    # Create agent with UI support if needed
                    agent = _create_agent_with_ui_if_needed(
                        McpAgent,
                        config,
                        app_instance.context,
                    )

                    await agent.initialize()

                    # Attach LLM to the agent
                    llm_factory = model_factory_func(model=config.model)
                    await agent.attach_llm(
                        llm_factory,
                        request_params=config.default_request_params,
                        api_key=config.api_key,
                    )
                    result_agents[name] = agent

                    # Log successful agent creation
                    logger.info(
                        f"Loaded {name}",
                        data={
                            "progress_action": ProgressAction.LOADED,
                            "agent_name": name,
                            "target": name,
                        },
                    )

            elif agent_type == AgentType.CUSTOM:
                # Get the class to instantiate (support legacy 'agent_class' and new 'cls')
                cls = agent_data.get("agent_class") or agent_data.get("cls")
                if cls is None:
                    raise AgentConfigError(
                        f"Custom agent '{name}' missing class reference ('agent_class' or 'cls')"
                    )

                # Create agent with UI support if needed
                agent = _create_agent_with_ui_if_needed(
                    cls,
                    config,
                    app_instance.context,
                )

                await agent.initialize()
                # Attach LLM to the agent
                llm_factory = model_factory_func(model=config.model)
                await agent.attach_llm(
                    llm_factory,
                    request_params=config.default_request_params,
                    api_key=config.api_key,
                )
                result_agents[name] = agent

                # Log successful agent creation
                logger.info(
                    f"Loaded {name}",
                    data={
                        "progress_action": ProgressAction.LOADED,
                        "agent_name": name,
                        "target": name,
                    },
                )

            elif agent_type == AgentType.ORCHESTRATOR or agent_type == AgentType.ITERATIVE_PLANNER:
                # Get base params configured with model settings
                base_params = (
                    config.default_request_params.model_copy()
                    if config.default_request_params
                    else RequestParams()
                )
                base_params.use_history = False  # Force no history for orchestrator

                # Get the child agents
                child_agents = []
                for agent_name in agent_data["child_agents"]:
                    if agent_name not in active_agents:
                        raise AgentConfigError(f"Agent {agent_name} not found")
                    agent = active_agents[agent_name]
                    child_agents.append(agent)

                orchestrator = IterativePlanner(
                    config=config,
                    context=app_instance.context,
                    agents=child_agents,
                    plan_iterations=agent_data.get("plan_iterations", 5),
                    plan_type=agent_data.get("plan_type", "full"),
                )

                # Initialize the orchestrator
                await orchestrator.initialize()

                # Attach LLM to the orchestrator
                llm_factory = model_factory_func(model=config.model)

                await orchestrator.attach_llm(
                    llm_factory,
                    request_params=config.default_request_params,
                    api_key=config.api_key,
                )

                result_agents[name] = orchestrator

            elif agent_type == AgentType.PARALLEL:
                # Get the fan-out and fan-in agents
                fan_in_name = agent_data.get("fan_in")
                fan_out_names = agent_data["fan_out"]

                # Create or retrieve the fan-in agent
                if not fan_in_name:
                    # Create default fan-in agent with auto-generated name
                    fan_in_name = f"{name}_fan_in"
                    fan_in_agent = await _create_default_fan_in_agent(
                        fan_in_name, app_instance.context, model_factory_func
                    )
                    # Add to result_agents so it's registered properly
                    result_agents[fan_in_name] = fan_in_agent
                elif fan_in_name not in active_agents:
                    raise AgentConfigError(f"Fan-in agent {fan_in_name} not found")
                else:
                    fan_in_agent = active_agents[fan_in_name]

                # Get the fan-out agents
                fan_out_agents = []
                for agent_name in fan_out_names:
                    if agent_name not in active_agents:
                        raise AgentConfigError(f"Fan-out agent {agent_name} not found")
                    fan_out_agents.append(active_agents[agent_name])

                # Create the parallel agent
                parallel = ParallelAgent(
                    config=config,
                    context=app_instance.context,
                    fan_in_agent=fan_in_agent,
                    fan_out_agents=fan_out_agents,
                    include_request=agent_data.get("include_request", True),
                )
                await parallel.initialize()
                result_agents[name] = parallel

            elif agent_type == AgentType.ROUTER:
                # Get the router agents
                router_agents = []
                for agent_name in agent_data["router_agents"]:
                    if agent_name not in active_agents:
                        raise AgentConfigError(f"Router agent {agent_name} not found")
                    router_agents.append(active_agents[agent_name])

                # Create the router agent
                router = RouterAgent(
                    config=config,
                    context=app_instance.context,
                    agents=router_agents,
                    routing_instruction=agent_data.get("instruction"),
                )
                await router.initialize()

                # Attach LLM to the router
                llm_factory = model_factory_func(model=config.model)
                await router.attach_llm(
                    llm_factory,
                    request_params=config.default_request_params,
                    api_key=config.api_key,
                )
                result_agents[name] = router

            elif agent_type == AgentType.CHAIN:
                # Get the chained agents
                chain_agents = []

                agent_names = agent_data["sequence"]
                if 0 == len(agent_names):
                    raise AgentConfigError("No agents in the chain")

                for agent_name in agent_data["sequence"]:
                    if agent_name not in active_agents:
                        raise AgentConfigError(f"Chain agent {agent_name} not found")
                    chain_agents.append(active_agents[agent_name])

                from fast_agent.agents.workflow.chain_agent import ChainAgent

                # Get the cumulative parameter
                cumulative = agent_data.get("cumulative", False)

                chain = ChainAgent(
                    config=config,
                    context=app_instance.context,
                    agents=chain_agents,
                    cumulative=cumulative,
                )
                await chain.initialize()
                result_agents[name] = chain

            elif agent_type == AgentType.EVALUATOR_OPTIMIZER:
                # Get the generator and evaluator agents
                generator_name = agent_data["generator"]
                evaluator_name = agent_data["evaluator"]

                if generator_name not in active_agents:
                    raise AgentConfigError(f"Generator agent {generator_name} not found")

                if evaluator_name not in active_agents:
                    raise AgentConfigError(f"Evaluator agent {evaluator_name} not found")

                generator_agent = active_agents[generator_name]
                evaluator_agent = active_agents[evaluator_name]

                # Get min_rating and max_refinements from agent_data
                min_rating_str = agent_data.get("min_rating", "GOOD")
                min_rating = QualityRating(min_rating_str)
                max_refinements = agent_data.get("max_refinements", 3)

                # Create the evaluator-optimizer agent
                evaluator_optimizer = EvaluatorOptimizerAgent(
                    config=config,
                    context=app_instance.context,
                    generator_agent=generator_agent,
                    evaluator_agent=evaluator_agent,
                    min_rating=min_rating,
                    max_refinements=max_refinements,
                    refinement_instruction=agent_data.get("refinement_instruction"),
                )

                # Initialize the agent
                await evaluator_optimizer.initialize()
                result_agents[name] = evaluator_optimizer

            elif agent_type == AgentType.MAKER:
                # MAKER: Massively decomposed Agentic processes with K-voting Error Reduction
                from fast_agent.agents.workflow.maker_agent import MakerAgent, MatchStrategy

                worker_name = agent_data["worker"]
                if worker_name not in active_agents:
                    raise AgentConfigError(f"Worker agent {worker_name} not found")

                worker_agent = active_agents[worker_name]

                # Parse match strategy
                match_strategy_str = agent_data.get("match_strategy", "exact")
                match_strategy = MatchStrategy(match_strategy_str)

                # Create the MAKER agent
                maker_agent = MakerAgent(
                    config=config,
                    context=app_instance.context,
                    worker_agent=worker_agent,
                    k=agent_data.get("k", 3),
                    max_samples=agent_data.get("max_samples", 50),
                    match_strategy=match_strategy,
                    red_flag_max_length=agent_data.get("red_flag_max_length"),
                )

                # Initialize the agent
                await maker_agent.initialize()
                result_agents[name] = maker_agent

            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

    return result_agents


async def active_agents_in_dependency_group(
    app_instance: Core,
    agents_dict: AgentConfigDict,
    model_factory_func: ModelFactoryFunctionProtocol,
    group: list[str],
    active_agents: AgentDict,
):
    """
    For each of the possible agent types, create agents and update the active agents dictionary.

    Notice: This function modifies the active_agents dictionary in-place which is a feature (no copies).
    """
    type_of_agents = list(map(lambda c: (c, c.value), AgentType))
    for agent_type, agent_type_value in type_of_agents:
        agents_dict_local = {
            name: agents_dict[name]
            for name in group
            if agents_dict[name]["type"] == agent_type_value
        }
        agents = await create_agents_by_type(
            app_instance,
            agents_dict_local,
            agent_type,
            model_factory_func,
            active_agents,
        )
        active_agents.update(agents)


async def create_agents_in_dependency_order(
    app_instance: Core,
    agents_dict: AgentConfigDict,
    model_factory_func: ModelFactoryFunctionProtocol,
    allow_cycles: bool = False,
) -> AgentDict:
    """
    Create agent instances in dependency order without proxies.

    Args:
        app_instance: The main application instance
        agents_dict: Dictionary of agent configurations
        model_factory_func: Function for creating model factories
        allow_cycles: Whether to allow cyclic dependencies

    Returns:
        Dictionary of initialized agent instances
    """
    # Get the dependencies between agents
    dependencies = get_dependencies_groups(agents_dict, allow_cycles)

    # Create a dictionary to store all active agents/workflows
    active_agents: AgentDict = {}

    active_agents_in_dependency_group_partial = partial(
        active_agents_in_dependency_group,
        app_instance,
        agents_dict,
        model_factory_func,
    )

    # Create agent proxies for each group in dependency order
    for group in dependencies:
        await active_agents_in_dependency_group_partial(group, active_agents)

    return active_agents


async def _create_default_fan_in_agent(
    fan_in_name: str,
    context,
    model_factory_func: ModelFactoryFunctionProtocol,
) -> AgentProtocol:
    """
    Create a default fan-in agent for parallel workflows when none is specified.

    Args:
        fan_in_name: Name for the new fan-in agent
        context: Application context
        model_factory_func: Function for creating model factories

    Returns:
        Initialized Agent instance for fan-in operations
    """
    # Create a simple config for the fan-in agent with passthrough model
    default_config = AgentConfig(
        name=fan_in_name,
        model="passthrough",
        instruction="You are a passthrough agent that combines outputs from parallel agents.",
    )

    # Create and initialize the default agent
    fan_in_agent = LlmAgent(
        config=default_config,
        context=context,
    )
    await fan_in_agent.initialize()

    # Attach LLM to the agent
    llm_factory = model_factory_func(model="passthrough")
    await fan_in_agent.attach_llm(llm_factory)

    return fan_in_agent
