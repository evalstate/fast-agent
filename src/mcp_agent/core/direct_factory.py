"""
Direct factory functions for creating agent and workflow instances without proxies.
Implements type-safe factories with improved error handling.
"""

from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

from mcp_agent.agents.agent import Agent, AgentConfig
from mcp_agent.agents.workflow.evaluator_optimizer import (
    EvaluatorOptimizerAgent,
    QualityRating,
)
from mcp_agent.agents.workflow.orchestrator_agent import OrchestratorAgent
from mcp_agent.agents.workflow.parallel_agent import ParallelAgent
from mcp_agent.agents.workflow.router_agent import RouterAgent
from mcp_agent.app import MCPApp
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.exceptions import AgentConfigError
from mcp_agent.core.validation import get_dependencies_groups
from mcp_agent.event_progress import ProgressAction
from mcp_agent.llm.augmented_llm import RequestParams
from mcp_agent.llm.model_factory import ModelFactory
from mcp_agent.logging.logger import get_logger

# Type aliases for improved readability and IDE support
AgentDict = Dict[str, Agent]
AgentConfigDict = Dict[str, Dict[str, Any]]
T = TypeVar("T")  # For generic types

# Type for model factory functions
ModelFactoryFn = Callable[[Optional[str], Optional[RequestParams]], Callable[[], Any]]


logger = get_logger(__name__)


class AgentCreatorProtocol(Protocol):
    """Protocol for agent creator functions."""

    async def __call__(
        self,
        app_instance: MCPApp,
        agents_dict: AgentConfigDict,
        agent_type: AgentType,
        active_agents: Optional[AgentDict] = None,
        model_factory_func: Optional[ModelFactoryFn] = None,
        **kwargs: Any,
    ) -> AgentDict: ...


def get_model_factory(
    context,
    model: Optional[str] = None,
    request_params: Optional[RequestParams] = None,
    default_model: Optional[str] = None,
    cli_model: Optional[str] = None,
) -> Callable:
    """
    Get model factory using specified or default model.
    Model string is parsed by ModelFactory to determine provider and reasoning effort.

    Args:
        context: Application context
        model: Optional model specification string (highest precedence)
        request_params: Optional RequestParams to configure LLM behavior
        default_model: Default model from configuration
        cli_model: Model specified via command line

    Returns:
        ModelFactory instance for the specified or default model
    """
    # Config has lowest precedence
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
    return ModelFactory.create_factory(model_spec, request_params=request_params)


async def create_agents_by_type(
    app_instance: MCPApp,
    agents_dict: AgentConfigDict,
    agent_type: AgentType,
    active_agents: Optional[AgentDict] = None,
    model_factory_func: Optional[ModelFactoryFn] = None,
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

    if model_factory_func is None:
        # Default factory that just returns the inputs - should be overridden
        def model_factory_func(model=None, request_params=None):
            return lambda: None

    # Create a dictionary to store the initialized agents
    result_agents: AgentDict = {}

    # Get all agents of the specified type
    for name, agent_data in agents_dict.items():
        logger.info(
            f"Loaded {name}",
            data={
                "progress_action": ProgressAction.LOADED,
                "agent_name": name,
            },
        )

        # Compare type string from config with Enum value
        if agent_data["type"] == agent_type.value:
            # Get common configuration
            config = agent_data["config"]

            # Type-specific initialization based on the Enum type
            # Note: Above we compared string values from config, here we compare Enum objects directly
            if agent_type == AgentType.BASIC:
                # Create a basic agent
                agent = Agent(
                    config=config,
                    context=app_instance.context,
                )
                await agent.initialize()

                # Attach LLM to the agent
                llm_factory = model_factory_func(model=config.model)
                await agent.attach_llm(llm_factory, request_params=config.default_request_params)
                result_agents[name] = agent

            elif agent_type == AgentType.CUSTOM:
                # Get the class to instantiate
                cls = agent_data["agent_class"]
                # Create the custom agent
                agent = cls(
                    config=config,
                    context=app_instance.context,
                )
                await agent.initialize()

                # Attach LLM to the agent
                llm_factory = model_factory_func(model=config.model)
                await agent.attach_llm(llm_factory, request_params=config.default_request_params)
                result_agents[name] = agent

            elif agent_type == AgentType.ORCHESTRATOR:
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

                # Create the orchestrator
                orchestrator = OrchestratorAgent(
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
                    llm_factory, request_params=config.default_request_params
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
                await router.attach_llm(llm_factory, request_params=config.default_request_params)
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

                from mcp_agent.agents.workflow.chain_agent import ChainAgent

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
                )

                # Initialize the agent
                await evaluator_optimizer.initialize()
                result_agents[name] = evaluator_optimizer

            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

    return result_agents


async def create_agents_in_dependency_order(
    app_instance: MCPApp,
    agents_dict: AgentConfigDict,
    model_factory_func: ModelFactoryFn,
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

    # Create agent proxies for each group in dependency order
    for group in dependencies:
        # Create basic agents first
        # Note: We compare string values from config with the Enum's string value
        if AgentType.BASIC.value in [agents_dict[name]["type"] for name in group]:
            basic_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.BASIC.value
                },
                AgentType.BASIC,
                active_agents,
                model_factory_func,
            )
            active_agents.update(basic_agents)

        # Create custom agents first
        if AgentType.CUSTOM.value in [agents_dict[name]["type"] for name in group]:
            basic_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.CUSTOM.value
                },
                AgentType.CUSTOM,
                active_agents,
                model_factory_func,
            )
            active_agents.update(basic_agents)

        # Create parallel agents
        if AgentType.PARALLEL.value in [agents_dict[name]["type"] for name in group]:
            parallel_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.PARALLEL.value
                },
                AgentType.PARALLEL,
                active_agents,
                model_factory_func,
            )
            active_agents.update(parallel_agents)

        # Create router agents
        if AgentType.ROUTER.value in [agents_dict[name]["type"] for name in group]:
            router_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.ROUTER.value
                },
                AgentType.ROUTER,
                active_agents,
                model_factory_func,
            )
            active_agents.update(router_agents)

        # Create chain agents
        if AgentType.CHAIN.value in [agents_dict[name]["type"] for name in group]:
            chain_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.CHAIN.value
                },
                AgentType.CHAIN,
                active_agents,
                model_factory_func,
            )
            active_agents.update(chain_agents)

        # Create evaluator-optimizer agents
        if AgentType.EVALUATOR_OPTIMIZER.value in [agents_dict[name]["type"] for name in group]:
            evaluator_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.EVALUATOR_OPTIMIZER.value
                },
                AgentType.EVALUATOR_OPTIMIZER,
                active_agents,
                model_factory_func,
            )
            active_agents.update(evaluator_agents)

        # Create orchestrator agents last since they might depend on other agents
        if AgentType.ORCHESTRATOR.value in [agents_dict[name]["type"] for name in group]:
            orchestrator_agents = await create_agents_by_type(
                app_instance,
                {
                    name: agents_dict[name]
                    for name in group
                    if agents_dict[name]["type"] == AgentType.ORCHESTRATOR.value
                },
                AgentType.ORCHESTRATOR,
                active_agents,
                model_factory_func,
            )
            active_agents.update(orchestrator_agents)

    return active_agents


async def _create_default_fan_in_agent(
    fan_in_name: str,
    context,
    model_factory_func: ModelFactoryFn,
) -> Agent:
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
    fan_in_agent = Agent(
        config=default_config,
        context=context,
    )
    await fan_in_agent.initialize()

    # Attach LLM to the agent
    llm_factory = model_factory_func(model="passthrough")
    await fan_in_agent.attach_llm(llm_factory)

    return fan_in_agent
