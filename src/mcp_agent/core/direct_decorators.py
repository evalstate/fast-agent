"""
Type-safe decorators for DirectFastAgent applications.
These decorators provide type-safe function signatures and IDE support
for creating agents in the DirectFastAgent framework.
"""

import inspect
from functools import wraps
from typing import (
    Awaitable,
    Callable,
    List,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
    cast,
)

from mcp_agent.agents.agent import AgentConfig
from mcp_agent.core.agent_types import AgentType
from mcp_agent.core.request_params import RequestParams

# Type variables for the decorated function
P = ParamSpec("P")  # Parameters
R = TypeVar("R")  # Return type

# Type for agent functions - can be either async or sync
AgentCallable = Callable[P, Union[Awaitable[R], R]]


# Protocol for decorated agent functions
class DecoratedAgentProtocol(Protocol[P, R]):
    """Protocol defining the interface of a decorated agent function."""

    _agent_type: AgentType
    _agent_config: AgentConfig

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Union[Awaitable[R], R]: ...


# Protocol for orchestrator functions
class DecoratedOrchestratorProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated orchestrator functions with additional metadata."""

    _child_agents: List[str]
    _plan_type: Literal["full", "iterative"]


# Protocol for router functions
class DecoratedRouterProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated router functions with additional metadata."""

    _router_agents: List[str]


# Protocol for chain functions
class DecoratedChainProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated chain functions with additional metadata."""

    _chain_agents: List[str]


# Protocol for parallel functions
class DecoratedParallelProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated parallel functions with additional metadata."""

    _fan_out: List[str]
    _fan_in: str


# Protocol for evaluator-optimizer functions
class DecoratedEvaluatorOptimizerProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated evaluator-optimizer functions with additional metadata."""

    _generator: str
    _evaluator: str


def _decorator_impl(
    self,
    agent_type: AgentType,
    name: str,
    instruction: str,
    *,
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    **extra_kwargs,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Core implementation for agent decorators with common behavior and type safety.

    Args:
        agent_type: Type of agent to create
        name: Name of the agent
        instruction: Base instruction for the agent
        servers: List of server names the agent should connect to
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        default: Whether to mark this as the default agent
        **extra_kwargs: Additional agent/workflow-specific parameters
    """

    def decorator(func: AgentCallable[P, R]) -> DecoratedAgentProtocol[P, R]:
        is_async = inspect.iscoroutinefunction(func)

        # Handle both async and sync functions consistently
        if is_async:

            @wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Call the original function
                return await func(*args, **kwargs)
        else:

            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Call the original function
                return func(*args, **kwargs)

        # Create agent configuration
        config = AgentConfig(
            name=name,
            instruction=instruction,
            servers=servers,
            model=model,
            use_history=use_history,
            human_input=human_input,
            default=default,
        )

        # Update request params if provided
        if request_params:
            config.default_request_params = request_params

        # Store metadata on the wrapper function
        agent_data = {
            "config": config,
            "type": agent_type.value,
            "func": func,
        }

        # Add extra parameters specific to this agent type
        for key, value in extra_kwargs.items():
            agent_data[key] = value

        # Store the configuration in the FastAgent instance
        self.agents[name] = agent_data

        # Store type information for IDE support
        setattr(wrapper, "_agent_type", agent_type)
        setattr(wrapper, "_agent_config", config)
        for key, value in extra_kwargs.items():
            setattr(wrapper, f"_{key}", value)

        return cast("DecoratedAgentProtocol[P, R]", wrapper)

    return decorator


def agent(
    self,
    name: str = "default",
    instruction_or_kwarg: Optional[str] = None,
    *,
    instruction: str = "You are a helpful agent.",
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Decorator to create and register a standard agent with type-safe signature.

    Args:
        name: Name of the agent
        instruction_or_kwarg: Optional positional parameter for instruction
        instruction: Base instruction for the agent (keyword arg)
        servers: List of server names the agent should connect to
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the agent with proper type annotations
    """
    final_instruction = instruction_or_kwarg if instruction_or_kwarg is not None else instruction

    return _decorator_impl(
        self,
        AgentType.BASIC,
        name=name,
        instruction=final_instruction,
        servers=servers,
        model=model,
        use_history=use_history,
        request_params=request_params,
        human_input=human_input,
        default=default,
    )


def custom(
    self,
    cls,
    name: str = "default",
    instruction_or_kwarg: Optional[str] = None,
    *,
    instruction: str = "You are a helpful agent.",
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Decorator to create and register a standard agent with type-safe signature.

    Args:
        name: Name of the agent
        instruction_or_kwarg: Optional positional parameter for instruction
        instruction: Base instruction for the agent (keyword arg)
        servers: List of server names the agent should connect to
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities

    Returns:
        A decorator that registers the agent with proper type annotations
    """
    final_instruction = instruction_or_kwarg if instruction_or_kwarg is not None else instruction

    return _decorator_impl(
        self,
        AgentType.CUSTOM,
        name=name,
        instruction=final_instruction,
        servers=servers,
        model=model,
        use_history=use_history,
        request_params=request_params,
        human_input=human_input,
        agent_class=cls,
        default=default,
    )


DEFAULT_INSTRUCTION_ORCHESTRATOR = """
You are an expert planner. Given an objective task and a list of Agents
(which are collections of capabilities), your job is to break down the objective
into a series of steps, which can be performed by these agents.
"""


def orchestrator(
    self,
    name: str,
    *,
    agents: List[str],
    instruction: str = DEFAULT_INSTRUCTION_ORCHESTRATOR,
    model: Optional[str] = None,
    request_params: RequestParams | None = None,
    use_history: bool = False,
    human_input: bool = False,
    plan_type: Literal["full", "iterative"] = "full",
    plan_iterations: int = 5,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]]:
    """
    Decorator to create and register an orchestrator agent with type-safe signature.

    Args:
        name: Name of the orchestrator
        agents: List of agent names this orchestrator can use
        instruction: Base instruction for the orchestrator
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        plan_type: Planning approach - "full" or "iterative"
        plan_iterations: Maximum number of planning iterations
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the orchestrator with proper type annotations
    """

    # Create final request params with plan_iterations

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.ORCHESTRATOR,
            name=name,
            instruction=instruction,
            servers=[],  # Orchestrators don't connect to servers directly
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            child_agents=agents,
            plan_type=plan_type,
            plan_iterations=plan_iterations,
            default=default,
        ),
    )


def router(
    self,
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str] = None,
    servers: List[str] = [],
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedRouterProtocol[P, R]]:
    """
    Decorator to create and register a router agent with type-safe signature.

    Args:
        name: Name of the router
        agents: List of agent names this router can route to
        instruction: Base instruction for the router
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the router with proper type annotations
    """
    default_instruction = """
    You are a router that determines which specialized agent should handle a given query.
    Analyze the query and select the most appropriate agent to handle it.
    """

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedRouterProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.ROUTER,
            name=name,
            instruction=instruction or default_instruction,
            servers=servers,
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            default=default,
            router_agents=agents,
        ),
    )


def chain(
    self,
    name: str,
    *,
    sequence: List[str],
    instruction: Optional[str] = None,
    cumulative: bool = False,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedChainProtocol[P, R]]:
    """
    Decorator to create and register a chain agent with type-safe signature.

    Args:
        name: Name of the chain
        sequence: List of agent names in the chain, executed in sequence
        instruction: Base instruction for the chain
        cumulative: Whether to use cumulative mode (each agent sees all previous responses)
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the chain with proper type annotations
    """
    # Validate sequence is not empty
    if not sequence:
        from mcp_agent.core.exceptions import AgentConfigError

        raise AgentConfigError(f"Chain '{name}' requires at least one agent in the sequence")

    default_instruction = """
    You are a chain that processes requests through a series of specialized agents in sequence.
    Pass the output of each agent to the next agent in the chain.
    """

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedChainProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.CHAIN,
            name=name,
            instruction=instruction or default_instruction,
            sequence=sequence,
            cumulative=cumulative,
            default=default,
        ),
    )


def parallel(
    self,
    name: str,
    *,
    fan_out: List[str],
    fan_in: str | None = None,
    instruction: Optional[str] = None,
    include_request: bool = True,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedParallelProtocol[P, R]]:
    """
    Decorator to create and register a parallel agent with type-safe signature.

    Args:
        name: Name of the parallel agent
        fan_out: List of agents to execute in parallel
        fan_in: Agent to aggregate results
        instruction: Base instruction for the parallel agent
        include_request: Whether to include the original request when aggregating
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the parallel agent with proper type annotations
    """
    default_instruction = """
    You are a parallel processor that executes multiple agents simultaneously
    and aggregates their results.
    """

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedParallelProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.PARALLEL,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Parallel agents don't connect to servers directly
            fan_in=fan_in,
            fan_out=fan_out,
            include_request=include_request,
            default=default,
        ),
    )


def evaluator_optimizer(
    self,
    name: str,
    *,
    generator: str,
    evaluator: str,
    instruction: Optional[str] = None,
    min_rating: str = "GOOD",
    max_refinements: int = 3,
    default: bool = False,
) -> Callable[[AgentCallable[P, R]], DecoratedEvaluatorOptimizerProtocol[P, R]]:
    """
    Decorator to create and register an evaluator-optimizer agent with type-safe signature.

    Args:
        name: Name of the evaluator-optimizer agent
        generator: Name of the agent that generates responses
        evaluator: Name of the agent that evaluates responses
        instruction: Base instruction for the evaluator-optimizer
        min_rating: Minimum acceptable quality rating (EXCELLENT, GOOD, FAIR, POOR)
        max_refinements: Maximum number of refinement iterations
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the evaluator-optimizer with proper type annotations
    """
    default_instruction = """
    You implement an iterative refinement process where content is generated,
    evaluated for quality, and then refined based on specific feedback until
    it reaches an acceptable quality standard.
    """

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedEvaluatorOptimizerProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.EVALUATOR_OPTIMIZER,
            name=name,
            instruction=instruction or default_instruction,
            servers=[],  # Evaluator-optimizer doesn't connect to servers directly
            generator=generator,
            evaluator=evaluator,
            min_rating=min_rating,
            max_refinements=max_refinements,
            default=default,
        ),
    )
