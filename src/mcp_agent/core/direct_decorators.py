"""
Type-safe decorators for DirectFastAgent applications.
These decorators provide type-safe function signatures and IDE support
for creating agents in the DirectFastAgent framework.
"""

import inspect
from functools import wraps
from pathlib import Path
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
    cast,
)

from mcp.client.session import ElicitationFnT
from pydantic import AnyUrl

from mcp_agent.agents.agent import AgentConfig
from mcp_agent.agents.workflow.iterative_planner import ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE
from mcp_agent.agents.workflow.router_agent import (
    ROUTING_SYSTEM_INSTRUCTION,
)
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


def _fetch_url_content(url: str) -> str:
    """
    Fetch content from a URL.

    Args:
        url: The URL to fetch content from

    Returns:
        The text content from the URL

    Raises:
        requests.RequestException: If the URL cannot be fetched
        UnicodeDecodeError: If the content cannot be decoded as UTF-8
    """
    import requests

    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise exception for HTTP errors
    return response.text


def _apply_templates(text: str) -> str:
    """
    Apply template substitutions to instruction text.

    Supported templates:
        {{currentDate}} - Current date in format "24 July 2025"
        {{url:https://...}} - Content fetched from the specified URL

    Args:
        text: The text to process

    Returns:
        Text with template substitutions applied

    Raises:
        requests.RequestException: If a URL in {{url:...}} cannot be fetched
        UnicodeDecodeError: If URL content cannot be decoded as UTF-8
    """
    import re
    from datetime import datetime

    # Apply {{currentDate}} template
    current_date = datetime.now().strftime("%d %B %Y")
    text = text.replace("{{currentDate}}", current_date)

    # Apply {{url:...}} templates
    url_pattern = re.compile(r"\{\{url:(https?://[^}]+)\}\}")

    def replace_url(match):
        url = match.group(1)
        return _fetch_url_content(url)

    text = url_pattern.sub(replace_url, text)

    return text


def _resolve_instruction(instruction: str | Path | AnyUrl) -> str:
    """
    Resolve instruction from either a string, Path, or URL with template support.

    Args:
        instruction: Either a string instruction, Path to a file, or URL containing the instruction

    Returns:
        The resolved instruction string with templates applied

    Raises:
        FileNotFoundError: If the Path doesn't exist
        PermissionError: If the Path can't be read
        UnicodeDecodeError: If the file/URL content can't be decoded as UTF-8
        requests.RequestException: If the URL cannot be fetched
    """
    if isinstance(instruction, Path):
        text = instruction.read_text(encoding="utf-8")
    elif isinstance(instruction, AnyUrl):
        text = _fetch_url_content(str(instruction))
    else:
        text = instruction

    # Apply template substitutions
    return _apply_templates(text)


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
    tools: Optional[Dict[str, List[str]]] = None,
    resources: Optional[Dict[str, List[str]]] = None,
    prompts: Optional[Dict[str, List[str]]] = None,
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
            tools=tools,
            resources=resources,
            prompts=prompts,
            model=model,
            use_history=use_history,
            human_input=human_input,
            default=default,
            elicitation_handler=extra_kwargs.get("elicitation_handler"),
            api_key=extra_kwargs.get("api_key"),
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
    instruction_or_kwarg: Optional[str | Path | AnyUrl] = None,
    *,
    instruction: str | Path | AnyUrl = "You are a helpful agent.",
    servers: List[str] = [],
    tools: Optional[Dict[str, List[str]]] = None,
    resources: Optional[Dict[str, List[str]]] = None,
    prompts: Optional[Dict[str, List[str]]] = None,
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: Optional[ElicitationFnT] = None,
    api_key: str | None = None,
) -> Callable[[AgentCallable[P, R]], DecoratedAgentProtocol[P, R]]:
    """
    Decorator to create and register a standard agent with type-safe signature.

    Args:
        name: Name of the agent
        instruction_or_kwarg: Optional positional parameter for instruction
        instruction: Base instruction for the agent (keyword arg)
        servers: List of server names the agent should connect to
        tools: Optional list of tool names or patterns to include
        resources: Optional list of resource names or patterns to include
        prompts: Optional list of prompt names or patterns to include
        model: Model specification string
        use_history: Whether to maintain conversation history
        request_params: Additional request parameters for the LLM
        human_input: Whether to enable human input capabilities
        default: Whether to mark this as the default agent
        elicitation_handler: Custom elicitation handler function (ElicitationFnT)
        api_key: Optional API key for the LLM provider

    Returns:
        A decorator that registers the agent with proper type annotations
    """
    final_instruction_raw = (
        instruction_or_kwarg if instruction_or_kwarg is not None else instruction
    )
    final_instruction = _resolve_instruction(final_instruction_raw)

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
        elicitation_handler=elicitation_handler,
        tools=tools,
        resources=resources,
        prompts=prompts,
        api_key=api_key,
    )


def custom(
    self,
    cls,
    name: str = "default",
    instruction_or_kwarg: Optional[str | Path | AnyUrl] = None,
    *,
    instruction: str | Path | AnyUrl = "You are a helpful agent.",
    servers: List[str] = [],
    tools: Optional[Dict[str, List[str]]] = None,
    resources: Optional[Dict[str, List[str]]] = None,
    prompts: Optional[Dict[str, List[str]]] = None,
    model: Optional[str] = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: Optional[ElicitationFnT] = None,
    api_key: str | None = None,
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
        elicitation_handler: Custom elicitation handler function (ElicitationFnT)

    Returns:
        A decorator that registers the agent with proper type annotations
    """
    final_instruction_raw = (
        instruction_or_kwarg if instruction_or_kwarg is not None else instruction
    )
    final_instruction = _resolve_instruction(final_instruction_raw)

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
        elicitation_handler=elicitation_handler,
        api_key=api_key,
        tools=tools,
        resources=resources,
        prompts=prompts,
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
    instruction: str | Path | AnyUrl = DEFAULT_INSTRUCTION_ORCHESTRATOR,
    model: Optional[str] = None,
    request_params: RequestParams | None = None,
    use_history: bool = False,
    human_input: bool = False,
    plan_type: Literal["full", "iterative"] = "full",
    plan_iterations: int = 5,
    default: bool = False,
    api_key: str | None = None,
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
    resolved_instruction = _resolve_instruction(instruction)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.ORCHESTRATOR,
            name=name,
            instruction=resolved_instruction,
            servers=[],  # Orchestrators don't connect to servers directly
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            child_agents=agents,
            plan_type=plan_type,
            plan_iterations=plan_iterations,
            default=default,
            api_key=api_key,
        ),
    )


def iterative_planner(
    self,
    name: str,
    *,
    agents: List[str],
    instruction: str | Path | AnyUrl = ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE,
    model: Optional[str] = None,
    request_params: RequestParams | None = None,
    plan_iterations: int = -1,
    default: bool = False,
    api_key: str | None = None,
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
        plan_iterations: Maximum number of planning iterations (0 for unlimited)
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the orchestrator with proper type annotations
    """

    # Create final request params with plan_iterations
    resolved_instruction = _resolve_instruction(instruction)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedOrchestratorProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.ITERATIVE_PLANNER,
            name=name,
            instruction=resolved_instruction,
            servers=[],  # Orchestrators don't connect to servers directly
            model=model,
            use_history=False,
            request_params=request_params,
            child_agents=agents,
            plan_iterations=plan_iterations,
            default=default,
            api_key=api_key,
        ),
    )


def router(
    self,
    name: str,
    *,
    agents: List[str],
    instruction: Optional[str | Path | AnyUrl] = None,
    servers: List[str] = [],
    tools: Optional[Dict[str, List[str]]] = None,
    resources: Optional[Dict[str, List[str]]] = None,
    prompts: Optional[Dict[str, List[str]]] = None,
    model: Optional[str] = None,
    use_history: bool = False,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: Optional[
        ElicitationFnT
    ] = None,  ## exclude from docs, decide whether allowable
    api_key: str | None = None,
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
        elicitation_handler: Custom elicitation handler function (ElicitationFnT)

    Returns:
        A decorator that registers the router with proper type annotations
    """
    resolved_instruction = _resolve_instruction(instruction or ROUTING_SYSTEM_INSTRUCTION)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedRouterProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.ROUTER,
            name=name,
            instruction=resolved_instruction,
            servers=servers,
            model=model,
            use_history=use_history,
            request_params=request_params,
            human_input=human_input,
            default=default,
            router_agents=agents,
            elicitation_handler=elicitation_handler,
            api_key=api_key,
            tools=tools,
            prompts=prompts,
            resources=resources,
        ),
    )


def chain(
    self,
    name: str,
    *,
    sequence: List[str],
    instruction: Optional[str | Path | AnyUrl] = None,
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
    resolved_instruction = _resolve_instruction(instruction or default_instruction)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedChainProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.CHAIN,
            name=name,
            instruction=resolved_instruction,
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
    instruction: Optional[str | Path | AnyUrl] = None,
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
    resolved_instruction = _resolve_instruction(instruction or default_instruction)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedParallelProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.PARALLEL,
            name=name,
            instruction=resolved_instruction,
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
    instruction: Optional[str | Path | AnyUrl] = None,
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
    resolved_instruction = _resolve_instruction(instruction or default_instruction)

    return cast(
        "Callable[[AgentCallable[P, R]], DecoratedEvaluatorOptimizerProtocol[P, R]]",
        _decorator_impl(
            self,
            AgentType.EVALUATOR_OPTIMIZER,
            name=name,
            instruction=resolved_instruction,
            servers=[],  # Evaluator-optimizer doesn't connect to servers directly
            generator=generator,
            evaluator=evaluator,
            min_rating=min_rating,
            max_refinements=max_refinements,
            default=default,
        ),
    )
