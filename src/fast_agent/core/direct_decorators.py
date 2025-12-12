"""
Type-safe decorators for DirectFastAgent applications.
These decorators provide type-safe function signatures and IDE support
for creating agents in the DirectFastAgent framework.
"""

from collections.abc import Coroutine
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
)

from mcp.client.session import ElicitationFnT
from pydantic import AnyUrl

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.workflow.iterative_planner import ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE
from fast_agent.agents.workflow.router_agent import (
    ROUTING_SYSTEM_INSTRUCTION,
)
from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
from fast_agent.skills import SkillManifest, SkillRegistry
from fast_agent.types import RequestParams

# Type variables for the decorated function
P = ParamSpec("P")  # Parameters
R = TypeVar("R", covariant=True)  # Return type


# Protocol for decorated agent functions
class DecoratedAgentProtocol(Protocol[P, R]):
    """Protocol defining the interface of a decorated agent function."""

    _agent_type: AgentType
    _agent_config: AgentConfig

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[R]: ...


# Protocol for orchestrator functions
class DecoratedOrchestratorProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated orchestrator functions with additional metadata."""

    _child_agents: list[str]
    _plan_type: Literal["full", "iterative"]


# Protocol for router functions
class DecoratedRouterProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated router functions with additional metadata."""

    _router_agents: list[str]


# Protocol for chain functions
class DecoratedChainProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated chain functions with additional metadata."""

    _chain_agents: list[str]


# Protocol for parallel functions
class DecoratedParallelProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated parallel functions with additional metadata."""

    _fan_out: list[str]
    _fan_in: str


# Protocol for evaluator-optimizer functions
class DecoratedEvaluatorOptimizerProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated evaluator-optimizer functions with additional metadata."""

    _generator: str
    _evaluator: str


# Protocol for maker functions
class DecoratedMakerProtocol(DecoratedAgentProtocol[P, R], Protocol):
    """Protocol for decorated MAKER functions with additional metadata."""

    _worker: str
    _k: int
    _max_samples: int


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

    Note: File templates ({{file:...}} and {{file_silent:...}}) are resolved later
    during runtime to ensure they're relative to the workspaceRoot.

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
    servers: list[str] = [],
    model: str | None = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    tools: dict[str, list[str]] | None = None,
    resources: dict[str, list[str]] | None = None,
    prompts: dict[str, list[str]] | None = None,
    skills: SkillManifest
    | SkillRegistry
    | Path
    | str
    | list[SkillManifest | SkillRegistry | Path | str | None]
    | None = None,
    **extra_kwargs,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    def decorator(func: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, Coroutine[Any, Any, R]]:
        # Create agent configuration
        config = AgentConfig(
            name=name,
            instruction=instruction,
            servers=servers,
            tools=tools,
            resources=resources,
            prompts=prompts,
            skills=skills,
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

        # Store metadata in the registry
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

        # Store type information on the function for IDE support
        setattr(func, "_agent_type", agent_type)
        setattr(func, "_agent_config", config)
        for key, value in extra_kwargs.items():
            setattr(func, f"_{key}", value)

        return func

    return decorator


def agent(
    self,
    name: str = "default",
    instruction_or_kwarg: str | Path | AnyUrl | None = None,
    *,
    instruction: str | Path | AnyUrl = DEFAULT_AGENT_INSTRUCTION,
    agents: list[str] | None = None,
    servers: list[str] = [],
    tools: dict[str, list[str]] | None = None,
    resources: dict[str, list[str]] | None = None,
    prompts: dict[str, list[str]] | None = None,
    skills: SkillManifest | SkillRegistry | Path | str | None = None,
    model: str | None = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: ElicitationFnT | None = None,
    api_key: str | None = None,
    history_mode: Any | None = None,
    max_parallel: int | None = None,
    child_timeout_sec: int | None = None,
    max_display_instances: int | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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
        child_agents=agents,
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
        skills=skills,
        api_key=api_key,
        agents_as_tools_options={
            "history_mode": history_mode,
            "max_parallel": max_parallel,
            "child_timeout_sec": child_timeout_sec,
            "max_display_instances": max_display_instances,
        },
    )


def custom(
    self,
    cls,
    name: str = "default",
    instruction_or_kwarg: str | Path | AnyUrl | None = None,
    *,
    instruction: str | Path | AnyUrl = "You are a helpful agent.",
    servers: list[str] = [],
    tools: dict[str, list[str]] | None = None,
    resources: dict[str, list[str]] | None = None,
    prompts: dict[str, list[str]] | None = None,
    skills: SkillManifest | SkillRegistry | Path | str | None = None,
    model: str | None = None,
    use_history: bool = True,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: ElicitationFnT | None = None,
    api_key: str | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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
        skills=skills,
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
    agents: list[str],
    instruction: str | Path | AnyUrl = DEFAULT_INSTRUCTION_ORCHESTRATOR,
    model: str | None = None,
    request_params: RequestParams | None = None,
    use_history: bool = False,
    human_input: bool = False,
    plan_type: Literal["full", "iterative"] = "full",
    plan_iterations: int = 5,
    default: bool = False,
    api_key: str | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    return _decorator_impl(
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
    )


def iterative_planner(
    self,
    name: str,
    *,
    agents: list[str],
    instruction: str | Path | AnyUrl = ITERATIVE_PLAN_SYSTEM_PROMPT_TEMPLATE,
    model: str | None = None,
    request_params: RequestParams | None = None,
    plan_iterations: int = -1,
    default: bool = False,
    api_key: str | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    return _decorator_impl(
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
    )


def router(
    self,
    name: str,
    *,
    agents: list[str],
    instruction: str | Path | AnyUrl | None = None,
    servers: list[str] = [],
    tools: dict[str, list[str]] | None = None,
    resources: dict[str, list[str]] | None = None,
    prompts: dict[str, list[str]] | None = None,
    model: str | None = None,
    use_history: bool = False,
    request_params: RequestParams | None = None,
    human_input: bool = False,
    default: bool = False,
    elicitation_handler: ElicitationFnT
    | None = None,  ## exclude from docs, decide whether allowable
    api_key: str | None = None,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    return _decorator_impl(
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
    )


def chain(
    self,
    name: str,
    *,
    sequence: list[str],
    instruction: str | Path | AnyUrl | None = None,
    cumulative: bool = False,
    default: bool = False,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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
        from fast_agent.core.exceptions import AgentConfigError

        raise AgentConfigError(f"Chain '{name}' requires at least one agent in the sequence")

    default_instruction = """Chain processes requests through a series of agents in sequence, the output of each agent is passed to the next."""
    resolved_instruction = _resolve_instruction(instruction or default_instruction)

    return _decorator_impl(
        self,
        AgentType.CHAIN,
        name=name,
        instruction=resolved_instruction,
        sequence=sequence,
        cumulative=cumulative,
        default=default,
    )


def parallel(
    self,
    name: str,
    *,
    fan_out: list[str],
    fan_in: str | None = None,
    instruction: str | Path | AnyUrl | None = None,
    include_request: bool = True,
    default: bool = False,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    return _decorator_impl(
        self,
        AgentType.PARALLEL,
        name=name,
        instruction=resolved_instruction,
        servers=[],  # Parallel agents don't connect to servers directly
        fan_in=fan_in,
        fan_out=fan_out,
        include_request=include_request,
        default=default,
    )


def evaluator_optimizer(
    self,
    name: str,
    *,
    generator: str,
    evaluator: str,
    instruction: str | Path | AnyUrl | None = None,
    min_rating: str = "GOOD",
    max_refinements: int = 3,
    refinement_instruction: str | None = None,
    default: bool = False,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
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

    return _decorator_impl(
        self,
        AgentType.EVALUATOR_OPTIMIZER,
        name=name,
        instruction=resolved_instruction,
        servers=[],  # Evaluator-optimizer doesn't connect to servers directly
        generator=generator,
        evaluator=evaluator,
        min_rating=min_rating,
        max_refinements=max_refinements,
        refinement_instruction=refinement_instruction,
        default=default,
    )


def maker(
    self,
    name: str,
    *,
    worker: str,
    k: int = 3,
    max_samples: int = 50,
    match_strategy: str = "exact",
    red_flag_max_length: int | None = None,
    instruction: str | Path | AnyUrl | None = None,
    default: bool = False,
) -> Callable[[Callable[P, Coroutine[Any, Any, R]]], Callable[P, Coroutine[Any, Any, R]]]:
    """
    Decorator to create a MAKER agent for statistical error correction via voting.

    MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.

    Based on the paper "Solving a Million-Step LLM Task with Zero Errors"
    (arXiv:2511.09030). Implements first-to-ahead-by-k voting where multiple
    samples are drawn from a worker agent, and the first response to achieve
    a k-vote margin over alternatives wins.

    This enables high reliability with cost-effective models by trading
    compute (multiple samples) for accuracy (statistical consensus).

    Args:
        name: Name of the MAKER agent
        worker: Name of the agent to sample from for voting
        k: Margin required to declare winner (first-to-ahead-by-k).
           Higher k = more reliable but more samples needed.
           Default of 3 provides strong guarantees for most use cases.
        max_samples: Maximum samples before falling back to plurality vote
        match_strategy: How to compare responses for voting:
            - "exact": Character-for-character match
            - "normalized": Ignore whitespace and case differences
            - "structured": Parse as JSON and compare structurally
        red_flag_max_length: Discard responses longer than this (characters).
                             Per the paper, overly long responses correlate
                             with errors. None = no length limit.
        instruction: Base instruction for the MAKER agent
        default: Whether to mark this as the default agent

    Returns:
        A decorator that registers the MAKER agent

    Example:
        @fast.agent(name="calculator", instruction="Return only the numeric result")
        @fast.maker(name="reliable_calc", worker="calculator", k=3)
        async def main():
            async with fast.run() as agent:
                result = await agent.reliable_calc.send("What is 17 * 23?")
    """
    default_instruction = """
    MAKER: Massively decomposed Agentic processes with K-voting Error Reduction.
    Implements statistical error correction through voting consensus.
    Multiple samples are drawn and the first response to achieve a k-vote
    margin wins, ensuring high reliability even with cost-effective models.
    """
    resolved_instruction = _resolve_instruction(instruction or default_instruction)

    return _decorator_impl(
        self,
        AgentType.MAKER,
        name=name,
        instruction=resolved_instruction,
        servers=[],  # MAKER doesn't connect to servers directly
        worker=worker,
        k=k,
        max_samples=max_samples,
        match_strategy=match_strategy,
        red_flag_max_length=red_flag_max_length,
        default=default,
    )
