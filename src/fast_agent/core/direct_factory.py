"""
Direct factory functions for creating agent and workflow instances without proxies.
Implements type-safe factories with improved error handling.
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    cast,
)

from fastmcp.tools import FunctionTool

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType, FunctionToolConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.agents.workflow.evaluator_optimizer import (
    EvaluatorOptimizerAgent,
    QualityRating,
)
from fast_agent.agents.workflow.iterative_planner import IterativePlanner
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.context import Context
from fast_agent.core import Core
from fast_agent.core.agent_card_types import AgentCardData
from fast_agent.core.exceptions import AgentConfigError, ModelConfigError
from fast_agent.core.function_tool_support import custom_class_supports_function_tools
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.model_resolution import (
    HARDCODED_DEFAULT_MODEL,
    get_context_cli_model_override,
    resolve_model_spec,
)
from fast_agent.core.validation import (
    get_dependencies_groups,
    is_basic_like_agent_type,
    normalize_agent_type_value,
)
from fast_agent.event_progress import ProgressAction
from fast_agent.hooks.hook_messages import show_hook_failure
from fast_agent.interfaces import (
    AgentProtocol,
    LLMFactoryProtocol,
    ModelFactoryFunctionProtocol,
)
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.ui_agent import McpAgentWithUI
from fast_agent.tools.function_tool_loader import load_function_tools
from fast_agent.tools.hook_loader import load_tool_runner_hooks
from fast_agent.types import RequestParams

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookAgentProtocol

# Type aliases for improved readability and IDE support
AgentDict = dict[str, AgentProtocol]
AgentConfigDict = Mapping[str, AgentCardData | dict[str, Any]]
AgentTypeBuilder = Callable[[str, Mapping[str, Any], "AgentBuildContext", AgentDict], Awaitable[None]]


@dataclass(frozen=True)
class AgentBuildContext:
    app_instance: "CoreContextProtocol"
    agents_dict: AgentConfigDict
    active_agents: AgentDict
    model_factory_func: ModelFactoryFunctionProtocol
    session_history_enabled: bool


@dataclass(frozen=True)
class AgentsAsToolsBuildInputs:
    config: AgentConfig
    function_tools: list[FunctionTool]
    child_agents: list[AgentProtocol]
    options: Any
    child_message_files: dict[str, list[Path]]


class CoreContextProtocol(Protocol):
    context: Context


class _ContextCoreShim:
    def __init__(self, context: Context) -> None:
        self.context = context


def _ensure_basic_only_agents(agents_dict: AgentConfigDict) -> None:
    for name, agent_data in agents_dict.items():
        agent_type = agent_data.get("type") if isinstance(agent_data, Mapping) else None
        if not is_basic_like_agent_type(agent_type):
            raise AgentConfigError(
                "Smart tool only supports 'agent' cards",
                f"Card '{name}' has unsupported type '{agent_type}'",
            )

def _load_configured_function_tools(
    config: AgentConfig,
    agent_data: Mapping[str, Any],
) -> list[FunctionTool]:
    """Load function tools from config or agent_data, resolving paths.

    Args:
        config: The agent configuration
        agent_data: The agent data dictionary from card or registry

    Returns:
        List of loaded function tools (may be empty)
    """
    tools_config_raw = config.function_tools
    if tools_config_raw is None:
        tools_config_raw = agent_data.get("function_tools")

    tools_config: list[FunctionToolConfig] | None = None
    if isinstance(tools_config_raw, str):
        tools_config = [tools_config_raw]
    elif isinstance(tools_config_raw, list):
        tools_config = cast("list[FunctionToolConfig]", tools_config_raw)

    if not tools_config:
        return []

    source_path = agent_data.get("source_path")
    base_path = Path(source_path).parent if source_path else None
    return load_function_tools(tools_config, base_path)


def _resolve_function_tools_with_globals(
    config: AgentConfig,
    agent_data: Mapping[str, Any],
    build_ctx: "AgentBuildContext",
) -> list[FunctionTool]:
    """Load per-agent function tools, falling back to global @fast.tool tools.

    If the agent has explicit function_tools configured (including an empty list),
    only those are used. Otherwise, globally registered tools from ``@fast.tool``
    are provided.

    Naming note:
    the returned value is a list of resolved executable function tools. In the
    custom-agent path, these are later passed to the constructor as ``tools=``,
    which is distinct from ``AgentConfig.tools`` MCP filter settings.
    """
    if config.function_tools is not None or agent_data.get("function_tools") is not None:
        return _load_configured_function_tools(config, agent_data)

    global_tools = getattr(build_ctx.app_instance, "_registered_tools", None)
    if global_tools:
        return list(global_tools)

    return []


def _register_loaded_agent(
    result_agents: AgentDict,
    name: str,
    agent: AgentProtocol,
) -> None:
    result_agents[name] = agent
    logger.info(
        f"Loaded {name}",
        data={
            "progress_action": ProgressAction.LOADED,
            "agent_name": name,
            "target": name,
        },
    )


def _iter_agents_of_type(
    agents_dict: AgentConfigDict,
    agent_type: AgentType,
) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (name, agent_data)
        for name, agent_data in agents_dict.items()
        if normalize_agent_type_value(agent_data.get("type")) == agent_type.value
    ]


def _resolve_child_agents(
    parent_name: str,
    child_names: Sequence[str],
    active_agents: AgentDict,
    *,
    skip_missing: bool,
) -> list[AgentProtocol]:
    child_agents: list[AgentProtocol] = []
    for agent_name in child_names:
        child_agent = active_agents.get(agent_name)
        if child_agent is None:
            if skip_missing:
                logger.warning(
                    "Skipping missing child agent",
                    data={"agent_name": agent_name, "parent": parent_name},
                )
                continue
            raise AgentConfigError(f"Agent {agent_name} not found")
        child_agents.append(child_agent)
    return child_agents


def _build_agents_as_tools_options(agent_data: Mapping[str, Any]) -> object:
    from fast_agent.agents.workflow.agents_as_tools_agent import AgentsAsToolsOptions

    raw_opts = agent_data.get("agents_as_tools_options") or {}
    opt_kwargs = {key: value for key, value in raw_opts.items() if value is not None}
    return AgentsAsToolsOptions(**opt_kwargs)


def _collect_child_message_files(
    child_names: Sequence[str],
    agents_dict: AgentConfigDict,
    options: object,
) -> dict[str, list[Path]]:
    from fast_agent.agents.workflow.agents_as_tools_agent import (
        HistoryMergeTarget,
        HistorySource,
    )

    child_message_files: dict[str, list[Path]] = {}
    history_source = getattr(options, "history_source", None)
    history_merge_target = getattr(options, "history_merge_target", None)
    requires_messages = (
        history_source == HistorySource.MESSAGES
        or history_merge_target == HistoryMergeTarget.MESSAGES
    )
    missing_messages: list[str] = []

    for agent_name in child_names:
        child_data = agents_dict.get(agent_name, {})
        message_files = child_data.get("message_files")
        if not message_files:
            if requires_messages:
                missing_messages.append(agent_name)
            continue
        child_message_files[agent_name] = list(message_files)

    if missing_messages:
        missing_list = ", ".join(sorted(set(missing_messages)))
        raise AgentConfigError(
            "history_source/history_merge_target=messages requires child agents with messages",
            f"Missing messages for: {missing_list}",
        )

    return child_message_files


def _build_agents_as_tools_inputs(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
) -> AgentsAsToolsBuildInputs:
    config = cast("AgentConfig", agent_data["config"])
    child_names = cast("Sequence[str]", agent_data.get("child_agents", []) or [])
    options = _build_agents_as_tools_options(agent_data)
    return AgentsAsToolsBuildInputs(
        config=config,
        function_tools=_resolve_function_tools_with_globals(config, agent_data, build_ctx),
        child_agents=_resolve_child_agents(
            name,
            child_names,
            build_ctx.active_agents,
            skip_missing=True,
        ),
        options=options,
        child_message_files=_collect_child_message_files(
            child_names,
            build_ctx.agents_dict,
            options,
        ),
    )


async def _initialize_agent_with_llm(
    agent: Any,
    config: AgentConfig,
    model_factory_func: ModelFactoryFunctionProtocol,
) -> None:
    await agent.initialize()
    llm_factory = model_factory_func(model=config.model)
    await agent.attach_llm(
        llm_factory,
        request_params=config.default_request_params,
        api_key=config.api_key,
    )


def _attach_child_agents_as_tools(
    agent: Any,
    parent_name: str,
    child_names: Sequence[str],
    active_agents: AgentDict,
) -> None:
    add_tool_fn = getattr(agent, "add_agent_tool", None)
    if not callable(add_tool_fn):
        return

    for child_agent in _resolve_child_agents(
        parent_name,
        child_names,
        active_agents,
        skip_missing=True,
    ):
        add_tool_fn(child_agent)




async def _finalize_agent(
    agent: LlmAgent,
    name: str,
    config: AgentConfig,
    agent_data: Mapping[str, Any],
    model_factory_func: ModelFactoryFunctionProtocol,
    result_agents: AgentDict,
    session_history_enabled: bool,
) -> None:
    """Complete agent setup: initialize, attach LLM, apply hooks, register.

    Args:
        agent: The agent instance to finalize
        name: Agent name for registration
        config: Agent configuration
        agent_data: Agent data dictionary (for source_path)
        model_factory_func: Factory function for LLM creation
        result_agents: Dictionary to register the agent in
        session_history_enabled: Whether session history tracking is enabled
    """
    await agent.initialize()

    llm_factory = model_factory_func(model=config.model)
    await agent.attach_llm(
        llm_factory,
        request_params=config.default_request_params,
        api_key=config.api_key,
    )

    if config.tool_hooks or config.trim_tool_history or session_history_enabled:
        _apply_tool_hooks(
            agent,
            config,
            agent_data.get("source_path"),
            enable_session_history=session_history_enabled,
        )

    _register_loaded_agent(result_agents, name, agent)


T = TypeVar("T")  # For generic types


logger = get_logger(__name__)


def _create_agent_with_ui_if_needed(
    agent_class: type,
    config: Any,
    context: Any,
    **kwargs: Any,
) -> Any:
    """
    Create an agent with UI support if MCP UI mode is enabled.

    Args:
        agent_class: The agent class to potentially enhance with UI
        config: Agent configuration
        context: Application context
        **kwargs: Additional arguments passed to agent constructor (e.g., tools)

    Returns:
        Either a UI-enhanced agent instance or the original agent instance
    """
    # Check UI mode from settings
    settings = context.config if hasattr(context, "config") else None
    ui_mode = getattr(settings, "mcp_ui_mode", "auto") if settings else "auto"

    if ui_mode != "disabled" and agent_class == McpAgent:
        # Use the UI-enhanced agent class instead of the base class
        return McpAgentWithUI(config=config, context=context, ui_mode=ui_mode, **kwargs)
    else:
        # Create the original agent instance
        return agent_class(config=config, context=context, **kwargs)


def _apply_tool_hooks(
    agent: LlmAgent,
    config: AgentConfig,
    source_path: str | None = None,
    *,
    enable_session_history: bool = False,
) -> None:
    """
    Apply tool runner hooks to an agent based on config.

    Handles both:
    - tool_hooks: dict mapping hook types to function specs
    - trim_tool_history: shortcut to apply built-in history trimmer

    Args:
        agent: The agent to apply hooks to
        config: Agent configuration with tool_hooks and/or trim_tool_history
        source_path: Path to the source file for resolving relative hook paths
    """
    # Import here to avoid circular imports
    from fast_agent.agents.tool_runner import ToolRunnerHooks
    from fast_agent.hooks import save_session_history, trim_tool_loop_history
    from fast_agent.hooks.hook_context import HookContext

    hooks_config = config.tool_hooks
    trim_history = config.trim_tool_history

    # If trim_tool_history is set and no after_turn_complete hook is configured,
    # add the built-in trimmer
    if trim_history:
        if hooks_config is None:
            hooks_config = {}
        if "after_turn_complete" not in hooks_config:
            # Use a wrapper that creates HookContext for the built-in trimmer
            async def _trimmer_wrapper(runner, message):
                ctx = HookContext(
                    runner=runner,
                    agent=cast("HookAgentProtocol", agent),
                    message=message,
                    hook_type="after_turn_complete",
                )
                try:
                    await trim_tool_loop_history(ctx)
                except Exception as exc:  # noqa: BLE001
                    show_hook_failure(
                        ctx,
                        hook_name="trim_tool_loop_history",
                        hook_kind="tool",
                        error=exc,
                    )
                    logger.exception(
                        "Tool hook failed",
                        hook_type="after_turn_complete",
                        hook_name="trim_tool_loop_history",
                    )
                    raise

            # Set the hooks directly since we have a callable, not a spec string
            existing_hooks = getattr(agent, "tool_runner_hooks", None) or ToolRunnerHooks()
            agent.tool_runner_hooks = ToolRunnerHooks(
                before_llm_call=existing_hooks.before_llm_call,
                after_llm_call=existing_hooks.after_llm_call,
                before_tool_call=existing_hooks.before_tool_call,
                after_tool_call=existing_hooks.after_tool_call,
                after_turn_complete=_trimmer_wrapper,
            )
            # Keep going so session history hooks can still be applied

    # Load custom hooks from config
    if hooks_config:
        base_path = Path(source_path).parent if source_path else None
        loaded_hooks = load_tool_runner_hooks(hooks_config, agent, base_path)
        if loaded_hooks:
            # Merge with any existing hooks (trim_tool_history wrapper)
            existing = getattr(agent, "tool_runner_hooks", None)
            if existing:
                agent.tool_runner_hooks = ToolRunnerHooks(
                    before_llm_call=loaded_hooks.before_llm_call or existing.before_llm_call,
                    after_llm_call=loaded_hooks.after_llm_call or existing.after_llm_call,
                    before_tool_call=loaded_hooks.before_tool_call or existing.before_tool_call,
                    after_tool_call=loaded_hooks.after_tool_call or existing.after_tool_call,
                    after_turn_complete=loaded_hooks.after_turn_complete or existing.after_turn_complete,
                )
            else:
                agent.tool_runner_hooks = loaded_hooks

    if enable_session_history:
        existing_hooks = getattr(agent, "tool_runner_hooks", None) or ToolRunnerHooks()
        existing_after_turn = existing_hooks.after_turn_complete

        async def _session_history_wrapper(runner, message):
            if existing_after_turn is not None:
                await existing_after_turn(runner, message)
            ctx = HookContext(
                runner=runner,
                agent=cast("HookAgentProtocol", agent),
                message=message,
                hook_type="after_turn_complete",
            )
            try:
                await save_session_history(ctx)
            except Exception as exc:  # noqa: BLE001
                show_hook_failure(
                    ctx,
                    hook_name="save_session_history",
                    hook_kind="tool",
                    error=exc,
                )
                logger.exception(
                    "Tool hook failed",
                    hook_type="after_turn_complete",
                    hook_name="save_session_history",
                )
                raise

        agent.tool_runner_hooks = ToolRunnerHooks(
            before_llm_call=existing_hooks.before_llm_call,
            after_llm_call=existing_hooks.after_llm_call,
            before_tool_call=existing_hooks.before_tool_call,
            after_tool_call=existing_hooks.after_tool_call,
            after_turn_complete=_session_history_wrapper,
        )


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
        1. Hardcoded default (gpt-5-mini?reasoning=low)
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
    cli_model = cli_model or get_context_cli_model_override(context)
    model_spec, source = resolve_model_spec(
        context,
        model=model,
        default_model=default_model,
        cli_model=cli_model,
        hardcoded_default=HARDCODED_DEFAULT_MODEL,
    )
    if model_spec is None:
        raise ModelConfigError(
            "No model configured",
            "Set --model, FAST_AGENT_MODEL, or default_model in config.",
        )
    logger.info(
        f"Resolved model '{model_spec}' via {source}",
        model=model_spec,
        source=source,
    )

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
    model_references: dict[str, dict[str, str]] | None = None,
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

    _, source = resolve_model_spec(
        context=None,
        default_model=config_default_model,
        cli_model=None,
        fallback_to_hardcoded=False,
        model_references=model_references,
    )
    if source == "config file":
        return "config file"
    if source and source.startswith("environment variable"):
        return "environment variable"

    return None


async def _create_basic_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    child_names = cast("Sequence[str]", agent_data.get("child_agents", []) or [])
    if child_names:
        inputs = _build_agents_as_tools_inputs(name, agent_data, build_ctx)

        from fast_agent.agents.workflow.agents_as_tools_agent import AgentsAsToolsAgent

        agent = AgentsAsToolsAgent(
            config=inputs.config,
            context=build_ctx.app_instance.context,
            agents=cast("list[LlmAgent]", inputs.child_agents),
            options=inputs.options,
            tools=inputs.function_tools,
            child_message_files=inputs.child_message_files,
        )
    else:
        function_tools = _resolve_function_tools_with_globals(config, agent_data, build_ctx)
        agent = _create_agent_with_ui_if_needed(
            McpAgent,
            config,
            build_ctx.app_instance.context,
            tools=function_tools,
        )

    await _finalize_agent(
        agent,
        name,
        config,
        agent_data,
        build_ctx.model_factory_func,
        result_agents,
        build_ctx.session_history_enabled,
    )


async def _create_smart_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    child_names = cast("Sequence[str]", agent_data.get("child_agents", []) or [])
    if child_names:
        inputs = _build_agents_as_tools_inputs(name, agent_data, build_ctx)

        from fast_agent.agents.smart_agent import SmartAgentsAsToolsAgent

        agent = SmartAgentsAsToolsAgent(
            config=inputs.config,
            context=build_ctx.app_instance.context,
            agents=cast("list[LlmAgent]", inputs.child_agents),
            options=inputs.options,
            tools=inputs.function_tools,
            child_message_files=inputs.child_message_files,
        )
    else:
        function_tools = _resolve_function_tools_with_globals(config, agent_data, build_ctx)

        from fast_agent.agents.smart_agent import SmartAgent, SmartAgentWithUI

        settings = (
            build_ctx.app_instance.context.config if build_ctx.app_instance.context else None
        )
        ui_mode = getattr(settings, "mcp_ui_mode", "auto") if settings else "auto"
        if ui_mode != "disabled":
            agent = SmartAgentWithUI(
                config=config,
                context=build_ctx.app_instance.context,
                ui_mode=ui_mode,
                tools=function_tools,
            )
        else:
            agent = SmartAgent(
                config=config,
                context=build_ctx.app_instance.context,
                tools=function_tools,
            )

    await _finalize_agent(
        agent,
        name,
        config,
        agent_data,
        build_ctx.model_factory_func,
        result_agents,
        build_ctx.session_history_enabled,
    )


async def _create_custom_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    cls = agent_data.get("agent_class") or agent_data.get("cls")
    if cls is None:
        raise AgentConfigError(
            f"Custom agent '{name}' missing class reference ('agent_class' or 'cls')"
        )

    explicit_function_tools = (
        config.function_tools is not None or agent_data.get("function_tools") is not None
    )
    function_tools = _resolve_function_tools_with_globals(config, agent_data, build_ctx)
    custom_supports_function_tools = custom_class_supports_function_tools(cls)
    if function_tools and explicit_function_tools and not custom_supports_function_tools:
        raise AgentConfigError(
            "Custom agent does not accept function tools",
            f"Custom agent '{name}' cannot use function_tools because "
            f"{getattr(cls, '__name__', cls)!r} does not accept tools=.",
        )

    create_kwargs: dict[str, Any] = {}
    if function_tools and custom_supports_function_tools:
        # Custom agent constructors follow the existing ToolAgent/McpAgent
        # convention: resolved function tools are passed as ``tools=``.
        create_kwargs["tools"] = function_tools

    agent = _create_agent_with_ui_if_needed(
        cls,
        config,
        build_ctx.app_instance.context,
        **create_kwargs,
    )
    await _initialize_agent_with_llm(agent, config, build_ctx.model_factory_func)

    child_names = cast("Sequence[str]", agent_data.get("child_agents", []) or [])
    if child_names:
        _attach_child_agents_as_tools(agent, name, child_names, build_ctx.active_agents)

    if (
        config.tool_hooks
        or config.trim_tool_history
        or build_ctx.session_history_enabled
    ):
        _apply_tool_hooks(
            agent,
            config,
            agent_data.get("source_path"),
            enable_session_history=build_ctx.session_history_enabled,
        )

    _register_loaded_agent(result_agents, name, agent)


async def _create_planner_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    child_names = cast("Sequence[str]", agent_data["child_agents"])
    child_agents = _resolve_child_agents(
        name,
        child_names,
        build_ctx.active_agents,
        skip_missing=False,
    )

    orchestrator = IterativePlanner(
        config=config,
        context=build_ctx.app_instance.context,
        agents=child_agents,
        plan_iterations=agent_data.get("plan_iterations", 5),
        plan_type=agent_data.get("plan_type", "full"),
    )
    await _initialize_agent_with_llm(orchestrator, config, build_ctx.model_factory_func)
    result_agents[name] = orchestrator


async def _create_parallel_workflow_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    fan_in_name = agent_data.get("fan_in")
    fan_out_names = cast("Sequence[str]", agent_data["fan_out"])

    if not fan_in_name:
        fan_in_name = f"{name}_fan_in"
        fan_in_agent = await _create_default_fan_in_agent(
            fan_in_name,
            build_ctx.app_instance.context,
            build_ctx.model_factory_func,
        )
        result_agents[fan_in_name] = fan_in_agent
    else:
        fan_in_agent = build_ctx.active_agents.get(fan_in_name)
        if fan_in_agent is None:
            raise AgentConfigError(f"Fan-in agent {fan_in_name} not found")

    fan_out_agents = _resolve_child_agents(
        name,
        fan_out_names,
        build_ctx.active_agents,
        skip_missing=False,
    )

    parallel = ParallelAgent(
        config=config,
        context=build_ctx.app_instance.context,
        fan_in_agent=fan_in_agent,
        fan_out_agents=fan_out_agents,
        include_request=agent_data.get("include_request", True),
    )
    await parallel.initialize()
    result_agents[name] = parallel


async def _create_router_workflow_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    router = RouterAgent(
        config=config,
        context=build_ctx.app_instance.context,
        agents=cast(
            "list[LlmAgent]",
            _resolve_child_agents(
                name,
                cast("Sequence[str]", agent_data["router_agents"]),
                build_ctx.active_agents,
                skip_missing=False,
            ),
        ),
        routing_instruction=agent_data.get("instruction"),
    )
    await _initialize_agent_with_llm(router, config, build_ctx.model_factory_func)
    result_agents[name] = router


async def _create_chain_workflow_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    from fast_agent.agents.workflow.chain_agent import ChainAgent

    config = cast("AgentConfig", agent_data["config"])
    agent_names = cast("Sequence[str]", agent_data["sequence"])
    if not agent_names:
        raise AgentConfigError("No agents in the chain")

    chain = ChainAgent(
        config=config,
        context=build_ctx.app_instance.context,
        agents=cast(
            "list[LlmAgent]",
            _resolve_child_agents(
                name,
                agent_names,
                build_ctx.active_agents,
                skip_missing=False,
            ),
        ),
        cumulative=agent_data.get("cumulative", False),
    )
    await chain.initialize()
    result_agents[name] = chain


async def _create_evaluator_optimizer_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    config = cast("AgentConfig", agent_data["config"])
    generator_name = cast("str", agent_data["generator"])
    evaluator_name = cast("str", agent_data["evaluator"])

    generator_agent = build_ctx.active_agents.get(generator_name)
    if generator_agent is None:
        raise AgentConfigError(f"Generator agent {generator_name} not found")

    evaluator_agent = build_ctx.active_agents.get(evaluator_name)
    if evaluator_agent is None:
        raise AgentConfigError(f"Evaluator agent {evaluator_name} not found")

    evaluator_optimizer = EvaluatorOptimizerAgent(
        config=config,
        context=build_ctx.app_instance.context,
        generator_agent=generator_agent,
        evaluator_agent=evaluator_agent,
        min_rating=QualityRating(agent_data.get("min_rating", "GOOD")),
        max_refinements=agent_data.get("max_refinements", 3),
        refinement_instruction=agent_data.get("refinement_instruction"),
    )
    await evaluator_optimizer.initialize()
    result_agents[name] = evaluator_optimizer


async def _create_maker_agent(
    name: str,
    agent_data: Mapping[str, Any],
    build_ctx: AgentBuildContext,
    result_agents: AgentDict,
) -> None:
    from fast_agent.agents.workflow.maker_agent import MakerAgent, MatchStrategy

    config = cast("AgentConfig", agent_data["config"])
    worker_name = cast("str", agent_data["worker"])
    worker_agent = build_ctx.active_agents.get(worker_name)
    if worker_agent is None:
        raise AgentConfigError(f"Worker agent {worker_name} not found")

    maker_agent = MakerAgent(
        config=config,
        context=build_ctx.app_instance.context,
        worker_agent=worker_agent,
        k=agent_data.get("k", 3),
        max_samples=agent_data.get("max_samples", 50),
        match_strategy=MatchStrategy(agent_data.get("match_strategy", "exact")),
        red_flag_max_length=agent_data.get("red_flag_max_length"),
    )
    await maker_agent.initialize()
    result_agents[name] = maker_agent


_AGENT_TYPE_BUILDERS: dict[AgentType, AgentTypeBuilder] = {
    AgentType.LLM: _create_basic_agent,
    AgentType.BASIC: _create_basic_agent,
    AgentType.SMART: _create_smart_agent,
    AgentType.CUSTOM: _create_custom_agent,
    AgentType.ORCHESTRATOR: _create_planner_agent,
    AgentType.ITERATIVE_PLANNER: _create_planner_agent,
    AgentType.PARALLEL: _create_parallel_workflow_agent,
    AgentType.ROUTER: _create_router_workflow_agent,
    AgentType.CHAIN: _create_chain_workflow_agent,
    AgentType.EVALUATOR_OPTIMIZER: _create_evaluator_optimizer_agent,
    AgentType.MAKER: _create_maker_agent,
}


async def create_agents_by_type(
    app_instance: CoreContextProtocol,
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

    result_agents: AgentDict = {}
    session_history_enabled = True
    if app_instance.context and app_instance.context.config:
        session_history_enabled = getattr(app_instance.context.config, "session_history", True)
    builder = _AGENT_TYPE_BUILDERS.get(agent_type)
    if builder is None:
        raise ValueError(f"Unknown agent type: {agent_type}")

    build_ctx = AgentBuildContext(
        app_instance=app_instance,
        agents_dict=agents_dict,
        active_agents=active_agents,
        model_factory_func=model_factory_func,
        session_history_enabled=session_history_enabled,
    )

    for name, agent_data in _iter_agents_of_type(agents_dict, agent_type):
        await builder(name, agent_data, build_ctx, result_agents)

    return result_agents


async def active_agents_in_dependency_group(
    app_instance: CoreContextProtocol,
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
            if normalize_agent_type_value(agents_dict[name].get("type")) == agent_type_value
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
    app_instance: CoreContextProtocol,
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


async def create_basic_agents_in_dependency_order(
    context: Context,
    agents_dict: AgentConfigDict,
    model_factory_func: ModelFactoryFunctionProtocol,
    allow_cycles: bool = False,
) -> AgentDict:
    """
    Create BASIC agents in dependency order without a full Core instance.

    Args:
        context: Application context
        agents_dict: Dictionary of agent configurations
        model_factory_func: Function for creating model factories
        allow_cycles: Whether to allow cyclic dependencies

    Returns:
        Dictionary of initialized agent instances
    """
    _ensure_basic_only_agents(agents_dict)
    return await create_agents_in_dependency_order(
        _ContextCoreShim(context),
        agents_dict,
        model_factory_func,
        allow_cycles,
    )


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
