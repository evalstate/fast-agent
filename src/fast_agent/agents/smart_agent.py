"""Smart agent implementation with built-in tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
)
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_card_loader import load_agent_cards
from fast_agent.core.agent_card_validation import AgentCardScanResult, scan_agent_card_path
from fast_agent.core.direct_factory import (
    create_basic_agents_in_dependency_order,
    get_model_factory,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.core.validation import validate_provider_keys_post_creation
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.mcp.ui_mixin import McpUIMixin
from fast_agent.paths import resolve_environment_paths
from fast_agent.tools.function_tool_loader import FastMCPTool

if TYPE_CHECKING:
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.context import Context
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.interfaces import AgentProtocol

logger = get_logger(__name__)


@dataclass(frozen=True)
class _SmartCardBundle:
    agents_dict: dict[str, "AgentCardData | dict[str, Any]"]
    message_files: dict[str, list[Path]]


def _resolve_agent_card_path(path_value: str, context: Context | None) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        if candidate.exists():
            return candidate.resolve()
    else:
        cwd_candidate = (Path.cwd() / candidate).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

    env_paths = resolve_environment_paths(
        settings=context.config if context else None, cwd=Path.cwd()
    )
    for base in (env_paths.agent_cards, env_paths.tool_cards):
        env_candidate = (base / candidate).resolve()
        if env_candidate.exists():
            return env_candidate

    raise AgentConfigError(
        "AgentCard path not found",
        f"Tried: {candidate} (cwd), {env_paths.agent_cards}, {env_paths.tool_cards}",
    )


def _ensure_basic_only_cards(cards: Sequence) -> _SmartCardBundle:
    agents_dict: dict[str, "AgentCardData | dict[str, Any]"] = {}
    message_files: dict[str, list[Path]] = {}
    for card in cards:
        agent_data = card.agent_data
        agent_type = agent_data.get("type")
        if agent_type != AgentType.BASIC.value:
            raise AgentConfigError(
                "Smart tool only supports 'agent' cards",
                f"Card '{card.name}' has unsupported type '{agent_type}'",
            )
        agents_dict[card.name] = agent_data
        if card.message_files:
            message_files[card.name] = list(card.message_files)
    return _SmartCardBundle(agents_dict=agents_dict, message_files=message_files)


def _apply_agent_card_histories(
    agents: dict[str, AgentProtocol],
    message_map: Mapping[str, list[Path]],
) -> None:
    for name, history_files in message_map.items():
        agent = agents.get(name)
        if agent is None:
            continue
        messages = []
        for history_file in history_files:
            messages.extend(load_prompt(history_file))
        agent.clear(clear_prompts=True)
        agent.message_history.extend(messages)


async def _apply_instruction_context(
    agents: dict[str, AgentProtocol],
    context_vars: Mapping[str, str],
) -> None:
    await apply_instruction_context(agents.values(), context_vars)


async def _shutdown_agents(agents: Mapping[str, AgentProtocol]) -> None:
    for agent in agents.values():
        try:
            await agent.shutdown()
        except Exception:
            pass


def _format_validation_results(results: Sequence[AgentCardScanResult]) -> str:
    if not results:
        return "No AgentCards found."

    lines: list[str] = []
    for entry in results:
        if entry.ignored_reason:
            status = f"ignored - {entry.ignored_reason}"
        elif entry.errors:
            status = "error"
        else:
            status = "ok"
        lines.append(f"{entry.name} ({entry.type}) - {status}")
        for error in entry.errors:
            lines.append(f"  - {error}")
    return "\n".join(lines)


async def _run_smart_call(
    context: "Context | None",
    agent_card_path: str,
    message: str,
    *,
    disable_streaming: bool = False,
) -> str:
    if context is None:
        raise AgentConfigError("Smart tool requires an initialized context")

    resolved_path = _resolve_agent_card_path(agent_card_path, context)
    cards = load_agent_cards(resolved_path)
    bundle = _ensure_basic_only_cards(cards)

    def model_factory_func(model=None, request_params=None):
        return get_model_factory(context, model=model, request_params=request_params)

    agents_map = await create_basic_agents_in_dependency_order(
        context,
        bundle.agents_dict,
        model_factory_func,
    )
    try:
        validate_provider_keys_post_creation(agents_map)
        tool_only_agents = {
            name for name, data in bundle.agents_dict.items() if data.get("tool_only", False)
        }
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)

        if disable_streaming:
            for agent in agents_map.values():
                setter = getattr(agent, "force_non_streaming_next_turn", None)
                if callable(setter):
                    setter(reason="parallel smart tool calls")
            logger.info(
                "Disabled streaming for smart tool child agents",
                data={"agent_count": len(agents_map)},
            )

        if bundle.message_files:
            _apply_agent_card_histories(agents_map, bundle.message_files)

        context_vars: dict[str, str] = {}
        enrich_with_environment_context(
            context_vars,
            str(Path.cwd()),
            {"name": "fast-agent"},
        )
        await _apply_instruction_context(agents_map, context_vars)

        return await app.send(message)
    finally:
        await _shutdown_agents(agents_map)


async def _run_validate_call(
    context: "Context | None",
    agent_card_path: str,
) -> str:
    resolved_path = _resolve_agent_card_path(agent_card_path, context)

    server_names = None
    if context and context.config and context.config.mcp and context.config.mcp.servers:
        server_names = set(context.config.mcp.servers.keys())

    results = scan_agent_card_path(resolved_path, server_names=server_names)
    return _format_validation_results(results)


def _enable_smart_tooling(agent: Any) -> None:
    """Register smart tool endpoints on a smart-capable agent."""
    setattr(agent, "_parallel_smart_tool_calls", False)
    smart_tool = FastMCPTool.from_function(
        agent.smart,
        name="smart",
        description="Load AgentCards from a path and send a message to the default agent.",
    )
    validate_tool = FastMCPTool.from_function(
        agent.validate,
        name="validate",
        description="Validate AgentCard files using the same checks as fast-agent check.",
    )
    agent.add_tool(smart_tool)
    agent.add_tool(validate_tool)


async def _dispatch_smart_tool(
    agent: Any,
    agent_card_path: str,
    message: str,
) -> str:
    disable_streaming = bool(getattr(agent, "_parallel_smart_tool_calls", False))
    context = getattr(agent, "context", None)
    return await _run_smart_call(
        context,
        agent_card_path,
        message,
        disable_streaming=disable_streaming,
    )


async def _dispatch_validate_tool(agent: Any, agent_card_path: str) -> str:
    context = getattr(agent, "context", None)
    return await _run_validate_call(context, agent_card_path)


class SmartAgent(McpAgent):
    """Smart agent with built-in tools for AgentCard execution and validation."""

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, context=context, **kwargs)
        _enable_smart_tooling(self)

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SMART

    async def smart(self, agent_card_path: str, message: str) -> str:
        """Load AgentCards and send a message to the default agent."""
        return await _dispatch_smart_tool(self, agent_card_path, message)

    async def validate(self, agent_card_path: str) -> str:
        """Validate AgentCard files for the provided path."""
        return await _dispatch_validate_tool(self, agent_card_path)


class SmartAgentsAsToolsAgent(AgentsAsToolsAgent):
    """Agents-as-tools wrapper with smart tools."""

    def __init__(
        self,
        config: AgentConfig,
        agents: list["LlmAgent"],
        options: AgentsAsToolsOptions | None = None,
        context: Any | None = None,
        child_message_files: dict[str, list[Path]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            agents=agents,
            options=options,
            context=context,
            child_message_files=child_message_files,
            **kwargs,
        )
        _enable_smart_tooling(self)

    @property
    def agent_type(self) -> AgentType:
        return AgentType.SMART

    async def smart(self, agent_card_path: str, message: str) -> str:
        return await _dispatch_smart_tool(self, agent_card_path, message)

    async def validate(self, agent_card_path: str) -> str:
        return await _dispatch_validate_tool(self, agent_card_path)


class SmartAgentWithUI(McpUIMixin, SmartAgent):
    """Smart agent with UI support."""

    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
        ui_mode: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, context=context, ui_mode=ui_mode, **kwargs)
