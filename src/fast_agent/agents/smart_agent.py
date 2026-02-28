"""Smart agent implementation with built-in tools."""

from __future__ import annotations

import asyncio
import base64
import shlex
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mcp.types import BlobResourceContents, ReadResourceResult, TextResourceContents

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
)
from fast_agent.commands.command_catalog import (
    SmartCommandOperation,
    parse_smart_operation,
    smart_operation_help_entries,
)
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import models_manager as models_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.renderers.command_markdown import render_command_outcome_markdown
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.agent_card_loader import load_agent_cards
from fast_agent.core.agent_card_validation import AgentCardScanResult, scan_agent_card_path
from fast_agent.core.direct_factory import (
    create_basic_agents_in_dependency_order,
    get_model_factory,
)
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.instruction_utils import apply_instruction_context
from fast_agent.core.internal_resources import (
    InternalResource,
    get_internal_resource,
    list_internal_resources,
    read_internal_resource,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.core.validation import validate_provider_keys_post_creation
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.mcp.ui_mixin import McpUIMixin
from fast_agent.paths import resolve_environment_paths
from fast_agent.tools.function_tool_loader import FastMCPTool

if TYPE_CHECKING:
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)

_INTERNAL_RESOURCE_SERVER = "internal"


@dataclass(frozen=True)
class _SmartCardBundle:
    agents_dict: dict[str, "AgentCardData | dict[str, Any]"]
    message_files: dict[str, list[Path]]


class _McpCapableAgent(Protocol):
    async def attach_mcp_server(
        self,
        *,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult: ...

    async def detach_mcp_server(self, server_name: str) -> MCPDetachResult: ...

    def list_attached_mcp_servers(self) -> list[str]: ...


@dataclass(frozen=True)
class _SmartConnectSummary:
    connected: list[str]
    warnings: list[str]


class _SmartToolMcpManager:
    """Minimal MCP runtime manager adapter for temporary smart-tool agents."""

    def __init__(
        self,
        agents: Mapping[str, AgentProtocol],
        configured_server_names: set[str],
    ) -> None:
        self._agents = agents
        self._configured_server_names = configured_server_names

    def _agent(self, name: str) -> _McpCapableAgent:
        agent = self._agents.get(name)
        if agent is None:
            raise AgentConfigError("Unknown agent", f"Agent '{name}' is not loaded")

        required = ("attach_mcp_server", "detach_mcp_server", "list_attached_mcp_servers")
        if not all(callable(getattr(agent, attr, None)) for attr in required):
            raise AgentConfigError(
                "Agent does not support runtime MCP connection",
                f"Agent '{name}' cannot attach MCP servers",
            )

        return agent  # type: ignore[return-value]

    async def attach_mcp_server(
        self,
        agent_name: str,
        server_name: str,
        server_config: MCPServerSettings | None = None,
        options: MCPAttachOptions | None = None,
    ) -> MCPAttachResult:
        agent = self._agent(agent_name)
        return await agent.attach_mcp_server(
            server_name=server_name,
            server_config=server_config,
            options=options,
        )

    async def detach_mcp_server(self, agent_name: str, server_name: str) -> MCPDetachResult:
        agent = self._agent(agent_name)
        return await agent.detach_mcp_server(server_name)

    async def list_attached_mcp_servers(self, agent_name: str) -> list[str]:
        agent = self._agent(agent_name)
        return list(agent.list_attached_mcp_servers())

    async def list_configured_detached_mcp_servers(self, agent_name: str) -> list[str]:
        attached = set(await self.list_attached_mcp_servers(agent_name))
        return sorted(self._configured_server_names - attached)


class _SmartCommandAgentProvider:
    """Minimal agent-provider adapter for smart command handlers."""

    def __init__(self, agents: Mapping[str, object]) -> None:
        self._agents = agents

    def _agent(self, name: str) -> object:
        return self._agents[name]

    def agent_names(self) -> Iterable[str]:
        return list(self._agents.keys())

    async def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> object:
        del namespace, agent_name
        return {}


@dataclass(slots=True)
class _SmartCommandIO:
    """Non-interactive command IO that buffers emitted messages."""

    messages: list[CommandMessage]

    async def emit(self, message: CommandMessage) -> None:
        self.messages.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        del prompt, allow_empty
        return default

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options: Sequence[str],
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        del prompt, options, allow_cancel, default
        return None

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        del arg_name, description, required
        return None

    async def display_history_turn(
        self,
        agent_name: str,
        turn: list[PromptMessageExtended],
        *,
        turn_index: int | None = None,
        total_turns: int | None = None,
    ) -> None:
        del agent_name, turn, turn_index, total_turns

    async def display_history_overview(
        self,
        agent_name: str,
        history: list[PromptMessageExtended],
        usage: UsageAccumulator | None = None,
    ) -> None:
        del agent_name, history, usage

    async def display_usage_report(self, agents: dict[str, object]) -> None:
        del agents

    async def display_system_prompt(
        self,
        agent_name: str,
        system_prompt: str,
        *,
        server_count: int = 0,
    ) -> None:
        del agent_name, system_prompt, server_count


def _resolve_default_agent_name(
    agents: Mapping[str, AgentProtocol],
    *,
    tool_only_agents: set[str],
) -> str:
    for name, agent in agents.items():
        if name in tool_only_agents:
            continue
        if bool(getattr(agent.config, "default", False)):
            return name

    for name in agents:
        if name not in tool_only_agents:
            return name

    return next(iter(agents.keys()))


def _collect_outcome_messages(outcome: "CommandOutcome") -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for message in outcome.messages:
        text = str(message.text)
        if message.channel == "error":
            errors.append(text)
        elif message.channel == "warning":
            warnings.append(text)
    return errors, warnings


def _format_command_outcome(outcome: "CommandOutcome") -> str:
    lines: list[str] = []
    for message in outcome.messages:
        text = str(message.text).strip()
        if text:
            lines.append(text)
    return "\n".join(lines) if lines else "Done."


def _resolve_smart_command_agent_map(agent: Any) -> dict[str, object]:
    registry = getattr(agent, "agent_registry", None)
    agents: dict[str, object] = {}
    if isinstance(registry, Mapping):
        agents.update({str(name): member for name, member in registry.items()})

    agent_name = str(getattr(agent, "name", ""))
    if agent_name:
        agents.setdefault(agent_name, agent)
    return agents


def _build_smart_command_context(agent: Any) -> tuple[CommandContext, _SmartCommandIO]:
    context = getattr(agent, "context", None)
    settings = context.config if context else None
    agent_name = str(getattr(agent, "name", ""))
    if not agent_name:
        raise AgentConfigError("Smart command requires named agent", "Agent has no name")

    io = _SmartCommandIO(messages=[])
    provider = _SmartCommandAgentProvider(_resolve_smart_command_agent_map(agent))
    return (
        CommandContext(
            agent_provider=provider,
            current_agent_name=agent_name,
            io=io,
            settings=settings,
        ),
        io,
    )


def _render_smart_command_outcome(
    outcome: CommandOutcome,
    *,
    heading: str,
    io: _SmartCommandIO,
) -> str:
    return render_command_outcome_markdown(
        outcome,
        heading=heading,
        extra_messages=io.messages,
    )


async def _run_check_subprocess(argument: str | None) -> CommandOutcome:
    command = [
        sys.executable,
        "-m",
        "fast_agent.cli.main",
        "--no-color",
        "check",
    ]
    if argument and argument.strip():
        try:
            command.extend(shlex.split(argument))
        except ValueError as exc:
            raise AgentConfigError("Invalid check arguments", str(exc)) from exc

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()

    outcome = CommandOutcome()
    stdout_text = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if stdout_text:
        outcome.add_message(stdout_text)
    if stderr_text:
        channel = "error" if process.returncode else "warning"
        outcome.add_message(stderr_text, channel=channel)
    if process.returncode != 0 and not stderr_text:
        outcome.add_message(
            f"fast-agent check exited with status code {process.returncode}.",
            channel="error",
        )
    return outcome


async def _run_smart_command_call(
    agent: Any,
    operation: SmartCommandOperation,
    argument: str | None,
) -> str:
    try:
        command_name, action = parse_smart_operation(operation)
    except ValueError as exc:
        raise AgentConfigError("Unknown smart command operation", str(exc)) from exc
    if command_name == "check":
        outcome = await _run_check_subprocess(argument)
        return render_command_outcome_markdown(outcome, heading="check")

    context, io = _build_smart_command_context(agent)
    agent_name = context.current_agent_name

    if command_name == "skills":
        outcome = await skills_handlers.handle_skills_command(
            context,
            agent_name=agent_name,
            action=action,
            argument=argument,
        )
    elif command_name == "cards":
        outcome = await cards_handlers.handle_cards_command(
            context,
            agent_name=agent_name,
            action=action,
            argument=argument,
        )
    elif command_name == "models":
        outcome = await models_handlers.handle_models_command(
            context,
            agent_name=agent_name,
            action=action,
            argument=argument,
        )
    else:
        raise AgentConfigError(
            "Unsupported smart command operation",
            f"Operation '{operation}' resolved to unsupported command '{command_name}'.",
        )

    return _render_smart_command_outcome(outcome, heading=operation, io=io)


def _context_server_names(context: "Context | None") -> set[str]:
    if context and context.config and context.config.mcp and context.config.mcp.servers:
        return set(context.config.mcp.servers.keys())
    return set()


async def _apply_runtime_mcp_connections(
    *,
    context: "Context | None",
    agents_map: Mapping[str, AgentProtocol],
    target_agent_name: str,
    mcp_connect: Sequence[str],
) -> _SmartConnectSummary:
    configured_names = _context_server_names(context)

    manager = _SmartToolMcpManager(agents_map, configured_server_names=configured_names)
    connected_names: list[str] = []
    warnings: list[str] = []

    for raw_target in mcp_connect:
        target = raw_target.strip()
        if not target:
            continue

        outcome = await mcp_runtime_handlers.handle_mcp_connect(
            None,
            manager=manager,
            agent_name=target_agent_name,
            target_text=target,
        )
        errors, target_warnings = _collect_outcome_messages(outcome)
        warnings.extend(target_warnings)
        if errors:
            raise AgentConfigError(
                "Failed to connect MCP server for smart tool call",
                "\n".join(errors),
            )

        parsed = mcp_runtime_handlers.parse_connect_input(target)
        mode = mcp_runtime_handlers.infer_connect_mode(parsed.target_text)
        resolved_name = parsed.server_name or mcp_runtime_handlers.infer_server_name(
            parsed.target_text,
            mode,
        )
        connected_names.append(resolved_name)

    return _SmartConnectSummary(connected=connected_names, warnings=warnings)


async def _run_mcp_connect_call(agent: Any, target: str) -> str:
    context = getattr(agent, "context", None)
    manager = _SmartToolMcpManager(
        {agent.name: agent},
        configured_server_names=_context_server_names(context),
    )
    outcome = await mcp_runtime_handlers.handle_mcp_connect(
        None,
        manager=manager,
        agent_name=agent.name,
        target_text=target,
    )
    return _format_command_outcome(outcome)


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


async def _create_smart_tool_agents(
    context: "Context | None",
    agent_card_path: str,
) -> tuple[dict[str, AgentProtocol], _SmartCardBundle, str, set[str]]:
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
    validate_provider_keys_post_creation(agents_map)

    tool_only_agents = {
        name for name, data in bundle.agents_dict.items() if data.get("tool_only", False)
    }
    default_agent_name = _resolve_default_agent_name(
        agents_map,
        tool_only_agents=tool_only_agents,
    )

    return agents_map, bundle, default_agent_name, tool_only_agents


async def _hydrate_smart_agents_for_execution(
    agents_map: dict[str, AgentProtocol],
    bundle: _SmartCardBundle,
) -> None:
    if bundle.message_files:
        _apply_agent_card_histories(agents_map, bundle.message_files)

    context_vars: dict[str, str] = {}
    enrich_with_environment_context(
        context_vars,
        str(Path.cwd()),
        {"name": "fast-agent"},
    )
    await _apply_instruction_context(agents_map, context_vars)


def _extract_read_resource_text(result: ReadResourceResult, *, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for idx, content in enumerate(result.contents, start=1):
        if isinstance(content, TextResourceContents):
            text = content.text
            lines.append(f"[{idx}] text ({content.mimeType or 'unknown'})")
            lines.append(text)
            continue

        if isinstance(content, BlobResourceContents):
            blob_len = len(content.blob)
            preview = ""
            try:
                decoded = base64.b64decode(content.blob)
                preview = decoded[:400].decode("utf-8", errors="replace")
            except Exception:
                preview = "<binary blob>"
            lines.append(f"[{idx}] blob ({content.mimeType or 'unknown'}, {blob_len} b64 chars)")
            if preview:
                lines.append(preview)
            continue

        text = get_text(content)
        if text:
            lines.append(f"[{idx}] content")
            lines.append(text)

    joined = "\n".join(lines).strip()
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 1] + "…\n[truncated]"


def _format_smart_resource_listing(
    resources: Mapping[str, list[str]],
    templates: Mapping[str, Sequence[Any]],
    *,
    internal_resources: Sequence[InternalResource] = (),
) -> str:
    lines: list[str] = []
    server_names = sorted(set(resources.keys()) | set(templates.keys()))
    if not server_names and not internal_resources:
        return "No resources available."

    if internal_resources:
        lines.append(f"[{_INTERNAL_RESOURCE_SERVER}]")
        lines.append("resources:")
        for resource in internal_resources:
            lines.append(f"  - {resource.uri}")
            lines.append(f"    description: {resource.description}")
            lines.append(f"    why: {resource.why}")
        lines.append("templates: []")
        if server_names:
            lines.append("")

    for server_name in server_names:
        lines.append(f"[{server_name}]")
        server_resources = resources.get(server_name, [])
        if server_resources:
            lines.append("resources:")
            for uri in server_resources:
                lines.append(f"  - {uri}")
        else:
            lines.append("resources: []")

        server_templates = templates.get(server_name, [])
        if server_templates:
            lines.append("templates:")
            for template in server_templates:
                uri_template = getattr(template, "uriTemplate", "")
                name = getattr(template, "name", "")
                if name:
                    lines.append(f"  - {name}: {uri_template}")
                else:
                    lines.append(f"  - {uri_template}")
        else:
            lines.append("templates: []")

    return "\n".join(lines)


def _include_internal_resources(server_name: str | None) -> bool:
    return server_name is None or server_name == _INTERNAL_RESOURCE_SERVER


def _include_mcp_resources(server_name: str | None) -> bool:
    return server_name != _INTERNAL_RESOURCE_SERVER


def _is_internal_resource_uri(uri: str) -> bool:
    return uri.strip().startswith("internal://")


async def _run_current_agent_list_resources_call(
    agent: Any,
    *,
    server_name: str | None = None,
) -> str:
    resources: Mapping[str, list[str]] = {}
    templates: Mapping[str, Sequence[Any]] = {}
    if _include_mcp_resources(server_name):
        resources = await agent.list_resources(namespace=server_name)

        aggregator = getattr(agent, "aggregator", None)
        list_templates = getattr(aggregator, "list_resource_templates", None)
        if callable(list_templates):
            templates = await list_templates(server_name)

    internal_resources: Sequence[InternalResource] = ()
    if _include_internal_resources(server_name):
        internal_resources = list_internal_resources()

    return _format_smart_resource_listing(
        resources,
        templates,
        internal_resources=internal_resources,
    )


async def _run_internal_resource_read_call(uri: str) -> str:
    resource = get_internal_resource(uri)
    content = read_internal_resource(resource.uri)
    header = f"Resource: {resource.uri}\nTitle: {resource.title}"
    return f"{header}\n\n{content}" if content else header


async def _run_smart_call(
    context: "Context | None",
    agent_card_path: str,
    message: str,
    *,
    mcp_connect: Sequence[str] | None = None,
    disable_streaming: bool = False,
) -> str:
    agents_map, bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)

        if mcp_connect:
            connect_summary = await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )
            if connect_summary.connected:
                logger.info(
                    "Connected runtime MCP servers for smart tool call",
                    data={
                        "agent": default_agent_name,
                        "servers": connect_summary.connected,
                    },
                )
            for warning in connect_summary.warnings:
                logger.warning(
                    "Runtime MCP connect warning in smart tool call",
                    data={"warning": warning, "agent": default_agent_name},
                )

        if disable_streaming:
            for agent in agents_map.values():
                setter = getattr(agent, "force_non_streaming_next_turn", None)
                if callable(setter):
                    setter(reason="parallel smart tool calls")
            logger.info(
                "Disabled streaming for smart tool child agents",
                data={"agent_count": len(agents_map)},
            )

        await _hydrate_smart_agents_for_execution(agents_map, bundle)

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


async def _run_smart_list_resources_call(
    context: "Context | None",
    agent_card_path: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    resources: Mapping[str, list[str]] = {}
    templates: Mapping[str, Sequence[Any]] = {}

    if _include_mcp_resources(server_name):
        agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
            context,
            agent_card_path,
        )
        try:
            app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
            if mcp_connect:
                await _apply_runtime_mcp_connections(
                    context=context,
                    agents_map=agents_map,
                    target_agent_name=default_agent_name,
                    mcp_connect=mcp_connect,
                )

            resources = await app.list_resources(
                server_name=server_name,
                agent_name=default_agent_name,
            )

            default_agent = agents_map.get(default_agent_name)
            aggregator = getattr(default_agent, "aggregator", None) if default_agent else None
            list_templates = getattr(aggregator, "list_resource_templates", None)
            if callable(list_templates):
                templates = await list_templates(server_name)
        finally:
            await _shutdown_agents(agents_map)

    internal_resources: Sequence[InternalResource] = ()
    if _include_internal_resources(server_name):
        internal_resources = list_internal_resources()

    return _format_smart_resource_listing(
        resources,
        templates,
        internal_resources=internal_resources,
    )


async def _run_current_agent_get_resource_call(
    agent: Any,
    resource_uri: str,
    *,
    server_name: str | None = None,
) -> str:
    result = await agent.get_resource(resource_uri=resource_uri, namespace=server_name)
    body = _extract_read_resource_text(result)
    header = f"Resource: {resource_uri}"
    if server_name:
        header += f" (server={server_name})"
    return f"{header}\n\n{body}" if body else header


async def _run_smart_get_resource_call(
    context: "Context | None",
    agent_card_path: str,
    resource_uri: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        result = await app.get_resource(
            resource_uri=resource_uri,
            server_name=server_name,
            agent_name=default_agent_name,
        )
        body = _extract_read_resource_text(result)
        header = f"Resource: {resource_uri}"
        if server_name:
            header += f" (server={server_name})"
        return f"{header}\n\n{body}" if body else header
    finally:
        await _shutdown_agents(agents_map)


async def _run_smart_with_resource_call(
    context: "Context | None",
    agent_card_path: str,
    message: str,
    resource_uri: str,
    *,
    server_name: str | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        app = AgentApp(agents_map, tool_only_agents=tool_only_agents)
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        await _hydrate_smart_agents_for_execution(agents_map, bundle)

        return await app.with_resource(
            prompt_content=message,
            resource_uri=resource_uri,
            server_name=server_name,
            agent_name=default_agent_name,
        )
    finally:
        await _shutdown_agents(agents_map)


def _resolve_template_server_name(
    template_uri: str,
    templates_by_server: Mapping[str, Sequence[Any]],
) -> str | None:
    matches: list[str] = []
    for server_name, templates in templates_by_server.items():
        for template in templates:
            if getattr(template, "uriTemplate", None) == template_uri:
                matches.append(server_name)
                break

    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    return None


async def _run_smart_complete_resource_argument_call(
    context: "Context | None",
    agent_card_path: str,
    template_uri: str,
    argument_name: str,
    value: str,
    *,
    server_name: str | None = None,
    context_args: Mapping[str, str] | None = None,
    mcp_connect: Sequence[str] | None = None,
) -> str:
    agents_map, _bundle, default_agent_name, tool_only_agents = await _create_smart_tool_agents(
        context,
        agent_card_path,
    )
    try:
        del tool_only_agents
        if mcp_connect:
            await _apply_runtime_mcp_connections(
                context=context,
                agents_map=agents_map,
                target_agent_name=default_agent_name,
                mcp_connect=mcp_connect,
            )

        default_agent = agents_map.get(default_agent_name)
        aggregator = getattr(default_agent, "aggregator", None) if default_agent else None
        if aggregator is None:
            raise AgentConfigError(
                "Smart resource completion requires MCP-capable card agent",
                f"Agent '{default_agent_name}' does not expose an MCP aggregator",
            )

        target_server = server_name
        if target_server is None:
            templates = await aggregator.list_resource_templates(None)
            target_server = _resolve_template_server_name(template_uri, templates)
            if target_server is None:
                raise AgentConfigError(
                    "Unable to resolve resource template server",
                    "Pass server_name explicitly when multiple servers expose the same template.",
                )

        completion = await aggregator.complete_resource_argument(
            server_name=target_server,
            template_uri=template_uri,
            argument_name=argument_name,
            value=value,
            context_args=dict(context_args) if context_args else None,
        )
        values = completion.values or []
        if not values:
            return "No completion values returned."
        return "\n".join(values)
    finally:
        await _shutdown_agents(agents_map)


def _smart_command_tool_description() -> str:
    lines = [
        "Run a shared fast-agent command operation by operation key.",
        "",
        "Supported operations:",
    ]
    for operation, help_text in smart_operation_help_entries():
        lines.append(f"- `{operation}`: {help_text}")
    return "\n".join(lines)


def _enable_smart_tooling(agent: Any) -> None:
    """Register smart tool endpoints on a smart-capable agent."""
    setattr(agent, "_parallel_smart_tool_calls", False)
    smart_tool_names = {
        "smart",
        "smart_command",
        "validate",
        "mcp_connect",
        "list_resources",
        "get_resource",
        "smart_with_resource",
        "smart_complete_resource_argument",
    }
    existing_smart_tools = set(getattr(agent, "_smart_tool_names", []) or [])
    existing_smart_tools.update(smart_tool_names)
    setattr(agent, "_smart_tool_names", existing_smart_tools)

    smart_tool = FastMCPTool.from_function(
        agent.smart,
        name="smart",
        description=(
            "Load AgentCards from a path and send a message to the resolved default card agent "
            "(default:true, otherwise first non-tool_only). Optional `mcp_connect` entries "
            "accept `/mcp connect` style target strings for runtime MCP attachment."
        ),
    )
    validate_tool = FastMCPTool.from_function(
        agent.validate,
        name="validate",
        description="Validate AgentCard files using the same checks as fast-agent check.",
    )
    smart_command_tool = FastMCPTool.from_function(
        agent.smart_command,
        name="smart_command",
        description=_smart_command_tool_description(),
    )
    mcp_connect_tool = FastMCPTool.from_function(
        agent.mcp_connect,
        name="mcp_connect",
        description=(
            "Connect an MCP server to this smart agent at runtime. "
            "Accepts `/mcp connect` style target strings, including flags "
            "like --name/--auth/--timeout/--oauth/--reconnect. "
            "`--auth` supports `$VAR`, `${VAR}`, and `${VAR:default}` env references. "
            "Pass token value only; fast-agent sends `Authorization: Bearer <token>` automatically "
            "(optional `Bearer ` input is normalized)."
        ),
    )
    resource_list_tool = FastMCPTool.from_function(
        agent.resource_list,
        name="list_resources",
        description=(
            "List resources available to this smart agent, including bundled internal "
            "resources and attached MCP resources/templates."
        ),
    )
    resource_read_tool = FastMCPTool.from_function(
        agent.resource_read,
        name="get_resource",
        description=(
            "Read a resource by URI. `internal://` URIs are read directly from bundled "
            "resources; other URIs are fetched from attached MCP resources."
        ),
    )
    smart_with_resource_tool = FastMCPTool.from_function(
        agent.smart_with_resource,
        name="smart_with_resource",
        description=(
            "Load AgentCards and run a message on the default agent with an attached "
            "MCP resource."
        ),
    )
    smart_complete_resource_argument_tool = FastMCPTool.from_function(
        agent.smart_complete_resource_argument,
        name="smart_complete_resource_argument",
        description=(
            "Load AgentCards and call MCP completion/complete for a resource template "
            "argument value."
        ),
    )
    agent.add_tool(smart_tool)
    agent.add_tool(smart_command_tool)
    agent.add_tool(validate_tool)
    agent.add_tool(mcp_connect_tool)
    agent.add_tool(resource_list_tool)
    agent.add_tool(resource_read_tool)
    agent.add_tool(smart_with_resource_tool)
    agent.add_tool(smart_complete_resource_argument_tool)


async def _dispatch_smart_tool(
    agent: Any,
    agent_card_path: str,
    message: str,
    mcp_connect: list[str] | None = None,
) -> str:
    disable_streaming = bool(getattr(agent, "_parallel_smart_tool_calls", False))
    context = getattr(agent, "context", None)
    return await _run_smart_call(
        context,
        agent_card_path,
        message,
        mcp_connect=mcp_connect,
        disable_streaming=disable_streaming,
    )


async def _dispatch_smart_command_tool(
    agent: Any,
    operation: SmartCommandOperation,
    argument: str | None = None,
) -> str:
    return await _run_smart_command_call(agent, operation, argument)


async def _dispatch_validate_tool(agent: Any, agent_card_path: str) -> str:
    context = getattr(agent, "context", None)
    return await _run_validate_call(context, agent_card_path)


async def _dispatch_mcp_connect_tool(agent: Any, target: str) -> str:
    return await _run_mcp_connect_call(agent, target)


async def _dispatch_resource_list_tool(
    agent: Any,
    server_name: str | None = None,
) -> str:
    return await _run_current_agent_list_resources_call(agent, server_name=server_name)


async def _dispatch_resource_read_tool(
    agent: Any,
    uri: str,
    server_name: str | None = None,
) -> str:
    if _is_internal_resource_uri(uri):
        return await _run_internal_resource_read_call(uri)
    return await _run_current_agent_get_resource_call(
        agent,
        uri,
        server_name=server_name,
    )


async def _dispatch_smart_list_resources_tool(
    agent: Any,
    agent_card_path: str,
    mcp_connect: list[str] | None = None,
    server_name: str | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_list_resources_call(
        context,
        agent_card_path,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_get_resource_tool(
    agent: Any,
    agent_card_path: str,
    resource_uri: str,
    server_name: str | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    if _is_internal_resource_uri(resource_uri):
        return await _run_internal_resource_read_call(resource_uri)

    context = getattr(agent, "context", None)
    return await _run_smart_get_resource_call(
        context,
        agent_card_path,
        resource_uri,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_with_resource_tool(
    agent: Any,
    agent_card_path: str,
    message: str,
    resource_uri: str,
    server_name: str | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_with_resource_call(
        context,
        agent_card_path,
        message,
        resource_uri,
        server_name=server_name,
        mcp_connect=mcp_connect,
    )


async def _dispatch_smart_complete_resource_argument_tool(
    agent: Any,
    agent_card_path: str,
    template_uri: str,
    argument_name: str,
    value: str,
    server_name: str | None = None,
    context_args: dict[str, str] | None = None,
    mcp_connect: list[str] | None = None,
) -> str:
    context = getattr(agent, "context", None)
    return await _run_smart_complete_resource_argument_call(
        context,
        agent_card_path,
        template_uri,
        argument_name,
        value,
        server_name=server_name,
        context_args=context_args,
        mcp_connect=mcp_connect,
    )


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

    async def smart(
        self,
        agent_card_path: str,
        message: str,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Load AgentCards and send a message to the default agent."""
        return await _dispatch_smart_tool(
            self,
            agent_card_path,
            message,
            mcp_connect=mcp_connect,
        )

    async def smart_command(
        self,
        operation: SmartCommandOperation,
        argument: str | None = None,
    ) -> str:
        """Run a shared command operation exposed by the smart command bridge."""
        return await _dispatch_smart_command_tool(
            self,
            operation=operation,
            argument=argument,
        )

    async def validate(self, agent_card_path: str) -> str:
        """Validate AgentCard files for the provided path."""
        return await _dispatch_validate_tool(self, agent_card_path)

    async def mcp_connect(self, target: str) -> str:
        """Connect an MCP server to this agent at runtime."""
        return await _dispatch_mcp_connect_tool(self, target)

    async def resource_list(self, server_name: str | None = None) -> str:
        """List internal and attached MCP resources for this smart agent."""
        return await _dispatch_resource_list_tool(self, server_name=server_name)

    async def resource_read(self, uri: str, server_name: str | None = None) -> str:
        """Read an internal resource or attached MCP resource by URI."""
        return await _dispatch_resource_read_tool(self, uri, server_name=server_name)

    async def smart_list_resources(
        self,
        agent_card_path: str,
        mcp_connect: list[str] | None = None,
        server_name: str | None = None,
    ) -> str:
        """List card-agent resources and bundled internal resources."""
        return await _dispatch_smart_list_resources_tool(
            self,
            agent_card_path,
            mcp_connect=mcp_connect,
            server_name=server_name,
        )

    async def smart_get_resource(
        self,
        agent_card_path: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Read a card-agent resource or bundled internal resource by URI."""
        return await _dispatch_smart_get_resource_tool(
            self,
            agent_card_path,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_with_resource(
        self,
        agent_card_path: str,
        message: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Run a card agent prompt with a resource attachment."""
        return await _dispatch_smart_with_resource_tool(
            self,
            agent_card_path,
            message,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_complete_resource_argument(
        self,
        agent_card_path: str,
        template_uri: str,
        argument_name: str,
        value: str,
        server_name: str | None = None,
        context_args: dict[str, str] | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        """Run MCP completion for a resource-template argument."""
        return await _dispatch_smart_complete_resource_argument_tool(
            self,
            agent_card_path,
            template_uri,
            argument_name,
            value,
            server_name=server_name,
            context_args=context_args,
            mcp_connect=mcp_connect,
        )


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

    async def smart(
        self,
        agent_card_path: str,
        message: str,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_tool(
            self,
            agent_card_path,
            message,
            mcp_connect=mcp_connect,
        )

    async def smart_command(
        self,
        operation: SmartCommandOperation,
        argument: str | None = None,
    ) -> str:
        return await _dispatch_smart_command_tool(
            self,
            operation=operation,
            argument=argument,
        )

    async def validate(self, agent_card_path: str) -> str:
        return await _dispatch_validate_tool(self, agent_card_path)

    async def mcp_connect(self, target: str) -> str:
        return await _dispatch_mcp_connect_tool(self, target)

    async def resource_list(self, server_name: str | None = None) -> str:
        return await _dispatch_resource_list_tool(self, server_name=server_name)

    async def resource_read(self, uri: str, server_name: str | None = None) -> str:
        return await _dispatch_resource_read_tool(self, uri, server_name=server_name)

    async def smart_list_resources(
        self,
        agent_card_path: str,
        mcp_connect: list[str] | None = None,
        server_name: str | None = None,
    ) -> str:
        return await _dispatch_smart_list_resources_tool(
            self,
            agent_card_path,
            mcp_connect=mcp_connect,
            server_name=server_name,
        )

    async def smart_get_resource(
        self,
        agent_card_path: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_get_resource_tool(
            self,
            agent_card_path,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_with_resource(
        self,
        agent_card_path: str,
        message: str,
        resource_uri: str,
        server_name: str | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_with_resource_tool(
            self,
            agent_card_path,
            message,
            resource_uri,
            server_name=server_name,
            mcp_connect=mcp_connect,
        )

    async def smart_complete_resource_argument(
        self,
        agent_card_path: str,
        template_uri: str,
        argument_name: str,
        value: str,
        server_name: str | None = None,
        context_args: dict[str, str] | None = None,
        mcp_connect: list[str] | None = None,
    ) -> str:
        return await _dispatch_smart_complete_resource_argument_tool(
            self,
            agent_card_path,
            template_uri,
            argument_name,
            value,
            server_name=server_name,
            context_args=context_args,
            mcp_connect=mcp_connect,
        )


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
