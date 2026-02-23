"""Smart agent implementation with built-in tools."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from mcp.types import BlobResourceContents, ReadResourceResult, TextResourceContents

from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.workflow.agents_as_tools_agent import (
    AgentsAsToolsAgent,
    AgentsAsToolsOptions,
)
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
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
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompts.prompt_load import load_prompt
from fast_agent.mcp.ui_mixin import McpUIMixin
from fast_agent.paths import resolve_environment_paths
from fast_agent.tools.function_tool_loader import FastMCPTool

if TYPE_CHECKING:
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.commands.results import CommandOutcome
    from fast_agent.config import MCPServerSettings
    from fast_agent.context import Context
    from fast_agent.core.agent_card_types import AgentCardData
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.mcp.mcp_aggregator import MCPAttachOptions, MCPAttachResult, MCPDetachResult

logger = get_logger(__name__)


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
) -> str:
    lines: list[str] = []
    server_names = sorted(set(resources.keys()) | set(templates.keys()))
    if not server_names:
        return "No resources available."

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

        templates: dict[str, Sequence[Any]] = {}
        default_agent = agents_map.get(default_agent_name)
        aggregator = getattr(default_agent, "aggregator", None) if default_agent else None
        list_templates = getattr(aggregator, "list_resource_templates", None)
        if callable(list_templates):
            templates = await list_templates(server_name)

        return _format_smart_resource_listing(resources, templates)
    finally:
        await _shutdown_agents(agents_map)


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


def _enable_smart_tooling(agent: Any) -> None:
    """Register smart tool endpoints on a smart-capable agent."""
    setattr(agent, "_parallel_smart_tool_calls", False)
    smart_tool_names = {
        "smart",
        "validate",
        "mcp_connect",
        "smart_list_resources",
        "smart_get_resource",
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
    smart_list_resources_tool = FastMCPTool.from_function(
        agent.smart_list_resources,
        name="smart_list_resources",
        description=(
            "Load AgentCards and list MCP resources/resource templates available to the "
            "resolved default agent. Optional `mcp_connect` entries use `/mcp connect` "
            "style target strings before listing."
        ),
    )
    smart_get_resource_tool = FastMCPTool.from_function(
        agent.smart_get_resource,
        name="smart_get_resource",
        description=(
            "Load AgentCards and fetch a specific MCP resource for inspection. "
            "Use `server_name` to disambiguate when needed."
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
    agent.add_tool(validate_tool)
    agent.add_tool(mcp_connect_tool)
    agent.add_tool(smart_list_resources_tool)
    agent.add_tool(smart_get_resource_tool)
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


async def _dispatch_validate_tool(agent: Any, agent_card_path: str) -> str:
    context = getattr(agent, "context", None)
    return await _run_validate_call(context, agent_card_path)


async def _dispatch_mcp_connect_tool(agent: Any, target: str) -> str:
    return await _run_mcp_connect_call(agent, target)


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

    async def validate(self, agent_card_path: str) -> str:
        """Validate AgentCard files for the provided path."""
        return await _dispatch_validate_tool(self, agent_card_path)

    async def mcp_connect(self, target: str) -> str:
        """Connect an MCP server to this agent at runtime."""
        return await _dispatch_mcp_connect_tool(self, target)

    async def smart_list_resources(
        self,
        agent_card_path: str,
        mcp_connect: list[str] | None = None,
        server_name: str | None = None,
    ) -> str:
        """List MCP resources/templates exposed by the resolved default card agent."""
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
        """Fetch and summarize a specific MCP resource exposed by a card agent."""
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

    async def validate(self, agent_card_path: str) -> str:
        return await _dispatch_validate_tool(self, agent_card_path)

    async def mcp_connect(self, target: str) -> str:
        return await _dispatch_mcp_connect_tool(self, target)

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
