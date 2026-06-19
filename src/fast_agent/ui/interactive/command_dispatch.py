"""Command payload dispatch for the TUI interactive loop."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal, Protocol, TypeGuard, cast

from rich import print as rich_print
from rich.text import Text

from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionRegistry,
    PluginCommandActionResult,
    PluginCommandActionSpec,
    PluginRuntimeFacade,
)
from fast_agent.command_actions.config import normalize_plugin_command_name
from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import compact as compact_handlers
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_manager_handlers
from fast_agent.commands.handlers import plugins as plugins_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import session_export as session_export_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.results import CommandOutcome
from fast_agent.commands.session_export_help import render_session_export_help_markdown
from fast_agent.commands.shared_command_intents import should_default_export_agent
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.command_payloads import (
    AgentCommand,
    AttachCommand,
    CardsCommand,
    CheckCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandError,
    CommandPayload,
    CommandsCommand,
    CompactCommand,
    CreateSessionCommand,
    ExportSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryViewCommand,
    HistoryWebClearCommand,
    InterruptCommand,
    ListPromptsCommand,
    ListSessionsCommand,
    ListSkillsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpReconnectCommand,
    ModelFastCommand,
    ModelReasoningCommand,
    ModelsCommand,
    ModelSwitchCommand,
    ModelTaskBudgetCommand,
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
    ModelXSearchCommand,
    PinSessionCommand,
    PluginsCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
    TitleSessionCommand,
    UnknownCommand,
)
from fast_agent.ui.history_display import display_history_show
from fast_agent.ui.progress_display import progress_display
from fast_agent.ui.prompt.attachment_tokens import (
    append_attachment_tokens,
    build_local_attachment_token,
    build_remote_attachment_token,
    is_remote_attachment_reference,
    normalize_local_attachment_reference,
    strip_local_attachment_tokens,
)
from fast_agent.utils.slash_commands import parse_slash_command_line

from .command_context import build_command_context, emit_command_outcome
from .mcp_connect_flow import handle_mcp_connect

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.command_actions.models import PluginCommandAgentProtocol
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.interactive_prompt import InteractivePrompt

logger = get_logger(__name__)


def _print_styled(message: str, style: str) -> None:
    rich_print(Text(message, style=style))


def _plugin_error_text(command_name: str, suffix: str, exc: Exception) -> Text:
    message = Text("Command /", style="red")
    message.append(command_name)
    message.append(suffix, style="red")
    message.append(str(exc))
    return message


@dataclass
class DispatchResult:
    handled: bool = False
    next_agent: str | None = None
    buffer_prefill: str | None = None
    hash_send_target: str | None = None
    hash_send_message: str | None = None
    hash_send_quiet: bool = False
    shell_execute_cmd: str | None = None
    should_return: bool = False
    available_agents: list[str] | None = None
    available_agents_set: set[str] | None = None


@dataclass(frozen=True)
class _PluginCommandRequest:
    command_name: str
    arguments: str
    agent: "PluginCommandAgentProtocol"
    spec: PluginCommandActionSpec
    base_path: Path | None


@dataclass(frozen=True)
class _DispatchStep:
    name: str
    run: Callable[[], Awaitable[DispatchResult | None]]


CommandOutcomeHandler = Callable[[CommandContext], Awaitable[CommandOutcome]]
CommandHandlerFunction = Callable[..., Awaitable[CommandOutcome]]
CatalogActionCommand = SkillsCommand | CardsCommand | PluginsCommand | ModelsCommand
_CommandRouteGroup = Literal["catalog", "display", "mcp", "model"]
_CommandRouteKind = Literal[
    "agent_name",
    "argument",
    "catalog_action",
    "mcp_list",
    "mcp_server",
    "value",
]


class _ValueCommandPayload(Protocol):
    value: str | None


class _ArgumentCommandPayload(Protocol):
    argument: str | None


class _McpServerCommandPayload(Protocol):
    server_name: str | None
    error: str | None


@dataclass(frozen=True)
class _CommandOutcomeRoute:
    payload_type: type[CommandPayload]
    group: _CommandRouteGroup
    kind: _CommandRouteKind
    handler: CommandHandlerFunction


_COMMAND_OUTCOME_ROUTES: tuple[_CommandOutcomeRoute, ...] = (
    _CommandOutcomeRoute(
        ListToolsCommand,
        "catalog",
        "agent_name",
        tools_handlers.handle_list_tools,
    ),
    _CommandOutcomeRoute(
        ListSkillsCommand,
        "catalog",
        "agent_name",
        skills_handlers.handle_list_skills,
    ),
    _CommandOutcomeRoute(
        SkillsCommand,
        "catalog",
        "catalog_action",
        skills_handlers.handle_skills_command,
    ),
    _CommandOutcomeRoute(
        CardsCommand,
        "catalog",
        "catalog_action",
        cards_handlers.handle_cards_command,
    ),
    _CommandOutcomeRoute(
        PluginsCommand,
        "catalog",
        "catalog_action",
        plugins_handlers.handle_plugins_command,
    ),
    _CommandOutcomeRoute(
        ModelsCommand,
        "catalog",
        "catalog_action",
        models_manager_handlers.handle_models_command,
    ),
    _CommandOutcomeRoute(
        ShowUsageCommand,
        "display",
        "agent_name",
        display_handlers.handle_show_usage,
    ),
    _CommandOutcomeRoute(
        ShowSystemCommand,
        "display",
        "agent_name",
        display_handlers.handle_show_system,
    ),
    _CommandOutcomeRoute(
        ShowMarkdownCommand,
        "display",
        "agent_name",
        display_handlers.handle_show_markdown,
    ),
    _CommandOutcomeRoute(
        ShowMcpStatusCommand,
        "display",
        "agent_name",
        display_handlers.handle_show_mcp_status,
    ),
    _CommandOutcomeRoute(
        CheckCommand,
        "display",
        "argument",
        display_handlers.handle_check,
    ),
    _CommandOutcomeRoute(
        CommandsCommand,
        "display",
        "argument",
        display_handlers.handle_commands,
    ),
    _CommandOutcomeRoute(
        McpListCommand,
        "mcp",
        "mcp_list",
        mcp_runtime_handlers.handle_mcp_list,
    ),
    _CommandOutcomeRoute(
        McpDisconnectCommand,
        "mcp",
        "mcp_server",
        mcp_runtime_handlers.handle_mcp_disconnect,
    ),
    _CommandOutcomeRoute(
        McpReconnectCommand,
        "mcp",
        "mcp_server",
        mcp_runtime_handlers.handle_mcp_reconnect,
    ),
    _CommandOutcomeRoute(
        ModelReasoningCommand,
        "model",
        "value",
        model_handlers.handle_model_reasoning,
    ),
    _CommandOutcomeRoute(
        ModelTaskBudgetCommand,
        "model",
        "value",
        model_handlers.handle_model_task_budget,
    ),
    _CommandOutcomeRoute(
        ModelVerbosityCommand,
        "model",
        "value",
        model_handlers.handle_model_verbosity,
    ),
    _CommandOutcomeRoute(
        ModelFastCommand,
        "model",
        "value",
        model_handlers.handle_model_fast,
    ),
    _CommandOutcomeRoute(
        ModelWebSearchCommand,
        "model",
        "value",
        model_handlers.handle_model_web_search,
    ),
    _CommandOutcomeRoute(
        ModelXSearchCommand,
        "model",
        "value",
        model_handlers.handle_model_x_search,
    ),
    _CommandOutcomeRoute(
        ModelWebFetchCommand,
        "model",
        "value",
        model_handlers.handle_model_web_fetch,
    ),
)
_COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE: dict[type[CommandPayload], _CommandOutcomeRoute] = {
    route.payload_type: route for route in _COMMAND_OUTCOME_ROUTES
}


def _without_context(
    handler: Callable[[], Awaitable[CommandOutcome]],
) -> CommandOutcomeHandler:
    async def run(_context: CommandContext) -> CommandOutcome:
        return await handler()

    return run


def _is_catalog_action_command(payload: CommandPayload) -> TypeGuard[CatalogActionCommand]:
    return isinstance(payload, (SkillsCommand, CardsCommand, PluginsCommand, ModelsCommand))


def _command_route(
    payload: CommandPayload,
    *,
    group: _CommandRouteGroup,
) -> _CommandOutcomeRoute | None:
    route = _COMMAND_OUTCOME_ROUTE_BY_PAYLOAD_TYPE.get(type(payload))
    if route is None or route.group != group:
        return None
    return route


def _command_route_handler(
    payload: CommandPayload,
    *,
    route: _CommandOutcomeRoute,
    prompt_provider: "AgentApp",
    agent: str,
) -> CommandOutcomeHandler | None:
    if route.kind == "agent_name":
        return partial(route.handler, agent_name=agent)

    if route.kind == "argument":
        argument_payload = cast("_ArgumentCommandPayload", payload)
        return partial(
            route.handler,
            agent_name=agent,
            argument=argument_payload.argument,
        )

    if route.kind == "catalog_action":
        if not _is_catalog_action_command(payload):
            return None
        if isinstance(payload, ModelsCommand):
            return partial(
                route.handler,
                agent_name=agent,
                action=payload.action,
                argument=payload.argument,
                command_name=payload.command_name,
            )
        return partial(
            route.handler,
            agent_name=agent,
            action=payload.action,
            argument=payload.argument,
        )

    if route.kind == "mcp_list":
        return _without_context(
            partial(
                route.handler,
                manager=prompt_provider,
                agent_name=agent,
            )
        )

    if route.kind == "mcp_server":
        server_payload = cast("_McpServerCommandPayload", payload)
        if message := _mcp_server_command_error(server_payload.server_name, server_payload.error):
            _print_styled(message, "red")
            return None
        return partial(
            route.handler,
            manager=prompt_provider,
            agent_name=agent,
            server_name=cast("str", server_payload.server_name),
        )

    value_payload = cast("_ValueCommandPayload", payload)
    return partial(route.handler, agent_name=agent, value=value_payload.value)


async def _run_command_handler(
    *,
    prompt_provider: "AgentApp",
    agent: str,
    handler: CommandOutcomeHandler,
) -> CommandOutcome:
    context = build_command_context(prompt_provider, agent)
    outcome = await handler(context)
    await emit_command_outcome(context, outcome)
    return outcome


async def _local_attach_paths(
    *,
    prompt_provider: "AgentApp",
    agent_name: str,
    paths: tuple[str, ...],
) -> list[str] | None:
    if paths:
        return list(paths)

    context = build_command_context(prompt_provider, agent_name)
    prompted_path = await context.io.prompt_text(
        "Attach file path or HTTP(S) URL:",
        allow_empty=False,
    )
    if not prompted_path:
        return None
    return [prompted_path]


def _attachment_token(raw_path: str, *, shell_working_dir: "Path | None") -> str:
    if is_remote_attachment_reference(raw_path):
        return build_remote_attachment_token(raw_path)

    attachment_path = normalize_local_attachment_reference(
        raw_path,
        cwd=shell_working_dir,
    )
    if not attachment_path.exists():
        raise FileNotFoundError(raw_path)
    if not attachment_path.is_file():
        raise IsADirectoryError(raw_path)
    return build_local_attachment_token(attachment_path)


def _attachment_tokens(
    paths: list[str],
    *,
    shell_working_dir: "Path | None",
) -> list[str]:
    tokens: list[str] = []
    for raw_path in paths:
        try:
            tokens.append(_attachment_token(raw_path, shell_working_dir=shell_working_dir))
        except Exception as exc:
            _print_styled(f"Unable to attach '{raw_path}': {exc}", "red")
    return tokens


async def _dispatch_attach_command(
    payload: AttachCommand,
    *,
    prompt_provider: "AgentApp",
    agent_name: str,
    buffer_prefill: str,
    shell_working_dir: "Path | None",
) -> DispatchResult:
    result = DispatchResult(handled=True)
    if payload.error:
        _print_styled(payload.error, "red")
        return result

    if payload.clear:
        result.buffer_prefill = strip_local_attachment_tokens(buffer_prefill)
        return result

    paths = await _local_attach_paths(
        prompt_provider=prompt_provider,
        agent_name=agent_name,
        paths=payload.paths,
    )
    if paths is None:
        result.buffer_prefill = buffer_prefill
        return result

    tokens = _attachment_tokens(paths, shell_working_dir=shell_working_dir)
    result.buffer_prefill = append_attachment_tokens(buffer_prefill, tokens)
    return result


async def _dispatch_switch_agent_command(
    payload: SwitchAgentCommand,
    *,
    prompt_provider: "AgentApp",
    available_agents_set: set[str],
) -> DispatchResult:
    result = DispatchResult(handled=True)
    if payload.agent_name in available_agents_set:
        result.next_agent = payload.agent_name
        rich_print()
        await enhanced_prompt._display_agent_info_helper(payload.agent_name, prompt_provider)
        return result

    _print_styled(f"Agent '{payload.agent_name}' not found", "red")
    return result


def _dispatch_hash_agent_command(
    payload: HashAgentCommand,
    *,
    available_agents_set: set[str],
) -> DispatchResult:
    result = DispatchResult(handled=True)
    if payload.agent_name not in available_agents_set:
        _print_styled(f"Agent '{payload.agent_name}' not found", "red")
        return result
    if not payload.message:
        prefix = "##" if payload.quiet else "#"
        _print_styled(f"Usage: {prefix}{payload.agent_name} <message>", "yellow")
        return result

    result.hash_send_target = payload.agent_name
    result.hash_send_message = payload.message
    result.hash_send_quiet = payload.quiet
    return result


async def _dispatch_local_ui_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    available_agents_set: set[str],
    agent_name: str,
    buffer_prefill: str,
    shell_working_dir: Path | None = None,
) -> DispatchResult | None:
    match payload:
        case InterruptCommand():
            raise KeyboardInterrupt()
        case SwitchAgentCommand():
            return await _dispatch_switch_agent_command(
                payload,
                prompt_provider=prompt_provider,
                available_agents_set=available_agents_set,
            )
        case HashAgentCommand():
            return _dispatch_hash_agent_command(
                payload,
                available_agents_set=available_agents_set,
            )
        case AttachCommand():
            return await _dispatch_attach_command(
                payload,
                prompt_provider=prompt_provider,
                agent_name=agent_name,
                buffer_prefill=buffer_prefill,
                shell_working_dir=shell_working_dir,
            )
        case _:
            return _dispatch_simple_local_ui_payload(payload)


def _dispatch_simple_local_ui_payload(payload: CommandPayload) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ShellCommand(command=shell_cmd):
            result.shell_execute_cmd = shell_cmd
        case UnknownCommand(command=command):
            _print_styled(f"Command not found: {command}", "red")
        case CommandError(message=message):
            _print_styled(message, "red")
        case _:
            return None
    return result


async def _dispatch_prompt_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    handler = _prompt_handler(payload, agent=agent)
    if handler is None:
        return None

    outcome = await _run_command_handler(
        prompt_provider=prompt_provider,
        agent=agent,
        handler=handler,
    )
    result = DispatchResult(handled=True)
    if isinstance(payload, (SelectPromptCommand, LoadPromptCommand)):
        result.buffer_prefill = outcome.buffer_prefill
    return result


def _prompt_handler(
    payload: CommandPayload,
    *,
    agent: str,
) -> CommandOutcomeHandler | None:
    match payload:
        case ListPromptsCommand():
            return partial(prompt_handlers.handle_list_prompts, agent_name=agent)
        case SelectPromptCommand(prompt_name=prompt_name, prompt_index=prompt_index):
            return partial(
                prompt_handlers.handle_select_prompt,
                agent_name=agent,
                requested_name=prompt_name,
                prompt_index=prompt_index,
            )
        case LoadPromptCommand(filename=filename, error=error):
            return partial(
                prompt_handlers.handle_load_prompt,
                agent_name=agent,
                filename=filename,
                error=error,
            )
        case _:
            return None


async def _dispatch_catalog_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    handler = _catalog_handler(payload, prompt_provider=prompt_provider, agent=agent)
    if handler is None:
        return None

    await _run_command_handler(
        prompt_provider=prompt_provider,
        agent=agent,
        handler=handler,
    )
    return DispatchResult(handled=True)


def _catalog_handler(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> CommandOutcomeHandler | None:
    route = _command_route(payload, group="catalog")
    if route is None:
        return None
    return _command_route_handler(
        payload,
        route=route,
        prompt_provider=prompt_provider,
        agent=agent,
    )


async def _dispatch_display_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    route = _command_route(payload, group="display")
    if route is None:
        return None
    handler = _command_route_handler(
        payload,
        route=route,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if handler is None:
        return None

    await _run_command_handler(
        prompt_provider=prompt_provider,
        agent=agent,
        handler=handler,
    )
    return DispatchResult(handled=True)


def _history_command_target_agent(payload: CommandPayload) -> str | None:
    if isinstance(
        payload,
        (HistoryViewCommand, HistoryFixCommand, HistoryWebClearCommand, ClearCommand),
    ):
        return payload.agent
    return None


def _history_handler(
    payload: CommandPayload,
    *,
    agent: str,
) -> CommandOutcomeHandler | None:
    match payload:
        case HistoryViewCommand(view="overview", agent=target_agent):
            handler = partial(
                history_handlers.handle_show_history,
                agent_name=agent,
                target_agent=target_agent,
            )
        case SaveHistoryCommand(filename=filename):
            handler = partial(
                history_handlers.handle_history_save,
                agent_name=agent,
                filename=filename,
                send_func=None,
            )
        case LoadHistoryCommand(filename=filename, error=error):
            handler = partial(
                history_handlers.handle_history_load,
                agent_name=agent,
                filename=filename,
                error=error,
            )
        case HistoryRewindCommand(turn_index=turn_index, error=error):
            handler = partial(
                history_handlers.handle_history_rewind,
                agent_name=agent,
                turn_index=turn_index,
                error=error,
            )
        case HistoryReviewCommand(turn_index=turn_index, error=error):
            handler = partial(
                history_handlers.handle_history_review,
                agent_name=agent,
                turn_index=turn_index,
                error=error,
            )
        case HistoryFixCommand(agent=target_agent):
            handler = partial(
                history_handlers.handle_history_fix,
                agent_name=agent,
                target_agent=target_agent,
            )
        case HistoryWebClearCommand(agent=target_agent):
            handler = partial(
                history_handlers.handle_history_webclear,
                agent_name=agent,
                target_agent=target_agent,
            )
        case ClearCommand(kind="clear_last", agent=target_agent):
            handler = partial(
                history_handlers.handle_history_clear_last,
                agent_name=agent,
                target_agent=target_agent,
            )
        case ClearCommand(kind="clear_history", agent=target_agent):
            handler = partial(
                history_handlers.handle_history_clear_all,
                agent_name=agent,
                target_agent=target_agent,
            )
        case _:
            handler = None
    return handler


def _history_target_missing(
    owner: "InteractivePrompt",
    *,
    prompt_provider: "AgentApp",
    payload: CommandPayload,
) -> bool:
    target_agent = _history_command_target_agent(payload)
    return bool(target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None)


def _dispatch_history_show_command(
    owner: "InteractivePrompt",
    payload: HistoryViewCommand,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult:
    result = DispatchResult(handled=True)
    target_name = payload.agent or agent
    target = owner._get_history_agent_or_warn(prompt_provider, target_name)
    if target is None:
        return result

    history = list(target.message_history)
    usage = target.usage_accumulator
    display_history_show(target_name, history, usage)
    return result


async def _dispatch_mcp_connect_command(
    payload: McpConnectCommand,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult:
    result = DispatchResult(handled=True)
    if payload.error:
        _print_styled(payload.error, "red")
        return result
    if payload.request is None:
        rich_print("[red]Connection target is required[/red]")
        return result

    context = build_command_context(prompt_provider, agent)
    outcome = await handle_mcp_connect(
        context=context,
        prompt_provider=prompt_provider,
        agent=agent,
        request=payload.request,
    )
    if outcome is not None:
        await emit_command_outcome(context, outcome)
    return result


def _mcp_server_command_error(server_name: str | None, error: str | None) -> str | None:
    if error:
        return error
    if not server_name:
        return "Server name is required"
    return None


def _mcp_handler(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> CommandOutcomeHandler | None:
    route = _command_route(payload, group="mcp")
    if route is None:
        return None
    return _command_route_handler(
        payload,
        route=route,
        prompt_provider=prompt_provider,
        agent=agent,
    )


async def _dispatch_compact_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    if not isinstance(payload, CompactCommand):
        return None

    result = DispatchResult(handled=True)
    context = build_command_context(prompt_provider, agent)

    if payload.action == "run":
        # Resume the streaming token progress display around the summarization
        # call so /compact shows tokens arriving, the same as a normal turn.
        progress_display.resume()
        try:
            outcome = await compact_handlers.handle_compact(
                context,
                agent_name=agent,
                instructions=payload.instructions,
            )
        finally:
            progress_display.pause(cancel_deferred_on_noop=True)
    elif payload.action == "preview":
        outcome = await compact_handlers.handle_compact_preview(context, agent_name=agent)
    else:  # "prompt"
        outcome = await compact_handlers.handle_compact_prompt(context, agent_name=agent)

    await emit_command_outcome(context, outcome)
    return result


async def _dispatch_history_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case HistoryViewCommand(view="table"):
            return _dispatch_history_show_command(
                owner,
                payload,
                prompt_provider=prompt_provider,
                agent=agent,
            )
        case _:
            handler = _history_handler(payload, agent=agent)
            if handler is None:
                return None
            if _history_target_missing(
                owner,
                prompt_provider=prompt_provider,
                payload=payload,
            ):
                return result
            outcome = await _run_command_handler(
                prompt_provider=prompt_provider,
                agent=agent,
                handler=handler,
            )
            if isinstance(payload, HistoryRewindCommand):
                result.buffer_prefill = outcome.buffer_prefill
            return result


async def _dispatch_mcp_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case McpConnectCommand():
            return await _dispatch_mcp_connect_command(
                payload,
                prompt_provider=prompt_provider,
                agent=agent,
            )
        case _:
            handler = _mcp_handler(
                payload,
                prompt_provider=prompt_provider,
                agent=agent,
            )
            if handler is None:
                if isinstance(
                    payload,
                    (
                        McpListCommand,
                        McpDisconnectCommand,
                        McpReconnectCommand,
                    ),
                ):
                    return result
                return None
            await _run_command_handler(
                prompt_provider=prompt_provider,
                agent=agent,
                handler=handler,
            )
            return result


async def _dispatch_model_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    handler = _model_handler(payload, prompt_provider=prompt_provider, agent=agent)
    if handler is not None:
        await _run_command_handler(
            prompt_provider=prompt_provider,
            agent=agent,
            handler=handler,
        )
        return result

    if not isinstance(payload, ModelSwitchCommand):
        return None

    context = build_command_context(prompt_provider, agent)
    outcome = await model_handlers.handle_model_switch(
        context,
        agent_name=agent,
        value=payload.value,
    )
    await model_handlers.apply_model_switch_session_reset(context, outcome)
    await emit_command_outcome(context, outcome)
    return result


def _model_handler(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> CommandOutcomeHandler | None:
    route = _command_route(payload, group="model")
    if route is None:
        return None
    return _command_route_handler(
        payload,
        route=route,
        prompt_provider=prompt_provider,
        agent=agent,
    )


async def _dispatch_create_session_command(
    payload: CreateSessionCommand,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult:
    result = DispatchResult(handled=True)
    context = build_command_context(prompt_provider, agent)
    outcome = await sessions_handlers.handle_create_session(
        context,
        session_name=payload.session_name,
    )
    sessions_handlers.apply_session_new_history_reset(context, outcome)
    await emit_command_outcome(context, outcome)
    return result


def _session_handler(
    payload: CommandPayload,
    *,
    agent: str,
) -> CommandOutcomeHandler | None:
    match payload:
        case ListSessionsCommand(show_help=show_help):
            handler = partial(
                sessions_handlers.handle_list_sessions,
                show_help=show_help,
            )
        case ClearSessionsCommand(target=target):
            handler = partial(
                sessions_handlers.handle_clear_sessions,
                target=target,
            )
        case PinSessionCommand(value=value, target=target):
            handler = partial(
                sessions_handlers.handle_pin_session,
                value=value,
                target=target,
            )
        case ResumeSessionCommand(session_id=session_id):
            handler = partial(
                sessions_handlers.handle_resume_session,
                agent_name=agent,
                session_id=session_id,
            )
        case TitleSessionCommand(title=title):
            handler = partial(
                sessions_handlers.handle_title_session,
                title=title,
            )
        case ForkSessionCommand(title=title):
            handler = partial(
                sessions_handlers.handle_fork_session,
                title=title,
            )
        case _:
            handler = None
    return handler


def _active_session_id_or_empty(context: CommandContext, target: str | None) -> str | None:
    if context.noenv:
        return None

    manager = context.resolve_session_manager()
    current_session = manager.current_session
    current_session_id = current_session.info.name if current_session is not None else None
    if target is None and current_session_id is None:
        return ""
    return current_session_id


async def _dispatch_session_export_command(
    payload: ExportSessionCommand,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult:
    result = DispatchResult(handled=True)
    context = build_command_context(prompt_provider, agent)
    if payload.show_help:
        outcome = CommandOutcome()
        outcome.add_message(render_session_export_help_markdown(), render_markdown=True)
        await emit_command_outcome(context, outcome)
        return result

    current_session_id = _active_session_id_or_empty(context, payload.target)
    if current_session_id == "":
        outcome = CommandOutcome()
        outcome.add_message(
            "No active session to export.",
            channel="error",
            right_info="session",
        )
        await emit_command_outcome(context, outcome)
        return result

    resolved_agent_name = payload.agent_name
    if resolved_agent_name is None and should_default_export_agent(
        payload.target,
        current_session_id=current_session_id,
    ):
        resolved_agent_name = agent

    outcome = await session_export_handlers.handle_session_export(
        context,
        target=payload.target,
        agent_name=resolved_agent_name,
        output_path=payload.output_path,
        hf_url=payload.hf_url,
        hf_dataset=payload.hf_dataset,
        hf_dataset_path=payload.hf_dataset_path,
        privacy_filter=payload.privacy_filter,
        privacy_filter_path=payload.privacy_filter_path,
        download_privacy_filter=payload.download_privacy_filter,
        privacy_filter_device=payload.privacy_filter_device,
        privacy_filter_variant=payload.privacy_filter_variant,
        show_redactions=payload.show_redactions,
        current_session_id=current_session_id,
        error=payload.error,
    )
    await emit_command_outcome(context, outcome)
    return result


async def _dispatch_session_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case CreateSessionCommand():
            return await _dispatch_create_session_command(
                payload,
                prompt_provider=prompt_provider,
                agent=agent,
            )
        case ExportSessionCommand():
            return await _dispatch_session_export_command(
                payload,
                prompt_provider=prompt_provider,
                agent=agent,
            )
        case _:
            handler = _session_handler(payload, agent=agent)
            if handler is None:
                return None
            outcome = await _run_command_handler(
                prompt_provider=prompt_provider,
                agent=agent,
                handler=handler,
            )
            if isinstance(payload, ResumeSessionCommand) and outcome.switch_agent:
                result.next_agent = outcome.switch_agent
            return result


def _refresh_available_agents(
    owner: "InteractivePrompt",
    prompt_provider: "AgentApp",
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> tuple[list[str], set[str]]:
    base_agent_names = list(prompt_provider.visible_agent_names())
    next_available_agents = merge_pinned_agents(base_agent_names)
    force_include = next_available_agents[0] if next_available_agents else None
    owner.agent_types = prompt_provider.visible_agent_types(force_include=force_include)
    next_available_agents_set = set(next_available_agents)
    enhanced_prompt.available_agents = set(next_available_agents)
    return next_available_agents, next_available_agents_set


def _apply_refresh_preferences(
    *,
    prompt_provider: "AgentApp",
    current_agent: str,
    next_available_agents: list[str],
    next_available_agents_set: set[str],
) -> str | None:
    refresh_result = prompt_provider.latest_refresh_result()
    for warning in refresh_result.warnings:
        rich_print(Text(warning, style="yellow"))
    preferred_agent = refresh_result.active_agent
    if preferred_agent and preferred_agent in next_available_agents_set:
        return preferred_agent
    if current_agent in next_available_agents_set:
        return None
    if next_available_agents:
        return next_available_agents[0]
    return None


def _refresh_dispatch_agents(
    result: DispatchResult,
    *,
    owner: "InteractivePrompt",
    prompt_provider: "AgentApp",
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> tuple[list[str], set[str]]:
    next_available_agents, next_available_agents_set = _refresh_available_agents(
        owner,
        prompt_provider,
        merge_pinned_agents,
    )
    result.available_agents = next_available_agents
    result.available_agents_set = next_available_agents_set
    return next_available_agents, next_available_agents_set


async def _dispatch_agent_card_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case LoadAgentCardCommand(
            filename=filename,
            add_tool=add_tool,
            remove_tool=remove_tool,
            error=error,
        ):
            if error:
                _print_styled(error, "red")
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await agent_card_handlers.handle_card_load(
                context,
                manager=prompt_provider,
                filename=filename,
                add_tool=add_tool,
                remove_tool=remove_tool,
                current_agent=agent,
            )
            await emit_command_outcome(context, outcome)
            if outcome.requires_refresh:
                next_available_agents, next_available_agents_set = _refresh_dispatch_agents(
                    result,
                    owner=owner,
                    prompt_provider=prompt_provider,
                    merge_pinned_agents=merge_pinned_agents,
                )
                if agent not in next_available_agents_set:
                    if next_available_agents:
                        result.next_agent = next_available_agents[0]
                    else:
                        rich_print("[red]No agents available after load.[/red]")
                        result.should_return = True
            return result
        case AgentCommand(
            agent_name=agent_name,
            add_tool=add_tool,
            remove_tool=remove_tool,
            dump=dump,
            error=error,
        ):
            if error:
                _print_styled(error, "red")
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await agent_card_handlers.handle_agent_command(
                context,
                manager=prompt_provider,
                current_agent=agent,
                target_agent=agent_name,
                add_tool=add_tool,
                remove_tool=remove_tool,
                dump=dump,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_reload_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ReloadAgentsCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await agent_card_handlers.handle_reload_agents(
                context,
                manager=prompt_provider,
            )
            await emit_command_outcome(context, outcome)
            if outcome.requires_refresh:
                next_available_agents, next_available_agents_set = _refresh_dispatch_agents(
                    result,
                    owner=owner,
                    prompt_provider=prompt_provider,
                    merge_pinned_agents=merge_pinned_agents,
                )
                next_agent = _apply_refresh_preferences(
                    prompt_provider=prompt_provider,
                    current_agent=agent,
                    next_available_agents=next_available_agents,
                    next_available_agents_set=next_available_agents_set,
                )
                if next_agent is not None:
                    result.next_agent = next_agent
                elif not next_available_agents:
                    rich_print("[red]No agents available after reload.[/red]")
                    result.should_return = True
            return result
        case _:
            return None


async def _first_dispatch_result(
    dispatchers: Sequence[_DispatchStep],
) -> DispatchResult | None:
    for dispatcher in dispatchers:
        result = await dispatcher.run()
        if result is not None:
            return result
    return None


async def dispatch_command_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
    available_agents: list[str],
    available_agents_set: set[str],
    merge_pinned_agents: Callable[[list[str]], list[str]],
    buffer_prefill: str = "",
    shell_working_dir: Path | None = None,
) -> DispatchResult:
    del available_agents

    result = await _first_dispatch_result(
        (
            _DispatchStep(
                name="plugin command fallback",
                run=lambda: _dispatch_plugin_command_payload(
                    owner,
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                    available_agents_set=available_agents_set,
                    merge_pinned_agents=merge_pinned_agents,
                    shell_working_dir=shell_working_dir,
                ),
            ),
            _DispatchStep(
                name="local UI command",
                run=lambda: _dispatch_local_ui_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    available_agents_set=available_agents_set,
                    agent_name=agent,
                    buffer_prefill=buffer_prefill,
                    shell_working_dir=shell_working_dir,
                ),
            ),
            _DispatchStep(
                name="prompt command",
                run=lambda: _dispatch_prompt_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="catalog command",
                run=lambda: _dispatch_catalog_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="display command",
                run=lambda: _dispatch_display_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="compact command",
                run=lambda: _dispatch_compact_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="history command",
                run=lambda: _dispatch_history_payload(
                    owner,
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="mcp command",
                run=lambda: _dispatch_mcp_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="model command",
                run=lambda: _dispatch_model_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="session command",
                run=lambda: _dispatch_session_payload(
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                ),
            ),
            _DispatchStep(
                name="agent/card command",
                run=lambda: _dispatch_agent_card_payload(
                    owner,
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                    merge_pinned_agents=merge_pinned_agents,
                ),
            ),
            _DispatchStep(
                name="reload command",
                run=lambda: _dispatch_reload_payload(
                    owner,
                    payload,
                    prompt_provider=prompt_provider,
                    agent=agent,
                    merge_pinned_agents=merge_pinned_agents,
                ),
            ),
        )
    )
    return result or DispatchResult(handled=False)


def _parse_unknown_plugin_command(payload: UnknownCommand) -> tuple[str, str] | None:
    parsed = parse_slash_command_line(payload.command)
    if parsed is None:
        return None

    command_name, arguments = parsed
    command_name = normalize_plugin_command_name(command_name)
    if not command_name:
        return None
    return command_name, arguments


def _resolve_plugin_command_spec(
    *,
    current_agent: "PluginCommandAgentProtocol",
    prompt_provider: "AgentApp",
    command_name: str,
) -> tuple[PluginCommandActionSpec, Path | None] | None:
    agent_commands = current_agent.config.commands
    if agent_commands is not None:
        spec = agent_commands.get(command_name)
        if spec is not None:
            base_path = None
            if current_agent.config.source_path is not None:
                base_path = current_agent.config.source_path.parent
            return spec, base_path

    if prompt_provider.plugin_commands is None:
        return None

    spec = prompt_provider.plugin_commands.get(command_name)
    if spec is None:
        return None
    return spec, prompt_provider.plugin_command_base_path


def _plugin_command_request(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> _PluginCommandRequest | None:
    if not isinstance(payload, UnknownCommand):
        return None

    parsed_command = _parse_unknown_plugin_command(payload)
    if parsed_command is None:
        return None
    command_name, arguments = parsed_command

    current_agent = prompt_provider.get_agent(agent)
    if current_agent is None:
        return None
    plugin_agent = cast("PluginCommandAgentProtocol", current_agent)

    resolved_command = _resolve_plugin_command_spec(
        current_agent=plugin_agent,
        prompt_provider=prompt_provider,
        command_name=command_name,
    )
    if resolved_command is None:
        return None
    spec, base_path = resolved_command
    return _PluginCommandRequest(
        command_name=command_name,
        arguments=arguments,
        agent=plugin_agent,
        spec=spec,
        base_path=base_path,
    )


def _plugin_runtime_facade(
    prompt_provider: "AgentApp",
    *,
    current_agent_name: str,
) -> PluginRuntimeFacade:
    return PluginRuntimeFacade(
        current_agent_name=current_agent_name,
        attach_mcp_server_callback=prompt_provider.attach_mcp_server,
        detach_mcp_server_callback=prompt_provider.detach_mcp_server,
        list_attached_mcp_servers_callback=prompt_provider.list_attached_mcp_servers,
        list_configured_detached_mcp_servers_callback=(
            prompt_provider.list_configured_detached_mcp_servers
        ),
    )


def _plugin_command_context(
    *,
    command_name: str,
    arguments: str,
    current_agent: "PluginCommandAgentProtocol",
    context: CommandContext,
    prompt_provider: "AgentApp",
    shell_working_dir: Path | None,
) -> PluginCommandActionContext:
    return PluginCommandActionContext(
        command_name=command_name,
        arguments=arguments,
        agent=cast("PluginCommandAgentProtocol", current_agent),
        settings=context.settings,
        session_cwd=shell_working_dir,
        runtime=_plugin_runtime_facade(
            prompt_provider,
            current_agent_name=current_agent.name,
        ),
        is_tui=True,
    )


async def _execute_plugin_command_action(
    *,
    command_name: str,
    spec: PluginCommandActionSpec,
    base_path: Path | None,
    plugin_context: PluginCommandActionContext,
) -> PluginCommandActionResult | None:
    registry = PluginCommandActionRegistry.from_specs(
        {command_name: spec},
        base_path=base_path,
    )
    return await registry.execute(command_name, plugin_context)


def _plugin_action_outcome(
    action_result: PluginCommandActionResult | None,
    *,
    plugin_context: PluginCommandActionContext | None = None,
) -> CommandOutcome:
    if action_result is None:
        return CommandOutcome()

    post_content = None
    if action_result.images and plugin_context is not None:
        from fast_agent.ui.terminal_images import render_plugin_command_images_for_settings

        terminal_images = (
            plugin_context.settings.logger.terminal_images
            if plugin_context.settings is not None
            else None
        )
        post_content = render_plugin_command_images_for_settings(
            terminal_images,
            action_result.images,
            base_dir=plugin_context.session_cwd,
        )

    outcome = CommandOutcome(
        buffer_prefill=action_result.buffer_prefill,
        switch_agent=action_result.switch_agent,
        requires_refresh=action_result.refresh_agents,
    )
    if action_result.markdown:
        outcome.add_message(action_result.markdown, render_markdown=True, post_content=post_content)
    elif action_result.message:
        outcome.add_message(action_result.message, post_content=post_content)
    elif post_content is not None:
        outcome.add_message("", render_markdown=True, post_content=post_content)
    return outcome


def _plugin_dispatch_result(outcome: CommandOutcome) -> DispatchResult:
    return DispatchResult(
        handled=True,
        buffer_prefill=outcome.buffer_prefill,
        next_agent=outcome.switch_agent,
    )


def _refresh_plugin_dispatch_agents(
    result: DispatchResult,
    *,
    owner: "InteractivePrompt",
    prompt_provider: "AgentApp",
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> set[str]:
    next_available_agents, next_available_agents_set = _refresh_available_agents(
        owner,
        prompt_provider,
        merge_pinned_agents,
    )
    result.available_agents = next_available_agents
    result.available_agents_set = next_available_agents_set
    return next_available_agents_set


async def _dispatch_plugin_command_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
    available_agents_set: set[str],
    merge_pinned_agents: Callable[[list[str]], list[str]],
    shell_working_dir: Path | None,
) -> DispatchResult | None:
    request = _plugin_command_request(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if request is None:
        return None

    try:
        context = build_command_context(prompt_provider, agent)
        plugin_context = _plugin_command_context(
            command_name=request.command_name,
            arguments=request.arguments,
            current_agent=request.agent,
            context=context,
            prompt_provider=prompt_provider,
            shell_working_dir=shell_working_dir,
        )
        action_result = await _execute_plugin_command_action(
            command_name=request.command_name,
            spec=request.spec,
            base_path=request.base_path,
            plugin_context=plugin_context,
        )
    except AgentConfigError as exc:
        logger.warning(
            "Failed to load plugin command action",
            command=request.command_name,
            error=str(exc),
        )
        rich_print(_plugin_error_text(request.command_name, " failed to load: ", exc))
        return DispatchResult(handled=True)
    except Exception as exc:
        logger.exception("Plugin command action failed", command=request.command_name)
        rich_print(_plugin_error_text(request.command_name, " failed: ", exc))
        return DispatchResult(handled=True)

    outcome = _plugin_action_outcome(action_result, plugin_context=plugin_context)
    await emit_command_outcome(context, outcome)
    result = _plugin_dispatch_result(outcome)

    if outcome.requires_refresh:
        available_agents_set = _refresh_plugin_dispatch_agents(
            result,
            owner=owner,
            prompt_provider=prompt_provider,
            merge_pinned_agents=merge_pinned_agents,
        )

    if result.next_agent is not None and result.next_agent not in available_agents_set:
        message = Text("Unknown agent: ", style="red")
        message.append(result.next_agent)
        rich_print(message)
        result.next_agent = None

    return result
