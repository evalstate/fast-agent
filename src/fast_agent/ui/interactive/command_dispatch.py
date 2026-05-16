"""Command payload dispatch for the TUI interactive loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from rich import print as rich_print

from fast_agent.command_actions import (
    PluginCommandActionContext,
    PluginCommandActionRegistry,
    PluginRuntimeFacade,
)
from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_manager_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import session_export as session_export_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
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
    ClearCommand,
    ClearSessionsCommand,
    CommandPayload,
    CreateSessionCommand,
    ExportSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryShowCommand,
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
    McpSessionCommand,
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
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowHistoryCommand,
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
from fast_agent.ui.prompt.attachment_tokens import (
    append_attachment_tokens,
    build_local_attachment_token,
    build_remote_attachment_token,
    normalize_local_attachment_reference,
    normalize_remote_attachment_reference,
    strip_local_attachment_tokens,
)

from .command_context import build_command_context, emit_command_outcome
from .mcp_connect_flow import handle_mcp_connect

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.command_actions.models import PluginCommandAgentProtocol
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.interactive_prompt import InteractivePrompt

logger = get_logger(__name__)


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
    return_result: str = ""
    available_agents: list[str] | None = None
    available_agents_set: set[str] | None = None


async def _apply_model_switch_session_reset(
    *,
    context,
    prompt_provider,
    outcome,
) -> None:
    if not outcome.reset_session:
        return

    if not context.noenv:
        outcome.add_message(
            "Model switch starts a new session to avoid mixing histories.",
            channel="info",
        )
        session_outcome = await sessions_handlers.handle_create_session(
            context,
            session_name=None,
        )
        outcome.messages.extend(session_outcome.messages)
    else:
        outcome.add_message(
            "Model switch cleared in-memory history (--noenv disables session persistence).",
            channel="info",
        )

    cleared = clear_agent_histories(prompt_provider.registered_agents())
    if cleared:
        outcome.add_message(
            f"Cleared agent history: {', '.join(sorted(cleared))}",
            channel="info",
        )


async def _dispatch_local_ui_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    available_agents_set: set[str],
    agent_name: str,
    buffer_prefill: str,
    shell_working_dir: Path | None = None,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case InterruptCommand():
            raise KeyboardInterrupt()
        case SwitchAgentCommand(agent_name=new_agent):
            if new_agent in available_agents_set:
                result.next_agent = new_agent
                rich_print()
                await enhanced_prompt._display_agent_info_helper(new_agent, prompt_provider)
                return result
            rich_print(f"[red]Agent '{new_agent}' not found[/red]")
            return result
        case HashAgentCommand(agent_name=target_agent, message=hash_message, quiet=quiet):
            if target_agent not in available_agents_set:
                rich_print(f"[red]Agent '{target_agent}' not found[/red]")
                return result
            if not hash_message:
                prefix = "##" if quiet else "#"
                rich_print(f"[yellow]Usage: {prefix}{target_agent} <message>[/yellow]")
                return result
            result.hash_send_target = target_agent
            result.hash_send_message = hash_message
            result.hash_send_quiet = quiet
            return result
        case ShellCommand(command=shell_cmd):
            result.shell_execute_cmd = shell_cmd
            return result
        case AttachCommand(paths=paths, clear=clear, error=error):
            if error:
                rich_print(f"[red]{error}[/red]")
                return result

            if clear:
                result.buffer_prefill = strip_local_attachment_tokens(buffer_prefill)
                return result

            resolved_paths = list(paths)
            if not resolved_paths:
                context = build_command_context(prompt_provider, agent_name)
                prompted_path = await context.io.prompt_text(
                    "Attach file path or HTTP(S) URL:",
                    allow_empty=False,
                )
                if not prompted_path:
                    result.buffer_prefill = buffer_prefill
                    return result
                resolved_paths = [prompted_path]

            tokens: list[str] = []
            for raw_path in resolved_paths:
                try:
                    if raw_path.strip().lower().startswith(("http://", "https://")):
                        token = build_remote_attachment_token(
                            normalize_remote_attachment_reference(raw_path)
                        )
                    else:
                        attachment_path = normalize_local_attachment_reference(
                            raw_path,
                            cwd=shell_working_dir,
                        )
                        if not attachment_path.exists():
                            raise FileNotFoundError(raw_path)
                        if not attachment_path.is_file():
                            raise IsADirectoryError(raw_path)
                        token = build_local_attachment_token(attachment_path)
                except Exception as exc:
                    rich_print(f"[red]Unable to attach '{raw_path}': {exc}[/red]")
                    continue
                tokens.append(token)

            result.buffer_prefill = append_attachment_tokens(buffer_prefill, tokens)
            return result
        case UnknownCommand(command=command):
            rich_print(f"[red]Command not found: {command}[/red]")
            return result
        case _:
            return None


async def _dispatch_prompt_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ListPromptsCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await prompt_handlers.handle_list_prompts(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case SelectPromptCommand(prompt_name=prompt_name, prompt_index=prompt_index):
            context = build_command_context(prompt_provider, agent)
            outcome = await prompt_handlers.handle_select_prompt(
                context,
                agent_name=agent,
                requested_name=prompt_name,
                prompt_index=prompt_index,
            )
            await emit_command_outcome(context, outcome)
            result.buffer_prefill = outcome.buffer_prefill
            return result
        case LoadPromptCommand(filename=filename, error=error):
            context = build_command_context(prompt_provider, agent)
            outcome = await prompt_handlers.handle_load_prompt(
                context,
                agent_name=agent,
                filename=filename,
                error=error,
            )
            await emit_command_outcome(context, outcome)
            result.buffer_prefill = outcome.buffer_prefill
            return result
        case _:
            return None


async def _dispatch_catalog_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ListToolsCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await tools_handlers.handle_list_tools(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case ListSkillsCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await skills_handlers.handle_list_skills(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case SkillsCommand(action=action, argument=argument):
            context = build_command_context(prompt_provider, agent)
            outcome = await skills_handlers.handle_skills_command(
                context,
                agent_name=agent,
                action=action,
                argument=argument,
            )
            await emit_command_outcome(context, outcome)
            return result
        case CardsCommand(action=action, argument=argument):
            context = build_command_context(prompt_provider, agent)
            outcome = await cards_handlers.handle_cards_command(
                context,
                agent_name=agent,
                action=action,
                argument=argument,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelsCommand(action=action, argument=argument):
            context = build_command_context(prompt_provider, agent)
            outcome = await models_manager_handlers.handle_models_command(
                context,
                agent_name=agent,
                action=action,
                argument=argument,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_display_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ShowUsageCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await display_handlers.handle_show_usage(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case ShowSystemCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await display_handlers.handle_show_system(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case ShowMarkdownCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await display_handlers.handle_show_markdown(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case ShowMcpStatusCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await display_handlers.handle_show_mcp_status(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_history_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ShowHistoryCommand(agent=target_agent):
            if target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None:
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_show_history(
                context,
                agent_name=agent,
                target_agent=target_agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case HistoryShowCommand(agent=target_agent):
            target_name = target_agent or agent
            target = owner._get_history_agent_or_warn(prompt_provider, target_name)
            if target is None:
                return result
            history = list(target.message_history)
            usage = target.usage_accumulator
            display_history_show(target_name, history, usage)
            return result
        case SaveHistoryCommand(filename=filename):
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_save(
                context,
                agent_name=agent,
                filename=filename,
                send_func=None,
            )
            await emit_command_outcome(context, outcome)
            return result
        case LoadHistoryCommand(filename=filename, error=error):
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_load(
                context,
                agent_name=agent,
                filename=filename,
                error=error,
            )
            await emit_command_outcome(context, outcome)
            return result
        case HistoryRewindCommand(turn_index=turn_index, error=error):
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_rewind(
                context,
                agent_name=agent,
                turn_index=turn_index,
                error=error,
            )
            await emit_command_outcome(context, outcome)
            result.buffer_prefill = outcome.buffer_prefill
            return result
        case HistoryReviewCommand(turn_index=turn_index, error=error):
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_review(
                context,
                agent_name=agent,
                turn_index=turn_index,
                error=error,
            )
            await emit_command_outcome(context, outcome)
            return result
        case HistoryFixCommand(agent=target_agent):
            if target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None:
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_fix(
                context,
                agent_name=agent,
                target_agent=target_agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case HistoryWebClearCommand(agent=target_agent):
            if target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None:
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_webclear(
                context,
                agent_name=agent,
                target_agent=target_agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ClearCommand(kind="clear_last", agent=target_agent):
            if target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None:
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_clear_last(
                context,
                agent_name=agent,
                target_agent=target_agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ClearCommand(kind="clear_history", agent=target_agent):
            if target_agent and owner._get_agent_or_warn(prompt_provider, target_agent) is None:
                return result
            context = build_command_context(prompt_provider, agent)
            outcome = await history_handlers.handle_history_clear_all(
                context,
                agent_name=agent,
                target_agent=target_agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_mcp_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case McpListCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await mcp_runtime_handlers.handle_mcp_list(
                context,
                manager=prompt_provider,
                agent_name=agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case McpConnectCommand(request=request, error=error):
            context = build_command_context(prompt_provider, agent)
            if error:
                rich_print(f"[red]{error}[/red]")
                return result
            if request is None:
                rich_print("[red]Connection target is required[/red]")
                return result

            outcome = await handle_mcp_connect(
                context=context,
                prompt_provider=prompt_provider,
                agent=agent,
                request=request,
            )
            if outcome is not None:
                await emit_command_outcome(context, outcome)
            return result
        case McpDisconnectCommand(server_name=server_name, error=error):
            context = build_command_context(prompt_provider, agent)
            if error or not server_name:
                rich_print(f"[red]{error or 'Server name is required'}[/red]")
                return result
            outcome = await mcp_runtime_handlers.handle_mcp_disconnect(
                context,
                manager=prompt_provider,
                agent_name=agent,
                server_name=server_name,
            )
            await emit_command_outcome(context, outcome)
            return result
        case McpReconnectCommand(server_name=server_name, error=error):
            context = build_command_context(prompt_provider, agent)
            if error or not server_name:
                rich_print(f"[red]{error or 'Server name is required'}[/red]")
                return result
            outcome = await mcp_runtime_handlers.handle_mcp_reconnect(
                context,
                manager=prompt_provider,
                agent_name=agent,
                server_name=server_name,
            )
            await emit_command_outcome(context, outcome)
            return result
        case McpSessionCommand(
            action=action,
            server_identity=server_identity,
            session_id=session_id,
            title=title,
            clear_all=clear_all,
            error=error,
        ):
            context = build_command_context(prompt_provider, agent)
            if error:
                rich_print(f"[red]{error}[/red]")
                return result
            outcome = await mcp_runtime_handlers.handle_mcp_session(
                context,
                agent_name=agent,
                action=action,
                server_identity=server_identity,
                session_id=session_id,
                title=title,
                clear_all=clear_all,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_model_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case ModelReasoningCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_reasoning(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelTaskBudgetCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_task_budget(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelVerbosityCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_verbosity(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelFastCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_fast(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelWebSearchCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_web_search(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelXSearchCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_x_search(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelWebFetchCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_web_fetch(
                context,
                agent_name=agent,
                value=value,
            )
            await emit_command_outcome(context, outcome)
            return result
        case ModelSwitchCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_switch(
                context,
                agent_name=agent,
                value=value,
            )
            await _apply_model_switch_session_reset(
                context=context,
                prompt_provider=prompt_provider,
                outcome=outcome,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


async def _dispatch_session_payload(
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
) -> DispatchResult | None:
    result = DispatchResult(handled=True)
    match payload:
        case CreateSessionCommand(session_name=session_name):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_create_session(context, session_name=session_name)
            cleared = clear_agent_histories(prompt_provider.registered_agents())
            if cleared:
                outcome.add_message(f"Cleared agent history: {', '.join(sorted(cleared))}", channel="info")
            await emit_command_outcome(context, outcome)
            return result
        case ListSessionsCommand(show_help=show_help):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_list_sessions(context, show_help=show_help)
            await emit_command_outcome(context, outcome)
            return result
        case ClearSessionsCommand(target=target):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_clear_sessions(context, target=target)
            await emit_command_outcome(context, outcome)
            return result
        case PinSessionCommand(value=value, target=target):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_pin_session(context, value=value, target=target)
            await emit_command_outcome(context, outcome)
            return result
        case ResumeSessionCommand(session_id=session_id):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_resume_session(
                context,
                agent_name=agent,
                session_id=session_id,
            )
            await emit_command_outcome(context, outcome)
            if outcome.switch_agent:
                result.next_agent = outcome.switch_agent
            return result
        case TitleSessionCommand(title=title):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_title_session(context, title=title)
            await emit_command_outcome(context, outcome)
            return result
        case ForkSessionCommand(title=title):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_fork_session(context, title=title)
            await emit_command_outcome(context, outcome)
            return result
        case ExportSessionCommand(
            target=target,
            agent_name=agent_name,
            output_path=output_path,
            hf_dataset=hf_dataset,
            hf_dataset_path=hf_dataset_path,
            privacy_filter=privacy_filter,
            privacy_filter_path=privacy_filter_path,
            download_privacy_filter=download_privacy_filter,
            privacy_filter_device=privacy_filter_device,
            privacy_filter_variant=privacy_filter_variant,
            show_redactions=show_redactions,
            show_help=show_help,
            error=error,
        ):
            context = build_command_context(prompt_provider, agent)
            if show_help:
                outcome = CommandOutcome()
                outcome.add_message(render_session_export_help_markdown(), render_markdown=True)
                await emit_command_outcome(context, outcome)
                return result
            manager = context.resolve_session_manager()
            current_session = manager.current_session
            current_session_id = current_session.info.name if current_session is not None else None
            if target is None and current_session_id is None:
                outcome = CommandOutcome()
                outcome.add_message(
                    "No active session to export.",
                    channel="error",
                    right_info="session",
                )
                await emit_command_outcome(context, outcome)
                return result
            resolved_agent_name = agent_name
            if resolved_agent_name is None and should_default_export_agent(
                target,
                current_session_id=current_session_id,
            ):
                resolved_agent_name = agent
            outcome = await session_export_handlers.handle_session_export(
                context,
                target=target,
                agent_name=resolved_agent_name,
                output_path=output_path,
                hf_dataset=hf_dataset,
                hf_dataset_path=hf_dataset_path,
                privacy_filter=privacy_filter,
                privacy_filter_path=privacy_filter_path,
                download_privacy_filter=download_privacy_filter,
                privacy_filter_device=privacy_filter_device,
                privacy_filter_variant=privacy_filter_variant,
                show_redactions=show_redactions,
                current_session_id=current_session_id,
                error=error,
            )
            await emit_command_outcome(context, outcome)
            return result
        case _:
            return None


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
        rich_print(f"[yellow]{warning}[/yellow]")
    preferred_agent = refresh_result.active_agent
    if preferred_agent and preferred_agent in next_available_agents_set:
        return preferred_agent
    if current_agent in next_available_agents_set:
        return None
    if next_available_agents:
        return next_available_agents[0]
    return None


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
                rich_print(f"[red]{error}[/red]")
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
                next_available_agents, next_available_agents_set = _refresh_available_agents(
                    owner,
                    prompt_provider,
                    merge_pinned_agents,
                )
                result.available_agents = next_available_agents
                result.available_agents_set = next_available_agents_set
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
                rich_print(f"[red]{error}[/red]")
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
            outcome = await agent_card_handlers.handle_reload_agents(context, manager=prompt_provider)
            await emit_command_outcome(context, outcome)
            if outcome.requires_refresh:
                next_available_agents, next_available_agents_set = _refresh_available_agents(
                    owner,
                    prompt_provider,
                    merge_pinned_agents,
                )
                result.available_agents = next_available_agents
                result.available_agents_set = next_available_agents_set
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

    plugin_result = await _dispatch_plugin_command_payload(
        owner,
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
        available_agents_set=available_agents_set,
        merge_pinned_agents=merge_pinned_agents,
        shell_working_dir=shell_working_dir,
    )
    if plugin_result is not None:
        return plugin_result

    local_result = await _dispatch_local_ui_payload(
        payload,
        prompt_provider=prompt_provider,
        available_agents_set=available_agents_set,
        agent_name=agent,
        buffer_prefill=buffer_prefill,
        shell_working_dir=shell_working_dir,
    )
    if local_result is not None:
        return local_result

    prompt_result = await _dispatch_prompt_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if prompt_result is not None:
        return prompt_result

    catalog_result = await _dispatch_catalog_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if catalog_result is not None:
        return catalog_result

    display_result = await _dispatch_display_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if display_result is not None:
        return display_result

    history_result = await _dispatch_history_payload(
        owner,
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if history_result is not None:
        return history_result

    mcp_result = await _dispatch_mcp_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if mcp_result is not None:
        return mcp_result

    model_result = await _dispatch_model_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if model_result is not None:
        return model_result

    session_result = await _dispatch_session_payload(
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
    )
    if session_result is not None:
        return session_result

    agent_card_result = await _dispatch_agent_card_payload(
        owner,
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
        merge_pinned_agents=merge_pinned_agents,
    )
    if agent_card_result is not None:
        return agent_card_result

    reload_result = await _dispatch_reload_payload(
        owner,
        payload,
        prompt_provider=prompt_provider,
        agent=agent,
        merge_pinned_agents=merge_pinned_agents,
    )
    if reload_result is not None:
        return reload_result

    return DispatchResult(handled=False)


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
    if not isinstance(payload, UnknownCommand):
        return None

    command_line = payload.command.strip()
    if not command_line.startswith("/"):
        return None

    command_name, _, arguments = command_line[1:].partition(" ")
    command_name = command_name.strip()
    arguments = arguments.lstrip()
    if not command_name:
        return None

    current_agent = prompt_provider.get_agent(agent)
    if current_agent is None:
        return None

    spec = None
    base_path = None
    agent_commands = current_agent.config.commands
    if agent_commands is not None:
        spec = agent_commands.get(command_name)
        if spec is not None and current_agent.config.source_path is not None:
            base_path = current_agent.config.source_path.parent

    if spec is None and prompt_provider.plugin_commands is not None:
        spec = prompt_provider.plugin_commands.get(command_name)
        base_path = prompt_provider.plugin_command_base_path

    if spec is None:
        return None

    try:
        registry = PluginCommandActionRegistry.from_specs(
            {command_name: spec},
            base_path=base_path,
        )
        context = build_command_context(prompt_provider, agent)
        plugin_context = PluginCommandActionContext(
            command_name=command_name,
            arguments=arguments,
            agent=cast("PluginCommandAgentProtocol", current_agent),
            settings=context.settings,
            session_cwd=shell_working_dir,
            runtime=PluginRuntimeFacade(
                current_agent_name=current_agent.name,
                attach_mcp_server_callback=prompt_provider.attach_mcp_server,
                detach_mcp_server_callback=prompt_provider.detach_mcp_server,
                list_attached_mcp_servers_callback=prompt_provider.list_attached_mcp_servers,
                list_configured_detached_mcp_servers_callback=(
                    prompt_provider.list_configured_detached_mcp_servers
                ),
            ),
            is_tui=True,
        )
        action_result = await registry.execute(command_name, plugin_context)
    except AgentConfigError as exc:
        logger.warning("Failed to load plugin command action", command=command_name, error=str(exc))
        rich_print(f"[red]Command /{command_name} failed to load:[/red] {exc}")
        return DispatchResult(handled=True)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Plugin command action failed", command=command_name)
        rich_print(f"[red]Command /{command_name} failed:[/red] {exc}")
        return DispatchResult(handled=True)

    if action_result is None:
        return DispatchResult(handled=True)

    outcome = CommandOutcome(
        buffer_prefill=action_result.buffer_prefill,
        switch_agent=action_result.switch_agent,
        requires_refresh=action_result.refresh_agents,
    )
    if action_result.markdown:
        outcome.add_message(action_result.markdown, render_markdown=True)
    elif action_result.message:
        outcome.add_message(action_result.message)

    await emit_command_outcome(context, outcome)

    result = DispatchResult(
        handled=True,
        buffer_prefill=outcome.buffer_prefill,
        next_agent=outcome.switch_agent,
    )

    if outcome.requires_refresh:
        next_available_agents, next_available_agents_set = _refresh_available_agents(
            owner,
            prompt_provider,
            merge_pinned_agents,
        )
        result.available_agents = next_available_agents
        result.available_agents_set = next_available_agents_set
        available_agents_set = next_available_agents_set

    if result.next_agent is not None and result.next_agent not in available_agents_set:
        rich_print(f"[red]Unknown agent:[/red] {result.next_agent}")
        result.next_agent = None

    return result
