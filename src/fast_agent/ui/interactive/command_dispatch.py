"""Command payload dispatch for the TUI interactive loop."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rich import print as rich_print

from fast_agent.commands.handlers import agent_cards as agent_card_handlers
from fast_agent.commands.handlers import cards_manager as cards_handlers
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.commands.handlers import history as history_handlers
from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.commands.handlers import model as model_handlers
from fast_agent.commands.handlers import models_manager as models_manager_handlers
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.commands.handlers import sessions as sessions_handlers
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.handlers.shared import clear_agent_histories
from fast_agent.ui import enhanced_prompt
from fast_agent.ui.command_payloads import (
    AgentCommand,
    CardsCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandPayload,
    CreateSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryWebClearCommand,
    InterruptCommand,
    ListPromptsCommand,
    ListSessionsCommand,
    ListSkillsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpSessionCommand,
    ModelReasoningCommand,
    ModelsCommand,
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
    PinSessionCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
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

from .command_context import build_command_context, emit_command_outcome
from .mcp_connect_flow import handle_mcp_connect

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.interactive_prompt import InteractivePrompt


@dataclass
class DispatchResult:
    handled: bool = False
    next_agent: str | None = None
    buffer_prefill: str | None = None
    hash_send_target: str | None = None
    hash_send_message: str | None = None
    shell_execute_cmd: str | None = None
    should_return: bool = False
    return_result: str = ""
    available_agents: list[str] | None = None
    available_agents_set: set[str] | None = None


async def dispatch_command_payload(
    owner: "InteractivePrompt",
    payload: CommandPayload,
    *,
    prompt_provider: "AgentApp",
    agent: str,
    available_agents: list[str],
    available_agents_set: set[str],
    merge_pinned_agents: Callable[[list[str]], list[str]],
) -> DispatchResult:
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
        case HashAgentCommand(agent_name=target_agent, message=hash_message):
            if target_agent not in available_agents_set:
                rich_print(f"[red]Agent '{target_agent}' not found[/red]")
                return result
            if not hash_message:
                rich_print(f"[yellow]Usage: #{target_agent} <message>[/yellow]")
                return result
            result.hash_send_target = target_agent
            result.hash_send_message = hash_message
            return result
        case ShellCommand(command=shell_cmd):
            result.shell_execute_cmd = shell_cmd
            return result
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
        case ShowUsageCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await display_handlers.handle_show_usage(context, agent_name=agent)
            await emit_command_outcome(context, outcome)
            return result
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
        case McpListCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await mcp_runtime_handlers.handle_mcp_list(
                context,
                manager=prompt_provider,
                agent_name=agent,
            )
            await emit_command_outcome(context, outcome)
            return result
        case McpConnectCommand(
            target_text=target_text,
            server_name=server_name,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            trigger_oauth=trigger_oauth,
            reconnect_on_disconnect=reconnect_on_disconnect,
            force_reconnect=force_reconnect,
            error=error,
        ):
            context = build_command_context(prompt_provider, agent)
            if error:
                rich_print(f"[red]{error}[/red]")
                return result
            runtime_target = target_text
            if server_name:
                runtime_target += f" --name {server_name}"
            if auth_token:
                runtime_target += f" --auth {shlex.quote(auth_token)}"
            if timeout_seconds is not None:
                runtime_target += f" --timeout {timeout_seconds}"
            if trigger_oauth is True:
                runtime_target += " --oauth"
            elif trigger_oauth is False:
                runtime_target += " --no-oauth"
            if reconnect_on_disconnect is False:
                runtime_target += " --no-reconnect"
            if force_reconnect:
                runtime_target += " --reconnect"

            outcome = await handle_mcp_connect(
                context=context,
                prompt_provider=prompt_provider,
                agent=agent,
                runtime_target=runtime_target,
                target_text=target_text,
                server_name=server_name,
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
        case ModelReasoningCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_reasoning(
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
        case ModelWebSearchCommand(value=value):
            context = build_command_context(prompt_provider, agent)
            outcome = await model_handlers.handle_model_web_search(
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
        case CreateSessionCommand(session_name=session_name):
            context = build_command_context(prompt_provider, agent)
            outcome = await sessions_handlers.handle_create_session(context, session_name=session_name)
            cleared = clear_agent_histories(prompt_provider._agents)
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
                next_available_agents = merge_pinned_agents(list(prompt_provider.agent_names()))
                next_available_agents_set = set(next_available_agents)
                owner.agent_types = prompt_provider.agent_types()
                enhanced_prompt.available_agents = set(next_available_agents)
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
        case ReloadAgentsCommand():
            context = build_command_context(prompt_provider, agent)
            outcome = await agent_card_handlers.handle_reload_agents(context, manager=prompt_provider)
            await emit_command_outcome(context, outcome)
            if outcome.requires_refresh:
                next_available_agents = merge_pinned_agents(list(prompt_provider.agent_names()))
                next_available_agents_set = set(next_available_agents)
                owner.agent_types = prompt_provider.agent_types()
                enhanced_prompt.available_agents = set(next_available_agents)
                result.available_agents = next_available_agents
                result.available_agents_set = next_available_agents_set
                if agent not in next_available_agents_set:
                    if next_available_agents:
                        result.next_agent = next_available_agents[0]
                    else:
                        rich_print("[red]No agents available after reload.[/red]")
                        result.should_return = True
            return result
        case UnknownCommand(command=command):
            rich_print(f"[red]Command not found: {command}[/red]")
            return result
        case _:
            result.handled = False
            return result
