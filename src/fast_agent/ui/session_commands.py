"""Session command handlers for interactive prompt."""

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.session import (
    format_history_summary,
    format_session_entries,
    get_session_manager,
    summarize_session_histories,
)
from fast_agent.ui.command_payloads import (
    CreateSessionCommand,
    ForkSessionCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SwitchSessionCommand,
    TitleSessionCommand,
)
from fast_agent.ui.enhanced_prompt import rich_print
from fast_agent.ui.shell_notice import format_shell_notice

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp


async def handle_list_sessions_cmd(agent_app: "AgentApp | None" = None) -> bool:
    """Handle /list_sessions command."""
    manager = get_session_manager()
    sessions = manager.list_sessions()
    
    if not sessions:
        rich_print("[yellow]No sessions found.[/yellow]")
        return True
    
    rich_print("\n[bold]Available Sessions:[/bold]")
    current = manager.current_session
    entries = format_session_entries(
        sessions,
        current.info.name if current else None,
        mode="verbose",
    )
    for line in entries:
        rich_print(line)

    rich_print("\n[dim]Use /create_session <name> to create a new session[/dim]")
    rich_print("[dim]Use /switch_session <name> to switch to a session[/dim]")
    return True


async def handle_create_session_cmd(
    command: CreateSessionCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /create_session command."""
    manager = get_session_manager()
    session = manager.create_session(command.session_name)
    
    rich_print(f"[green]Created and switched to session: {session.info.name}[/green]")
    return True


async def handle_switch_session_cmd(
    command: SwitchSessionCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /switch_session command."""
    manager = get_session_manager()
    session = manager.load_session(command.session_name)
    
    if session:
        rich_print(f"[green]Switched to session: {session.info.name}[/green]")
        return True
    else:
        rich_print(f"[red]Session not found: {command.session_name}[/red]")
        return False


async def handle_save_history_cmd(
    command: SaveHistoryCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /save_history command with session support."""
    if not agent_app:
        rich_print("[red]No agent context available[/red]")
        return False
    
    manager = get_session_manager()
    
    # If there's a current session, save to it
    if manager.current_session:
        try:
            agent_obj = agent_app._agent(None)
            filepath = await manager.save_current_session(agent_obj)
            if filepath:
                rich_print(f"[green]Saved to session: {Path(filepath).name}[/green]")
            return True
        except Exception as e:
            rich_print(f"[red]Failed to save to session: {e}[/red]")
            return False
    
    # Fallback to original behavior if no session
    try:
        from fast_agent.history.history_exporter import HistoryExporter

        agent_obj = agent_app._agent(None)
        filepath = await HistoryExporter.save(agent_obj, command.filename)
        rich_print(f"[green]History saved to {Path(filepath).name}[/green]")
        return True
    except Exception as exc:
        rich_print(f"[red]Failed to save history: {exc}[/red]")
        return False


async def handle_resume_session_cmd(
    command: ResumeSessionCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /resume command."""
    if not agent_app:
        rich_print("[red]No agent context available[/red]")
        return False

    manager = get_session_manager()
    default_agent = agent_app._agent(None)
    result = manager.resume_session_agents(
        agent_app._agents,
        command.session_id,
        default_agent_name=getattr(default_agent, "name", None),
    )
    if not result:
        if command.session_id:
            rich_print(f"[red]Session not found: {command.session_id}[/red]")
        else:
            rich_print("[yellow]No sessions found.[/yellow]")
        return False

    session, loaded, missing_agents = result
    if loaded:
        loaded_list = ", ".join(sorted(loaded.keys()))
        rich_print(f"[green]Resumed session: {session.info.name} ({loaded_list})[/green]")
        if (
            isinstance(default_agent, McpAgentProtocol)
            and default_agent.shell_runtime_enabled
        ):
            rich_print(
                format_shell_notice(
                    default_agent.shell_access_modes,
                    default_agent.shell_runtime,
                )
            )
    else:
        rich_print(f"[yellow]Resumed session: {session.info.name} (no history yet)[/yellow]")
        if (
            isinstance(default_agent, McpAgentProtocol)
            and default_agent.shell_runtime_enabled
        ):
            rich_print(
                format_shell_notice(
                    default_agent.shell_access_modes,
                    default_agent.shell_runtime,
                )
            )
    if missing_agents:
        missing_list = ", ".join(sorted(missing_agents))
        rich_print(f"[yellow]Missing agents from session: {missing_list}[/yellow]")
    if missing_agents or not loaded:
        summary = summarize_session_histories(session)
        summary_text = format_history_summary(summary)
        if summary_text:
            rich_print(f"[dim]Available histories:[/dim] {summary_text}")
    return True


async def handle_title_session_cmd(
    command: TitleSessionCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /session title command."""
    if not agent_app:
        rich_print("[red]No agent context available[/red]")
        return False
    if not command.title:
        rich_print("[red]Usage: /session title <text>[/red]")
        return False

    manager = get_session_manager()
    session = manager.current_session or manager.create_session()
    session.set_title(command.title)
    rich_print(f"[green]Session title set: {command.title}[/green]")
    return True


async def handle_fork_session_cmd(
    command: ForkSessionCommand, agent_app: "AgentApp | None" = None
) -> bool:
    """Handle /session fork command."""
    manager = get_session_manager()
    forked = manager.fork_current_session(title=command.title)
    if not forked:
        rich_print("[yellow]No session available to fork.[/yellow]")
        return False
    label = forked.info.metadata.get("title") or forked.info.name
    rich_print(f"[green]Forked session: {label}[/green]")
    return True


async def handle_session_command(
    command: str | None, agent_app: "AgentApp | None" = None
) -> bool | None:
    """Handle session-related commands."""
    if not command:
        return False
    
    # Parse session commands
    if isinstance(command, str):
        cmd_lower = command.lower()
        
        if cmd_lower == "/list_sessions":
            return await handle_list_sessions_cmd(agent_app)
        
        elif cmd_lower.startswith("/create_session"):
            parts = command.split(maxsplit=1)
            session_name = parts[1] if len(parts) > 1 else None
            return await handle_create_session_cmd(
                CreateSessionCommand(session_name=session_name), agent_app
            )
        
        elif cmd_lower.startswith("/switch_session"):
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                rich_print("[red]Usage: /switch_session <session_name>[/red]")
                return False
            
            session_name = parts[1]
            return await handle_switch_session_cmd(
                SwitchSessionCommand(session_name=session_name), agent_app
            )

        elif cmd_lower.startswith("/resume"):
            parts = command.split(maxsplit=1)
            session_id = parts[1] if len(parts) > 1 else None
            return await handle_resume_session_cmd(
                ResumeSessionCommand(session_id=session_id), agent_app
            )

        elif cmd_lower.startswith("/session"):
            parts = command.split(maxsplit=2)
            if len(parts) < 2:
                rich_print(
                    "[yellow]Usage: /session resume [id] | /session title <text> | /session fork [title][/yellow]"
                )
                return False
            subcmd = parts[1].lower()
            arg = parts[2] if len(parts) > 2 else ""
            if subcmd == "resume":
                session_id = arg if arg else None
                return await handle_resume_session_cmd(
                    ResumeSessionCommand(session_id=session_id), agent_app
                )
            if subcmd == "title":
                return await handle_title_session_cmd(
                    TitleSessionCommand(title=arg), agent_app
                )
            if subcmd == "fork":
                title = arg if arg else None
                return await handle_fork_session_cmd(
                    ForkSessionCommand(title=title), agent_app
                )
            rich_print("[yellow]Unknown /session command.[/yellow]")
            return False
        
        elif cmd_lower == "/sessions" or cmd_lower == "/list_sessions":
            return await handle_list_sessions_cmd(agent_app)
    
    return False
