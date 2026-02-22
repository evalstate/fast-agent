"""
Enhanced prompt functionality with advanced prompt_toolkit features.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich import print as rich_print

from fast_agent.agents.agent_types import AgentType
from fast_agent.agents.workflow.parallel_agent import ParallelAgent
from fast_agent.agents.workflow.router_agent import RouterAgent
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.command_payloads import (
    AgentCommand,
    ClearCommand,
    CommandPayload,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryWebClearCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpConnectMode,
    McpDisconnectCommand,
    McpListCommand,
    ReloadAgentsCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShowHistoryCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
)
from fast_agent.ui.mcp_display import render_mcp_status
from fast_agent.ui.model_display import format_model_display
from fast_agent.ui.prompt.alert_flags import _resolve_alert_flags_from_history
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.keybindings import ShellPrefixLexer, create_keybindings
from fast_agent.ui.prompt.session_runtime import (
    build_prompt_style,
    cleanup_prompt_session,
    create_prompt_session,
    run_prompt_once,
    start_toolbar_switch_task,
)
from fast_agent.ui.prompt.special_commands import handle_special_commands_async
from fast_agent.ui.prompt.toolbar import (
    _can_fit_shell_path_and_version,
    _fit_shell_identity_for_toolbar,
    _fit_shell_path_for_toolbar,
    _format_toolbar_agent_identity,
    _resolve_toolbar_width,
    _toolbar_markup_width,
)
from fast_agent.ui.reasoning_effort_display import render_reasoning_effort_gauge
from fast_agent.ui.shell_notice import format_shell_notice
from fast_agent.ui.text_verbosity_display import render_text_verbosity_gauge

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fast_agent.core.agent_app import AgentApp

# Get the application version
try:
    app_version = version("fast-agent-mcp")
except:  # noqa: E722
    app_version = "unknown"

# Map of agent names to their history
agent_histories = {}

# Store available agents for auto-completion
available_agents = set()

# Keep track of multi-line mode state
in_multiline_mode: bool = False

# Track last copyable output (shell output or assistant response)
_last_copyable_output: str | None = None

# Track transient copy notice for the toolbar.
_copy_notice: str | None = None
_copy_notice_until: float = 0.0

_SHELL_PATH_SWITCH_DELAY_SECONDS = 8.0
_ELLIPSIS = "…"


def set_last_copyable_output(output: str) -> None:
    """Set the last copyable output for Ctrl+Y clipboard functionality."""
    global _last_copyable_output
    _last_copyable_output = output


def _show_system_cmd() -> ShowSystemCommand:
    return ShowSystemCommand()


def _show_usage_cmd() -> ShowUsageCommand:
    return ShowUsageCommand()


def _show_markdown_cmd() -> ShowMarkdownCommand:
    return ShowMarkdownCommand()


def _show_mcp_status_cmd() -> ShowMcpStatusCommand:
    return ShowMcpStatusCommand()


def _mcp_list_cmd() -> McpListCommand:
    return McpListCommand()


def _mcp_connect_cmd(
    target_text: str,
    *,
    parsed_mode: McpConnectMode,
    server_name: str | None,
    auth_token: str | None,
    timeout_seconds: float | None,
    trigger_oauth: bool | None,
    reconnect_on_disconnect: bool | None,
    force_reconnect: bool,
    error: str | None,
) -> McpConnectCommand:
    return McpConnectCommand(
        target_text=target_text,
        parsed_mode=parsed_mode,
        server_name=server_name,
        auth_token=auth_token,
        timeout_seconds=timeout_seconds,
        trigger_oauth=trigger_oauth,
        reconnect_on_disconnect=reconnect_on_disconnect,
        force_reconnect=force_reconnect,
        error=error,
    )


def _mcp_disconnect_cmd(server_name: str | None, error: str | None) -> McpDisconnectCommand:
    return McpDisconnectCommand(server_name=server_name, error=error)


def _list_tools_cmd() -> ListToolsCommand:
    return ListToolsCommand()


def _switch_agent_cmd(agent_name: str) -> SwitchAgentCommand:
    return SwitchAgentCommand(agent_name=agent_name)


def _hash_agent_cmd(agent_name: str, message: str) -> HashAgentCommand:
    return HashAgentCommand(agent_name=agent_name, message=message)


def _default_shell_command() -> str:
    """Best-effort shell choice for interactive prompt commands."""
    if platform.system() == "Windows":
        for shell_name in ["pwsh", "powershell", "cmd"]:
            shell_path = shutil.which(shell_name)
            if shell_path:
                return shell_path
        return os.environ.get("COMSPEC", "cmd.exe")

    shell_env = os.environ.get("SHELL")
    if shell_env and Path(shell_env).exists():
        return shell_env

    for shell_name in ["bash", "zsh", "sh"]:
        shell_path = shutil.which(shell_name)
        if shell_path:
            return shell_path

    return "sh"


def _show_history_cmd(target_agent: str | None) -> ShowHistoryCommand:
    return ShowHistoryCommand(agent=target_agent)


def _clear_last_cmd(target_agent: str | None) -> ClearCommand:
    return ClearCommand(kind="clear_last", agent=target_agent)


def _clear_history_cmd(target_agent: str | None) -> ClearCommand:
    return ClearCommand(kind="clear_history", agent=target_agent)


def _save_history_cmd(filename: str | None) -> SaveHistoryCommand:
    return SaveHistoryCommand(filename=filename)


def _load_history_cmd(filename: str | None, error: str | None) -> LoadHistoryCommand:
    return LoadHistoryCommand(filename=filename, error=error)


def _load_prompt_cmd(filename: str | None, error: str | None) -> LoadPromptCommand:
    return LoadPromptCommand(filename=filename, error=error)


def _history_rewind_cmd(turn_index: int | None, error: str | None) -> HistoryRewindCommand:
    return HistoryRewindCommand(turn_index=turn_index, error=error)


def _history_review_cmd(turn_index: int | None, error: str | None) -> HistoryReviewCommand:
    return HistoryReviewCommand(turn_index=turn_index, error=error)


def _history_fix_cmd(target_agent: str | None) -> HistoryFixCommand:
    return HistoryFixCommand(agent=target_agent)


def _history_webclear_cmd(target_agent: str | None) -> HistoryWebClearCommand:
    return HistoryWebClearCommand(agent=target_agent)


def _load_agent_card_cmd(
    filename: str | None, add_tool: bool, remove_tool: bool, error: str | None
) -> LoadAgentCardCommand:
    return LoadAgentCardCommand(
        filename=filename, add_tool=add_tool, remove_tool=remove_tool, error=error
    )


def _agent_cmd(
    agent_name: str | None, add_tool: bool, remove_tool: bool, dump: bool, error: str | None
) -> AgentCommand:
    return AgentCommand(
        agent_name=agent_name,
        add_tool=add_tool,
        remove_tool=remove_tool,
        dump=dump,
        error=error,
    )


def _reload_agents_cmd() -> ReloadAgentsCommand:
    return ReloadAgentsCommand()


def _select_prompt_cmd(prompt_index: int | None, prompt_name: str | None) -> SelectPromptCommand:
    return SelectPromptCommand(prompt_index=prompt_index, prompt_name=prompt_name)


def _skills_cmd(action: str, argument: str | None) -> SkillsCommand:
    return SkillsCommand(action=action, argument=argument)


def _infer_mcp_connect_mode(target_text: str) -> McpConnectMode:
    stripped = target_text.strip()
    if stripped.startswith(("http://", "https://")):
        return "url"
    if stripped.startswith("@"):
        return "npx"
    if stripped.startswith("npx "):
        return "npx"
    if stripped.startswith("uvx "):
        return "uvx"
    return "stdio"


def _rebuild_mcp_target_text(tokens: list[str]) -> str:
    """Rebuild target text while preserving whitespace-grouped arguments."""
    if not tokens:
        return ""

    rebuilt_parts: list[str] = []
    for token in tokens:
        if token == "" or any(char.isspace() for char in token):
            rebuilt_parts.append(shlex.quote(token))
        else:
            rebuilt_parts.append(token)
    return " ".join(rebuilt_parts)



# Track whether help text has been shown globally
help_message_shown: bool = False

# Track which agents have shown their info
_agent_info_shown = set()

# One-off notices to render at the top of the prompt UI
_startup_notices: list[str] = []


def queue_startup_notice(notice: str) -> None:
    if notice:
        _startup_notices.append(notice)


async def show_mcp_status(agent_name: str, agent_provider: "AgentApp | None") -> None:
    if agent_provider is None:
        rich_print("[red]No agent provider available[/red]")
        return

    try:
        agent = agent_provider._agent(agent_name)
    except Exception as exc:
        rich_print(f"[red]Unable to load agent '{agent_name}': {exc}[/red]")
        return

    await render_mcp_status(agent)


async def _display_agent_info_helper(agent_name: str, agent_provider: "AgentApp | None") -> None:
    """Helper function to display agent information."""
    # Only show once per agent
    if agent_name in _agent_info_shown:
        return

    try:
        # Get agent info from AgentApp
        if agent_provider is None:
            return
        agent = agent_provider._agent(agent_name)

        # Get counts TODO -- add this to the type library or adjust the way aggregator/reporting works
        server_count = 0
        if isinstance(agent, McpAgentProtocol):
            server_names = agent.aggregator.server_names
            server_count = len(server_names) if server_names else 0

        tools_result = await agent.list_tools()
        tool_count = (
            len(tools_result.tools) if tools_result and hasattr(tools_result, "tools") else 0
        )

        resources_dict = await agent.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )

        prompts_dict = await agent.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0

        skill_count = 0
        skill_manifests = getattr(agent, "_skill_manifests", None)
        if skill_manifests:
            try:
                skill_count = len(list(skill_manifests))
            except TypeError:
                skill_count = 0
        tool_children = _collect_tool_children(agent)

        # Handle different agent types
        if isinstance(agent, ParallelAgent):
            # Count child agents for parallel agents
            child_count = 0
            if agent.fan_out_agents:
                child_count += len(agent.fan_out_agents)
            if agent.fan_in_agent:
                child_count += 1

            if child_count > 0:
                child_word = "child agent" if child_count == 1 else "child agents"
                rich_print(
                    f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {child_count:,}[dim] {child_word}[/dim]"
                )
        elif isinstance(agent, RouterAgent):
            # Count child agents for router agents
            child_count = 0
            if agent.agents:
                child_count = len(agent.agents)

            if child_count > 0:
                child_word = "child agent" if child_count == 1 else "child agents"
                rich_print(
                    f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {child_count:,}[dim] {child_word}[/dim]"
                )
        else:
            content_parts = []

            if tool_children:
                child_count = len(tool_children)
                child_word = "child agent" if child_count == 1 else "child agents"
                content_parts.append(f"{child_count:,}[dim] {child_word}[/dim]")

            if server_count > 0:
                sub_parts = []
                if tool_count > 0:
                    tool_word = "tool" if tool_count == 1 else "tools"
                    sub_parts.append(f"{tool_count:,}[dim] {tool_word}[/dim]")
                if prompt_count > 0:
                    prompt_word = "prompt" if prompt_count == 1 else "prompts"
                    sub_parts.append(f"{prompt_count:,}[dim] {prompt_word}[/dim]")
                if resource_count > 0:
                    resource_word = "resource" if resource_count == 1 else "resources"
                    sub_parts.append(f"{resource_count:,}[dim] {resource_word}[/dim]")

                server_word = "Server" if server_count == 1 else "Servers"
                server_text = f"{server_count:,}[dim] MCP {server_word}[/dim]"
                if sub_parts:
                    server_text = (
                        f"{server_text}[dim] ([/dim]"
                        + "[dim], [/dim]".join(sub_parts)
                        + "[dim])[/dim]"
                    )
                content_parts.append(server_text)

            if skill_count > 0:
                skill_word = "skill" if skill_count == 1 else "skills"
                content_parts.append(
                    f"{skill_count:,}[dim] {skill_word}[/dim][dim] available[/dim]"
                )

            if content_parts:
                content = "[dim]. [/dim]".join(content_parts)
                rich_print(f"[dim]Agent [/dim][blue]{agent_name}[/blue][dim]:[/dim] {content}")
        #               await _render_mcp_status(agent)

        # Display Skybridge status (if aggregator discovered any)
        try:
            aggregator = agent.aggregator if isinstance(agent, McpAgentProtocol) else None
            display = getattr(agent, "display", None)
            if aggregator and display and hasattr(display, "show_skybridge_summary"):
                skybridge_configs = await aggregator.get_skybridge_configs()
                display.show_skybridge_summary(agent_name, skybridge_configs)
        except Exception:
            # Ignore Skybridge rendering issues to avoid interfering with startup
            pass

        # Mark as shown
        _agent_info_shown.add(agent_name)

    except Exception:
        # Silently ignore errors to not disrupt the user experience
        pass


async def _display_all_agents_with_hierarchy(
    available_agents: Iterable[str], agent_provider: "AgentApp | None"
) -> None:
    """Display all agents with tree structure for workflow agents."""
    agent_list = list(available_agents)
    # Track which agents are children to avoid displaying them twice
    child_agents = set()

    # First pass: identify all child agents
    for agent_name in agent_list:
        try:
            if agent_provider is None:
                continue
            agent = agent_provider._agent(agent_name)

            if isinstance(agent, ParallelAgent):
                if agent.fan_out_agents:
                    for child_agent in agent.fan_out_agents:
                        if child_agent.name:
                            child_agents.add(child_agent.name)
                if agent.fan_in_agent and agent.fan_in_agent.name:
                    child_agents.add(agent.fan_in_agent.name)
            elif isinstance(agent, RouterAgent):
                if agent.agents:
                    for child_agent in agent.agents:
                        if child_agent.name:
                            child_agents.add(child_agent.name)
            else:
                tool_children = _collect_tool_children(agent)
                for child_agent in tool_children:
                    child_name = getattr(child_agent, "name", None)
                    if child_name:
                        child_agents.add(child_name)
        except Exception:
            continue

    # Second pass: display agents (parents with children, standalone agents without children)
    for agent_name in sorted(agent_list):
        # Skip if this agent is a child of another agent
        if agent_name in child_agents:
            continue

        try:
            if agent_provider is None:
                continue
            agent = agent_provider._agent(agent_name)

            # Display parent agent
            await _display_agent_info_helper(agent_name, agent_provider)

            # If it's a workflow agent, display its children
            if agent.agent_type == AgentType.PARALLEL:
                await _display_parallel_children(agent, agent_provider)
            elif agent.agent_type == AgentType.ROUTER:
                await _display_router_children(agent, agent_provider)
            else:
                tool_children = _collect_tool_children(agent)
                if tool_children:
                    await _display_tool_children(tool_children, agent_provider)

        except Exception:
            continue


async def _display_parallel_children(parallel_agent, agent_provider: "AgentApp | None") -> None:
    """Display child agents of a parallel agent in tree format."""
    children = []

    # Collect fan-out agents
    if parallel_agent.fan_out_agents:
        for child_agent in parallel_agent.fan_out_agents:
            children.append(child_agent)

    # Collect fan-in agent
    if parallel_agent.fan_in_agent is not None:
        children.append(parallel_agent.fan_in_agent)

    # Display children with tree formatting
    for i, child_agent in enumerate(children):
        is_last = i == len(children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


async def _display_router_children(router_agent, agent_provider: "AgentApp | None") -> None:
    """Display child agents of a router agent in tree format."""
    children = []

    # Collect routing agents
    if router_agent.agents:
        children = list(router_agent.agents)

    # Display children with tree formatting
    for i, child_agent in enumerate(children):
        is_last = i == len(children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


async def _display_tool_children(tool_children, agent_provider: "AgentApp | None") -> None:
    """Display tool-exposed child agents in tree format."""
    for i, child_agent in enumerate(tool_children):
        is_last = i == len(tool_children) - 1
        prefix = "└─" if is_last else "├─"
        await _display_child_agent_info(child_agent, prefix, agent_provider)


def _collect_tool_children(agent) -> list[Any]:
    """Collect child agents exposed as tools."""
    children: list[Any] = []
    child_map = getattr(agent, "_child_agents", None)
    if isinstance(child_map, dict):
        children.extend(child_map.values())
    agent_tools = getattr(agent, "_agent_tools", None)
    if isinstance(agent_tools, dict):
        children.extend(agent_tools.values())

    seen: set[str] = set()
    unique_children: list[Any] = []
    for child in children:
        name = getattr(child, "name", None)
        if not name or name in seen:
            continue
        seen.add(name)
        unique_children.append(child)
    return unique_children


async def _display_child_agent_info(
    child_agent, prefix: str, agent_provider: "AgentApp | None"
) -> None:
    """Display info for a child agent with tree prefix."""
    try:
        # Get counts for child agent
        servers = await child_agent.list_servers()
        server_count = len(servers) if servers else 0

        tools_result = await child_agent.list_tools()
        tool_count = (
            len(tools_result.tools) if tools_result and hasattr(tools_result, "tools") else 0
        )

        resources_dict = await child_agent.list_resources()
        resource_count = (
            sum(len(resources) for resources in resources_dict.values()) if resources_dict else 0
        )

        prompts_dict = await child_agent.list_prompts()
        prompt_count = sum(len(prompts) for prompts in prompts_dict.values()) if prompts_dict else 0

        # Only display if child has MCP servers
        if server_count > 0:
            # Pluralization helpers
            server_word = "Server" if server_count == 1 else "Servers"
            tool_word = "tool" if tool_count == 1 else "tools"
            resource_word = "resource" if resource_count == 1 else "resources"
            prompt_word = "prompt" if prompt_count == 1 else "prompts"

            rich_print(
                f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]:[/dim] {server_count:,}[dim] MCP {server_word}, [/dim]{tool_count:,}[dim] {tool_word}, [/dim]{resource_count:,}[dim] {resource_word}, [/dim]{prompt_count:,}[dim] {prompt_word} available[/dim]"
            )
        else:
            # Show child even without MCP servers for context
            rich_print(
                f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue][dim]: No MCP Servers[/dim]"
            )

    except Exception:
        # Fallback: just show the name
        rich_print(f"[dim]  {prefix} [/dim][blue]{child_agent.name}[/blue]")


# AgentCompleter moved to fast_agent.ui.prompt.completer



def parse_special_input(text: str) -> str | CommandPayload:
    """Compatibility wrapper around the prompt parser module."""
    from fast_agent.ui.prompt.parser import parse_special_input as _parse_special_input

    return _parse_special_input(text)


async def get_enhanced_input(
    agent_name: str,
    default: str = "",
    show_default: bool = False,
    show_stop_hint: bool = False,
    multiline: bool = False,
    available_agent_names: list[str] | None = None,
    agent_types: dict[str, AgentType] | None = None,
    is_human_input: bool = False,
    toolbar_color: str = "ansiblue",
    agent_provider: "AgentApp | None" = None,
    noenv_mode: bool = False,
    pre_populate_buffer: str = "",
) -> str | CommandPayload:
    """
    Enhanced input with advanced prompt_toolkit features.

    Args:
        agent_name: Name of the agent (used for prompt and history)
        default: Default value if user presses enter
        show_default: Whether to show the default value in the prompt
        show_stop_hint: Whether to show the STOP hint
        multiline: Start in multiline mode
        available_agent_names: List of agent names for auto-completion
        agent_types: Dictionary mapping agent names to their types for display
        is_human_input: Whether this is a human input request (disables agent selection features)
        toolbar_color: Color to use for the agent name in the toolbar (default: "ansiblue")
        agent_provider: Optional AgentApp for displaying agent info
        noenv_mode: Whether session operations should be disabled for --noenv mode
        pre_populate_buffer: Text to pre-populate in the input buffer for editing (one-off)

    Returns:
        User input string or parsed command payload
    """
    global in_multiline_mode, available_agents, help_message_shown

    # Update global state
    in_multiline_mode = multiline
    if available_agent_names:
        available_agents = set(available_agent_names)
    if agent_provider:
        try:
            available_agents = set(agent_provider.agent_names())
        except Exception:
            pass

    # Get or create history object for this agent
    if agent_name not in agent_histories:
        agent_histories[agent_name] = InMemoryHistory()

    # Define callback for multiline toggle
    def on_multiline_toggle(enabled) -> None:
        nonlocal session
        if hasattr(session, "app") and session.app:
            session.app.invalidate()

    shell_enabled = False
    shell_access_modes: tuple[str, ...] = ()
    shell_name: str | None = None
    shell_runtime = None
    shell_working_dir: Path | None = None
    toolbar_started_at = time.monotonic()
    show_shell_path_segment = False
    toolbar_switch_task = None

    # Define toolbar function that will update dynamically
    def get_toolbar():
        global _copy_notice
        nonlocal show_shell_path_segment
        if in_multiline_mode:
            mode_style = "ansired"  # More noticeable for multiline mode
            mode_text = "MLTI"
        #           toggle_text = "Normal"
        else:
            mode_style = "ansigreen"
            mode_text = "NRML"
        #            toggle_text = "Multiline"

        # No shortcut hints in the toolbar for now
        shortcuts = []

        # Only show relevant shortcuts based on mode
        shortcuts = [(k, v) for k, v in shortcuts if v]

        shortcut_text = " | ".join(f"{key}:{action}" for key, action in shortcuts)

        # Resolve model name, turn counter, and TDV from the current agent if available
        model_display = None
        tdv_segment = None
        turn_count = 0
        agent = None
        if agent_provider:
            try:
                agent = agent_provider._agent(agent_name)
            except Exception:
                agent = None

        if agent:
            for message in agent.message_history:
                if message.role == "user":
                    turn_count += 1

            # Resolve LLM reference safely (avoid assertion when unattached)
            llm = None
            try:
                llm = agent.llm
            except AssertionError:
                llm = getattr(agent, "_llm", None)
            except Exception as exc:
                print(f"[toolbar debug] agent.llm access failed for '{agent_name}': {exc}")

            model_name = None
            if llm:
                model_name = getattr(llm, "model_name", None)
                if not model_name:
                    model_name = getattr(
                        getattr(llm, "default_request_params", None), "model", None
                    )

            if not model_name:
                model_name = agent.config.model
            if not model_name and agent.config.default_request_params:
                model_name = agent.config.default_request_params.model
            if not model_name:
                try:
                    context = agent.context
                except Exception:
                    context = None
                if context and context.config:
                    model_name = context.config.default_model

            codex_suffix = ""
            reasoning_gauge = None
            verbosity_gauge = None
            model_suffix = ""
            if model_name:
                display_name = format_model_display(model_name) or model_name
                if llm and getattr(llm, "provider", None) == Provider.CODEX_RESPONSES:
                    codex_suffix = " <style bg='ansiyellow'>$</style>"
                if llm:
                    try:
                        reasoning_gauge = render_reasoning_effort_gauge(
                            llm.reasoning_effort,
                            llm.reasoning_effort_spec,
                        )
                        verbosity_gauge = render_text_verbosity_gauge(
                            llm.text_verbosity,
                            llm.text_verbosity_spec,
                        )
                        if llm.web_search_enabled:
                            model_suffix = "⊕"  # ◉ # 🌐
                    except Exception:
                        reasoning_gauge = None
                        verbosity_gauge = None
                        model_suffix = ""
                max_len = 25
                model_display = (
                    display_name[: max_len - 1] + "…"
                    if len(display_name) > max_len
                    else display_name
                )
            else:
                if isinstance(agent, ParallelAgent):
                    parallel_models: list[str] = []
                    for fan_out_agent in agent.fan_out_agents:
                        child_llm = None
                        try:
                            child_llm = fan_out_agent.llm
                        except AssertionError:
                            child_llm = getattr(fan_out_agent, "_llm", None)
                        except Exception:
                            child_llm = None

                        child_model_name = None
                        if child_llm:
                            child_model_name = getattr(child_llm, "model_name", None)
                            if not child_model_name:
                                child_model_name = getattr(
                                    getattr(child_llm, "default_request_params", None),
                                    "model",
                                    None,
                                )
                        if not child_model_name:
                            child_model_name = fan_out_agent.config.model
                        if not child_model_name and fan_out_agent.config.default_request_params:
                            child_model_name = fan_out_agent.config.default_request_params.model
                        if child_model_name:
                            display_name = (
                                format_model_display(child_model_name) or child_model_name
                            )
                            parallel_models.append(display_name)

                    if parallel_models:
                        deduped_models = list(dict.fromkeys(parallel_models))
                        display_name = ",".join(deduped_models)
                        max_len = 25
                        model_display = (
                            display_name[: max_len - 1] + "…"
                            if len(display_name) > max_len
                            else display_name
                        )
                    else:
                        model_display = "parallel"

                if not model_display:
                    print(f"[toolbar debug] no model resolved for agent '{agent_name}'")
                    model_display = "unknown"

            # Build TDV capability segment based on model database
            info = None
            if llm:
                info = ModelInfo.from_llm(llm)
            if not info and model_name:
                info = ModelInfo.from_name(model_name)

            # Default to text-only if info resolution fails for any reason
            t, d, v = (True, False, False)
            if info:
                t, d, v = info.tdv_flags

            # Check for alert flags in persisted history.
            alert_flags = _resolve_alert_flags_from_history(agent.message_history)

            def _style_flag(letter: str, supported: bool) -> str:
                # Enabled uses the same color as NORMAL mode (ansigreen), disabled is dim
                if letter in alert_flags:
                    return f"<style fg='ansired' bg='ansiblack'>{letter}</style>"

                enabled_color = "ansigreen"
                if supported:
                    return f"<style fg='{enabled_color}' bg='ansiblack'>{letter}</style>"
                return f"<style fg='ansiblack' bg='ansiwhite'>{letter}</style>"

            tdv_segment = f"{_style_flag('T', t)}{_style_flag('V', v)}{_style_flag('D', d)}"
        else:
            model_display = None
            tdv_segment = None

        agent_identity_segment = _format_toolbar_agent_identity(agent_name, toolbar_color, agent)

        # Build dynamic middle segments: model (in green), turn counter, and optional shortcuts
        middle_segments = []
        if model_display:
            # Model chip + inline TDV flags
            model_label = f"{model_display}{model_suffix}"
            if tdv_segment:
                gauge_segment = ""
                if reasoning_gauge or verbosity_gauge:
                    gauges = "".join(g for g in (reasoning_gauge, verbosity_gauge) if g)
                    gauge_segment = f" {gauges}"
                middle_segments.append(
                    f"{tdv_segment}{gauge_segment} <style bg='ansigreen'>{model_label}</style>{codex_suffix}"
                )
            else:
                gauge_segment = ""
                if reasoning_gauge or verbosity_gauge:
                    gauges = "".join(g for g in (reasoning_gauge, verbosity_gauge) if g)
                    gauge_segment = f" {gauges}"
                middle_segments.append(
                    f"{gauge_segment} <style bg='ansigreen'>{model_label}</style>{codex_suffix}"
                )

        # Add turn counter (formatted as 3 digits)
        middle_segments.append(f"{turn_count:03d}")

        if shortcut_text:
            middle_segments.append(shortcut_text)
        middle = " | ".join(middle_segments)

        # Version/app label in green (dynamic version)
        version_segment = f"fast-agent {app_version}"
        toolbar_identity_segment = version_segment

        # Add notifications - prioritize active events over completed ones
        from fast_agent.ui import notification_tracker

        notification_segment = ""

        # Check for active events first (highest priority)
        active_status = notification_tracker.get_active_status()
        if active_status:
            event_type = active_status["type"].upper()
            server = active_status["server"]
            notification_segment = (
                f" | <style fg='ansired' bg='ansiblack'>◀ {event_type} ({server})</style>"
            )
        elif notification_tracker.get_count() > 0:
            # Show completed events summary when no active events
            counts_by_type = notification_tracker.get_counts_by_type()
            total_events = sum(counts_by_type.values()) if counts_by_type else 0

            if len(counts_by_type) == 1:
                event_type, count = next(iter(counts_by_type.items()))
                label_text = notification_tracker.format_event_label(event_type, count)
                notification_segment = f" | ◀ {label_text}"
            else:
                summary = notification_tracker.get_summary(compact=True)
                heading = "event" if total_events == 1 else "events"
                notification_segment = f" | ◀ {total_events} {heading} ({summary})"

        copy_notice = ""
        if _copy_notice:
            if time.monotonic() < _copy_notice_until:
                copy_notice = f" | <style fg='{mode_style}' bg='ansiblack'> {_copy_notice} </style>"
            else:
                # Expire the notice once the timer elapses.
                _copy_notice = None

        if shell_enabled:
            working_dir = shell_working_dir or Path.cwd()

            if middle:
                left_prefix = (
                    f" {agent_identity_segment} "
                    f" {middle} | <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                )
            else:
                left_prefix = (
                    f" {agent_identity_segment} "
                    f"Mode: <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                )

            right_suffix = f"{notification_segment}{copy_notice}"
            available_width = (
                _resolve_toolbar_width()
                - _toolbar_markup_width(left_prefix)
                - _toolbar_markup_width(right_suffix)
            )

            if _can_fit_shell_path_and_version(working_dir, version_segment, available_width):
                # If both can fit, show normal path+version behavior immediately.
                toolbar_identity_segment = _fit_shell_identity_for_toolbar(
                    working_dir,
                    version_segment,
                    available_width,
                )
                show_shell_path_segment = True
            else:
                if not show_shell_path_segment:
                    elapsed = time.monotonic() - toolbar_started_at
                    if elapsed >= _SHELL_PATH_SWITCH_DELAY_SECONDS:
                        show_shell_path_segment = True

                if show_shell_path_segment:
                    toolbar_identity_segment = _fit_shell_path_for_toolbar(
                        working_dir,
                        available_width,
                    )

        if middle:
            return HTML(
                f" {agent_identity_segment} "
                f" {middle} | <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                f"{toolbar_identity_segment}{notification_segment}{copy_notice}"
            )
        else:
            return HTML(
                f" {agent_identity_segment} "
                f"Mode: <style fg='{mode_style}' bg='ansiblack'> {mode_text} </style> | "
                f"{toolbar_identity_segment}{notification_segment}{copy_notice}"
            )

    custom_style = build_prompt_style()

    session = create_prompt_session(
        history=agent_histories[agent_name],
        completer=AgentCompleter(
            agents=list(available_agents) if available_agents else [],
            agent_types=agent_types or {},
            is_human_input=is_human_input,
            current_agent=agent_name,
            agent_provider=agent_provider,
            noenv_mode=noenv_mode,
        ),
        lexer=ShellPrefixLexer(),
        multiline_filter=Condition(lambda: in_multiline_mode),
        toolbar=get_toolbar,
        style=custom_style,
    )

    # Create key bindings with a reference to the app
    bindings = create_keybindings(
        on_toggle_multiline=on_multiline_toggle,
        app=session.app,
        agent_provider=agent_provider,
        agent_name=agent_name,
    )
    session.app.key_bindings = bindings

    shell_agent = None
    if agent_provider:
        try:
            shell_agent = agent_provider._agent(agent_name)
        except Exception:
            shell_agent = None

    if isinstance(shell_agent, McpAgentProtocol):
        direct_shell_enabled = shell_agent.shell_runtime_enabled
        shell_access_modes = shell_agent.shell_access_modes

        sub_agent_shells = [
            child
            for child in _collect_tool_children(shell_agent)
            if isinstance(child, McpAgentProtocol) and child.shell_runtime_enabled
        ]
        if sub_agent_shells:
            if direct_shell_enabled:
                if "sub-agent" not in shell_access_modes:
                    shell_access_modes = (*shell_access_modes, "sub-agent")
            else:
                shell_access_modes = ("sub-agent",)
                if len(sub_agent_shells) == 1:
                    shell_runtime = sub_agent_shells[0].shell_runtime

        shell_enabled = direct_shell_enabled or bool(sub_agent_shells)
        if direct_shell_enabled:
            shell_runtime = shell_agent.shell_runtime

        # Get the detected shell name from the runtime
        if shell_enabled and shell_runtime:
            runtime_info = shell_runtime.runtime_info()
            shell_name = runtime_info.get("name")
            try:
                shell_working_dir = shell_runtime.working_directory()
            except Exception:
                shell_working_dir = None

    if shell_enabled:
        toolbar_switch_task = start_toolbar_switch_task(
            session,
            _SHELL_PATH_SWITCH_DELAY_SECONDS,
        )

    def _resolve_prompt_text() -> HTML:
        buffer_text = ""
        try:
            buffer_text = session.default_buffer.text
        except Exception:
            buffer_text = ""

        if buffer_text.lstrip().startswith("!"):
            arrow_segment = "<ansired>❯</ansired>"
        else:
            arrow_segment = "<ansibrightyellow>❯</ansibrightyellow>" if shell_enabled else "❯"

        prompt_text = f"<ansibrightblue>{agent_name}</ansibrightblue> {arrow_segment} "

        # Add default value display if requested
        if show_default and default and default != "STOP":
            prompt_text = f"{prompt_text} [<ansigreen>{default}</ansigreen>] "

        return HTML(prompt_text)

    # Only show hints at startup if requested
    if show_stop_hint:
        if default == "STOP":
            rich_print("Enter a prompt, [red]STOP[/red] or [red]Ctrl+D[/red] to finish")
            if default:
                rich_print(f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]")

    # Mention available features but only on first usage globally
    if not help_message_shown:
        if is_human_input:
            rich_print("[dim]Type /help for commands. Ctrl+T toggles multiline mode.[/dim]")
        else:
            rich_print(
                "[dim]Use '/' for commands, '!' for shell. '#' to query, '@' to switch agents\nCTRL+T multiline, CTRL+Y copy last message, CTRL+E external editor.[/dim]\n"
            )

        if shell_enabled:
            modes_display = ", ".join(shell_access_modes or ("direct",))
            shell_display = f"{modes_display}, {shell_name}" if shell_name else modes_display

            # Add working directory info
            if shell_runtime:
                working_dir = shell_runtime.working_directory()
                try:
                    # Try to show relative to cwd for cleaner display
                    working_dir_display = str(working_dir.relative_to(Path.cwd()))
                    if working_dir_display == ".":
                        # Show last 2 parts of the path (e.g., "source/fast-agent")
                        parts = Path.cwd().parts
                        if len(parts) >= 2:
                            working_dir_display = "/".join(parts[-2:])
                        elif len(parts) == 1:
                            working_dir_display = parts[0]
                        else:
                            working_dir_display = str(Path.cwd())
                except ValueError:
                    # If not relative to cwd, show absolute path
                    working_dir_display = str(working_dir)
                shell_display = f"{shell_display} | cwd: {working_dir_display}"

            rich_print(format_shell_notice(shell_access_modes, shell_runtime))

            # Display agent info right after help text if agent_provider is available
            if agent_provider and not is_human_input:
                # Display info for all available agents with tree structure for workflows
                await _display_all_agents_with_hierarchy(available_agents, agent_provider)

            # Show streaming status message
            if agent_provider:
                # Get logger settings from the agent's context (not agent_provider)
                logger_settings = None
                try:
                    active_agent = shell_agent
                    if active_agent is None:
                        active_agent = agent_provider._agent(agent_name)
                    try:
                        agent_context = active_agent.context
                    except Exception:
                        agent_context = None
                    if agent_context and agent_context.config:
                        logger_settings = agent_context.config.logger
                except Exception:
                    # If we can't get the agent or its context, logger_settings stays None
                    pass

                # Only show streaming messages if chat display is enabled AND we have logger_settings
                if logger_settings:
                    show_chat = getattr(logger_settings, "show_chat", True)

                    if show_chat:
                        # Check for parallel agents
                        has_parallel = any(
                            agent.agent_type == AgentType.PARALLEL
                            for agent in agent_provider._agents.values()
                        )

                        # Note: streaming may have been disabled by fastagent.py if parallel agents exist
                        # So we check has_parallel first to show the appropriate message
                        if has_parallel:
                            # Streaming is disabled due to parallel agents
                            rich_print(
                                "[dim]Markdown Streaming disabled (Parallel Agents configured)[/dim]"
                            )
                        else:
                            # Check if streaming is enabled
                            streaming_enabled = getattr(logger_settings, "streaming_display", True)
                            streaming_mode = getattr(logger_settings, "streaming", "markdown")
                            if streaming_enabled and streaming_mode != "none":
                                # Streaming is enabled - notify users since it's experimental
                                rich_print(
                                    f"[dim]Experimental: Streaming Enabled - {streaming_mode} mode[/dim]"
                                )

                        # Show model source if configured via env var or config file
                        model_source = (
                            getattr(agent_context.config, "model_source", None)
                            if agent_context and agent_context.config
                            else None
                        )
                        if model_source:
                            rich_print(f"[dim]Model selected via {model_source}[/dim]")

                        # Show HuggingFace model and provider info if applicable
                        try:
                            if active_agent and active_agent.llm:
                                get_hf_info = getattr(active_agent.llm, "get_hf_display_info", None)
                                if get_hf_info:
                                    hf_info = get_hf_info()
                                    model = hf_info.get("model", "unknown")
                                    provider = hf_info.get("provider", "auto-routing")
                                    rich_print(f"[dim]HuggingFace: {model} via {provider}[/dim]")
                        except Exception:
                            pass

            if agent_provider and not is_human_input and _startup_notices:
                for notice in _startup_notices:
                    rich_print(notice)
                _startup_notices.clear()

        rich_print()
        help_message_shown = True

    # Process special commands

    # Determine what to use as the buffer's initial content:
    # - pre_populate_buffer takes priority (one-off, for # command results)
    # - otherwise use the default parameter
    buffer_default = pre_populate_buffer if pre_populate_buffer else default

    try:
        return await run_prompt_once(
            session=session,
            agent_name=agent_name,
            default_buffer=buffer_default,
            resolve_prompt_text=_resolve_prompt_text,
            parse_special_input=parse_special_input,
        )
    finally:
        await cleanup_prompt_session(
            session=session,
            toolbar_switch_task=toolbar_switch_task,
        )


async def get_selection_input(
    prompt_text: str,
    options: list[str] | None = None,
    default: str | None = None,
    allow_cancel: bool = True,
    complete_options: bool = True,
) -> str | None:
    """
    Display a selection prompt and return the user's selection.

    Args:
        prompt_text: Text to display as the prompt
        options: List of valid options (for auto-completion)
        default: Default value if user presses enter
        allow_cancel: Whether to allow cancellation with empty input
        complete_options: Whether to use the options for auto-completion

    Returns:
        Selected value, or None if cancelled
    """
    try:
        # Initialize completer if options provided and completion requested
        completer = WordCompleter(options) if options and complete_options else None

        # Create prompt session
        prompt_session = PromptSession(completer=completer)

        try:
            # Get user input
            selection = await prompt_session.prompt_async(
                prompt_text,
                default=default or "",
                set_exception_handler=False,
            )

            # Handle cancellation
            if allow_cancel and not selection.strip():
                return None

            return selection
        finally:
            # Ensure prompt session cleanup
            if prompt_session.app.is_running:
                prompt_session.app.exit()
    except (KeyboardInterrupt, EOFError):
        return None
    except Exception as e:
        rich_print(f"\n[red]Error getting selection: {e}[/red]")
        return None


async def get_argument_input(
    arg_name: str,
    description: str | None = None,
    required: bool = True,
) -> str | None:
    """
    Prompt for an argument value with formatting and help text.

    Args:
        arg_name: Name of the argument
        description: Optional description of the argument
        required: Whether this argument is required

    Returns:
        Input value, or None if cancelled/skipped
    """
    # Format the prompt differently based on whether it's required
    required_text = "(required)" if required else "(optional, press Enter to skip)"

    # Show description if available
    if description:
        rich_print(f"  [dim]{arg_name}: {description}[/dim]")

    prompt_text = HTML(
        f"Enter value for <ansibrightcyan>{arg_name}</ansibrightcyan> {required_text}: "
    )

    # Create prompt session
    prompt_session = PromptSession()

    try:
        # Get user input
        arg_value = await prompt_session.prompt_async(
            prompt_text,
            set_exception_handler=False,
        )

        # For optional arguments, empty input means skip
        if not required and not arg_value:
            return None

        return arg_value
    except (KeyboardInterrupt, EOFError):
        return None
    except Exception as e:
        rich_print(f"\n[red]Error getting input: {e}[/red]")
        return None
    finally:
        # Ensure prompt session cleanup
        if prompt_session.app.is_running:
            prompt_session.app.exit()


async def handle_special_commands(
    command: str | CommandPayload | None, agent_app: "AgentApp | bool | None" = None
) -> bool | CommandPayload:
    """Handle special input commands."""
    return await handle_special_commands_async(
        command,
        agent_app,
        available_agents=available_agents,
    )
