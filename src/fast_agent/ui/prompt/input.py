"""
Enhanced prompt functionality with advanced prompt_toolkit features.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich import print as rich_print
from rich.text import Text

from fast_agent.agents.agent_types import AgentType
from fast_agent.mcp.connect_targets import parse_connect_command_text
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
from fast_agent.ui.message_primitives import MessageType
from fast_agent.ui.model_shortcuts import (
    build_model_shortcut_hints,
    cycle_reasoning_setting,
    cycle_text_verbosity,
)
from fast_agent.ui.prompt.agent_info import (
    collect_tool_children as _collect_tool_children_impl,
)
from fast_agent.ui.prompt.agent_info import (
    display_agent_info as _display_agent_info_impl,
)
from fast_agent.ui.prompt.agent_info import (
    display_all_agents_with_hierarchy as _display_all_agents_with_hierarchy_impl,
)
from fast_agent.ui.prompt.completer import AgentCompleter
from fast_agent.ui.prompt.input_runtime import (
    build_prompt_style,
    cleanup_prompt_session,
    create_prompt_session,
    is_default_agent_name,
    run_prompt_once,
    start_toolbar_switch_task,
)
from fast_agent.ui.prompt.input_toolbar import (
    ShellToolbarState,
    render_input_toolbar,
    resolve_active_llm,
)
from fast_agent.ui.prompt.keybindings import ShellPrefixLexer, create_keybindings
from fast_agent.ui.prompt.special_commands import handle_special_commands_async
from fast_agent.ui.service_tier_display import cycle_service_tier
from fast_agent.ui.shell_notice import format_shell_notice
from fast_agent.utils.async_utils import suppress_known_runtime_warnings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

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
    del parsed_mode
    if error or not target_text:
        return McpConnectCommand(request=None, error=error)

    argv = [target_text]
    if server_name:
        argv.extend(["--name", shlex.quote(server_name)])
    if auth_token:
        argv.extend(["--auth", shlex.quote(auth_token)])
    if timeout_seconds is not None:
        argv.extend(["--timeout", str(timeout_seconds)])
    if trigger_oauth is True:
        argv.append("--oauth")
    elif trigger_oauth is False:
        argv.append("--no-oauth")
    if reconnect_on_disconnect is False:
        argv.append("--no-reconnect")
    if force_reconnect:
        argv.append("--reconnect")
    return McpConnectCommand(
        request=parse_connect_command_text(" ".join(argv)),
        error=None,
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


@dataclass(slots=True)
class StartupNotice:
    text: str
    render_markdown: bool = False
    title: str | None = None
    right_info: str | None = None
    agent_name: str | None = None


# One-off notices to render at the top of the prompt UI
_startup_notices: list[object] = []


def queue_startup_notice(notice: object) -> None:
    if notice:
        _startup_notices.append(notice)


def queue_startup_markdown_notice(
    text: str,
    *,
    title: str | None = None,
    style: str | None = None,
    right_info: str | None = None,
    agent_name: str | None = None,
) -> None:
    """Queue a markdown notice for display at next interactive prompt render."""
    if not text:
        return

    if style is not None and right_info is None and agent_name is None:
        from rich.markdown import Markdown

        if title:
            queue_startup_notice(title)
        queue_startup_notice(Markdown(text, style=style or ""))
        return

    _startup_notices.append(
        StartupNotice(
            text=text,
            render_markdown=True,
            title=title,
            right_info=right_info,
            agent_name=agent_name,
        )
    )


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
    """Compatibility wrapper for prompt agent-info rendering."""
    await _display_agent_info_impl(
        agent_name,
        agent_provider,
        shown_agents=_agent_info_shown,
    )


async def _display_all_agents_with_hierarchy(
    available_agents: Iterable[str],
    agent_provider: "AgentApp | None",
) -> None:
    """Compatibility wrapper for prompt agent hierarchy rendering."""
    await _display_all_agents_with_hierarchy_impl(
        available_agents,
        agent_provider,
        shown_agents=_agent_info_shown,
    )


def _collect_tool_children(agent: object) -> list[Any]:
    return _collect_tool_children_impl(agent)


# AgentCompleter moved to fast_agent.ui.prompt.completer


def parse_special_input(text: str) -> str | CommandPayload:
    """Compatibility wrapper around the prompt parser module."""
    from fast_agent.ui.prompt.parser import parse_special_input as _parse_special_input

    return _parse_special_input(text)


@dataclass(slots=True)
class ShellInputContext:
    enabled: bool = False
    access_modes: tuple[str, ...] = ()
    name: str | None = None
    runtime: Any | None = None
    working_dir: Path | None = None


@dataclass(slots=True)
class InputCycleCallbacks:
    on_cycle_service_tier: "Callable[[], None]"
    on_cycle_reasoning: "Callable[[], None]"
    on_cycle_verbosity: "Callable[[], None]"
    on_cycle_web_search: "Callable[[], None]"
    on_cycle_web_fetch: "Callable[[], None]"


def _initialize_prompt_input_state(
    *,
    agent_name: str,
    multiline: bool,
    available_agent_names: list[str] | None,
    agent_provider: "AgentApp | None",
) -> None:
    global in_multiline_mode, available_agents

    in_multiline_mode = multiline
    if available_agent_names:
        available_agents = set(available_agent_names)
    if agent_provider is not None:
        try:
            available_agents = set(agent_provider.visible_agent_names(force_include=agent_name))
        except Exception:
            pass

    if agent_name not in agent_histories:
        agent_histories[agent_name] = InMemoryHistory()


def _build_multiline_toggle(
    session_factory: "Callable[[], PromptSession]",
) -> "Callable[[bool], None]":
    def on_multiline_toggle(enabled: bool) -> None:
        del enabled
        session = session_factory()
        if hasattr(session, "app") and session.app:
            session.app.invalidate()

    return on_multiline_toggle


def _build_toolbar(
    *,
    agent_name: str,
    toolbar_color: str,
    agent_provider: "AgentApp | None",
    shell_context: ShellInputContext,
    session_factory: "Callable[[], PromptSession]",
) -> "Callable[[], HTML]":
    shell_state = ShellToolbarState(
        enabled=shell_context.enabled,
        working_dir=shell_context.working_dir,
        started_at=time.monotonic(),
    )

    def get_toolbar() -> HTML:
        global _copy_notice
        try:
            current_input_text = session_factory().default_buffer.text
        except Exception:
            current_input_text = ""
        result = render_input_toolbar(
            agent_name=agent_name,
            toolbar_color=toolbar_color,
            agent_provider=agent_provider,
            multiline_mode=in_multiline_mode,
            shell_state=shell_state,
            app_version=app_version,
            copy_notice=_copy_notice,
            copy_notice_until=_copy_notice_until,
            shell_path_switch_delay_seconds=_SHELL_PATH_SWITCH_DELAY_SECONDS,
            current_input_text=current_input_text,
        )
        shell_state.show_path_segment = result.show_shell_path_segment
        if result.clear_copy_notice:
            _copy_notice = None
        return result.html

    return get_toolbar


def _build_cycle_callbacks(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> InputCycleCallbacks:
    def on_cycle_service_tier() -> None:
        llm = resolve_active_llm(agent_provider, agent_name)
        if llm is None or not llm.service_tier_supported:
            return

        next_service_tier = cycle_service_tier(
            llm.service_tier,
            allowed_tiers=getattr(llm, "available_service_tiers", ()),
        )
        try:
            llm.set_service_tier(next_service_tier)
        except ValueError:
            return

    def on_cycle_reasoning() -> None:
        llm = resolve_active_llm(agent_provider, agent_name)
        if llm is None:
            return

        next_setting = cycle_reasoning_setting(llm.reasoning_effort, llm.reasoning_effort_spec)
        if next_setting is None:
            return
        try:
            llm.set_reasoning_effort(next_setting)
        except ValueError:
            return

    def on_cycle_verbosity() -> None:
        llm = resolve_active_llm(agent_provider, agent_name)
        if llm is None:
            return

        next_value = cycle_text_verbosity(llm.text_verbosity, llm.text_verbosity_spec)
        if next_value is None:
            return
        try:
            llm.set_text_verbosity(next_value)
        except ValueError:
            return

    def on_cycle_web_search() -> None:
        llm = resolve_active_llm(agent_provider, agent_name)
        if llm is None or not llm.web_search_supported:
            return

        try:
            llm.set_web_search_enabled(not llm.web_search_enabled)
        except ValueError:
            return

    def on_cycle_web_fetch() -> None:
        llm = resolve_active_llm(agent_provider, agent_name)
        if llm is None or not llm.web_fetch_supported:
            return

        try:
            llm.set_web_fetch_enabled(not llm.web_fetch_enabled)
        except ValueError:
            return

    return InputCycleCallbacks(
        on_cycle_service_tier=on_cycle_service_tier,
        on_cycle_reasoning=on_cycle_reasoning,
        on_cycle_verbosity=on_cycle_verbosity,
        on_cycle_web_search=on_cycle_web_search,
        on_cycle_web_fetch=on_cycle_web_fetch,
    )


def _resolve_shell_context(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> tuple[ShellInputContext, object | None]:
    shell_context = ShellInputContext()
    shell_agent = None
    if agent_provider is None:
        return shell_context, None

    try:
        shell_agent = agent_provider._agent(agent_name)
    except Exception:
        return shell_context, None

    if not isinstance(shell_agent, McpAgentProtocol):
        return shell_context, shell_agent

    direct_shell_enabled = shell_agent.shell_runtime_enabled
    shell_access_modes = shell_agent.shell_access_modes
    shell_runtime = None

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

    if direct_shell_enabled:
        shell_runtime = shell_agent.shell_runtime

    shell_context.enabled = direct_shell_enabled or bool(sub_agent_shells)
    shell_context.access_modes = shell_access_modes
    shell_context.runtime = shell_runtime
    if shell_context.enabled and shell_runtime:
        runtime_info = shell_runtime.runtime_info()
        shell_context.name = runtime_info.get("name")
        try:
            shell_context.working_dir = shell_runtime.working_directory()
        except Exception:
            shell_context.working_dir = None
    return shell_context, shell_agent


def resolve_shell_working_dir(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> Path | None:
    shell_context, _ = _resolve_shell_context(agent_name=agent_name, agent_provider=agent_provider)
    return shell_context.working_dir


def _build_prompt_text_resolver(
    *,
    session_factory: "Callable[[], PromptSession]",
    agent_name: str,
    default_agent_name: str | None,
    show_default: bool,
    default: str,
    shell_enabled: bool,
) -> "Callable[[], HTML]":
    def _resolve_prompt_text() -> HTML:
        buffer_text = ""
        try:
            buffer_text = session_factory().default_buffer.text
        except Exception:
            buffer_text = ""

        if buffer_text.lstrip().startswith("!"):
            arrow_segment = "<ansired>❯</ansired>"
        else:
            arrow_segment = "<ansibrightyellow>❯</ansibrightyellow>" if shell_enabled else "❯"

        if is_default_agent_name(agent_name, default_agent_name=default_agent_name):
            prompt_text = f"{arrow_segment} "
        else:
            prompt_text = f"<ansibrightblue>{agent_name}</ansibrightblue> {arrow_segment} "
        if show_default and default and default != "STOP":
            prompt_text = f"{prompt_text} [<ansigreen>{default}</ansigreen>] "
        return HTML(prompt_text)

    return _resolve_prompt_text


def _resolve_default_agent_name(agent_provider: "AgentApp | None") -> str | None:
    if agent_provider is None:
        return None
    try:
        return agent_provider.get_default_agent_name()
    except Exception:
        try:
            return getattr(agent_provider._agent(None), "name", None)
        except Exception:
            return None


def _show_stop_hint_message(
    *,
    default: str,
    show_stop_hint: bool,
) -> None:
    if not show_stop_hint:
        return
    if default == "STOP":
        rich_print("Enter a prompt, [red]STOP[/red] or [red]Ctrl+D[/red] to finish")
        if default:
            rich_print(f"Press <ENTER> to use the default prompt:\n[cyan]{default}[/cyan]")


def _show_input_help_banner(
    *,
    is_human_input: bool,
) -> None:
    if is_human_input:
        rich_print("[dim]Type /help for commands. Ctrl+T toggles multiline mode.[/dim]")
        return

    rich_print(
        """[dim]Use '/' for commands, '!' for shell. '#' to query, '@' to switch agents\n"""
        """CTRL+T multiline, CTRL+Y copy last message, CTRL+E external editor.\n"""
        """CTRL+Space or Tab for path completion. Use /attach, `^file:`, or `^url:` for attachments. F10 to clear.[/dim]"""
    )


def _show_model_shortcut_hints(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> None:
    startup_llm = resolve_active_llm(agent_provider, agent_name)
    shortcut_hints = build_model_shortcut_hints(startup_llm)
    if not shortcut_hints:
        return
    rich_print("[dim]Model shortcuts:[/dim]")
    for hint in shortcut_hints:
        rich_print(f"[dim]  {hint.key} = {hint.label} ({hint.values_text})[/dim]")
    rich_print()


async def _show_shell_startup(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
    shell_context: ShellInputContext,
    shell_agent: object | None,
    is_human_input: bool,
) -> None:
    if not shell_context.enabled:
        return

    rich_print(format_shell_notice(shell_context.access_modes, shell_context.runtime))
    if agent_provider and not is_human_input:
        await _display_all_agents_with_hierarchy(available_agents, agent_provider)
    await _show_streaming_status(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_agent=shell_agent,
    )
    if agent_provider and not is_human_input and _startup_notices:
        _render_startup_notices(agent_name=agent_name, agent_provider=agent_provider)


async def _show_streaming_status(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
    shell_agent: object | None,
) -> None:
    if agent_provider is None:
        return

    logger_settings = None
    agent_context = None
    try:
        active_agent = shell_agent or agent_provider._agent(agent_name)
        try:
            agent_context = getattr(active_agent, "context")
        except Exception:
            agent_context = None
        if agent_context and agent_context.config:
            logger_settings = agent_context.config.logger
    except Exception:
        logger_settings = None

    if not logger_settings or not getattr(logger_settings, "show_chat", True):
        return

    _show_streaming_mode_notice(agent_provider, logger_settings)
    if agent_context and agent_context.config:
        model_source = getattr(agent_context.config, "model_source", None)
        if model_source:
            rich_print(f"[dim]Model selected via {model_source}[/dim]")

    try:
        active_agent = shell_agent or agent_provider._agent(agent_name)
        llm = getattr(active_agent, "llm", None)
        get_hf_info = getattr(llm, "get_hf_display_info", None) if llm else None
        if get_hf_info:
            hf_info = get_hf_info()
            model = hf_info.get("model", "unknown")
            provider = hf_info.get("provider", "auto-routing")
            rich_print(f"[dim]HuggingFace: {model} via {provider}[/dim]")
    except Exception:
        return


def _show_streaming_mode_notice(agent_provider: "AgentApp", logger_settings: object) -> None:
    agent_types = agent_provider.registered_agent_types().values()
    has_parallel = any(agent_type == AgentType.PARALLEL for agent_type in agent_types)
    if has_parallel:
        rich_print("[dim]Markdown Streaming disabled (Parallel Agents configured)[/dim]")
        return

    streaming_enabled = getattr(logger_settings, "streaming_display", True)
    streaming_mode = getattr(logger_settings, "streaming", "markdown")
    if streaming_enabled and streaming_mode != "none":
        rich_print(f"[dim]Experimental: Streaming Enabled - {streaming_mode} mode[/dim]")


def _render_startup_notices(
    *,
    agent_name: str,
    agent_provider: "AgentApp",
) -> None:
    for notice in _startup_notices:
        if isinstance(notice, StartupNotice) and notice.render_markdown:
            target_agent_name = notice.agent_name or agent_name
            target_display = None
            try:
                target_agent = agent_provider._agent(target_agent_name)
                target_display = getattr(target_agent, "display", None)
            except Exception:
                target_display = None

            if target_display is not None:
                if notice.title:
                    target_display.show_status_message(Text(notice.title, style="bold"))
                target_display.display_message(
                    content=notice.text,
                    message_type=MessageType.ASSISTANT,
                    name=target_agent_name,
                    right_info=notice.right_info or "",
                    truncate_content=False,
                    render_markdown=True,
                )
            else:
                rich_print(notice.text)
            continue

        rich_print(notice)
    _startup_notices.clear()


async def _show_input_startup(
    *,
    agent_name: str,
    default: str,
    show_stop_hint: bool,
    is_human_input: bool,
    shell_context: ShellInputContext,
    shell_agent: object | None,
    agent_provider: "AgentApp | None",
) -> None:
    global help_message_shown
    _show_stop_hint_message(default=default, show_stop_hint=show_stop_hint)
    if help_message_shown:
        return

    _show_input_help_banner(is_human_input=is_human_input)
    _show_model_shortcut_hints(agent_name=agent_name, agent_provider=agent_provider)
    await _show_shell_startup(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_context=shell_context,
        shell_agent=shell_agent,
        is_human_input=is_human_input,
    )
    rich_print()
    help_message_shown = True


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
    _initialize_prompt_input_state(
        agent_name=agent_name,
        multiline=multiline,
        available_agent_names=available_agent_names,
        agent_provider=agent_provider,
    )
    shell_context, shell_agent = _resolve_shell_context(
        agent_name=agent_name,
        agent_provider=agent_provider,
    )
    session: PromptSession | None = None

    def session_factory() -> PromptSession:
        return cast("PromptSession", session)

    toolbar = _build_toolbar(
        agent_name=agent_name,
        toolbar_color=toolbar_color,
        agent_provider=agent_provider,
        shell_context=shell_context,
        session_factory=session_factory,
    )
    session = create_prompt_session(
        history=agent_histories[agent_name],
        completer=AgentCompleter(
            agents=list(available_agents) if available_agents else [],
            agent_types=agent_types or {},
            is_human_input=is_human_input,
            current_agent=agent_name,
            agent_provider=agent_provider,
            noenv_mode=noenv_mode,
            cwd=resolve_shell_working_dir(
                agent_name=agent_name,
                agent_provider=agent_provider,
            ),
        ),
        lexer=ShellPrefixLexer(),
        multiline_filter=Condition(lambda: in_multiline_mode),
        toolbar=toolbar,
        style=build_prompt_style(),
    )

    cycle_callbacks = _build_cycle_callbacks(
        agent_name=agent_name,
        agent_provider=agent_provider,
    )
    bindings = create_keybindings(
        on_toggle_multiline=_build_multiline_toggle(session_factory),
        on_cycle_service_tier=cycle_callbacks.on_cycle_service_tier,
        on_cycle_reasoning=cycle_callbacks.on_cycle_reasoning,
        on_cycle_verbosity=cycle_callbacks.on_cycle_verbosity,
        on_cycle_web_search=cycle_callbacks.on_cycle_web_search,
        on_cycle_web_fetch=cycle_callbacks.on_cycle_web_fetch,
        app=session.app,
        agent_provider=agent_provider,
        agent_name=agent_name,
    )
    session.app.key_bindings = bindings

    toolbar_switch_task = None
    if shell_context.enabled:
        toolbar_switch_task = start_toolbar_switch_task(
            session,
            _SHELL_PATH_SWITCH_DELAY_SECONDS,
        )

    await _show_input_startup(
        agent_name=agent_name,
        default=default,
        show_stop_hint=show_stop_hint,
        is_human_input=is_human_input,
        shell_context=shell_context,
        shell_agent=shell_agent,
        agent_provider=agent_provider,
    )
    buffer_default = pre_populate_buffer if pre_populate_buffer else default
    default_agent_name = _resolve_default_agent_name(agent_provider)
    resolve_prompt_text = _build_prompt_text_resolver(
        session_factory=session_factory,
        agent_name=agent_name,
        default_agent_name=default_agent_name,
        show_default=show_default,
        default=default,
        shell_enabled=shell_context.enabled,
    )

    try:
        return await run_prompt_once(
            session=session,
            agent_name=agent_name,
            default_agent_name=default_agent_name,
            default_buffer=buffer_default,
            resolve_prompt_text=resolve_prompt_text,
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
            with suppress_known_runtime_warnings():
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
    default: str | None = None,
) -> str | None:
    """
    Prompt for an argument value with formatting and help text.

    Args:
        arg_name: Name of the argument
        description: Optional description of the argument
        required: Whether this argument is required
        default: Optional default value pre-filled in the prompt

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
        with suppress_known_runtime_warnings():
            arg_value = await prompt_session.prompt_async(
                prompt_text,
                default=default or "",
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
