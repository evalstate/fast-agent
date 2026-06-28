"""
Enhanced prompt functionality with advanced prompt_toolkit features.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from html import escape as escape_html
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, TypeVar, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, WordCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from rich import print as rich_print
from rich.markup import escape as escape_markup
from rich.text import Text

from fast_agent.commands.model_capabilities import (
    available_service_tier_values,
    resolve_reasoning_effort,
    resolve_reasoning_effort_spec,
    resolve_service_tier,
    resolve_service_tier_supported,
    resolve_text_verbosity,
    resolve_text_verbosity_spec,
    set_reasoning_effort,
    set_service_tier,
    set_text_verbosity,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.ui.mcp_display import render_mcp_status
from fast_agent.ui.model_binary_toggles import (
    WEB_FETCH_TOGGLE,
    WEB_SEARCH_TOGGLE,
    ModelBinaryToggle,
    cycle_model_binary_toggle,
)
from fast_agent.ui.model_shortcuts import (
    cycle_reasoning_setting,
    cycle_text_verbosity,
)
from fast_agent.ui.prompt import input_startup
from fast_agent.ui.prompt.agent_info import collect_tool_children
from fast_agent.ui.prompt.agent_info import (
    display_agent_info as _display_agent_info_impl,
)
from fast_agent.ui.prompt.agent_info import (
    display_all_agents_with_hierarchy as _display_all_agents_with_hierarchy_impl,
)
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
    ToolbarRenderCache,
    render_input_toolbar,
    resolve_active_llm,
)
from fast_agent.ui.prompt.keybindings import ShellPrefixLexer, create_keybindings
from fast_agent.ui.prompt.special_commands import handle_special_commands_async
from fast_agent.ui.service_tier_display import cycle_service_tier
from fast_agent.utils.env import env_flag

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from fast_agent.agents.agent_types import AgentType
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.ui.command_payloads import CommandPayload


# Get the application version
try:
    app_version = version("fast-agent-mcp")
except Exception:
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
logger = get_logger(__name__)
ModelSettingT = TypeVar("ModelSettingT")


class _LazyAgentCompleter(Completer):
    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._delegate: Completer | None = None
        self._lock = threading.Lock()

    def _get_delegate(self) -> Completer:
        if self._delegate is None:
            with self._lock:
                if self._delegate is None:
                    from fast_agent.ui.prompt.completer import AgentCompleter

                    self._delegate = AgentCompleter(**self._kwargs)
                    self._kwargs = {}
        return self._delegate

    def get_completions(self, document: Any, complete_event: Any):
        yield from self._get_delegate().get_completions(document, complete_event)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_TOOLBAR_DIAGNOSTICS_ENABLED = env_flag("FAST_AGENT_TOOLBAR_DIAGNOSTICS")
_TOOLBAR_DIAGNOSTICS_THRESHOLD_MS = _env_float("FAST_AGENT_TOOLBAR_DIAGNOSTICS_THRESHOLD_MS", 50.0)


def set_last_copyable_output(output: str) -> None:
    """Set the last copyable output for Ctrl+Y clipboard functionality."""
    global _last_copyable_output
    _last_copyable_output = output


# Track whether help text has been shown globally
help_message_shown: bool = False

# Track which agents have shown their info
_agent_info_shown = set()


StartupNotice = input_startup.StartupNotice


# One-off notices to render at the top of the prompt UI
_startup_notices: list[object] = []


def _sync_startup_output() -> None:
    input_startup.rich_print = rich_print


def queue_startup_notice(notice: object) -> None:
    input_startup.queue_startup_notice(_startup_notices, notice)


def queue_startup_markdown_notice(
    text: str,
    *,
    title: str | None = None,
    style: str | None = None,
    right_info: str | None = None,
    agent_name: str | None = None,
) -> None:
    input_startup.queue_startup_markdown_notice(
        _startup_notices,
        text,
        title=title,
        style=style,
        right_info=right_info,
        agent_name=agent_name,
    )


async def show_mcp_status(agent_name: str, agent_provider: "AgentApp | None") -> None:
    if agent_provider is None:
        rich_print("[red]No agent provider available[/red]")
        return

    try:
        agent = agent_provider._agent(agent_name)
    except Exception as exc:
        rich_print(Text(f"Unable to load agent '{agent_name}': {exc}", style="red"))
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


def _show_fast_agent_home_summary(agent_provider: "AgentApp | None") -> None:
    _sync_startup_output()
    input_startup.show_fast_agent_home_summary(agent_provider)


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


@dataclass(frozen=True, slots=True)
class ResolvedShellInput:
    context: ShellInputContext
    agent: object | None = None


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
        with suppress(Exception):
            available_agents = set(agent_provider.visible_agent_names(force_include=agent_name))

    if agent_name not in agent_histories:
        agent_histories[agent_name] = InMemoryHistory()


def _build_multiline_toggle(
    session_factory: "Callable[[], PromptSession]",
) -> "Callable[[bool], None]":
    def on_multiline_toggle(enabled: bool) -> None:
        del enabled
        session = session_factory()
        if session.app:
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
    toolbar_cache = ToolbarRenderCache()

    def get_toolbar() -> HTML:
        global _copy_notice
        started_at = time.perf_counter() if _TOOLBAR_DIAGNOSTICS_ENABLED else 0.0
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
            cache=toolbar_cache,
        )
        shell_state.show_path_segment = result.show_shell_path_segment
        if result.clear_copy_notice:
            _copy_notice = None
        if _TOOLBAR_DIAGNOSTICS_ENABLED:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            if elapsed_ms >= _TOOLBAR_DIAGNOSTICS_THRESHOLD_MS:
                logger.warning(
                    "Slow prompt toolbar render",
                    data={
                        "elapsed_ms": round(elapsed_ms, 2),
                        "agent_name": agent_name,
                        "input_length": len(current_input_text),
                        "agent_state_cache_hit": result.agent_state_cache_hit,
                        "attachment_summary_cache_hit": result.attachment_summary_cache_hit,
                        "attachment_summary_skipped": result.attachment_summary_skipped,
                    },
                )
        return result.html

    return get_toolbar


def _cycle_active_llm(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> "FastAgentLLMProtocol | None":
    return resolve_active_llm(agent_provider, agent_name)


def _cycle_service_tier(llm: "FastAgentLLMProtocol | None") -> None:
    if llm is None or not resolve_service_tier_supported(llm):
        return

    next_service_tier = cycle_service_tier(
        resolve_service_tier(llm),
        allowed_tiers=available_service_tier_values(llm),
    )
    _set_model_setting_if_valid(llm, set_service_tier, next_service_tier)


def _cycle_reasoning(llm: "FastAgentLLMProtocol | None") -> None:
    if llm is None:
        return

    next_setting = cycle_reasoning_setting(
        resolve_reasoning_effort(llm),
        resolve_reasoning_effort_spec(llm),
    )
    if next_setting is None:
        return
    _set_model_setting_if_valid(llm, set_reasoning_effort, next_setting)


def _cycle_verbosity(llm: "FastAgentLLMProtocol | None") -> None:
    if llm is None:
        return

    next_value = cycle_text_verbosity(
        resolve_text_verbosity(llm),
        resolve_text_verbosity_spec(llm),
    )
    if next_value is None:
        return
    _set_model_setting_if_valid(llm, set_text_verbosity, next_value)


def _cycle_binary_model_toggle(
    llm: "FastAgentLLMProtocol | None", toggle: ModelBinaryToggle
) -> None:
    cycle_model_binary_toggle(llm, toggle)


def _set_model_setting_if_valid(
    llm: "FastAgentLLMProtocol",
    setter: "Callable[[FastAgentLLMProtocol, ModelSettingT], None]",
    value: ModelSettingT,
) -> None:
    try:
        setter(llm, value)
    except ValueError:
        return


def _build_cycle_callbacks(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> InputCycleCallbacks:
    def on_cycle_service_tier() -> None:
        _cycle_service_tier(_cycle_active_llm(agent_name=agent_name, agent_provider=agent_provider))

    def on_cycle_reasoning() -> None:
        _cycle_reasoning(_cycle_active_llm(agent_name=agent_name, agent_provider=agent_provider))

    def on_cycle_verbosity() -> None:
        _cycle_verbosity(_cycle_active_llm(agent_name=agent_name, agent_provider=agent_provider))

    def on_cycle_web_search() -> None:
        _cycle_binary_model_toggle(
            _cycle_active_llm(agent_name=agent_name, agent_provider=agent_provider),
            WEB_SEARCH_TOGGLE,
        )

    def on_cycle_web_fetch() -> None:
        _cycle_binary_model_toggle(
            _cycle_active_llm(agent_name=agent_name, agent_provider=agent_provider),
            WEB_FETCH_TOGGLE,
        )

    return InputCycleCallbacks(
        on_cycle_service_tier=on_cycle_service_tier,
        on_cycle_reasoning=on_cycle_reasoning,
        on_cycle_verbosity=on_cycle_verbosity,
        on_cycle_web_search=on_cycle_web_search,
        on_cycle_web_fetch=on_cycle_web_fetch,
    )


def _llm_supports_clipboard_image_paste(llm: "FastAgentLLMProtocol | None") -> bool:
    if llm is None:
        return False
    model_info = llm.model_info
    if model_info is None:
        return False
    return model_info.supports_vision


def _shell_agent_for_input(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> object | None:
    if agent_provider is None:
        return None
    try:
        return agent_provider._agent(agent_name)
    except Exception:
        return None


def _sub_agent_shells(shell_agent: McpAgentProtocol) -> list[McpAgentProtocol]:
    return [
        child
        for child in collect_tool_children(shell_agent)
        if isinstance(child, McpAgentProtocol) and child.shell_runtime_enabled
    ]


def _shell_access_modes(
    shell_agent: McpAgentProtocol,
    sub_agent_shells: list[McpAgentProtocol],
) -> tuple[tuple[str, ...], Any | None]:
    if not sub_agent_shells:
        return shell_agent.shell_access_modes, None

    if not shell_agent.shell_runtime_enabled:
        shell_runtime = sub_agent_shells[0].shell_runtime if len(sub_agent_shells) == 1 else None
        return ("sub-agent",), shell_runtime

    if "sub-agent" in shell_agent.shell_access_modes:
        return shell_agent.shell_access_modes, None
    return (*shell_agent.shell_access_modes, "sub-agent"), None


def _populate_shell_runtime_context(
    shell_context: ShellInputContext,
    shell_runtime: Any | None,
) -> None:
    shell_context.runtime = shell_runtime
    if not shell_context.enabled or shell_runtime is None:
        return

    runtime_info = shell_runtime.runtime_info()
    shell_context.name = runtime_info.get("name")
    try:
        shell_context.working_dir = shell_runtime.working_directory()
    except Exception:
        shell_context.working_dir = None


def _resolve_shell_context(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> ResolvedShellInput:
    shell_context = ShellInputContext()
    shell_agent = _shell_agent_for_input(
        agent_name=agent_name,
        agent_provider=agent_provider,
    )
    if shell_agent is None:
        return ResolvedShellInput(context=shell_context)

    if not isinstance(shell_agent, McpAgentProtocol):
        return ResolvedShellInput(context=shell_context, agent=shell_agent)

    direct_shell_enabled = shell_agent.shell_runtime_enabled
    sub_agent_shells = _sub_agent_shells(shell_agent)
    shell_access_modes, shell_runtime = _shell_access_modes(
        shell_agent,
        sub_agent_shells,
    )

    if direct_shell_enabled:
        shell_runtime = shell_agent.shell_runtime

    shell_context.enabled = direct_shell_enabled or bool(sub_agent_shells)
    shell_context.access_modes = shell_access_modes
    _populate_shell_runtime_context(shell_context, shell_runtime)
    return ResolvedShellInput(context=shell_context, agent=shell_agent)


def resolve_shell_working_dir(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> Path | None:
    shell_input = _resolve_shell_context(agent_name=agent_name, agent_provider=agent_provider)
    return shell_input.context.working_dir


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
    _sync_startup_output()
    input_startup.show_stop_hint_message(default=default, show_stop_hint=show_stop_hint)


def _show_input_help_banner(
    *,
    is_human_input: bool,
    supports_clipboard_image_paste: bool,
) -> None:
    _sync_startup_output()
    input_startup.show_input_help_banner(
        is_human_input=is_human_input,
        supports_clipboard_image_paste=supports_clipboard_image_paste,
    )


def _show_model_shortcut_hints(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> None:
    _sync_startup_output()
    input_startup.show_model_shortcut_hints(
        agent_name=agent_name,
        agent_provider=agent_provider,
    )


async def _show_shell_startup(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
    shell_context: ShellInputContext,
    shell_agent: object | None,
    is_human_input: bool,
) -> None:
    _sync_startup_output()
    await input_startup.show_shell_startup(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_context=shell_context,
        shell_agent=shell_agent,
        is_human_input=is_human_input,
        available_agents=available_agents,
        display_all_agents_with_hierarchy=_display_all_agents_with_hierarchy,
    )


def _render_startup_notices(
    *,
    agent_name: str,
    agent_provider: "AgentApp",
) -> None:
    _sync_startup_output()
    input_startup.render_startup_notices(
        _startup_notices,
        agent_name=agent_name,
        agent_provider=agent_provider,
    )


async def _show_input_startup(
    *,
    agent_name: str,
    default: str,
    show_stop_hint: bool,
    is_human_input: bool,
    shell_context: ShellInputContext,
    shell_agent: object | None,
    agent_provider: "AgentApp | None",
    supports_clipboard_image_paste: bool,
) -> None:
    global help_message_shown
    _show_stop_hint_message(default=default, show_stop_hint=show_stop_hint)
    if help_message_shown:
        return

    _show_input_help_banner(
        is_human_input=is_human_input,
        supports_clipboard_image_paste=supports_clipboard_image_paste,
    )
    _show_model_shortcut_hints(agent_name=agent_name, agent_provider=agent_provider)
    if agent_provider and not is_human_input:
        _show_fast_agent_home_summary(agent_provider)
    await _show_shell_startup(
        agent_name=agent_name,
        agent_provider=agent_provider,
        shell_context=shell_context,
        shell_agent=shell_agent,
        is_human_input=is_human_input,
    )
    if agent_provider and _startup_notices:
        _render_startup_notices(agent_name=agent_name, agent_provider=agent_provider)
    rich_print()
    help_message_shown = True


def _show_a2a_prompt_status(
    *,
    agent_name: str,
    agent_provider: "AgentApp | None",
) -> None:
    _sync_startup_output()
    input_startup.show_a2a_prompt_status(
        agent_name=agent_name,
        agent_provider=agent_provider,
    )


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
    session_manager: "SessionManager | None" = None,
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
        session_manager: Optional session manager for session completions

    Returns:
        User input string or parsed command payload
    """
    _initialize_prompt_input_state(
        agent_name=agent_name,
        multiline=multiline,
        available_agent_names=available_agent_names,
        agent_provider=agent_provider,
    )
    shell_input = _resolve_shell_context(
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
        shell_context=shell_input.context,
        session_factory=session_factory,
    )
    session = create_prompt_session(
        history=agent_histories[agent_name],
        completer=_LazyAgentCompleter(
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
            session_manager=session_manager,
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
    supports_clipboard_image_paste = _llm_supports_clipboard_image_paste(
        resolve_active_llm(agent_provider, agent_name)
    )
    bindings = create_keybindings(
        on_toggle_multiline=_build_multiline_toggle(session_factory),
        on_cycle_service_tier=cycle_callbacks.on_cycle_service_tier,
        on_cycle_reasoning=cycle_callbacks.on_cycle_reasoning,
        on_cycle_verbosity=cycle_callbacks.on_cycle_verbosity,
        on_cycle_web_search=cycle_callbacks.on_cycle_web_search,
        on_cycle_web_fetch=cycle_callbacks.on_cycle_web_fetch,
        enable_clipboard_image_paste=supports_clipboard_image_paste,
        app=session.app,
        agent_provider=agent_provider,
        agent_name=agent_name,
    )
    session.app.key_bindings = bindings

    toolbar_switch_task = None
    if shell_input.context.enabled:
        toolbar_switch_task = start_toolbar_switch_task(
            session,
            _SHELL_PATH_SWITCH_DELAY_SECONDS,
        )

    await _show_input_startup(
        agent_name=agent_name,
        default=default,
        show_stop_hint=show_stop_hint,
        is_human_input=is_human_input,
        shell_context=shell_input.context,
        shell_agent=shell_input.agent,
        agent_provider=agent_provider,
        supports_clipboard_image_paste=supports_clipboard_image_paste,
    )
    _show_a2a_prompt_status(agent_name=agent_name, agent_provider=agent_provider)
    buffer_default = pre_populate_buffer if pre_populate_buffer else default
    default_agent_name = _resolve_default_agent_name(agent_provider)
    resolve_prompt_text = _build_prompt_text_resolver(
        session_factory=session_factory,
        agent_name=agent_name,
        default_agent_name=default_agent_name,
        show_default=show_default,
        default=default,
        shell_enabled=shell_input.context.enabled,
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
        rich_print(f"\n[red]Error getting selection: {escape_markup(str(e))}[/red]")
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
        rich_print(f"  [dim]{escape_markup(arg_name)}: {escape_markup(description)}[/dim]")

    prompt_text = HTML(
        f"Enter value for <ansibrightcyan>{escape_html(arg_name)}</ansibrightcyan> "
        f"{required_text}: "
    )

    # Create prompt session
    prompt_session = PromptSession()

    try:
        # Get user input
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
        rich_print(f"\n[red]Error getting input: {escape_markup(str(e))}[/red]")
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
