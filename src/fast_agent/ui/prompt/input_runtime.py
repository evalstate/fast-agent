"""Runtime helpers for interactive prompt session lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import Condition, is_done
from prompt_toolkit.formatted_text import StyleAndTextTuples, to_formatted_text
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.renderer import CPR_Support
from prompt_toolkit.shortcuts.prompt import CompleteStyle
from prompt_toolkit.styles import Style
from rich.markup import escape as escape_markup

from fast_agent.ui.agent_identity import is_default_agent_name
from fast_agent.ui.command_payloads import CommandPayload, EOFCommand, InterruptCommand
from fast_agent.ui.console import rich_print
from fast_agent.ui.prompt.keybindings import PromptInputInterrupt, reset_renderer_cpr_state
from fast_agent.ui.prompt.turn_input import acknowledge_replayed_input
from fast_agent.ui.prompt_marks import emit_prompt_mark, prompt_mark_sequence
from fast_agent.ui.terminal_streams import is_tty_stream

if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.formatted_text.base import AnyFormattedText


_ERASE_PREVIOUS_LINE_SEQ = "\x1b[1A\x1b[2K\r"


def _format_prompt_prefix(agent_name: str, *, default_agent_name: str | None = None) -> str:
    if is_default_agent_name(agent_name, default_agent_name=default_agent_name):
        return "❯"
    return f"{agent_name} ❯"


def _clear_prompt_echo_line(result: str, *, stream: TextIO | None = None) -> None:
    """Erase the just-submitted prompt echo for regular chat input.

    Slash (`/`) and shell (`!`) commands are intentionally left visible because
    we explicitly reprint those command lines below.
    """
    stripped = result.lstrip()
    if not stripped:
        return
    if stripped.startswith(("/", "!")):
        return
    if "\n" in result:
        return

    target = stream or sys.stdout
    if not is_tty_stream(target):
        return

    try:
        target.write(_ERASE_PREVIOUS_LINE_SEQ)
        target.flush()
    except Exception:
        return


def build_prompt_style() -> Style:
    """Build the shared prompt-toolkit style used by enhanced input."""
    return Style.from_dict(
        {
            "completion-menu.completion": "bg:#ansiblack #ansigreen",
            "completion-menu.completion.current": "bg:#ansiblack bold #ansigreen",
            "completion-menu.meta.completion": "bg:#ansiblack #ansiblue",
            "completion-menu.meta.completion.current": "bg:#ansibrightblack #ansiblue",
            "bottom-toolbar": "#ansiblack bg:#ansigray",
            "shell-command": "#ansired",
            "comment-command": "#ansiblue",
        }
    )


def _configure_inline_toolbar(session: PromptSession[str]) -> None:
    """Pin the toolbar directly below the prompt and clear any stale lines below it."""
    # 1. Disable CPR-based full-height stretching so the prompt never pads blank
    #    rows down to the bottom of the terminal (and never hangs on CPR replies).
    renderer = session.app.renderer
    renderer.cpr_support = CPR_Support.NOT_SUPPORTED
    renderer._min_available_height = 0

    original_render = renderer.render

    def _inline_render(app: Any, layout: Any, is_done: bool = False) -> None:
        renderer._min_available_height = 0
        output = renderer.output
        if isinstance(output, Vt100_Output):
            # Wrap in DEC synchronized output (mode 2026) and flush once so
            # erase_down + 2-line redraw renders atomically without flicker.
            original_flush = output.flush
            output.write_raw("\x1b[?2026h")
            output.flush = lambda: None
            try:
                original_render(app, layout, is_done=is_done)
            finally:
                output.flush = original_flush
                output.write_raw("\x1b[?2026l")
                original_flush()
        else:
            original_render(app, layout, is_done=is_done)

    renderer.render = _inline_render

    # 2. Remove `renderer_height_is_known` from the bottom_toolbar container so it
    #    renders immediately below the prompt without waiting for CPR.
    root = session.layout.container
    if isinstance(root, HSplit):
        for child in root.children:
            if (
                isinstance(child, ConditionalContainer)
                and isinstance(child.content, Window)
                and child.content.style == "class:bottom-toolbar"
            ):
                child.filter = Condition(lambda: session.bottom_toolbar is not None) & ~is_done
                break

    # 3. Only reserve completion-menu lines when there are actual completions to
    #    display (Buffer.start_completion creates an empty CompletionState while
    #    completing in a thread, which would otherwise flicker the toolbar down).
    def _inline_default_buffer_control_height() -> Dimension:
        if session.app.is_done:
            return Dimension()

        if session.completer is not None and session.complete_style != CompleteStyle.READLINE_LIKE:
            space = session.reserve_space_for_menu
        else:
            space = 0

        if space:
            complete_state = session.default_buffer.complete_state
            if complete_state is not None and complete_state.completions:
                return Dimension(min=space)

        # Keep one blank breathing line between the prompt and the status bar.
        return Dimension(min=2)

    session._get_default_buffer_control_height = _inline_default_buffer_control_height
    current_window = session.layout.current_window
    if isinstance(current_window, Window):
        current_window.height = _inline_default_buffer_control_height


def _install_resilient_resize_handler(
    session: PromptSession[str],
    *,
    debounce_seconds: float = 0.08,
) -> None:
    app = session.app
    original_on_resize = app._on_resize
    pending_handle: asyncio.TimerHandle | None = None

    def _finish_resize() -> None:
        nonlocal pending_handle
        pending_handle = None
        if app.is_done or not app.is_running:
            return
        renderer = app.renderer
        reset_renderer_cpr_state(renderer)
        renderer._min_available_height = 0
        original_on_resize()

    def _resilient_on_resize() -> None:
        nonlocal pending_handle
        renderer = app.renderer
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            reset_renderer_cpr_state(renderer)
            renderer._min_available_height = 0
            original_on_resize()
            return

        if pending_handle is None:
            reset_renderer_cpr_state(renderer)
            renderer._min_available_height = 0
            renderer.erase(leave_alternate_screen=False)
        else:
            pending_handle.cancel()

        pending_handle = loop.call_later(debounce_seconds, _finish_resize)

    app._on_resize = _resilient_on_resize


def create_prompt_session(
    *, history, completer, lexer, multiline_filter, toolbar, style
) -> PromptSession:
    """Create a configured PromptSession for enhanced input."""
    session = PromptSession(
        history=history,
        completer=completer,
        lexer=lexer,
        complete_while_typing=True,
        multiline=multiline_filter,
        complete_in_thread=True,
        mouse_support=False,
        bottom_toolbar=toolbar,
        style=style,
        erase_when_done=True,
    )
    _configure_inline_toolbar(session)
    _install_resilient_resize_handler(session)
    return session


@contextlib.contextmanager
def _prompt_start_guard(start_code: str):
    """Emit OSC 133 prompt-start and leave prompt-end placement to prompt text."""
    emit_prompt_mark(start_code)
    yield


def _completion_menu_reserved_lines(default: int = 8) -> int:
    try:
        from fast_agent.config import get_settings

        return get_settings().tui.completion_menu_reserved_lines
    except Exception:
        return default


def _emit_prompt_end_mark_if_needed(*, emitted: bool) -> None:
    if not emitted:
        emit_prompt_mark("B")


def _track_accept_state(
    accept_state: dict[str, Any],
    original_accept_handler,
):
    def _track_accept(buffer_obj) -> bool:
        accept_state["accepted_at"] = time.perf_counter()
        accept_state["text"] = buffer_obj.text
        accept_state["completer"] = type(buffer_obj.completer).__name__
        accept_state["had_completions"] = buffer_obj.complete_state is not None
        if original_accept_handler is not None:
            return original_accept_handler(buffer_obj)
        return True

    return _track_accept


def _prompt_text_with_end_mark(
    resolve_prompt_text: "Callable[[], AnyFormattedText]",
    prompt_end_state: dict[str, bool],
) -> StyleAndTextTuples:
    fragments = list(to_formatted_text(resolve_prompt_text()))
    if not prompt_end_state["emitted"]:
        sequence = prompt_mark_sequence("B")
        if sequence:
            fragments.append(("[ZeroWidthEscape]", sequence))
        prompt_end_state["emitted"] = True
    return fragments


async def _prompt_async_result(
    *,
    session: PromptSession,
    default_buffer: str,
    resolve_prompt_text: "Callable[[], AnyFormattedText]",
    prompt_end_state: dict[str, bool],
) -> tuple[str | CommandPayload, float | None]:
    try:
        if isinstance(session, PromptSession):
            renderer = session.app.renderer
            reset_renderer_cpr_state(renderer)
            renderer._min_available_height = 0
        with _prompt_start_guard("A"):
            result = await session.prompt_async(
                lambda: _prompt_text_with_end_mark(resolve_prompt_text, prompt_end_state),
                default=default_buffer,
                set_exception_handler=False,
                reserve_space_for_menu=_completion_menu_reserved_lines(),
            )
        acknowledge_replayed_input()
        return result, time.perf_counter()
    except (KeyboardInterrupt, PromptInputInterrupt):
        _emit_prompt_end_mark_if_needed(emitted=prompt_end_state["emitted"])
        return InterruptCommand(), None
    except EOFError:
        _emit_prompt_end_mark_if_needed(emitted=prompt_end_state["emitted"])
        return EOFCommand(), None
    except Exception as exc:
        _emit_prompt_end_mark_if_needed(emitted=prompt_end_state["emitted"])
        print(f"\nInput error: {type(exc).__name__}: {exc}")
        return "STOP", None


def _truncate_prompt_preview(text: str, *, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _maybe_warn_prompt_shutdown_delay(
    *,
    accept_state: dict[str, Any],
    prompt_returned_at: float,
    stripped: str,
    warn_seconds: float = 0.5,
) -> None:
    accepted_at = accept_state.get("accepted_at")
    if not accepted_at:
        return

    shutdown_delay = prompt_returned_at - accepted_at
    if shutdown_delay < warn_seconds or not stripped.startswith("!"):
        return

    text_preview = _truncate_prompt_preview(str(accept_state.get("text") or "").strip())
    rich_print(
        "[yellow]Prompt shutdown delay[/yellow] "
        f"{shutdown_delay:.2f}s | "
        f"completer={accept_state.get('completer')} "
        f"completions_active={accept_state.get('had_completions')} "
        f"cwd={Path.cwd()} "
        f"input={text_preview!r}"
    )


def _echo_visible_command(
    *,
    stripped: str,
    agent_name: str,
    default_agent_name: str | None,
) -> None:
    if not stripped.startswith(("/", "!")):
        return
    prompt_prefix = _format_prompt_prefix(agent_name, default_agent_name=default_agent_name)
    visible_line = escape_markup(stripped.splitlines()[0])
    rich_print(f"[dim]{escape_markup(prompt_prefix)} {visible_line}[/dim]")


async def run_prompt_once(
    *,
    session: PromptSession,
    agent_name: str,
    default_agent_name: str | None,
    default_buffer: str,
    resolve_prompt_text: "Callable[[], AnyFormattedText]",
    parse_special_input: "Callable[[str], str | CommandPayload]",
) -> str | CommandPayload:
    """Run a single prompt cycle and normalize command/signal outcomes."""
    accept_state: dict[str, Any] = {}
    prompt_end_state = {"emitted": False}
    buffer = session.default_buffer
    original_accept_handler = buffer.accept_handler
    buffer.accept_handler = _track_accept_state(accept_state, original_accept_handler)
    try:
        prompt_result, prompt_returned_at = await _prompt_async_result(
            session=session,
            default_buffer=default_buffer,
            resolve_prompt_text=resolve_prompt_text,
            prompt_end_state=prompt_end_state,
        )
    finally:
        buffer.accept_handler = original_accept_handler

    if not isinstance(prompt_result, str):
        return prompt_result
    result = prompt_result
    if prompt_returned_at is None:
        return result

    _clear_prompt_echo_line(result)

    stripped = result.lstrip()
    _maybe_warn_prompt_shutdown_delay(
        accept_state=accept_state,
        prompt_returned_at=prompt_returned_at,
        stripped=stripped,
    )
    _echo_visible_command(
        stripped=stripped,
        agent_name=agent_name,
        default_agent_name=default_agent_name,
    )

    return parse_special_input(result)


def start_toolbar_switch_task(session: PromptSession, delay_seconds: float) -> asyncio.Task[None]:
    """Start delayed toolbar invalidation task used by shell mode."""

    async def _invalidate_toolbar_on_switch() -> None:
        await asyncio.sleep(delay_seconds)
        if session.app and not session.app.is_done:
            session.app.invalidate()

    return asyncio.create_task(_invalidate_toolbar_on_switch())


async def cleanup_prompt_session(
    *,
    session: PromptSession,
    toolbar_switch_task: asyncio.Task[None] | None,
) -> None:
    """Cancel helper tasks and terminate active prompt app state."""
    if toolbar_switch_task and not toolbar_switch_task.done():
        toolbar_switch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await toolbar_switch_task

    if session.app.is_running:
        session.app.exit()
