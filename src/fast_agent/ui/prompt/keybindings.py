"""Prompt key binding helpers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import Lexer
from rich import print as rich_print
from rich.text import Text

from fast_agent.command_actions.accessors import (
    lookup_agent,
    plugin_commands_for_agent,
    plugin_commands_for_provider,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.ui.prompt.attachment_tokens import (
    append_attachment_tokens,
    build_local_attachment_token,
    strip_local_attachment_tokens,
)
from fast_agent.ui.prompt.clipboard_image import paste_clipboard_image_to_temp_png
from fast_agent.ui.prompt.editor import get_text_from_editor
from fast_agent.ui.prompt.parser import try_parse_hash_agent_command
from fast_agent.utils.async_utils import run_in_thread

if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.buffer import Buffer

    from fast_agent.core.agent_app import AgentApp


class ShellPrefixLexer(Lexer):
    """Lexer that highlights shell (!) and comment (#) commands."""

    def lex_document(self, document):
        first_line = document.lines[0] if document.lines else ""
        first_stripped = first_line.lstrip()
        first_line_is_shell = first_stripped.startswith("!")
        first_line_is_hash_command = try_parse_hash_agent_command(first_stripped) is not None

        def get_line_tokens(line_number):
            line = document.lines[line_number]
            if line_number == 0 and first_line_is_shell:
                return [("class:shell-command", line)]
            if line_number == 0 and first_line_is_hash_command:
                return [("class:comment-command", line)]
            return [("", line)]

        return get_line_tokens


class AgentKeyBindings(KeyBindings):
    agent_provider: "AgentApp | None" = None
    current_agent_name: str | None = None


class PromptInputInterrupt(Exception):
    """Internal prompt-toolkit interrupt used instead of raw KeyboardInterrupt."""


logger = get_logger(__name__)


def _print_styled(message: str, style: str) -> None:
    rich_print(Text(message, style=style))


async def paste_clipboard_image_attachment_into_buffer(
    buffer: Buffer,
    *,
    app_ref: Any | None = None,
) -> None:
    """Paste the clipboard image into the input buffer as a local attachment token."""
    try:
        pasted = await run_in_thread(paste_clipboard_image_to_temp_png)
    except asyncio.CancelledError:
        _print_styled("Clipboard image paste cancelled.", "yellow")
        return
    except Exception as exc:
        _print_styled(f"Failed to paste clipboard image: {exc}", "red")
        if app_ref:
            app_ref.invalidate()
        return

    token = build_local_attachment_token(pasted.path)
    buffer.text = append_attachment_tokens(buffer.text, [token])
    buffer.cursor_position = len(buffer.text)
    if app_ref:
        app_ref.invalidate()


def _cycle_completion(buffer: Buffer, *, backwards: bool) -> bool:
    """Cycle through current completion menu items.

    Returns ``True`` when a completion menu was active and the selection moved.
    """
    state = buffer.complete_state
    if state is None:
        return False

    completions = state.completions
    if not completions:
        return False

    total = len(completions)
    current_index = state.complete_index

    if backwards:
        next_index = total - 1 if current_index is None else (current_index - 1) % total
    else:
        next_index = 0 if current_index is None else (current_index + 1) % total

    buffer.go_to_completion(next_index)
    return True


def _accept_completion(buffer: Buffer) -> bool:
    """Accept the currently highlighted completion without submitting input."""
    state = buffer.complete_state
    if state is None:
        return False

    if not state.completions:
        return False

    if state.complete_index is None:
        buffer.go_to_completion(0)

    # Keep the inserted completion text and close the menu.
    buffer.complete_state = None
    return True


def _session_state_module() -> Any:
    from fast_agent.ui.prompt import input as input_state

    return input_state


def _has_any_completions() -> bool:
    from prompt_toolkit.application.current import get_app

    state = get_app().current_buffer.complete_state
    if state is None:
        return False
    return bool(state.completions)


def _event_or_fallback_app(event: Any, fallback_app: Any | None) -> Any | None:
    return event.app or fallback_app


def _invalidate_event_app(event: Any, fallback_app: Any | None) -> None:
    app_ref = _event_or_fallback_app(event, fallback_app)
    if app_ref:
        app_ref.invalidate()


def _invoke_key_callback(
    callback: Callable[[], None] | None,
    event: Any,
    fallback_app: Any | None,
) -> bool:
    if callback is None:
        return False
    callback()
    _invalidate_event_app(event, fallback_app)
    return True


def _clear_local_attachments(event: Any, fallback_app: Any | None) -> None:
    cleared = strip_local_attachment_tokens(event.current_buffer.text)
    if cleared == event.current_buffer.text:
        return
    event.current_buffer.text = cleared
    event.current_buffer.cursor_position = len(cleared)
    _invalidate_event_app(event, fallback_app)


def _toggle_multiline_mode(
    event: Any,
    fallback_app: Any | None,
    on_toggle_multiline: Callable[[bool], None] | None,
) -> None:
    session_state = _session_state_module()
    session_state.in_multiline_mode = not session_state.in_multiline_mode

    _invalidate_event_app(event, fallback_app)

    if on_toggle_multiline:
        on_toggle_multiline(session_state.in_multiline_mode)


def _clear_terminal_screen(event: Any, fallback_app: Any | None) -> None:
    app_ref = _event_or_fallback_app(event, fallback_app)
    if not app_ref:
        return
    renderer = getattr(app_ref, "renderer", None)
    if renderer:
        renderer.clear()
        app_ref.invalidate()


async def _edit_current_buffer_in_editor(event: Any) -> None:
    app_ref = event.app
    current_text = app_ref.current_buffer.text
    try:
        edited_text = await run_in_terminal(
            lambda: get_text_from_editor(current_text),
            render_cli_done=False,
            in_executor=True,
        )
        app_ref.current_buffer.text = edited_text
        app_ref.current_buffer.cursor_position = len(edited_text)
    except asyncio.CancelledError:
        _print_styled("Editor interaction cancelled.", "yellow")
    except Exception as exc:
        _print_styled(f"Error during editor interaction: {exc}", "red")
    finally:
        app_ref.renderer.clear()
        app_ref.invalidate()


def _show_copy_notice(event: Any) -> None:
    session_state = _session_state_module()
    session_state._copy_notice = "COPIED"
    session_state._copy_notice_until = time.monotonic() + 2.0
    if event.app:
        event.app.invalidate()
    else:
        rich_print("\n[green]✓ Copied to clipboard[/green]")


def _copy_last_output(event: Any, kb: AgentKeyBindings) -> None:
    session_state = _session_state_module()
    if session_state._last_copyable_output:
        try:
            import pyperclip

            pyperclip.copy(session_state._last_copyable_output)
            _show_copy_notice(event)
            return
        except Exception:
            pass

    if kb.agent_provider and kb.current_agent_name:
        try:
            agent = kb.agent_provider._agent(kb.current_agent_name)
            for msg in reversed(agent.message_history):
                if msg.role == "assistant":
                    content = msg.last_text()
                    import pyperclip

                    pyperclip.copy(content)
                    _show_copy_notice(event)
                    return
        except Exception:
            pass


def _add_completion_keybindings(
    kb: AgentKeyBindings,
    *,
    on_cycle_service_tier: Callable[[], None] | None,
    app: Any | None,
) -> None:
    @kb.add("c-space")
    @kb.add("c-@")
    def _(event) -> None:
        event.current_buffer.start_completion()

    @kb.add("tab")
    @kb.add("c-i")
    def _(event) -> None:
        if _cycle_completion(event.current_buffer, backwards=False):
            return
        event.current_buffer.start_completion(insert_common_part=True)

    @kb.add("s-tab")
    def _(event) -> None:
        if _cycle_completion(event.current_buffer, backwards=True):
            return
        if _invoke_key_callback(on_cycle_service_tier, event, app):
            return
        event.current_buffer.start_completion(select_last=True)

    @kb.add("c-m", filter=Condition(lambda: _has_any_completions()), eager=True)
    @kb.add("enter", filter=Condition(lambda: _has_any_completions()), eager=True)
    def _(event) -> None:
        _accept_completion(event.current_buffer)


def _add_cycle_keybindings(
    kb: AgentKeyBindings,
    *,
    app: Any | None,
    on_cycle_reasoning: Callable[[], None] | None,
    on_cycle_verbosity: Callable[[], None] | None,
    on_cycle_web_search: Callable[[], None] | None,
    on_cycle_web_fetch: Callable[[], None] | None,
) -> None:
    @kb.add("f6")
    def _(event) -> None:
        if _invoke_key_callback(on_cycle_reasoning, event, app):
            return

    @kb.add("f7")
    def _(event) -> None:
        if _invoke_key_callback(on_cycle_verbosity, event, app):
            return

    @kb.add("f8")
    def _(event) -> None:
        if _invoke_key_callback(on_cycle_web_search, event, app):
            return

    @kb.add("f9")
    def _(event) -> None:
        if _invoke_key_callback(on_cycle_web_fetch, event, app):
            return


def _add_attachment_keybindings(
    kb: AgentKeyBindings,
    *,
    enable_clipboard_image_paste: bool,
    app: Any | None,
) -> None:
    @kb.add("f10")
    def _(event) -> None:
        _clear_local_attachments(event, app)

    if not enable_clipboard_image_paste:
        return

    @kb.add("escape", "c-v")
    @kb.add("escape", "v")
    async def _(event) -> None:
        """Ctrl+Alt+V / Alt+V: Paste an image from the clipboard as a local attachment."""
        await paste_clipboard_image_attachment_into_buffer(
            event.current_buffer,
            app_ref=event.app or app,
        )


def _add_multiline_keybindings(
    kb: AgentKeyBindings,
    *,
    app: Any | None,
    on_toggle_multiline: Callable[[bool], None] | None,
) -> None:
    @kb.add("c-m", filter=Condition(lambda: not _session_state_module().in_multiline_mode))
    def _(event) -> None:
        """Enter: accept input when not in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-j", filter=Condition(lambda: not _session_state_module().in_multiline_mode))
    def _(event) -> None:
        """Ctrl+J: Insert newline when in normal mode."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-m", filter=Condition(lambda: _session_state_module().in_multiline_mode))
    def _(event) -> None:
        """Enter: Insert newline when in multiline mode."""
        event.current_buffer.insert_text("\n")

    @kb.add("c-j", filter=Condition(lambda: _session_state_module().in_multiline_mode))
    def _(event) -> None:
        """Ctrl+J (equivalent to Ctrl+Enter): Submit in multiline mode."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-t")
    def _(event) -> None:
        """Ctrl+T: Toggle multiline mode."""
        _toggle_multiline_mode(event, app, on_toggle_multiline)


def _add_prompt_control_keybindings(kb: AgentKeyBindings, *, app: Any | None) -> None:
    @kb.add("c-l")
    def _(event) -> None:
        """Ctrl+L: Clear and redraw the terminal screen."""
        _clear_terminal_screen(event, app)

    @kb.add("c-u")
    def _(event) -> None:
        """Ctrl+U: Clear the input buffer."""
        event.current_buffer.text = ""

    @kb.add("c-c")
    def _(event) -> None:
        """Ctrl+C: interrupt prompt input (handled by caller policy)."""
        event.app.exit(exception=PromptInputInterrupt())

    @kb.add("c-d")
    def _(event) -> None:
        """Ctrl+D: signal EOF (mapped to STOP by prompt input handler)."""
        event.app.exit(exception=EOFError())

    @kb.add("c-e")
    async def _(event) -> None:
        """Ctrl+E: Edit current buffer in $EDITOR."""
        await _edit_current_buffer_in_editor(event)


def _add_copy_keybinding(
    kb: AgentKeyBindings,
    *,
    agent_provider: "AgentApp | None",
    agent_name: str | None,
) -> None:
    kb.agent_provider = agent_provider
    kb.current_agent_name = agent_name

    @kb.add("c-y")
    async def _(event) -> None:
        """Ctrl+Y: Copy last output (shell or assistant) to clipboard."""
        _copy_last_output(event, kb)


def create_keybindings(
    on_toggle_multiline: Callable[[bool], None] | None = None,
    on_cycle_service_tier: Callable[[], None] | None = None,
    on_cycle_reasoning: Callable[[], None] | None = None,
    on_cycle_verbosity: Callable[[], None] | None = None,
    on_cycle_web_search: Callable[[], None] | None = None,
    on_cycle_web_fetch: Callable[[], None] | None = None,
    enable_clipboard_image_paste: bool = False,
    app: Any | None = None,
    agent_provider: "AgentApp | None" = None,
    agent_name: str | None = None,
) -> AgentKeyBindings:
    """Create custom key bindings."""
    kb = AgentKeyBindings()
    _add_completion_keybindings(
        kb,
        on_cycle_service_tier=on_cycle_service_tier,
        app=app,
    )
    _add_cycle_keybindings(
        kb,
        app=app,
        on_cycle_reasoning=on_cycle_reasoning,
        on_cycle_verbosity=on_cycle_verbosity,
        on_cycle_web_search=on_cycle_web_search,
        on_cycle_web_fetch=on_cycle_web_fetch,
    )
    _add_attachment_keybindings(
        kb,
        enable_clipboard_image_paste=enable_clipboard_image_paste,
        app=app,
    )
    _add_multiline_keybindings(
        kb,
        app=app,
        on_toggle_multiline=on_toggle_multiline,
    )
    _add_prompt_control_keybindings(kb, app=app)
    _add_copy_keybinding(
        kb,
        agent_provider=agent_provider,
        agent_name=agent_name,
    )
    _add_plugin_command_keybindings(kb, agent_provider=agent_provider, agent_name=agent_name)

    return kb


def _add_plugin_command_keybindings(
    kb: AgentKeyBindings,
    *,
    agent_provider: "AgentApp | None",
    agent_name: str | None,
) -> None:
    if agent_provider is None or agent_name is None:
        return

    commands = {}
    global_commands = plugin_commands_for_provider(agent_provider)
    if global_commands:
        commands.update(global_commands)
    agent = lookup_agent(agent_provider, agent_name)
    agent_commands = plugin_commands_for_agent(agent)
    if agent_commands:
        commands.update(agent_commands)

    for command_name, spec in commands.items():
        if not spec.key:
            continue
        keys = tuple(part for part in spec.key.split() if part)
        if not keys:
            continue

        try:

            @kb.add(*keys)
            def _(event, command_name=command_name) -> None:
                command = f"/{command_name}"
                event.current_buffer.text = command
                event.current_buffer.cursor_position = len(command)
                event.current_buffer.validate_and_handle()

        except Exception as exc:
            logger.warning(
                "Ignoring invalid plugin command keybinding",
                agent=agent_name,
                command=command_name,
                key=spec.key,
                error=str(exc),
            )
