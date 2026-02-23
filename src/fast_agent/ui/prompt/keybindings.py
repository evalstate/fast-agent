"""Prompt key binding helpers."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import Lexer
from rich import print as rich_print

from fast_agent.ui.prompt.editor import get_text_from_editor

if TYPE_CHECKING:
    from collections.abc import Callable

    from prompt_toolkit.buffer import Buffer

    from fast_agent.core.agent_app import AgentApp


class ShellPrefixLexer(Lexer):
    """Lexer that highlights shell (!) and comment (#) commands."""

    def lex_document(self, document):
        def get_line_tokens(line_number):
            line = document.lines[line_number]
            stripped = line.lstrip()
            if stripped.startswith("!"):
                return [("class:shell-command", line)]
            if stripped.startswith("#"):
                return [("class:comment-command", line)]
            return [("", line)]

        return get_line_tokens


class AgentKeyBindings(KeyBindings):
    agent_provider: "AgentApp | None" = None
    current_agent_name: str | None = None


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


def create_keybindings(
    on_toggle_multiline: Callable[[bool], None] | None = None,
    app: Any | None = None,
    agent_provider: "AgentApp | None" = None,
    agent_name: str | None = None,
) -> AgentKeyBindings:
    """Create custom key bindings."""
    kb = AgentKeyBindings()

    def _session_state_module():
        from fast_agent.ui.prompt import session as session_state

        return session_state

    def _should_start_completion(text: str) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return True
        if stripped.startswith("!"):
            return True
        if stripped.startswith(("/", "@", "#")):
            return True
        return True

    @kb.add("c-space")
    @kb.add("c-@")
    def _(event) -> None:
        text = event.current_buffer.document.text_before_cursor
        if not _should_start_completion(text):
            return
        event.current_buffer.start_completion()

    @kb.add("tab")
    @kb.add("c-i")
    def _(event) -> None:
        if _cycle_completion(event.current_buffer, backwards=False):
            return

        text = event.current_buffer.document.text_before_cursor
        if not _should_start_completion(text):
            return
        event.current_buffer.start_completion(insert_common_part=True)

    @kb.add("s-tab")
    def _(event) -> None:
        if _cycle_completion(event.current_buffer, backwards=True):
            return

        text = event.current_buffer.document.text_before_cursor
        if not _should_start_completion(text):
            return
        event.current_buffer.start_completion(select_last=True)

    @kb.add("c-m", filter=Condition(lambda: _has_any_completions()), eager=True)
    @kb.add("enter", filter=Condition(lambda: _has_any_completions()), eager=True)
    def _(event) -> None:
        _accept_completion(event.current_buffer)

    def _has_any_completions() -> bool:
        from prompt_toolkit.application.current import get_app

        state = get_app().current_buffer.complete_state
        if state is None:
            return False
        return len(state.completions) > 0

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
        session_state = _session_state_module()
        session_state.in_multiline_mode = not session_state.in_multiline_mode

        if event.app:
            event.app.invalidate()
        elif app:
            app.invalidate()

        if on_toggle_multiline:
            on_toggle_multiline(session_state.in_multiline_mode)

    @kb.add("c-l")
    def _(event) -> None:
        """Ctrl+L: Clear and redraw the terminal screen."""
        app_ref = event.app or app
        if app_ref and getattr(app_ref, "renderer", None):
            app_ref.renderer.clear()
            app_ref.invalidate()

    @kb.add("c-u")
    def _(event) -> None:
        """Ctrl+U: Clear the input buffer."""
        event.current_buffer.text = ""

    @kb.add("c-c")
    def _(event) -> None:
        """Ctrl+C: interrupt prompt input (handled by caller policy)."""
        event.app.exit(exception=KeyboardInterrupt())

    @kb.add("c-d")
    def _(event) -> None:
        """Ctrl+D: signal EOF (mapped to STOP by prompt input handler)."""
        event.app.exit(exception=EOFError())

    @kb.add("c-e")
    async def _(event) -> None:
        """Ctrl+E: Edit current buffer in $EDITOR."""
        current_text = event.app.current_buffer.text
        try:
            edited_text = await event.app.loop.run_in_executor(
                None, get_text_from_editor, current_text
            )
            event.app.current_buffer.text = edited_text
            event.app.current_buffer.cursor_position = len(edited_text)
        except asyncio.CancelledError:
            rich_print("[yellow]Editor interaction cancelled.[/yellow]")
        except Exception as exc:
            rich_print(f"[red]Error during editor interaction: {exc}[/red]")
        finally:
            if event.app:
                event.app.invalidate()

    kb.agent_provider = agent_provider
    kb.current_agent_name = agent_name

    @kb.add("c-y")
    async def _(event) -> None:
        """Ctrl+Y: Copy last output (shell or assistant) to clipboard."""

        def _show_copy_notice() -> None:
            session_state = _session_state_module()
            session_state._copy_notice = "COPIED"
            session_state._copy_notice_until = time.monotonic() + 2.0
            if event.app:
                event.app.invalidate()
            else:
                rich_print("\n[green]✓ Copied to clipboard[/green]")

        session_state = _session_state_module()
        if session_state._last_copyable_output:
            try:
                import pyperclip

                pyperclip.copy(session_state._last_copyable_output)
                _show_copy_notice()
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
                        _show_copy_notice()
                        return
            except Exception:
                pass

    return kb
