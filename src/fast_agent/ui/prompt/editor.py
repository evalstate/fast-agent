"""External editor integration helpers."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from rich import print as rich_print
from rich.text import Text

from fast_agent.utils.commandline import split_commandline


def _print_styled(message: str, style: str) -> None:
    rich_print(Text(message, style=style))


def _editor_command() -> list[str] | None:
    editor_cmd_str = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor_cmd_str:
        editor_cmd_str = "notepad" if os.name == "nt" else "nano"

    try:
        editor_cmd_list = split_commandline(editor_cmd_str, syntax="posix")
        if not editor_cmd_list:
            raise ValueError("Editor command string is empty or invalid.")
    except ValueError as e:
        _print_styled(
            f"Error: Invalid editor command string ('{editor_cmd_str}'): {e}",
            "red",
        )
        return None
    return editor_cmd_list


def _create_editor_temp_file(initial_text: str) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp_file:
            if initial_text:
                tmp_file.write(initial_text)
                tmp_file.flush()
            return tmp_file.name
    except Exception as e:
        _print_styled(f"Error: Could not create temporary file for editor: {e}", "red")
        return None


def _read_editor_file(temp_file_path: str) -> str:
    with Path(temp_file_path).open("r", encoding="utf-8") as f:
        return f.read()


def _remove_editor_temp_file(temp_file_path: str | None) -> None:
    if temp_file_path is None:
        return
    temp_file = Path(temp_file_path)
    if not temp_file.exists():
        return
    try:
        temp_file.unlink()
    except Exception as e:
        _print_styled(
            f"Warning: Could not remove temporary file {temp_file_path}: {e}",
            "yellow",
        )


def _run_editor(editor_cmd_list: list[str], temp_file_path: str) -> str | None:
    try:
        subprocess.run([*editor_cmd_list, temp_file_path], check=True)
        return _read_editor_file(temp_file_path)
    except FileNotFoundError:
        _print_styled(
            f"Error: Editor command '{editor_cmd_list[0]}' not found. "
            f"Please set $VISUAL or $EDITOR correctly, or install '{editor_cmd_list[0]}'.",
            "red",
        )
    except subprocess.CalledProcessError as e:
        _print_styled(
            f"Error: Editor '{editor_cmd_list[0]}' closed with an error (code {e.returncode}).",
            "red",
        )
    except Exception as e:
        _print_styled(
            f"An unexpected error occurred while launching or using the editor: {e}",
            "red",
        )
    return None


def get_text_from_editor(initial_text: str = "") -> str:
    """
    Opens the user\'s configured editor ($VISUAL or $EDITOR) to edit the initial_text.
    Falls back to \'nano\' (Unix) or \'notepad\' (Windows) if neither is set.
    Returns the edited text, or the original text if an error occurs.
    """
    editor_cmd_list = _editor_command()
    if editor_cmd_list is None:
        return initial_text

    temp_file_path = _create_editor_temp_file(initial_text)
    if temp_file_path is None:
        return initial_text

    try:
        edited_text = _run_editor(editor_cmd_list, temp_file_path)
    finally:
        _remove_editor_temp_file(temp_file_path)

    if edited_text is None:
        return initial_text
    return edited_text.strip()
