"""External editor integration helpers."""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile

from rich import print as rich_print


def get_text_from_editor(initial_text: str = "") -> str:
    """
    Opens the user\'s configured editor ($VISUAL or $EDITOR) to edit the initial_text.
    Falls back to \'nano\' (Unix) or \'notepad\' (Windows) if neither is set.
    Returns the edited text, or the original text if an error occurs.
    """
    editor_cmd_str = os.environ.get("VISUAL") or os.environ.get("EDITOR")

    if not editor_cmd_str:
        if os.name == "nt":  # Windows
            editor_cmd_str = "notepad"
        else:  # Unix-like (Linux, macOS)
            editor_cmd_str = "nano"  # A common, usually available, simple editor

    # Use shlex.split to handle editors with arguments (e.g., "code --wait")
    try:
        editor_cmd_list = shlex.split(editor_cmd_str)
        if not editor_cmd_list:  # Handle empty string from shlex.split
            raise ValueError("Editor command string is empty or invalid.")
    except ValueError as e:
        rich_print(f"[red]Error: Invalid editor command string ('{editor_cmd_str}'): {e}[/red]")
        return initial_text

    # Create a temporary file for the editor to use.
    # Using a suffix can help some editors with syntax highlighting or mode.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as tmp_file:
            if initial_text:
                tmp_file.write(initial_text)
                tmp_file.flush()  # Ensure content is written to disk before editor opens it
            temp_file_path = tmp_file.name
    except Exception as e:
        rich_print(f"[red]Error: Could not create temporary file for editor: {e}[/red]")
        return initial_text

    try:
        # Construct the full command: editor_parts + [temp_file_path]
        # e.g., [\'vim\', \'/tmp/somefile.txt\'] or [\'code\', \'--wait\', \'/tmp/somefile.txt\']
        full_cmd = editor_cmd_list + [temp_file_path]

        # Run the editor. This is a blocking call.
        subprocess.run(full_cmd, check=True)

        # Read the content back from the temporary file.
        with open(temp_file_path, "r", encoding="utf-8") as f:
            edited_text = f.read()

    except FileNotFoundError:
        rich_print(
            f"[red]Error: Editor command '{editor_cmd_list[0]}' not found. "
            f"Please set $VISUAL or $EDITOR correctly, or install '{editor_cmd_list[0]}'.[/red]"
        )
        return initial_text
    except subprocess.CalledProcessError as e:
        rich_print(
            f"[red]Error: Editor '{editor_cmd_list[0]}' closed with an error (code {e.returncode}).[/red]"
        )
        return initial_text
    except Exception as e:
        rich_print(
            f"[red]An unexpected error occurred while launching or using the editor: {e}[/red]"
        )
        return initial_text
    finally:
        # Always attempt to clean up the temporary file.
        if "temp_file_path" in locals() and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                rich_print(
                    f"[yellow]Warning: Could not remove temporary file {temp_file_path}: {e}[/yellow]"
                )

    return edited_text.strip()  # Added strip() to remove trailing newlines often added by editors
