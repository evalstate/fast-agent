"""Hook to normalize common ripgrep tool-call mistakes before execution."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookContext

logger = get_logger(__name__)

TOOL_NAME_CORRECTIONS = {
    "exec": "execute",
    "executescript": "execute",
    "execscript": "execute",
    "executor": "execute",
    "exec_command": "execute",
}

_COMMAND_DELIMITERS = {"&&", "||", "|", ";"}
_INVALID_RIPGREP_FLAGS = {"-R", "--recursive"}
_RIPGREP_BINARIES = {"rg", "ripgrep", "rg.exe", "ripgrep.exe"}


def _is_ripgrep_executable(token: str) -> bool:
    return Path(token).name.lower() in _RIPGREP_BINARIES


def _normalize_tool_name(name: str) -> tuple[str, bool]:
    corrected = TOOL_NAME_CORRECTIONS.get(name)
    if corrected is not None:
        return corrected, True
    if name.startswith("exec") and name != "execute":
        return "execute", True
    return name, False


def _strip_invalid_ripgrep_flags(command: str) -> tuple[str, bool]:
    """Remove invalid `-R/--recursive` flags from rg invocations.

    Uses shell tokenization so we only remove standalone flag tokens from rg
    command segments (and stop parsing options after `--`).
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return command, False

    rewritten: list[str] = []
    modified = False
    in_ripgrep_command = False
    parsing_options = False

    for token in tokens:
        if token in _COMMAND_DELIMITERS:
            in_ripgrep_command = False
            parsing_options = False
            rewritten.append(token)
            continue

        if _is_ripgrep_executable(token):
            in_ripgrep_command = True
            parsing_options = True
            rewritten.append(token)
            continue

        if in_ripgrep_command and parsing_options:
            if token == "--":
                parsing_options = False
                rewritten.append(token)
                continue
            if token in _INVALID_RIPGREP_FLAGS:
                modified = True
                continue

        rewritten.append(token)

    if not modified:
        return command, False

    return shlex.join(rewritten), True


async def fix_ripgrep_tool_calls(ctx: "HookContext") -> None:
    """Fix hallucinated execute tool names and invalid rg recursive flags."""
    if ctx.hook_type != "before_tool_call":
        return

    message = ctx.message
    if not message.tool_calls:
        return

    for tool_id, tool_call in message.tool_calls.items():
        original_name = tool_call.params.name
        normalized_name, corrected = _normalize_tool_name(original_name)
        if corrected:
            tool_call.params.name = normalized_name
            logger.warning(
                "Corrected hallucinated tool name",
                data={
                    "tool_id": tool_id,
                    "original": original_name,
                    "corrected": normalized_name,
                },
            )

        if tool_call.params.name != "execute":
            continue

        args = tool_call.params.arguments
        if not isinstance(args, dict):
            continue

        command_value = args.get("command")
        if not isinstance(command_value, str):
            continue
        if "rg" not in command_value and "ripgrep" not in command_value:
            continue

        normalized_command, changed = _strip_invalid_ripgrep_flags(command_value)
        if not changed:
            continue

        args["command"] = normalized_command
        logger.warning(
            "Removed invalid recursive flags from ripgrep command",
            data={
                "tool_id": tool_id,
                "original": command_value,
                "modified": normalized_command,
            },
        )
