"""Lightweight guard for the smart card-pack ripgrep helper."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.hooks.hook_context import HookContext

logger = get_logger(__name__)

_TOOL_NAME_CORRECTIONS = {
    "exec": "execute",
    "executescript": "execute",
    "execscript": "execute",
    "executor": "execute",
    "exec_command": "execute",
}
_INVALID_RIPGREP_FLAGS = {"-R", "--recursive"}
_RIPGREP_BINARIES = {"rg", "ripgrep", "rg.exe", "ripgrep.exe"}
_ALLOWED_BINARIES = {
    "rg",
    "ripgrep",
    "rg.exe",
    "ripgrep.exe",
    "find",
    "fd",
    "fdfind",
    "ls",
    "wc",
    "sort",
    "head",
    "tail",
    "cut",
    "uniq",
    "tr",
    "grep",
    "sed",
    "awk",
    "xargs",
    "printf",
    "echo",
}
_DEFAULT_COMMAND_BUDGET = 6


def _first_token(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    return tokens[0] if tokens else None


def _is_ripgrep_command(command: str) -> bool:
    first = _first_token(command)
    return bool(first and Path(first).name.lower() in _RIPGREP_BINARIES)


def _split_shell_segments(command: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return None

    segments: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token in {"|", "||", "&&", ";"}:
            if current:
                segments.append(shlex.join(current))
                current = []
            continue
        current.append(token)

    if current:
        segments.append(shlex.join(current))
    return segments


def _normalize_tool_name(name: str) -> tuple[str, bool]:
    corrected = _TOOL_NAME_CORRECTIONS.get(name)
    if corrected is not None:
        return corrected, True
    if name.startswith("exec") and name != "execute":
        return "execute", True
    return name, False


def _is_allowed_shell_command(command: str) -> bool:
    if not command.strip():
        return False
    if any(token in command for token in (">", "<", "$(", "`")):
        return False

    segments = _split_shell_segments(command)
    if not segments:
        return False

    for segment in segments:
        first = _first_token(segment)
        if not first or Path(first).name.lower() not in _ALLOWED_BINARIES:
            return False

    return True


def _extract_text_items(content: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(content, list):
        return texts

    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
            continue

        item_type = getattr(item, "type", None)
        item_text = getattr(item, "text", None)
        if item_type == "text" and isinstance(item_text, str):
            texts.append(item_text)

    return texts


def _recent_messages(ctx: "HookContext", *, limit: int = 8) -> list[Any]:
    recent = list(ctx.message_history[-limit:])
    delta_messages = getattr(ctx.runner, "delta_messages", None)
    if isinstance(delta_messages, list):
        for message in delta_messages[-limit:]:
            if message not in recent:
                recent.append(message)
    return recent[-limit:]


def _extract_command_budget(ctx: "HookContext") -> int:
    for message in reversed(_recent_messages(ctx)):
        if getattr(message, "role", None) != "user":
            continue

        for text in _extract_text_items(getattr(message, "content", None)):
            candidate = text.strip()
            if not (candidate.startswith("{") and candidate.endswith("}")):
                continue
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            value = payload.get("max_commands")
            if isinstance(value, int):
                return max(1, min(value, _DEFAULT_COMMAND_BUDGET))

    return _DEFAULT_COMMAND_BUDGET


def _extract_repo_root(ctx: "HookContext") -> Path | None:
    for message in reversed(_recent_messages(ctx)):
        if getattr(message, "role", None) != "user":
            continue

        for text in _extract_text_items(getattr(message, "content", None)):
            candidate = text.strip()
            if not (candidate.startswith("{") and candidate.endswith("}")):
                continue
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            value = payload.get("repo_root")
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_absolute() and path.exists() and path.is_dir():
                return path.resolve()

    return None


def _strip_invalid_ripgrep_flags(command: str) -> tuple[str, bool]:
    if not _is_ripgrep_command(command):
        return command, False

    segments = _split_shell_segments(command)
    if not segments or len(segments) > 1:
        return command, False

    try:
        tokens = shlex.split(command)
    except ValueError:
        return command, False

    rewritten = [token for token in tokens if token not in _INVALID_RIPGREP_FLAGS]
    normalized = shlex.join(rewritten)
    return normalized, normalized != command


def _strip_absolute_glob_operands(command: str) -> tuple[str, bool]:
    if not _is_ripgrep_command(command):
        return command, False

    segments = _split_shell_segments(command)
    if not segments or len(segments) > 1:
        return command, False

    try:
        tokens = shlex.split(command)
    except ValueError:
        return command, False

    is_rg_files = "--files" in tokens
    rewritten: list[str] = []
    salvaged_paths: list[str] = []
    changed = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"-g", "--glob"} and i + 1 < len(tokens):
            operand = tokens[i + 1]
            if Path(operand).is_absolute():
                changed = True
                if is_rg_files and Path(operand).exists():
                    salvaged_paths.append(operand)
                i += 2
                continue
            rewritten.extend([token, operand])
            i += 2
            continue

        if token.startswith("--glob="):
            operand = token.split("=", 1)[1]
            if Path(operand).is_absolute():
                changed = True
                if is_rg_files and Path(operand).exists():
                    salvaged_paths.append(operand)
                i += 1
                continue

        rewritten.append(token)
        i += 1

    if salvaged_paths:
        rewritten.extend(salvaged_paths)

    return shlex.join(rewritten), changed


def _normalize_relative_rg_paths(command: str, repo_root: Path | None) -> str:
    if repo_root is None or not _is_ripgrep_command(command):
        return command

    segments = _split_shell_segments(command)
    if not segments or len(segments) > 1:
        return command

    try:
        tokens = shlex.split(command)
    except ValueError:
        return command

    rewritten: list[str] = []
    for idx, token in enumerate(tokens):
        if idx == 0 or token.startswith("-") or "/" not in token:
            rewritten.append(token)
            continue

        token_path = Path(token)
        if token_path.is_absolute() or token_path.exists():
            rewritten.append(token)
            continue

        candidate = (repo_root / token).resolve()
        if candidate.exists():
            rewritten.append(str(candidate))
            continue

        rewritten.append(token)

    return shlex.join(rewritten)


async def fix_ripgrep_tool_calls(ctx: "HookContext") -> None:
    """Normalize tool calls and keep ripgrep search loops bounded."""
    if ctx.hook_type != "before_tool_call":
        return

    message = ctx.message
    if not message.tool_calls:
        return

    seen_commands: set[str] = getattr(ctx.runner, "_ripgrep_seen_commands", set())
    command_count: int = getattr(ctx.runner, "_ripgrep_command_count", 0)
    command_budget: int = getattr(ctx.runner, "_ripgrep_command_budget", 0) or _extract_command_budget(ctx)
    budget_exhausted: bool = bool(getattr(ctx.runner, "_ripgrep_budget_exhausted", False))
    repo_root = _extract_repo_root(ctx)

    for tool_id, tool_call in message.tool_calls.items():
        normalized_name, corrected = _normalize_tool_name(tool_call.params.name)
        if corrected:
            logger.warning(
                "Corrected hallucinated tool name",
                data={"tool_id": tool_id, "original": tool_call.params.name, "corrected": normalized_name},
            )
            tool_call.params.name = normalized_name

        if tool_call.params.name != "execute":
            continue

        args = tool_call.params.arguments
        if not isinstance(args, dict):
            continue

        command = args.get("command")
        if not isinstance(command, str):
            continue

        if budget_exhausted:
            args["command"] = (
                "printf 'Search command budget reached; STOP. Do not call tools again; return final best-effort summary now.\\n'"
            )
            continue

        cleaned, changed_flags = _strip_invalid_ripgrep_flags(command)
        cleaned, changed_globs = _strip_absolute_glob_operands(cleaned)
        cleaned = _normalize_relative_rg_paths(cleaned, repo_root)
        normalized = " ".join(cleaned.split())

        if changed_flags:
            logger.warning(
                "Removed invalid recursive flags from ripgrep command",
                data={"tool_id": tool_id, "original": command, "modified": cleaned},
            )
        elif changed_globs:
            logger.warning(
                "Removed invalid absolute glob operand from ripgrep command",
                data={"tool_id": tool_id, "original": command, "modified": cleaned},
            )

        if not _is_allowed_shell_command(normalized):
            args["command"] = (
                "printf 'Only simple allowed read-only command chains are allowed in this ripgrep helper; summarize with existing results.\\n'"
            )
            continue

        if normalized in seen_commands:
            args["command"] = "printf 'Skipped duplicate rg command to avoid loop.\\n'"
            continue

        if command_count >= command_budget:
            args["command"] = (
                "printf 'Search command budget reached; STOP. Do not call tools again; return final best-effort summary now.\\n'"
            )
            budget_exhausted = True
            continue

        command_count += 1
        seen_commands.add(normalized)
        args["command"] = cleaned

    setattr(ctx.runner, "_ripgrep_seen_commands", seen_commands)
    setattr(ctx.runner, "_ripgrep_command_count", command_count)
    setattr(ctx.runner, "_ripgrep_command_budget", command_budget)
    setattr(ctx.runner, "_ripgrep_budget_exhausted", budget_exhausted)
