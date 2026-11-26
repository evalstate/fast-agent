"""Build OSC 7577 resume commands for interactive CLI runs."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.integrations.program_resume import ProgramResumeCommand

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.cli.runtime.run_request import AgentRunRequest

_CAPTURED_ARGV: tuple[str, ...] | None = None
_CAPTURED_CWD: Path | None = None

# Flags whose presence makes resuming either invalid (--local/--model-url
# conflict with --resume) or unsafe (raw bearer tokens should not be persisted
# in terminal resume records).
_NON_RESUMABLE_FLAGS = frozenset({"--local", "--model-url", "--model-auth", "--auth"})


def capture_cli_invocation(
    argv: Sequence[str] | None = None,
    *,
    cwd: Path | None = None,
) -> None:
    """Record the original CLI ``argv`` and ``cwd`` before flag normalization."""
    global _CAPTURED_ARGV, _CAPTURED_CWD

    _CAPTURED_ARGV = tuple(sys.argv if argv is None else argv)
    _CAPTURED_CWD = (Path.cwd() if cwd is None else cwd).resolve()


def _resolve_cli_executable(argv0: str) -> tuple[str, ...]:
    """Return an OSC 7577-compliant executable prefix (bare name or absolute path)."""
    if argv0 == "-m" or argv0.endswith("__main__.py"):
        return (os.path.abspath(sys.executable), "-m", "fast_agent.cli")

    resolved: Path | None = None
    if argv0:
        if os.path.isabs(argv0) or os.sep in argv0 or (os.altsep and os.altsep in argv0):
            candidate = Path(argv0).resolve()
            if candidate.exists():
                resolved = candidate
        else:
            found = shutil.which(argv0)
            if found:
                resolved = Path(found).resolve()

    # Prefer bare "fast-agent" when it resolves to the same binary on PATH
    # outside an active virtual environment (so upgrades keep working). Inside
    # a virtualenv / uv run, report the absolute path per OSC 7577 guidance.
    path_binary = shutil.which("fast-agent")
    if (
        resolved is not None
        and path_binary is not None
        and not os.environ.get("VIRTUAL_ENV")
        and Path(path_binary).resolve() == resolved
    ):
        return ("fast-agent",)

    if resolved is not None:
        return (str(resolved),)

    return ("fast-agent",)


def _is_option_token(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _replay_args(raw_args: Sequence[str]) -> list[str] | None:
    """Clean original CLI arguments for ``--resume <session_id>`` replay.

    Returns ``None`` when the invocation carries flags that cannot be resumed
    safely (e.g. ``--local``, ``--model-url``, or raw bearer-token flags).
    """
    cleaned: list[str] = []
    has_go_subcommand = False
    index = 0
    total = len(raw_args)

    while index < total:
        arg = raw_args[index]

        if arg == "local" and not has_go_subcommand and not cleaned:
            return None
        if arg in _NON_RESUMABLE_FLAGS or any(
            arg.startswith(f"{flag}=") for flag in _NON_RESUMABLE_FLAGS
        ):
            return None

        # Strip any prior --resume / -r flag (with optional value).
        if arg in ("--resume", "-r"):
            index += 1
            if index < total and not _is_option_token(raw_args[index]):
                index += 1
            continue
        if arg.startswith(("--resume=", "-r=")):
            index += 1
            continue

        # Strip single-model --model / --models so --resume restores the
        # session's active model (including /model switches). Keep multi-model
        # fan-out specs (containing ",") since resume needs the fan-out setup.
        if arg in ("--model", "--models"):
            if index + 1 < total and not _is_option_token(raw_args[index + 1]):
                value = raw_args[index + 1]
                if "," in value:
                    cleaned.extend((arg, value))
                index += 2
            else:
                index += 1
            continue
        if arg.startswith(("--model=", "--models=")):
            value = arg.split("=", 1)[1]
            if "," in value:
                cleaned.append(arg)
            index += 1
            continue

        if arg == "go" and not has_go_subcommand:
            has_go_subcommand = True
        cleaned.append(arg)
        index += 1

    if not has_go_subcommand:
        cleaned.insert(0, "go")
    return cleaned


def build_cli_resume_command(
    request: AgentRunRequest,
    session_id: str | None,
    *,
    raw_argv: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> ProgramResumeCommand | None:
    """Construct the OSC 7577 resume command by replaying the original CLI argv."""
    if (
        not session_id
        or request.mode != "interactive"
        or not request.is_repl
        or request.no_home
        or request.isolated
        or request.model_base_url is not None
    ):
        return None

    source_argv = (
        tuple(raw_argv)
        if raw_argv is not None
        else _CAPTURED_ARGV
        if _CAPTURED_ARGV is not None
        else tuple(sys.argv)
    )
    argv0 = source_argv[0] if source_argv else ""
    replayed = _replay_args(source_argv[1:])
    if replayed is None:
        return None

    argv = (*_resolve_cli_executable(argv0), *replayed, "--resume", session_id)
    resolved_cwd = (
        cwd
        if cwd is not None
        else _CAPTURED_CWD
        if _CAPTURED_CWD is not None
        else (request.workspace or Path.cwd())
    ).resolve()
    return ProgramResumeCommand(argv=argv, cwd=str(resolved_cwd))
