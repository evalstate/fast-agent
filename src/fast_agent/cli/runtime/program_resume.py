"""Build OSC 7577 resume commands for interactive CLI runs."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.integrations.program_resume import ProgramResumeCommand

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


def _resolve_cli_executable() -> tuple[str, ...]:
    """Return an OSC 7577-compliant executable prefix (bare name or absolute path)."""
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 == "-m" or argv0.endswith("__main__.py"):
        return (os.path.abspath(sys.executable), "-m", "fast_agent")

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


def build_cli_resume_command(
    request: AgentRunRequest,
    session_id: str | None,
    *,
    cwd: Path | None = None,
) -> ProgramResumeCommand | None:
    """Construct the ``fast-agent go --resume <session_id>`` command for OSC 7577."""
    if (
        not session_id
        or request.mode != "interactive"
        or not request.is_repl
        or request.no_home
        or request.isolated
        or request.model_base_url is not None
    ):
        return None

    argv: list[str] = [*_resolve_cli_executable(), "go", "--resume", session_id]

    if request.name and request.name != "fast-agent":
        argv.extend(["--name", request.name])
    if request.instruction:
        argv.extend(["--instruction", request.instruction])
    if request.config_path:
        argv.extend(["--config-path", str(Path(request.config_path).resolve())])
    if request.server_list:
        argv.extend(["--servers", ",".join(request.server_list)])
    if request.agent_cards:
        for card in request.agent_cards:
            argv.extend(["--card", card])
    if request.card_tools:
        for tool in request.card_tools:
            argv.extend(["--card-tool", tool])
    if request.agent_name:
        argv.extend(["--agent", request.agent_name])
    if request.environment:
        argv.extend(["--environment", request.environment])
    if request.skills_directory is not None:
        argv.extend(["--skills-dir", str(request.skills_directory.resolve())])
    if request.home is not None:
        argv.extend(["--home", str(request.home.resolve())])
    if request.shell_runtime:
        argv.append("--shell")
    elif request.no_shell:
        argv.append("--no-shell")
    if request.subagent_model:
        argv.extend(["--subagent-model", request.subagent_model])
    elif request.subagents is True:
        argv.append("--subagents")
    elif request.subagents is False:
        argv.append("--no-subagents")
    if request.image_gen_model:
        argv.extend(["--image-gen-model", request.image_gen_model])
    elif request.image_gen is True:
        argv.append("--image-gen")
    elif request.image_gen is False:
        argv.append("--no-image-gen")
    if request.ptc is True:
        argv.append("--ptc")
    elif request.ptc is False:
        argv.append("--no-ptc")
    if request.ptc_search is True:
        argv.append("--ptc-search")
    if request.reload:
        argv.append("--reload")
    if request.watch:
        argv.append("--watch")
    if request.quiet:
        argv.append("--quiet")

    resolved_cwd = (cwd or request.workspace or Path.cwd()).resolve()
    return ProgramResumeCommand(argv=tuple(argv), cwd=str(resolved_cwd))
