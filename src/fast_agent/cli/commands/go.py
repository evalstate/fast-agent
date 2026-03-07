"""Run an interactive agent directly from the command line."""

from __future__ import annotations

import os
import sys
from pathlib import Path  # noqa: TC003 - typer resolves Path annotations at runtime
from typing import Any, Literal

import typer

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.runtime.agent_setup import run_agent_request
from fast_agent.cli.runtime.request_builders import (
    CARD_EXTENSIONS as _CARD_EXTENSIONS,
)
from fast_agent.cli.runtime.request_builders import (
    DEFAULT_AGENT_CARDS_DIR as _DEFAULT_AGENT_CARDS_DIR,
)
from fast_agent.cli.runtime.request_builders import (
    DEFAULT_TOOL_CARDS_DIR as _DEFAULT_TOOL_CARDS_DIR,
)
from fast_agent.cli.runtime.request_builders import (
    build_command_run_request,
    build_run_agent_kwargs,
    is_multi_model,
    merge_card_sources,
    resolve_default_instruction,
    use_smart_agent,
)
from fast_agent.cli.runtime.request_builders import (
    collect_stdio_commands as _collect_stdio_commands,
)
from fast_agent.cli.runtime.request_builders import (
    resolve_instruction_option as _resolve_instruction_option,
)
from fast_agent.cli.runtime.run_request import (
    AgentRunRequest,
)
from fast_agent.cli.runtime.runner import run_request
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV

CARD_EXTENSIONS = _CARD_EXTENSIONS
DEFAULT_AGENT_CARDS_DIR = _DEFAULT_AGENT_CARDS_DIR
DEFAULT_TOOL_CARDS_DIR = _DEFAULT_TOOL_CARDS_DIR


app = typer.Typer(
    help="Run an interactive agent directly from the command line without creating an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def _is_multi_model(model: str | None) -> bool:
    return is_multi_model(model)


def _use_smart_agent(model: str | None, mode: Literal["interactive", "serve"]) -> bool:
    return use_smart_agent(model, mode)


def _resolve_default_instruction(model: str | None, mode: Literal["interactive", "serve"]) -> str:
    return resolve_default_instruction(model, mode)


def resolve_instruction_option(
    instruction: str | None,
    model: str | None,
    mode: Literal["interactive", "serve"],
) -> tuple[str, str]:
    return _resolve_instruction_option(instruction, model, mode)


def collect_stdio_commands(
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
) -> list[str]:
    return _collect_stdio_commands(npx, uvx, stdio)


_build_run_agent_kwargs = build_run_agent_kwargs


def _merge_card_sources(
    sources: list[str] | None,
    default_dir: Path,
) -> list[str] | None:
    return merge_card_sources(sources, default_dir)


def _build_compat_run_request(**kwargs: Any) -> AgentRunRequest:
    """Build an AgentRunRequest from legacy compatibility keyword arguments.

    This wrapper intentionally accepts ``Any`` because it preserves the legacy
    dynamic call surface used by older integrations while converting into the
    strongly typed ``AgentRunRequest`` model at the boundary.
    """
    return AgentRunRequest(
        name=kwargs.get("name", "fast-agent cli"),
        instruction=kwargs.get("instruction"),
        config_path=kwargs.get("config_path"),
        server_list=kwargs.get("server_list"),
        agent_cards=kwargs.get("agent_cards"),
        card_tools=kwargs.get("card_tools"),
        model=kwargs.get("model"),
        message=kwargs.get("message"),
        prompt_file=kwargs.get("prompt_file"),
        result_file=kwargs.get("result_file"),
        resume=kwargs.get("resume"),
        url_servers=kwargs.get("url_servers"),
        stdio_servers=kwargs.get("stdio_servers"),
        agent_name=kwargs.get("agent_name", "agent"),
        target_agent_name=kwargs.get("target_agent_name"),
        skills_directory=kwargs.get("skills_directory"),
        environment_dir=kwargs.get("environment_dir"),
        noenv=kwargs.get("noenv", False),
        force_smart=kwargs.get("force_smart", False),
        shell_runtime=kwargs.get("shell_runtime", False),
        mode=kwargs.get("mode", "interactive"),
        transport=kwargs.get("transport", "http"),
        host=kwargs.get("host", "0.0.0.0"),
        port=kwargs.get("port", 8000),
        tool_description=kwargs.get("tool_description"),
        tool_name_template=kwargs.get("tool_name_template"),
        instance_scope=kwargs.get("instance_scope", "shared"),
        permissions_enabled=kwargs.get("permissions_enabled", True),
        reload=kwargs.get("reload", False),
        watch=kwargs.get("watch", False),
        quiet=kwargs.get("quiet", False),
        missing_shell_cwd_policy=kwargs.get("missing_shell_cwd_policy"),
    )


async def _run_agent(
    request: AgentRunRequest | None = None,
    **kwargs: Any,
) -> None:
    """Compatibility wrapper for async request execution."""
    if request is not None and kwargs:
        raise ValueError("request cannot be combined with compatibility keyword arguments")

    await run_agent_request(request or _build_compat_run_request(**kwargs))


def run_async_agent(
    name: str,
    instruction: str,
    config_path: str | None = None,
    servers: str | None = None,
    urls: str | None = None,
    auth: str | None = None,
    client_metadata_url: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    model: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    result_file: str | None = None,
    resume: str | None = None,
    stdio_commands: list[str] | None = None,
    agent_name: str | None = None,
    target_agent_name: str | None = None,
    skills_directory: Path | None = None,
    environment_dir: Path | None = None,
    noenv: bool = False,
    force_smart: bool = False,
    shell_enabled: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    transport: str = "http",
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_description: str | None = None,
    tool_name_template: str | None = None,
    instance_scope: str = "shared",
    permissions_enabled: bool = True,
    reload: bool = False,
    watch: bool = False,
    quiet: bool = False,
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
) -> None:
    """Run the async agent function with proper loop handling."""
    try:
        run_kwargs = _build_run_agent_kwargs(
            name=name,
            mode=mode,
            noenv=noenv,
            resume=resume,
            model=model,
            agent_name=agent_name,
            target_agent_name=target_agent_name,
            message=message,
            prompt_file=prompt_file,
            result_file=result_file,
            skills_directory=skills_directory,
            environment_dir=environment_dir,
            instruction=instruction,
            force_smart=force_smart,
            config_path=config_path,
            servers=servers,
            urls=urls,
            auth=auth,
            client_metadata_url=client_metadata_url,
            agent_cards=agent_cards,
            card_tools=card_tools,
            stdio_commands=stdio_commands,
            shell_enabled=shell_enabled,
            transport=transport,
            instance_scope=instance_scope,
            host=host,
            port=port,
            tool_description=tool_description,
            tool_name_template=tool_name_template,
            permissions_enabled=permissions_enabled,
            reload=reload,
            watch=watch,
            quiet=quiet,
            missing_shell_cwd_policy=missing_shell_cwd_policy,
        )
        request = AgentRunRequest(**run_kwargs)
    except ValueError as exc:
        print(f"Error parsing URLs: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    run_request(request)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the agent"),
    instruction: str | None = CommonAgentOptions.instruction(),
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    client_metadata_url: str | None = CommonAgentOptions.client_metadata_url(),
    model: str | None = CommonAgentOptions.model(),
    agent: str | None = CommonAgentOptions.agent(),
    message: str | None = typer.Option(
        None,
        "--message",
        "-m",
        help="Message to send to the agent (skips interactive mode)",
    ),
    prompt_file: str | None = typer.Option(
        None,
        "--prompt-file",
        "-p",
        help="Path to a prompt file to use (either text or JSON)",
    ),
    results: str | None = typer.Option(
        None,
        "--results",
        help=("Write resulting history to file (single model) or per-model suffixed files "),
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume the last session or the specified session id",
    ),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    noenv: bool = CommonAgentOptions.noenv(),
    smart: bool = CommonAgentOptions.smart(),
    skills_dir: Path | None = CommonAgentOptions.skills_dir(),
    npx: str | None = CommonAgentOptions.npx(),
    uvx: str | None = CommonAgentOptions.uvx(),
    stdio: str | None = CommonAgentOptions.stdio(),
    shell: bool = CommonAgentOptions.shell(),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable manual AgentCard reloads (/reload)",
    ),
    watch: bool = CommonAgentOptions.watch(),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Disable progress/chat/tool output and print only direct command output",
    ),
) -> None:
    """Run an interactive agent directly from the command line."""
    if os.getenv(FAST_AGENT_SHELL_CHILD_ENV):
        typer.echo(
            "fast-agent is already running inside a fast-agent shell command. "
            "Exit the shell or unset FAST_AGENT_SHELL_CHILD to continue.",
            err=True,
        )
        raise typer.Exit(1)

    resolved_env_dir = resolve_environment_dir_option(ctx, env_dir, set_env_var=not noenv)

    request = build_command_run_request(
        name=name,
        instruction_option=instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        client_metadata_url=client_metadata_url,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        result_file=results,
        resume=resume,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        target_agent_name=agent,
        skills_directory=skills_dir,
        environment_dir=resolved_env_dir,
        noenv=noenv,
        force_smart=smart,
        shell_enabled=shell,
        mode="interactive",
        instance_scope="shared",
        reload=reload,
        watch=watch,
        quiet=quiet,
    )
    run_request(request)
