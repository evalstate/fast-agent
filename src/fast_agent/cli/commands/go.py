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
    build_agent_run_request,
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
    StdioServerConfig,
    UrlServerConfig,
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


def _build_run_agent_kwargs(
    *,
    name: str,
    instruction: str,
    config_path: str | None,
    servers: str | None,
    urls: str | None,
    auth: str | None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    resume: str | None,
    stdio_commands: list[str] | None,
    agent_name: str | None,
    target_agent_name: str | None,
    skills_directory: Path | None,
    environment_dir: Path | None,
    shell_enabled: bool,
    mode: Literal["interactive", "serve"],
    transport: str,
    host: str,
    port: int,
    tool_description: str | None,
    tool_name_template: str | None,
    instance_scope: str,
    permissions_enabled: bool,
    reload: bool,
    watch: bool,
    noenv: bool = False,
) -> dict[str, Any]:
    """Build keyword arguments for the async agent runner."""
    return build_run_agent_kwargs(
        name=name,
        instruction=instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        resume=resume,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
        target_agent_name=target_agent_name,
        skills_directory=skills_directory,
        environment_dir=environment_dir,
        noenv=noenv,
        shell_enabled=shell_enabled,
        mode=mode,
        transport=transport,
        host=host,
        port=port,
        tool_description=tool_description,
        tool_name_template=tool_name_template,
        instance_scope=instance_scope,
        permissions_enabled=permissions_enabled,
        reload=reload,
        watch=watch,
    )


def _merge_card_sources(
    sources: list[str] | None,
    default_dir: Path,
) -> list[str] | None:
    return merge_card_sources(sources, default_dir)


async def _run_agent(
    name: str = "fast-agent cli",
    instruction: str | None = None,
    config_path: str | None = None,
    server_list: list[str] | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    model: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    resume: str | None = None,
    url_servers: dict[str, UrlServerConfig] | None = None,
    stdio_servers: dict[str, StdioServerConfig] | None = None,
    agent_name: str | None = "agent",
    target_agent_name: str | None = None,
    skills_directory: Path | None = None,
    environment_dir: Path | None = None,
    noenv: bool = False,
    shell_runtime: bool = False,
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
) -> None:
    """Compatibility wrapper for async request execution."""
    request = AgentRunRequest(
        name=name,
        instruction=instruction,
        config_path=config_path,
        server_list=server_list,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        resume=resume,
        url_servers=url_servers,
        stdio_servers=stdio_servers,
        agent_name=agent_name,
        target_agent_name=target_agent_name,
        skills_directory=skills_directory,
        environment_dir=environment_dir,
        noenv=noenv,
        shell_runtime=shell_runtime,
        mode=mode,
        transport=transport,
        host=host,
        port=port,
        tool_description=tool_description,
        tool_name_template=tool_name_template,
        instance_scope=instance_scope,
        permissions_enabled=permissions_enabled,
        reload=reload,
        watch=watch,
    )
    await run_agent_request(request)


def run_async_agent(
    name: str,
    instruction: str,
    config_path: str | None = None,
    servers: str | None = None,
    urls: str | None = None,
    auth: str | None = None,
    agent_cards: list[str] | None = None,
    card_tools: list[str] | None = None,
    model: str | None = None,
    message: str | None = None,
    prompt_file: str | None = None,
    resume: str | None = None,
    stdio_commands: list[str] | None = None,
    agent_name: str | None = None,
    target_agent_name: str | None = None,
    skills_directory: Path | None = None,
    environment_dir: Path | None = None,
    noenv: bool = False,
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
) -> None:
    """Run the async agent function with proper loop handling."""
    try:
        request = build_agent_run_request(
            name=name,
            instruction=instruction,
            config_path=config_path,
            servers=servers,
            urls=urls,
            auth=auth,
            agent_cards=agent_cards,
            card_tools=card_tools,
            model=model,
            message=message,
            prompt_file=prompt_file,
            resume=resume,
            stdio_commands=stdio_commands,
            agent_name=agent_name,
            target_agent_name=target_agent_name,
            skills_directory=skills_directory,
            environment_dir=environment_dir,
            noenv=noenv,
            shell_enabled=shell_enabled,
            mode=mode,
            transport=transport,
            host=host,
            port=port,
            tool_description=tool_description,
            tool_name_template=tool_name_template,
            instance_scope=instance_scope,
            permissions_enabled=permissions_enabled,
            reload=reload,
            watch=watch,
        )
    except ValueError as exc:
        print(f"Error parsing URLs: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    run_request(request)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the agent"),
    instruction: str | None = typer.Option(
        None,
        "--instruction",
        "-i",
        help="Path to file or URL containing instruction for the agent",
    ),
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
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
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume the last session or the specified session id",
    ),
    env_dir: Path | None = CommonAgentOptions.env_dir(),
    noenv: bool = CommonAgentOptions.noenv(),
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
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        resume=resume,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        target_agent_name=agent,
        skills_directory=skills_dir,
        environment_dir=resolved_env_dir,
        noenv=noenv,
        shell_enabled=shell,
        mode="interactive",
        instance_scope="shared",
        reload=reload,
        watch=watch,
    )
    run_request(request)
