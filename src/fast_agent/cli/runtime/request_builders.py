"""Request-building helpers for CLI runtime commands."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Final, Literal

import typer

from fast_agent.cli.commands.server_helpers import generate_server_name
from fast_agent.cli.commands.url_parser import generate_server_configs, parse_server_urls
from fast_agent.constants import (
    DEFAULT_AGENT_INSTRUCTION,
    DEFAULT_GO_AGENT_TYPE,
    DEFAULT_SERVE_AGENT_TYPE,
    SMART_AGENT_INSTRUCTION,
)
from fast_agent.paths import resolve_environment_paths

from .run_request import AgentRunRequest, StdioServerConfig, UrlServerConfig

CARD_EXTENSIONS: Final[frozenset[str]] = frozenset({".md", ".markdown", ".yaml", ".yml"})

DEFAULT_ENV_PATHS = resolve_environment_paths()
DEFAULT_AGENT_CARDS_DIR: Final[Path] = DEFAULT_ENV_PATHS.agent_cards
DEFAULT_TOOL_CARDS_DIR: Final[Path] = DEFAULT_ENV_PATHS.tool_cards


def is_multi_model(model: str | None) -> bool:
    return bool(model and "," in model)


def use_smart_agent(model: str | None, mode: Literal["interactive", "serve"]) -> bool:
    return resolve_smart_agent_enabled(model, mode, force_smart=False)


def resolve_smart_agent_enabled(
    model: str | None,
    mode: Literal["interactive", "serve"],
    *,
    force_smart: bool,
) -> bool:
    if is_multi_model(model):
        return False
    if force_smart:
        return True
    if mode == "serve":
        return DEFAULT_SERVE_AGENT_TYPE == "smart"
    return DEFAULT_GO_AGENT_TYPE == "smart"


def resolve_default_instruction(
    model: str | None,
    mode: Literal["interactive", "serve"],
    *,
    force_smart: bool = False,
) -> str:
    return (
        SMART_AGENT_INSTRUCTION
        if resolve_smart_agent_enabled(model, mode, force_smart=force_smart)
        else DEFAULT_AGENT_INSTRUCTION
    )


def merge_card_sources(
    sources: list[str] | None,
    default_dir: Path,
) -> list[str] | None:
    merged: list[str] = []
    seen: set[str] = set()

    if sources:
        for entry in sources:
            if entry not in seen:
                merged.append(entry)
                seen.add(entry)
        return merged

    if default_dir.is_dir():
        has_cards = any(
            entry.is_file() and entry.suffix.lower() in CARD_EXTENSIONS
            for entry in default_dir.iterdir()
        )
        if has_cards:
            merged.append(str(default_dir))

    return merged or None


def normalize_explicit_card_sources(sources: list[str] | None) -> list[str] | None:
    """Normalize explicit card sources without implicit directory scans."""
    if not sources:
        return None

    merged: list[str] = []
    seen: set[str] = set()
    for entry in sources:
        if entry not in seen:
            merged.append(entry)
            seen.add(entry)
    return merged or None


def validate_noenv_conflicts(
    *,
    noenv: bool,
    environment_dir: Path | None,
    resume: str | None,
) -> None:
    """Validate unsupported option combinations for --noenv mode."""
    if not noenv:
        return

    if environment_dir is not None:
        raise typer.BadParameter("Cannot combine --noenv with --env.")

    if resume is not None:
        raise typer.BadParameter("Cannot combine --noenv with --resume.")


def resolve_instruction_option(
    instruction: str | None,
    model: str | None,
    mode: Literal["interactive", "serve"],
    *,
    force_smart: bool = False,
) -> tuple[str, str]:
    """Resolve the instruction option (file or URL) to text and inferred agent name."""
    resolved_instruction = resolve_default_instruction(model, mode, force_smart=force_smart)
    agent_name = "agent"

    if instruction:
        try:
            from pydantic import AnyUrl

            from fast_agent.core.direct_decorators import _resolve_instruction

            if instruction.startswith(("http://", "https://")):
                resolved_instruction = _resolve_instruction(AnyUrl(instruction))
            else:
                resolved_instruction = _resolve_instruction(Path(instruction))
                instruction_path = Path(instruction)
                if instruction_path.exists() and instruction_path.is_file():
                    agent_name = instruction_path.stem
        except Exception as exc:
            typer.echo(f"Error loading instruction from {instruction}: {exc}", err=True)
            raise typer.Exit(1) from exc

    return resolved_instruction, agent_name


def collect_stdio_commands(npx: str | None, uvx: str | None, stdio: str | None) -> list[str]:
    """Collect STDIO command definitions from convenience options."""
    stdio_commands: list[str] = []

    if npx:
        stdio_commands.append(f"npx {npx}")
    if uvx:
        stdio_commands.append(f"uvx {uvx}")
    if stdio:
        stdio_commands.append(stdio)

    return stdio_commands


def _merge_url_servers(
    server_list: list[str] | None,
    urls: str | None,
    auth: str | None,
    client_metadata_url: str | None,
) -> tuple[dict[str, UrlServerConfig] | None, list[str] | None]:
    url_servers: dict[str, UrlServerConfig] | None = None

    if urls:
        try:
            parsed_urls = parse_server_urls(urls, auth)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--url") from exc
        raw_url_servers = generate_server_configs(parsed_urls)
        url_servers = {}
        for server_name, server_config in raw_url_servers.items():
            normalized_config: UrlServerConfig = {
                "transport": str(server_config["transport"]),
                "url": str(server_config["url"]),
            }
            headers = server_config.get("headers")
            if isinstance(headers, dict):
                normalized_config["headers"] = {
                    str(key): str(value)
                    for key, value in headers.items()
                }
            if client_metadata_url:
                normalized_config["auth"] = {
                    "oauth": True,
                    "client_metadata_url": client_metadata_url,
                }
            url_servers[server_name] = normalized_config

        if url_servers and not server_list:
            server_list = list(url_servers.keys())
        elif url_servers and server_list:
            server_list.extend(list(url_servers.keys()))

    return url_servers, server_list


def _merge_stdio_servers(
    server_list: list[str] | None,
    stdio_commands: list[str] | None,
) -> tuple[dict[str, StdioServerConfig] | None, list[str] | None]:
    if not stdio_commands:
        return None, server_list

    stdio_servers: dict[str, StdioServerConfig] = {}

    for i, stdio_cmd in enumerate(stdio_commands):
        try:
            parsed_command = shlex.split(stdio_cmd)
        except ValueError as exc:
            print(f"Error parsing stdio command '{stdio_cmd}': {exc}", file=sys.stderr)
            continue

        if not parsed_command:
            print(f"Error: Empty stdio command: {stdio_cmd}", file=sys.stderr)
            continue

        command = parsed_command[0]
        initial_args = parsed_command[1:] if len(parsed_command) > 1 else []

        if initial_args:
            for arg in initial_args:
                if arg.endswith((".py", ".js", ".ts")):
                    base_name = generate_server_name(arg)
                    break
            else:
                base_name = generate_server_name(command)
        else:
            base_name = generate_server_name(command)

        server_name = base_name
        if len(stdio_commands) > 1:
            server_name = f"{base_name}_{i + 1}"

        stdio_config: StdioServerConfig = {
            "transport": "stdio",
            "command": command,
            "args": initial_args.copy(),
        }
        stdio_servers[server_name] = stdio_config

        if not server_list:
            server_list = [server_name]
        else:
            server_list.append(server_name)

    return stdio_servers, server_list


def build_agent_run_request(
    *,
    name: str,
    instruction: str,
    config_path: str | None,
    servers: str | None,
    urls: str | None,
    auth: str | None,
    client_metadata_url: str | None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    result_file: str | None,
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
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
    force_smart: bool = False,
    noenv: bool = False,
) -> AgentRunRequest:
    """Build a normalized runtime request from legacy CLI kwargs."""
    validate_noenv_conflicts(
        noenv=noenv,
        environment_dir=environment_dir,
        resume=resume,
    )

    server_list = servers.split(",") if servers else None

    url_servers, server_list = _merge_url_servers(
        server_list,
        urls,
        auth,
        client_metadata_url,
    )
    stdio_servers, server_list = _merge_stdio_servers(server_list, stdio_commands)

    if environment_dir:
        env_paths = resolve_environment_paths(override=environment_dir)
        default_agent_cards_dir = env_paths.agent_cards
        default_tool_cards_dir = env_paths.tool_cards
    else:
        default_agent_cards_dir = DEFAULT_AGENT_CARDS_DIR
        default_tool_cards_dir = DEFAULT_TOOL_CARDS_DIR

    merged_agent_cards = (
        normalize_explicit_card_sources(agent_cards)
        if noenv
        else merge_card_sources(agent_cards, default_agent_cards_dir)
    )
    merged_card_tools = (
        normalize_explicit_card_sources(card_tools)
        if noenv
        else merge_card_sources(card_tools, default_tool_cards_dir)
    )

    effective_permissions_enabled = (
        permissions_enabled if not (noenv and mode == "serve") else False
    )

    return AgentRunRequest(
        name=name,
        instruction=instruction,
        config_path=config_path,
        server_list=server_list,
        agent_cards=merged_agent_cards,
        card_tools=merged_card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        result_file=result_file,
        resume=resume,
        url_servers=url_servers,
        stdio_servers=stdio_servers,
        agent_name=agent_name,
        target_agent_name=target_agent_name,
        skills_directory=skills_directory,
        environment_dir=None if noenv else environment_dir,
        noenv=noenv,
        force_smart=force_smart,
        shell_runtime=shell_enabled,
        mode=mode,
        transport=transport,
        host=host,
        port=port,
        tool_description=tool_description,
        tool_name_template=tool_name_template,
        instance_scope=instance_scope,
        permissions_enabled=effective_permissions_enabled,
        reload=reload,
        watch=watch,
        missing_shell_cwd_policy=missing_shell_cwd_policy,
    )


def build_run_agent_kwargs(
    *,
    name: str,
    instruction: str,
    config_path: str | None,
    servers: str | None,
    urls: str | None,
    auth: str | None,
    client_metadata_url: str | None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    result_file: str | None,
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
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
    force_smart: bool = False,
    noenv: bool = False,
) -> dict[str, Any]:
    request = build_agent_run_request(
        name=name,
        instruction=instruction,
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
        result_file=result_file,
        resume=resume,
        stdio_commands=stdio_commands,
        agent_name=agent_name,
        target_agent_name=target_agent_name,
        skills_directory=skills_directory,
        environment_dir=environment_dir,
        noenv=noenv,
        force_smart=force_smart,
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
        missing_shell_cwd_policy=missing_shell_cwd_policy,
    )
    return request.to_agent_setup_kwargs()


def build_command_run_request(
    *,
    name: str,
    instruction_option: str | None,
    config_path: str | None,
    servers: str | None,
    urls: str | None,
    auth: str | None,
    client_metadata_url: str | None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    result_file: str | None,
    resume: str | None,
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
    target_agent_name: str | None,
    skills_directory: Path | None,
    environment_dir: Path | None,
    shell_enabled: bool,
    mode: Literal["interactive", "serve"],
    transport: str = "http",
    host: str = "0.0.0.0",
    port: int = 8000,
    tool_description: str | None = None,
    tool_name_template: str | None = None,
    instance_scope: str = "shared",
    permissions_enabled: bool = True,
    reload: bool = False,
    watch: bool = False,
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
    force_smart: bool = False,
    noenv: bool = False,
) -> AgentRunRequest:
    """Build a normalized request directly from command option values."""
    validate_noenv_conflicts(
        noenv=noenv,
        environment_dir=environment_dir,
        resume=resume,
    )

    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    resolved_instruction, inferred_agent_name = resolve_instruction_option(
        instruction_option,
        model,
        mode,
        force_smart=force_smart,
    )

    return build_agent_run_request(
        name=name,
        instruction=resolved_instruction,
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
        result_file=result_file,
        resume=resume,
        stdio_commands=stdio_commands,
        agent_name=inferred_agent_name,
        target_agent_name=target_agent_name,
        skills_directory=skills_directory,
        environment_dir=environment_dir,
        noenv=noenv,
        force_smart=force_smart,
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
        missing_shell_cwd_policy=missing_shell_cwd_policy,
    )
