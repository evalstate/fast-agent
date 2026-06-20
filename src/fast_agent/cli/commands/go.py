"""Run an interactive agent directly from the command line."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path  # noqa: TC003 - typer resolves Path annotations at runtime
from typing import Any, Literal

import typer

from fast_agent.a2a.connect import normalize_a2a_transport, normalize_a2a_url
from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.runtime.request_builders import (
    DEFAULT_AGENT_CARDS_DIR as _DEFAULT_AGENT_CARDS_DIR,
)
from fast_agent.cli.runtime.request_builders import (
    DEFAULT_TOOL_CARDS_DIR as _DEFAULT_TOOL_CARDS_DIR,
)
from fast_agent.cli.runtime.request_builders import (
    ResolvedInstructionOption,
    build_command_run_request,
    build_run_agent_kwargs,
    is_multi_model,
    merge_card_sources,
    resolve_default_instruction,
    resolve_instance_scope,
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
from fast_agent.core.agent_card_paths import AGENT_CARD_EXTENSIONS as _CARD_EXTENSIONS
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.mcp.hf_auth import add_explicit_bearer_auth_header

CARD_EXTENSIONS = _CARD_EXTENSIONS
DEFAULT_AGENT_CARDS_DIR = _DEFAULT_AGENT_CARDS_DIR
DEFAULT_TOOL_CARDS_DIR = _DEFAULT_TOOL_CARDS_DIR


app = typer.Typer(
    help="Run an interactive agent directly from the command line without creating an agent.py file",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_completion=False,
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
) -> ResolvedInstructionOption:
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


def _merge_pack_card_sources(
    sources: list[str] | None,
    pack_dir: Path,
) -> list[str] | None:
    pack_sources = merge_card_sources(None, pack_dir)
    if not pack_sources:
        return sources
    return merge_card_sources([*(sources or []), *pack_sources], pack_dir)


def _materialize_a2a_agent_cards(
    urls: list[str],
    *,
    transport: str | None,
    oauth: bool | None = None,
    auth_token: str | None = None,
) -> tuple[tempfile.TemporaryDirectory[str], list[str]]:
    normalized_transport = None
    if transport:
        normalized_transport = normalize_a2a_transport(transport)
        if normalized_transport is None:
            raise typer.BadParameter(
                f"Unsupported A2A transport: {transport}",
                param_hint="--a2a-transport",
            )

    tempdir = tempfile.TemporaryDirectory(prefix="fast-agent-a2a-")
    paths: list[str] = []
    for index, raw_url in enumerate(urls, start=1):
        url, card_path, error = normalize_a2a_url(raw_url)
        if error:
            tempdir.cleanup()
            raise typer.BadParameter(error, param_hint="--a2a")
        name = "a2a_remote" if index == 1 else f"a2a_remote_{index}"
        lines = [
            "type: a2a",
            f"name: {name}",
            f"url: {url}",
        ]
        if auth_token:
            normalized_token = auth_token.strip()
            if normalized_token.lower().startswith("bearer "):
                normalized_token = normalized_token[7:].strip()
            if not normalized_token:
                tempdir.cleanup()
                raise typer.BadParameter("Auth token cannot be empty", param_hint="--auth")
            headers = add_explicit_bearer_auth_header(url, None, normalized_token)
            lines.append("headers:")
            for key, value in headers.items():
                lines.append(f"  {key}: {value!r}")
        if normalized_transport:
            lines.append(f"transport: {normalized_transport}")
        if oauth is not None:
            lines.extend(
                [
                    "auth:",
                    f"  oauth: {str(oauth).lower()}",
                ]
            )
        if card_path:
            lines.append(f"relative_card_path: {card_path}")
        path = Path(tempdir.name) / f"{name}.yaml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(str(path))
    return tempdir, paths


def _build_compat_run_request(**kwargs: Any) -> AgentRunRequest:
    """Build an AgentRunRequest from legacy compatibility keyword arguments.

    This wrapper intentionally accepts ``Any`` because it preserves the legacy
    dynamic call surface used by older integrations while converting into the
    strongly typed ``AgentRunRequest`` model at the boundary.
    """
    transport = kwargs.get("transport", "http")
    instance_scope = kwargs.get("instance_scope", "shared")
    if transport == "acp" and instance_scope == "shared":
        instance_scope = None

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
        attachments=kwargs.get("attachments"),
        json_schema=kwargs.get("json_schema"),
        schema_model=kwargs.get("schema_model"),
        structured_tool_policy=kwargs.get("structured_tool_policy"),
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
        no_shell=kwargs.get("no_shell", False),
        mode=kwargs.get("mode", "interactive"),
        transport=kwargs.get("transport", "http"),
        host=kwargs.get("host", "127.0.0.1"),
        port=kwargs.get("port", 8000),
        tool_description=kwargs.get("tool_description"),
        tool_name_template=kwargs.get("tool_name_template"),
        instance_scope=resolve_instance_scope(
            transport=transport,
            instance_scope=instance_scope,
        ),
        permissions_enabled=kwargs.get("permissions_enabled", True),
        reload=kwargs.get("reload", False),
        watch=kwargs.get("watch", False),
        execution_mode=kwargs.get("execution_mode"),
        quiet=kwargs.get("quiet", False),
        missing_shell_cwd_policy=kwargs.get("missing_shell_cwd_policy"),
    )


async def _run_agent(
    request: AgentRunRequest | None = None,
    **kwargs: Any,
) -> None:
    """Compatibility wrapper for async request execution."""
    from fast_agent.cli.runtime.agent_setup import run_agent_request

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
    attachments: list[str] | None = None,
    json_schema: str | None = None,
    schema_model: str | None = None,
    structured_tool_policy: str | None = None,
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
    no_shell: bool = False,
    mode: Literal["interactive", "serve"] = "interactive",
    transport: str = "http",
    host: str = "127.0.0.1",
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
        normalized_instance_scope: str | None = instance_scope
        if transport == "acp" and instance_scope == "shared":
            normalized_instance_scope = None
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
            attachments=attachments,
            json_schema=json_schema,
            schema_model=schema_model,
            structured_tool_policy=structured_tool_policy,
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
            no_shell=no_shell,
            transport=transport,
            instance_scope=resolve_instance_scope(
                transport=transport,
                instance_scope=normalized_instance_scope,
            ),
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


def _resolve_effective_environment_dir(
    *,
    settings: Any | None,
    env_dir: Path | None,
) -> Path:
    from fast_agent.paths import resolve_environment_paths

    if env_dir is not None:
        return env_dir
    return resolve_environment_paths(settings=settings).root


def _maybe_queue_pack_readme_notice(
    *,
    pack_name: str,
    readme: str | None,
    message: str | None,
    prompt_file: str | None,
) -> None:
    if not readme or message is not None or prompt_file is not None:
        return

    from fast_agent.ui.enhanced_prompt import queue_startup_markdown_notice, queue_startup_notice

    queue_startup_notice(f"[dim]Card pack README:[/dim] [cyan]{pack_name}[/cyan]")
    queue_startup_markdown_notice(
        readme,
        title=f"{pack_name} README",
        right_info="card pack",
    )


def _resolve_request_update_notice(
    *,
    ctx: typer.Context,
    request: AgentRunRequest,
    environment_dir: Path | None,
) -> str | None:
    if request.noenv:
        return None
    if not request.is_repl:
        return None
    if request.quiet:
        return None
    context_payload = ensure_context_object(ctx)
    no_update_check_value = context_payload.get("no_update_check")
    no_update_check = no_update_check_value if isinstance(no_update_check_value, bool) else False

    from fast_agent.cli.update_check import check_for_update_notice, should_run_update_check

    if not should_run_update_check(
        disabled=no_update_check,
    ):
        return None
    return check_for_update_notice(environment_dir=environment_dir)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def go(
    ctx: typer.Context,
    name: str = typer.Option("fast-agent", "--name", help="Name for the agent"),
    instruction: str | None = CommonAgentOptions.instruction(),
    config_path: str | None = CommonAgentOptions.config_path(),
    servers: str | None = CommonAgentOptions.servers(),
    agent_cards: list[str] | None = CommonAgentOptions.agent_cards(),
    a2a: list[str] | None = typer.Option(
        None,
        "--a2a",
        metavar="<url>",
        help="Connect to a remote A2A agent by base URL or agent-card URL (repeatable).",
    ),
    a2a_transport: str | None = typer.Option(
        None,
        "--a2a-transport",
        help="Preferred A2A transport for --a2a: JSONRPC or HTTP+JSON.",
    ),
    a2a_oauth: bool | None = typer.Option(
        None,
        "--a2a-oauth/--no-a2a-oauth",
        help="Force or disable browser OAuth for --a2a remote agents.",
    ),
    card_tools: list[str] | None = CommonAgentOptions.card_tools(),
    urls: str | None = CommonAgentOptions.urls(),
    auth: str | None = CommonAgentOptions.auth(),
    client_metadata_url: str | None = CommonAgentOptions.client_metadata_url(),
    model: str | None = CommonAgentOptions.model(),
    pack: str | None = typer.Option(
        None,
        "--pack",
        "--card-pack",
        help="Ensure a named card pack is installed in the selected environment before starting.",
    ),
    pack_registry: str | None = typer.Option(
        None,
        "--pack-registry",
        metavar="<path-or-uri>",
        help="Marketplace path, HTTP(S) URL, file:// URI, or hf:// URI used to resolve --pack when it is not already installed.",
    ),
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
        metavar="<path-or-uri>",
        help="Path, HTTP(S) URL, file:// URI, or hf:// URI to a prompt file to send once and exit (either text or JSON)",
    ),
    attach: list[str] | None = typer.Option(
        None,
        "--attach",
        "-a",
        metavar="<path-or-url>",
        help="Attach a local file or HTTP(S) URL to the one-shot message. May be repeated.",
    ),
    json_schema: str | None = CommonAgentOptions.json_schema(),
    schema_model: str | None = CommonAgentOptions.schema_model(),
    structured_tool_policy: str | None = CommonAgentOptions.structured_tool_policy(),
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
    no_shell: bool = CommonAgentOptions.no_shell(),
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
    effective_env_dir = resolved_env_dir

    if pack:
        from fast_agent.cards import service as card_service

        if noenv:
            raise typer.BadParameter("Cannot combine --pack with --noenv.", param_hint="--pack")

        from fast_agent.cli.command_support import get_settings_or_exit
        from fast_agent.paths import resolve_environment_paths

        settings = (
            get_settings_or_exit(config_path)
            if (resolved_env_dir is None or pack_registry is None)
            else None
        )
        effective_env_dir = _resolve_effective_environment_dir(
            settings=settings,
            env_dir=resolved_env_dir,
        )
        env_paths = resolve_environment_paths(override=effective_env_dir)
        resolved_pack_registry = card_service.resolve_registry(pack_registry, settings=settings)

        try:
            ensured_pack = card_service.ensure_pack_available_sync(
                selector=pack,
                environment_paths=env_paths,
                registry=resolved_pack_registry,
            )
        except card_service.CardPackLookupError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        except Exception as exc:
            typer.echo(f"Failed to prepare card pack: {exc}", err=True)
            raise typer.Exit(1) from exc

        pack_readme = (
            ensured_pack.install_record.readme
            if ensured_pack.install_record is not None
            else card_service.read_installed_pack_readme(
                environment_paths=env_paths,
                selector=ensured_pack.name,
            ).readme
        )

        status = "Installed" if ensured_pack.installed else "Using installed"
        typer.echo(f"{status} card pack: {ensured_pack.name}")
        typer.echo(f"Launching fast-agent go with environment: {effective_env_dir}")
        _maybe_queue_pack_readme_notice(
            pack_name=ensured_pack.name,
            readme=pack_readme,
            message=message,
            prompt_file=prompt_file,
        )

        agent_cards = _merge_pack_card_sources(agent_cards, env_paths.agent_cards)
        card_tools = _merge_pack_card_sources(card_tools, env_paths.tool_cards)

    a2a_tempdir: tempfile.TemporaryDirectory[str] | None = None
    if a2a:
        a2a_tempdir, a2a_cards = _materialize_a2a_agent_cards(
            a2a,
            transport=a2a_transport,
            oauth=a2a_oauth,
            auth_token=auth,
        )
        agent_cards = [*(agent_cards or []), *a2a_cards]
        if agent is None and len(a2a_cards) == 1:
            agent = Path(a2a_cards[0]).stem

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
        attachments=attach,
        json_schema=json_schema,
        schema_model=schema_model,
        structured_tool_policy=structured_tool_policy,
        result_file=results,
        resume=resume,
        npx=npx,
        uvx=uvx,
        stdio=stdio,
        target_agent_name=agent,
        skills_directory=skills_dir,
        environment_dir=effective_env_dir,
        noenv=noenv,
        force_smart=smart,
        shell_enabled=shell,
        no_shell=no_shell,
        mode="interactive",
        instance_scope="shared",
        reload=reload,
        watch=watch,
        quiet=quiet,
    )

    update_notice = _resolve_request_update_notice(
        ctx=ctx,
        request=request,
        environment_dir=effective_env_dir,
    )
    if update_notice and not request.quiet:
        from fast_agent.ui.enhanced_prompt import queue_startup_notice

        queue_startup_notice(update_notice)

    try:
        run_request(request)
    except AgentConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    finally:
        if a2a_tempdir is not None:
            a2a_tempdir.cleanup()
