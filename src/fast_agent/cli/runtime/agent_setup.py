"""Agent setup and execution branch logic for CLI runtime requests."""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.markup import escape as escape_markup

from fast_agent.cli.commands.server_helpers import add_servers_to_config
from fast_agent.cli.constants import RESUME_LATEST_SENTINEL
from fast_agent.core.exceptions import AgentConfigError

from .request_builders import resolve_default_instruction, resolve_smart_agent_enabled
from .shell_cwd_policy import (
    can_prompt_for_missing_cwd,
    collect_shell_cwd_issues,
    create_missing_shell_cwd_directories,
    effective_missing_shell_cwd_policy,
    format_shell_cwd_issues,
    resolve_missing_shell_cwd_policy,
)

if TYPE_CHECKING:
    from .run_request import AgentRunRequest


def _find_last_assistant_text(history: list[Any]) -> str | None:
    for message in reversed(history):
        if getattr(message, "role", None) != "assistant":
            continue
        text = getattr(message, "last_text", None)
        if callable(text):
            value = text()
            if value:
                return str(value)
    return None


def _is_interactive_startup_notice_context(request: AgentRunRequest) -> bool:
    return request.mode == "interactive" and request.message is None and request.prompt_file is None


def _emit_startup_notice(request: AgentRunRequest, message: str) -> None:
    if _is_interactive_startup_notice_context(request):
        from fast_agent.ui.enhanced_prompt import queue_startup_notice

        queue_startup_notice(message)
        return

    typer.echo(message, err=True)


def _format_shell_cwd_policy_message(
    *,
    policy: str,
    lines: list[str],
) -> str:
    title = f"Shell cwd policy ({policy}):"
    return "\n".join([title, *lines])


def _apply_shell_cwd_policy_preflight(fast: Any, request: AgentRunRequest) -> None:
    issues = collect_shell_cwd_issues(
        fast.agents,
        shell_runtime_requested=request.shell_runtime,
        cwd=Path.cwd(),
    )
    if not issues:
        return

    settings = getattr(fast.app.context, "config", None)
    shell_settings = getattr(settings, "shell_execution", None)
    configured_policy = getattr(shell_settings, "missing_cwd_policy", None)
    resolved_policy = resolve_missing_shell_cwd_policy(
        cli_override=request.missing_shell_cwd_policy,
        configured_policy=configured_policy,
    )
    interactive_startup_context = _is_interactive_startup_notice_context(request)
    can_prompt = can_prompt_for_missing_cwd(
        mode=request.mode,
        message=request.message,
        prompt_file=request.prompt_file,
        stdin_is_tty=sys.stdin.isatty(),
        tty_device_available=False,
    )
    policy = effective_missing_shell_cwd_policy(resolved_policy, can_prompt=can_prompt)
    issue_lines = format_shell_cwd_issues(issues)

    if policy == "warn":
        _emit_startup_notice(
            request,
            _format_shell_cwd_policy_message(policy=policy, lines=issue_lines),
        )
        return

    if policy == "error":
        typer.echo(_format_shell_cwd_policy_message(policy=policy, lines=issue_lines), err=True)
        raise typer.Exit(1)

    if policy == "ask":
        if interactive_startup_context:
            # Keep interactive confirmation inside the prompt lifecycle for ask mode.
            return
        policy = "warn"

    if policy == "warn":
        _emit_startup_notice(
            request,
            _format_shell_cwd_policy_message(policy=policy, lines=issue_lines),
        )
        return

    created_paths, creation_errors = create_missing_shell_cwd_directories(issues)
    if created_paths:
        created_lines = [
            "Created missing shell cwd directories:",
            *[f" - {path}" for path in created_paths],
        ]
        _emit_startup_notice(request, "\n".join(created_lines))

    if creation_errors:
        error_lines = ["Failed to create one or more shell cwd directories:"]
        error_lines.extend(f" - {item.path}: {item.message}" for item in creation_errors)
        typer.echo("\n".join(error_lines), err=True)
        raise typer.Exit(1)

    remaining_issues = collect_shell_cwd_issues(
        fast.agents,
        shell_runtime_requested=request.shell_runtime,
        cwd=Path.cwd(),
    )
    if remaining_issues:
        typer.echo(
            _format_shell_cwd_policy_message(
                policy="error",
                lines=format_shell_cwd_issues(remaining_issues),
            ),
            err=True,
        )
        raise typer.Exit(1)


async def _resume_session_if_requested(agent_app, request: AgentRunRequest) -> None:
    if request.noenv:
        if request.resume:
            typer.echo("Error: --resume cannot be used with --noenv.", err=True)
            raise typer.Exit(1)
        return

    if not request.resume:
        return

    from fast_agent.session import get_session_manager
    from fast_agent.ui.enhanced_prompt import queue_startup_notice

    manager = get_session_manager()
    session_id = None if request.resume in ("", RESUME_LATEST_SENTINEL) else request.resume
    default_agent = agent_app._agent(None)
    result = manager.resume_session_agents(
        agent_app._agents,
        session_id,
        default_agent_name=getattr(default_agent, "name", None),
    )
    interactive_notice = request.message is None and request.prompt_file is None
    if not result:
        if session_id:
            notice = f"[yellow]Session not found:[/yellow] {session_id}"
        else:
            notice = "[yellow]No sessions found to resume.[/yellow]"
        if interactive_notice:
            queue_startup_notice(notice)
        else:
            typer.echo(
                notice.replace("[yellow]", "").replace("[/yellow]", ""),
                err=True,
            )
        return

    session, loaded, missing_agents = result
    session_time = session.info.last_activity.strftime("%y-%m-%d %H:%M")
    resume_notice = (
        f"[dim]Resumed session[/dim] [cyan]{session.info.name}[/cyan] [dim]({session_time})[/dim]"
    )
    if interactive_notice:
        queue_startup_notice(resume_notice)
    else:
        typer.echo(
            f"Resumed session {session.info.name} ({session_time})",
            err=True,
        )
    if missing_agents:
        missing_list = ", ".join(sorted(missing_agents))
        missing_notice = f"[yellow]Missing agents from session:[/yellow] {missing_list}"
        if interactive_notice:
            queue_startup_notice(missing_notice)
        else:
            typer.echo(f"Missing agents from session: {missing_list}", err=True)
    if missing_agents or not loaded:
        from fast_agent.session import format_history_summary, summarize_session_histories

        summary = summarize_session_histories(session)
        summary_text = format_history_summary(summary)
        if summary_text:
            summary_notice = f"[dim]Available histories:[/dim] {summary_text}"
            if interactive_notice:
                queue_startup_notice(summary_notice)
            else:
                typer.echo(f"Available histories: {summary_text}", err=True)

    preview_agent = default_agent
    default_name = getattr(default_agent, "name", None)
    if loaded and default_name not in loaded:
        first_loaded_name = sorted(loaded.keys())[0]
        preview_agent = agent_app._agents.get(first_loaded_name, default_agent)

    preview_history = getattr(preview_agent, "message_history", [])
    assistant_text = _find_last_assistant_text(list(preview_history))
    if assistant_text:
        if interactive_notice:
            queue_startup_notice("[dim]Last assistant message:[/dim]")
            queue_startup_notice(escape_markup(assistant_text))
        else:
            typer.echo("Last assistant message:", err=True)
            typer.echo(assistant_text, err=True)


def _validate_target_agent_name(fast, request: AgentRunRequest) -> None:
    if not request.target_agent_name:
        return
    if request.target_agent_name in fast.agents:
        return

    available_agents = ", ".join(sorted(fast.agents.keys()))
    typer.echo(
        (
            f"Error: Agent '{request.target_agent_name}' not found. "
            f"Available agents: {available_agents}"
        ),
        err=True,
    )
    raise typer.Exit(1)


def _attach_cli_servers_to_selected_agent(fast, request: AgentRunRequest) -> None:
    if not request.server_list:
        return

    selected_agent_data = None
    if request.target_agent_name and request.target_agent_name in fast.agents:
        selected_agent_data = fast.agents.get(request.target_agent_name)

    if selected_agent_data is None:
        for agent_data in fast.agents.values():
            config = agent_data.get("config")
            if config and getattr(config, "default", False):
                selected_agent_data = agent_data
                break

    if selected_agent_data is None and fast.agents:
        selected_agent_data = next(iter(fast.agents.values()))

    if selected_agent_data:
        config = selected_agent_data.get("config")
        if config:
            existing = list(config.servers) if config.servers else []
            config.servers = existing + [
                server for server in request.server_list if server not in existing
            ]


def _sanitize_result_suffix(label: str) -> str:
    normalized = re.sub(r"[\\/\s]+", "_", label.strip())
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", normalized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._-")
    return sanitized or "agent"


def _build_result_file_with_suffix(base_file: Path, suffix: str) -> Path:
    if base_file.suffix:
        return base_file.with_name(f"{base_file.stem}-{suffix}{base_file.suffix}")
    return base_file.with_name(f"{base_file.name}-{suffix}")


def _build_fan_out_result_paths(
    result_file: str,
    fan_out_agent_names: list[str],
) -> list[tuple[str, Path]]:
    base_path = Path(result_file)
    suffix_counts: dict[str, int] = {}
    exports: list[tuple[str, Path]] = []

    for agent_name in fan_out_agent_names:
        suffix = _sanitize_result_suffix(agent_name)
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix_counts[suffix] > 1:
            suffix = f"{suffix}-{suffix_counts[suffix]}"
        exports.append((agent_name, _build_result_file_with_suffix(base_path, suffix)))

    return exports


async def _save_result_history(agent_app: Any, *, agent_name: str, output_path: Path) -> None:
    from fast_agent.history.history_exporter import HistoryExporter

    output_path.parent.mkdir(parents=True, exist_ok=True)
    agent_obj = agent_app._agent(agent_name)
    await HistoryExporter.save(agent_obj, str(output_path))


async def _export_result_histories(
    agent_app: Any,
    request: AgentRunRequest,
    *,
    fan_out_agent_names: list[str] | None = None,
) -> None:
    if not request.result_file:
        return

    try:
        if fan_out_agent_names and request.target_agent_name is None:
            for agent_name, output_path in _build_fan_out_result_paths(
                request.result_file,
                fan_out_agent_names,
            ):
                await _save_result_history(
                    agent_app,
                    agent_name=agent_name,
                    output_path=output_path,
                )
            return

        selected_agent = agent_app._agent(request.target_agent_name)
        await _save_result_history(
            agent_app,
            agent_name=selected_agent.name,
            output_path=Path(request.result_file),
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Error exporting result file: {exc}", err=True)
        raise typer.Exit(1) from exc


async def _run_single_agent_cli_flow(agent_app: Any, request: AgentRunRequest) -> None:
    from fast_agent.mcp.prompts.prompt_load import load_prompt

    # Allow interactive prompt startup checks to honor per-run CLI override policy.
    setattr(agent_app, "_missing_shell_cwd_policy_override", request.missing_shell_cwd_policy)

    async def _run_interactive_with_interrupt_recovery() -> None:
        ctrl_c_exit_window_seconds = 2.0
        ctrl_c_deadline: float | None = None

        while True:
            try:
                await agent_app.interactive(agent_name=request.target_agent_name)
                return
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None:
                    while task.uncancel() > 0:
                        pass
                await asyncio.sleep(0)
                continue
            except KeyboardInterrupt:
                now = time.monotonic()
                if ctrl_c_deadline is not None and now <= ctrl_c_deadline:
                    typer.echo("Second Ctrl+C received; exiting fast-agent.", err=True)
                    raise

                ctrl_c_deadline = now + ctrl_c_exit_window_seconds
                typer.echo(
                    "Interrupted operation; returning to fast-agent prompt. "
                    "Press Ctrl+C again within 2 seconds to exit.",
                    err=True,
                )
                continue

    await _resume_session_if_requested(agent_app, request)
    if request.message:
        response = await agent_app.send(
            request.message,
            agent_name=request.target_agent_name,
        )
        print(response)
    elif request.prompt_file:
        prompt = load_prompt(Path(request.prompt_file))
        agent_obj = agent_app._agent(request.target_agent_name)
        await agent_obj.generate(prompt)
        print(
            "\nLoaded "
            f"{len(prompt)} messages from prompt file '{request.prompt_file}'"
        )
        await _run_interactive_with_interrupt_recovery()
    else:
        await _run_interactive_with_interrupt_recovery()

    await _export_result_histories(agent_app, request)


async def run_agent_request(request: AgentRunRequest) -> None:
    """Run the normalized CLI request."""
    serve_permissions_enabled = request.permissions_enabled and not (
        request.noenv and request.mode == "serve"
    )

    instruction = request.instruction
    if instruction is None:
        instruction = resolve_default_instruction(
            request.model,
            request.mode,
            force_smart=request.force_smart,
        )

    smart_agent_enabled = resolve_smart_agent_enabled(
        request.model,
        request.mode,
        force_smart=request.force_smart,
    )
    smart_unavailable_warning = (
        "Warning: --smart requested, but smart defaults are unavailable when using "
        "multiple models. Continuing with non-smart defaults."
    )
    if request.mode == "serve" and request.transport in ["stdio", "acp"]:
        from fast_agent.ui.console import configure_console_stream

        configure_console_stream("stderr")

    from fast_agent import FastAgent
    from fast_agent.agents.llm_agent import LlmAgent
    from fast_agent.ui.console_display import ConsoleDisplay

    fast = FastAgent(
        name=request.name,
        config_path=request.config_path,
        ignore_unknown_args=True,
        parse_cli_args=False,
        quiet=request.mode == "serve",
        skills_directory=request.skills_directory,
        environment_dir=request.environment_dir,
    )

    if request.model:
        fast.args.model = request.model
    fast.args.noenv = request.noenv
    fast.args.reload = request.reload
    fast.args.watch = request.watch
    fast.args.agent = request.target_agent_name or request.agent_name or "agent"

    if request.noenv or request.shell_runtime:
        await fast.app.initialize()
        if request.noenv:
            config = fast.app.context.config
            if config is not None:
                config.session_history = False
        if request.shell_runtime:
            setattr(fast.app.context, "shell_runtime", True)

    if request.url_servers:
        await add_servers_to_config(
            fast,
            cast("dict[str, dict[str, Any]]", request.url_servers),
        )
    if request.stdio_servers:
        await add_servers_to_config(
            fast,
            cast("dict[str, dict[str, Any]]", request.stdio_servers),
        )

    if request.agent_cards or request.card_tools:
        try:
            if request.agent_cards:
                for card_source in request.agent_cards:
                    if card_source.startswith(("http://", "https://")):
                        fast.load_agents_from_url(card_source)
                    else:
                        fast.load_agents(card_source)

            has_explicit_default = False
            explicit_default_type: str | None = None
            for agent_data in fast.agents.values():
                config = agent_data.get("config")
                if config and getattr(config, "default", False):
                    has_explicit_default = True
                    explicit_default_type = (
                        str(agent_data.get("type")) if agent_data.get("type") is not None else None
                    )
                    break

            if has_explicit_default and request.force_smart:
                from fast_agent.agents.agent_types import AgentType

                if explicit_default_type != AgentType.SMART.value:
                    typer.echo(
                        "Warning: --smart requested, but loaded AgentCards already define a "
                        "non-smart default agent. Keeping the card-defined default.",
                        err=True,
                    )
            elif request.force_smart and not smart_agent_enabled:
                typer.echo(smart_unavailable_warning, err=True)

            if not has_explicit_default:
                agent_decorator = fast.smart if smart_agent_enabled else fast.agent

                @agent_decorator(
                    name="agent",
                    instruction=instruction,
                    servers=request.server_list or [],
                    model=request.model,
                    default=True,
                )
                async def default_fallback_agent() -> None:
                    pass

            tool_loaded_names: list[str] = []
            if request.card_tools:
                for card_source in request.card_tools:
                    if card_source.startswith(("http://", "https://")):
                        tool_loaded_names.extend(fast.load_agents_from_url(card_source))
                    else:
                        tool_loaded_names.extend(fast.load_agents(card_source))

            if tool_loaded_names:
                target_name = (
                    request.target_agent_name
                    if request.target_agent_name and request.target_agent_name in fast.agents
                    else None
                )
                if not target_name:
                    target_name = (
                        request.agent_name
                        if request.agent_name and request.agent_name in fast.agents
                        else None
                    )
                if not target_name:
                    target_name = fast.get_default_agent_name()
                if target_name:
                    fast.attach_agent_tools(target_name, tool_loaded_names)

            _validate_target_agent_name(fast, request)
            _apply_shell_cwd_policy_preflight(fast, request)
        except AgentConfigError as exc:
            fast._handle_error(exc)
            raise typer.Exit(1) from exc

        _attach_cli_servers_to_selected_agent(fast, request)

        async def cli_agent() -> None:
            async with fast.run() as agent:
                await _run_single_agent_cli_flow(agent, request)

    elif request.model and "," in request.model:
        if request.force_smart and not smart_agent_enabled:
            typer.echo(smart_unavailable_warning, err=True)

        models = [m.strip() for m in request.model.split(",") if m.strip()]

        fan_out_agents: list[str] = []
        for model_name in models:
            branch_agent_name = f"{model_name}"

            @fast.agent(
                name=branch_agent_name,
                instruction=instruction,
                servers=request.server_list or [],
                model=model_name,
            )
            async def model_agent() -> None:
                pass

            fan_out_agents.append(branch_agent_name)

        _validate_target_agent_name(fast, request)

        class SilentFanInAgent(LlmAgent):
            async def show_assistant_message(self, *args, **kwargs):  # type: ignore[override]
                return None

            def show_user_message(self, *args, **kwargs):  # type: ignore[override]
                return None

        @fast.custom(
            SilentFanInAgent,
            name="aggregate",
            model="passthrough",
            instruction="You aggregate parallel outputs without displaying intermediate messages.",
        )
        async def aggregate() -> None:
            pass

        @fast.parallel(
            name="parallel",
            fan_out=fan_out_agents,
            fan_in="aggregate",
            include_request=True,
            default=True,
        )
        async def cli_agent() -> None:
            async with fast.run() as agent:
                await _resume_session_if_requested(agent, request)
                if request.message:
                    if request.target_agent_name:
                        response = await agent.send(
                            request.message,
                            agent_name=request.target_agent_name,
                        )
                        print(response)
                    else:
                        await agent.parallel.send(request.message)
                        display = ConsoleDisplay(config=None)
                        display.show_parallel_results(agent.parallel)
                elif request.prompt_file:
                    from fast_agent.mcp.prompts.prompt_load import load_prompt

                    prompt = load_prompt(Path(request.prompt_file))
                    if request.target_agent_name:
                        agent_obj = agent._agent(request.target_agent_name)
                        await agent_obj.generate(prompt)
                        print(
                            "\nLoaded "
                            f"{len(prompt)} messages from prompt file '{request.prompt_file}'"
                        )
                    else:
                        await agent.parallel.generate(prompt)
                        display = ConsoleDisplay(config=None)
                        display.show_parallel_results(agent.parallel)
                else:
                    await agent.interactive(
                        agent_name=request.target_agent_name,
                        pretty_print_parallel=True,
                    )

                await _export_result_histories(
                    agent,
                    request,
                    fan_out_agent_names=fan_out_agents,
                )

    else:
        agent_decorator = fast.smart if smart_agent_enabled else fast.agent

        @agent_decorator(
            name=request.agent_name or "agent",
            instruction=instruction,
            servers=request.server_list or [],
            model=request.model,
            default=True,
        )
        async def cli_agent() -> None:
            async with fast.run() as agent:
                await _run_single_agent_cli_flow(agent, request)

        _validate_target_agent_name(fast, request)

    if request.mode == "serve":
        await fast.start_server(
            transport=request.transport,
            host=request.host,
            port=request.port,
            tool_description=request.tool_description,
            tool_name_template=request.tool_name_template,
            instance_scope=request.instance_scope,
            permissions_enabled=serve_permissions_enabled,
        )
    else:
        await cli_agent()
