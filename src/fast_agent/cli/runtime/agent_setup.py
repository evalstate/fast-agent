"""Agent setup and execution branch logic for CLI runtime requests."""

from __future__ import annotations

import asyncio
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
from prompt_toolkit import PromptSession

from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.cli.commands.server_helpers import add_servers_to_config
from fast_agent.cli.constants import RESUME_LATEST_SENTINEL
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.keyring_utils import emit_keyring_access_notice
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
from fast_agent.llm.provider_types import Provider
from fast_agent.session.preview import find_last_assistant_preview_text
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.ui.model_picker_common import (
    has_explicit_provider_prefix,
    infer_initial_picker_provider,
    normalize_generic_model_spec,
)
from fast_agent.utils.async_utils import suppress_known_runtime_warnings

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
    from fast_agent.types import PromptMessageExtended

    from .run_request import AgentRunRequest

logger = get_logger(__name__)


def _find_last_assistant_text(history: list[Any]) -> str | None:
    typed_history = [message for message in history if hasattr(message, "role")]
    return find_last_assistant_preview_text(typed_history)


def _is_interactive_startup_notice_context(request: AgentRunRequest) -> bool:
    return request.mode == "interactive" and request.is_repl


def _should_prompt_for_model_picker(
    request: AgentRunRequest,
    *,
    stdin_is_tty: bool,
    stdout_is_tty: bool,
) -> bool:
    """Return True when interactive startup can safely prompt for model selection."""
    if not _is_interactive_startup_notice_context(request):
        return False
    return stdin_is_tty and stdout_is_tty


def _resolve_model_without_hardcoded_default(
    *,
    model: str | None,
    config_default_model: str | None,
    model_references: Mapping[str, Mapping[str, str]] | None,
) -> tuple[str | None, str | None]:
    """Resolve model precedence without falling back to the hardcoded system default."""
    from fast_agent.core.model_resolution import resolve_model_spec

    return resolve_model_spec(
        context=None,
        model=model,
        default_model=config_default_model,
        cli_model=model,
        fallback_to_hardcoded=False,
        model_references=model_references,
    )


def _load_request_settings(request: "AgentRunRequest"):
    from fast_agent import config as config_module

    if request.config_path is None:
        config_module._settings = None

    return get_settings_or_exit(request.config_path)


def _resolve_model_picker_initial_selection(
    *,
    settings: Any,
) -> tuple[str | None, str | None]:
    from fast_agent.core.exceptions import ModelConfigError
    from fast_agent.core.model_resolution import resolve_model_reference
    from fast_agent.llm.model_overlays import load_model_overlay_registry
    from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
    from fast_agent.ui.model_picker_common import model_identity

    references = getattr(settings, "model_references", None)
    if not isinstance(references, Mapping):
        return None, None

    system_references = references.get("system")
    if not isinstance(system_references, Mapping):
        return None, None

    raw_last_used = system_references.get("last_used")
    if not isinstance(raw_last_used, str):
        return None, None

    initial_model_spec = raw_last_used.strip()
    if not initial_model_spec:
        return None, None

    overlay_registry = load_model_overlay_registry(
        start_path=resolve_model_reference_start_path(
            settings=settings,
            fallback_path=Path.cwd(),
        ),
        env_dir=getattr(settings, "environment_dir", None),
    )
    if overlay_registry.resolve_model_string(initial_model_spec) is not None:
        return "overlays", initial_model_spec

    provider_name: str | None = None
    if model_identity(initial_model_spec) is not None:
        provider_name = infer_initial_picker_provider(initial_model_spec)
        return provider_name, initial_model_spec

    try:
        resolved_model_spec = resolve_model_reference(initial_model_spec, references)
    except ModelConfigError:
        return None, initial_model_spec

    if overlay_registry.resolve_model_string(resolved_model_spec) is not None:
        return "overlays", resolved_model_spec

    resolved_identity = model_identity(resolved_model_spec)
    if resolved_identity is not None:
        provider_name = infer_initial_picker_provider(resolved_model_spec)
        return provider_name, resolved_model_spec

    return None, initial_model_spec


def _persist_model_picker_last_used_selection(
    request: AgentRunRequest,
    *,
    settings: Any,
    model_spec: str,
) -> bool:
    from fast_agent.llm.model_reference_config import (
        ModelReferenceConfigService,
        resolve_model_reference_start_path,
    )
    from fast_agent.paths import resolve_environment_dir

    normalized_model = model_spec.strip()
    if request.noenv or not normalized_model:
        return False

    start_path = resolve_model_reference_start_path(settings=settings, fallback_path=Path.cwd())
    explicit_config_path = None
    if request.config_path is not None:
        loaded_config_file = getattr(settings, "_config_file", None)
        if isinstance(loaded_config_file, str) and loaded_config_file.strip():
            explicit_config_path = Path(loaded_config_file).expanduser().resolve()
        else:
            explicit_config_path = Path(request.config_path).expanduser().resolve()

    env_dir = resolve_environment_dir(
        settings=settings,
        cwd=Path.cwd(),
        override=request.environment_dir or getattr(settings, "environment_dir", None),
    )
    write_target = "project" if explicit_config_path is not None else "env"

    try:
        ModelReferenceConfigService(
            start_path=start_path,
            env_dir=env_dir,
            project_write_path=explicit_config_path,
        ).set_reference(
            "$system.last_used",
            normalized_model,
            target=write_target,
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist model picker last-used selection",
            env_dir=str(env_dir) if env_dir is not None else None,
            config_path=str(explicit_config_path) if explicit_config_path is not None else None,
            target=write_target,
            model_spec=normalized_model,
            error=str(exc),
        )
        return False

    references = getattr(settings, "model_references", None)
    if isinstance(references, dict):
        system_references = references.get("system")
        if isinstance(system_references, dict):
            system_references["last_used"] = normalized_model
        else:
            references["system"] = {"last_used": normalized_model}
    else:
        settings.model_references = {"system": {"last_used": normalized_model}}

    return True


def _normalize_generic_model_spec(raw_model: str) -> str | None:
    return normalize_generic_model_spec(raw_model)


def _generic_model_prompt_default(initial_model_spec: str | None) -> str:
    candidate = (initial_model_spec or "").strip()
    if candidate.startswith("generic."):
        candidate = candidate.removeprefix("generic.")
        return candidate or "llama3.2"
    if has_explicit_provider_prefix(candidate):
        return "llama3.2"
    return candidate or "llama3.2"


async def _prompt_for_generic_model_spec(*, default_model: str = "llama3.2") -> str:
    prompt_session = PromptSession()
    while True:
        try:
            with suppress_known_runtime_warnings():
                entered = await prompt_session.prompt_async(
                    "Local model (e.g. llama3.2): ",
                    default=default_model,
                )
        except (EOFError, KeyboardInterrupt):
            typer.echo("Model selection cancelled.", err=True)
            raise typer.Exit(1)

        normalized = _normalize_generic_model_spec(entered)
        if normalized:
            return normalized

        typer.echo("Please enter a non-empty model string.", err=True)


def _activate_model_picker_provider(action: str) -> bool:
    if action != "codex-login":
        typer.echo(f"Unsupported provider activation action: {action}", err=True)
        return False

    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
    from fast_agent.llm.provider.openai.codex_oauth import login_codex_oauth
    from fast_agent.ui import console

    typer.echo("Starting Codex OAuth login...", err=True)
    try:
        console.ensure_blocking_console()
        login_codex_oauth()
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        return False
    except (EOFError, KeyboardInterrupt):
        typer.echo("Codex OAuth login cancelled.", err=True)
        return False

    typer.echo("Codex OAuth login complete. Choose a Codex model to continue.", err=True)
    return True


async def _select_model_from_picker(
    request: AgentRunRequest,
    *,
    config_payload: dict[str, Any] | None = None,
    initial_provider: str | None = None,
    initial_model_spec: str | None = None,
) -> str:
    """Prompt user for model selection and return a resolved model string."""
    from fast_agent.ui.model_picker import run_model_picker_async

    config_path = Path(request.config_path) if request.config_path else None
    picker_start_path = (
        config_path.parent
        if config_path is not None
        else resolve_model_reference_start_path(settings=_load_request_settings(request))
    )
    while True:
        picker_result = await run_model_picker_async(
            config_path=config_path,
            config_payload=config_payload,
            start_path=picker_start_path,
            initial_provider=initial_provider,
            initial_model_spec=initial_model_spec,
        )
        if picker_result is None:
            typer.echo("Model selection cancelled.", err=True)
            raise typer.Exit(1)

        initial_provider = picker_result.provider

        if picker_result.activation_action is not None:
            _activate_model_picker_provider(picker_result.activation_action)
            continue

        if (
            picker_result.provider == Provider.GENERIC.config_name
            and picker_result.resolved_model is None
        ):
            return await _prompt_for_generic_model_spec(
                default_model=_generic_model_prompt_default(initial_model_spec),
            )

        if picker_result.refer_to_docs or not picker_result.resolved_model:
            typer.echo(
                "Selected provider requires manual model IDs/options. "
                "Please choose a concrete model (or press q to cancel).",
                err=True,
            )
            continue

        selected_model = picker_result.resolved_model or picker_result.selected_model
        assert selected_model is not None
        return selected_model


def _emit_startup_notice(request: AgentRunRequest, message: str) -> None:
    if _is_interactive_startup_notice_context(request):
        from fast_agent.ui.enhanced_prompt import queue_startup_notice

        queue_startup_notice(message)
        return

    typer.echo(message, err=True)


def _emit_immediate_notice(message: str) -> None:
    try:
        if not sys.stderr.isatty():
            return
    except Exception:
        return

    typer.echo(message, err=True)


def _emit_model_picker_keyring_notice(request: AgentRunRequest) -> None:
    """Explain the one-time keyring probe that happens while building the model picker."""
    del request
    emit_keyring_access_notice(
        purpose="checking stored Codex OAuth tokens for model setup",
        emitter=_emit_immediate_notice,
    )


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
        execution_mode=request.execution_mode or "repl",
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
    from fast_agent.ui.enhanced_prompt import queue_startup_markdown_notice, queue_startup_notice

    manager = get_session_manager()
    session_id = None if request.resume in ("", RESUME_LATEST_SENTINEL) else request.resume
    default_agent = agent_app._agent(None)
    agents_map = agent_app.registered_agents()
    fallback_agent_name = (
        agent_app.resolve_target_agent_name(request.target_agent_name)
        or getattr(default_agent, "name", None)
    )
    result = manager.resume_session_agents(
        agents_map,
        session_id,
        fallback_agent_name=fallback_agent_name,
    )
    interactive_notice = request.is_repl
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

    session = result.session
    loaded = result.loaded
    missing_agents = result.missing_agents
    usage_notices = result.usage_notices
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

    for usage_notice in usage_notices:
        if not usage_notice:
            continue
        if interactive_notice:
            queue_startup_notice(usage_notice)
        else:
            typer.echo(
                usage_notice
                .replace("[yellow]", "")
                .replace("[/yellow]", "")
                .replace("[dim]", "")
                .replace("[/dim]", ""),
                err=True,
            )

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
        preview_agent = agent_app.get_agent(first_loaded_name) or default_agent

    preview_history = getattr(preview_agent, "message_history", [])
    assistant_text = _find_last_assistant_text(list(preview_history))
    if assistant_text:
        if interactive_notice:
            queue_startup_notice("[dim]Last assistant message:[/dim]")
            queue_startup_markdown_notice(
                assistant_text,
                title="Last assistant message",
                right_info="session",
                agent_name=getattr(preview_agent, "name", None),
            )
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


def _build_transient_result_messages(
    request_messages: str | list["PromptMessageExtended"],
    response: "PromptMessageExtended",
) -> list["PromptMessageExtended"]:
    from fast_agent.types import normalize_to_extended_list

    export_messages = [
        message.model_copy(deep=True) for message in normalize_to_extended_list(request_messages)
    ]
    export_messages.append(response.model_copy(deep=True))
    return export_messages



def _response_was_persisted(
    history_before: list["PromptMessageExtended"],
    history_after: list["PromptMessageExtended"],
    response: "PromptMessageExtended",
) -> bool:
    if len(history_after) <= len(history_before):
        return False

    last_message = history_after[-1]
    return (
        last_message.role == response.role
        and last_message.last_text() == response.last_text()
        and last_message.stop_reason == response.stop_reason
    )


async def _save_result_history(
    agent_app: Any,
    *,
    agent_name: str,
    output_path: Path,
    messages_override: list["PromptMessageExtended"] | None = None,
) -> None:
    from fast_agent.history.history_exporter import HistoryExporter
    from fast_agent.mcp.prompt_serialization import save_messages

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if messages_override is not None:
        save_messages(messages_override, str(output_path))
        return

    agent_obj = agent_app._agent(agent_name)
    await HistoryExporter.save(agent_obj, str(output_path))


async def _export_result_histories(
    agent_app: Any,
    request: AgentRunRequest,
    *,
    fan_out_agent_names: list[str] | None = None,
    transient_messages_by_agent: Mapping[str, list["PromptMessageExtended"]] | None = None,
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
                    messages_override=(
                        transient_messages_by_agent.get(agent_name)
                        if transient_messages_by_agent is not None
                        else None
                    ),
                )
            return

        selected_agent = agent_app._agent(request.target_agent_name)
        await _save_result_history(
            agent_app,
            agent_name=selected_agent.name,
            output_path=Path(request.result_file),
            messages_override=(
                transient_messages_by_agent.get(selected_agent.name)
                if transient_messages_by_agent is not None
                else None
            ),
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
                write_interactive_trace(
                    "cli.interactive.enter",
                    agent=request.target_agent_name,
                )
                await agent_app.interactive(agent_name=request.target_agent_name)
                write_interactive_trace(
                    "cli.interactive.return",
                    agent=request.target_agent_name,
                )
                return
            except asyncio.CancelledError:
                write_interactive_trace(
                    "cli.interactive.cancelled_error",
                    agent=request.target_agent_name,
                )
                task = asyncio.current_task()
                if task is not None:
                    while task.uncancel() > 0:
                        pass
                await asyncio.sleep(0)
                continue
            except KeyboardInterrupt:
                now = time.monotonic()
                write_interactive_trace(
                    "cli.interactive.keyboard_interrupt",
                    agent=request.target_agent_name,
                    had_deadline=ctrl_c_deadline is not None,
                    exiting=ctrl_c_deadline is not None and now <= ctrl_c_deadline,
                )
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
    transient_messages_by_agent: dict[str, list[PromptMessageExtended]] | None = None
    if request.execution_mode == "one_shot_message":
        assert request.message is not None
        agent_obj = agent_app._agent(request.target_agent_name)
        history_before = [message.model_copy(deep=True) for message in agent_obj.message_history]
        response = await agent_obj.generate(request.message)
        print(response.last_text() or "")
        if request.result_file and not _response_was_persisted(
            history_before,
            agent_obj.message_history,
            response,
        ):
            transient_messages_by_agent = {
                agent_obj.name: _build_transient_result_messages(request.message, response)
            }
    elif request.execution_mode == "one_shot_prompt_file":
        assert request.prompt_file is not None
        prompt = load_prompt(Path(request.prompt_file))
        agent_obj = agent_app._agent(request.target_agent_name)
        history_before = [message.model_copy(deep=True) for message in agent_obj.message_history]
        response = await agent_obj.generate(prompt)
        print(response.last_text() or "")
        if request.result_file and not _response_was_persisted(
            history_before,
            agent_obj.message_history,
            response,
        ):
            transient_messages_by_agent = {
                agent_obj.name: _build_transient_result_messages(prompt, response)
            }
    else:
        await _run_interactive_with_interrupt_recovery()

    await _export_result_histories(
        agent_app,
        request,
        transient_messages_by_agent=transient_messages_by_agent,
    )


async def run_agent_request(request: AgentRunRequest) -> None:
    """Run the normalized CLI request."""
    picker_model_source_override: str | None = None

    if request.model is None:
        settings = _load_request_settings(request)
        _, explicit_source = _resolve_model_without_hardcoded_default(
            model=request.model,
            config_default_model=getattr(settings, "default_model", None),
            model_references=getattr(settings, "model_references", None),
        )

        if explicit_source is None and _should_prompt_for_model_picker(
            request,
            stdin_is_tty=sys.stdin.isatty(),
            stdout_is_tty=sys.stdout.isatty(),
        ):
            _emit_model_picker_keyring_notice(request)
            initial_provider, initial_model_spec = _resolve_model_picker_initial_selection(
                settings=settings,
            )
            request.model = await _select_model_from_picker(
                request,
                config_payload=settings.model_dump(),
                initial_provider=initial_provider,
                initial_model_spec=initial_model_spec,
            )
            _persist_model_picker_last_used_selection(
                request,
                settings=settings,
                model_spec=request.model,
            )
            picker_model_source_override = "model picker"

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
        quiet=request.mode == "serve" or request.quiet,
        skills_directory=request.skills_directory,
        environment_dir=request.environment_dir,
    )

    if request.model:
        fast.args.model = request.model
    if picker_model_source_override:
        fast.args.model_source_override = picker_model_source_override
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
            async def show_assistant_message(self, *args, **kwargs):
                return None

            def show_user_message(self, *args, **kwargs):
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
                transient_messages_by_agent: dict[str, list[PromptMessageExtended]] | None = None
                if request.execution_mode == "one_shot_message":
                    assert request.message is not None
                    if request.target_agent_name:
                        agent_obj = agent._agent(request.target_agent_name)
                        history_before = [
                            message.model_copy(deep=True) for message in agent_obj.message_history
                        ]
                        response = await agent_obj.generate(request.message)
                        print(response.last_text() or "")
                        if request.result_file and not _response_was_persisted(
                            history_before,
                            agent_obj.message_history,
                            response,
                        ):
                            transient_messages_by_agent = {
                                agent_obj.name: _build_transient_result_messages(
                                    request.message,
                                    response,
                                )
                            }
                    else:
                        await agent.parallel.send(request.message)
                        display = ConsoleDisplay(config=None)
                        display.show_parallel_results(agent.parallel)
                elif request.execution_mode == "one_shot_prompt_file":
                    assert request.prompt_file is not None
                    from fast_agent.mcp.prompts.prompt_load import load_prompt

                    prompt = load_prompt(Path(request.prompt_file))
                    if request.target_agent_name:
                        agent_obj = agent._agent(request.target_agent_name)
                        history_before = [
                            message.model_copy(deep=True) for message in agent_obj.message_history
                        ]
                        response = await agent_obj.generate(prompt)
                        print(response.last_text() or "")
                        if request.result_file and not _response_was_persisted(
                            history_before,
                            agent_obj.message_history,
                            response,
                        ):
                            transient_messages_by_agent = {
                                agent_obj.name: _build_transient_result_messages(prompt, response)
                            }
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
                    transient_messages_by_agent=transient_messages_by_agent,
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
