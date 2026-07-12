"""Agent setup and execution branch logic for CLI runtime requests."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import typer

from fast_agent.cli.commands.server_helpers import add_servers_to_config
from fast_agent.core.card_tool_attachment import load_and_attach_card_tool_agents
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.ui.interactive_diagnostics import write_interactive_trace
from fast_agent.utils.filename import sanitize_filename_suffix
from fast_agent.utils.text import strip_casefold, strip_to_none
from fast_agent.utils.transports import uses_protocol_stdio

from .harness_startup import run_cli_flow, run_parallel_cli_flow
from .model_bootstrap import (
    agent_config_defines_startup_model as _agent_config_defines_startup_model,
)
from .model_bootstrap import (
    explicit_agent_cards_define_startup_model as _explicit_agent_cards_define_startup_model,
)
from .model_bootstrap import (
    generic_model_prompt_default as _generic_model_prompt_default,
)
from .model_bootstrap import (
    last_used_model_reference as _last_used_model_reference,
)
from .model_bootstrap import (
    load_request_settings as _load_request_settings,
)
from .model_bootstrap import (
    persist_model_picker_last_used_selection as _persist_model_picker_last_used_selection,
)
from .model_bootstrap import (
    resolve_model_picker_initial_selection as _resolve_model_picker_initial_selection,
)
from .model_bootstrap import (
    resolve_model_without_hardcoded_default as _resolve_model_without_hardcoded_default,
)
from .model_bootstrap import (
    select_model_from_picker as _select_model_from_picker,
)
from .model_bootstrap import (
    settings_model_references as _settings_model_references,
)
from .model_bootstrap import (
    should_prompt_for_model_picker as _should_prompt_for_model_picker,
)
from .model_bootstrap import (
    should_prompt_for_unpinned_system_default as _should_prompt_for_unpinned_system_default,
)
from .model_bootstrap import (
    system_default_reference_is_missing as _system_default_reference_is_missing,
)
from .one_shot import run_one_shot_payload as _run_one_shot_payload
from .request_builders import resolve_default_instruction, resolve_smart_agent_enabled
from .session_resume import (
    find_last_assistant_text as _find_last_assistant_text,
)
from .session_resume import (
    resume_session_id as _resume_session_id,
)
from .session_resume import (
    resume_session_if_requested,
)
from .session_resume import (
    validate_resume_request as _validate_resume_request,
)
from .shell_cwd_preflight import (
    apply_shell_cwd_policy_preflight as _apply_shell_cwd_policy_preflight,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from mcp.types import ContentBlock

    from fast_agent.core.agent_app import AgentApp
    from fast_agent.core.harness import HarnessSession
    from fast_agent.session.session_manager import SessionManager
    from fast_agent.types import PromptMessageExtended

    from .run_request import AgentRunRequest

logger = get_logger(__name__)

__all__ = [
    "_agent_config_defines_startup_model",
    "_find_last_assistant_text",
    "_generic_model_prompt_default",
    "_last_used_model_reference",
]


async def _resume_session_if_requested(agent_app: Any, request: AgentRunRequest) -> None:
    await resume_session_if_requested(cast("AgentApp", agent_app), request)


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


def _select_loaded_card_agent(
    fast,
    request: AgentRunRequest,
    loaded_agent_names: list[str],
) -> str | None:
    if request.target_agent_name and request.target_agent_name in fast.agents:
        return request.target_agent_name
    if request.agent_name and request.agent_name in fast.agents:
        request.target_agent_name = request.agent_name
        return request.agent_name

    runnable_names: list[str] = []
    for name in loaded_agent_names:
        agent_data = fast.agents.get(name)
        if not agent_data or agent_data.get("tool_only", False):
            continue
        runnable_names.append(name)

    if len(runnable_names) != 1:
        return None

    selected_name = runnable_names[0]
    request.target_agent_name = selected_name
    return selected_name


def _attach_cli_servers_to_selected_agent(fast, request: AgentRunRequest) -> None:
    if not request.server_list:
        return

    from fast_agent.agents.agent_types import AgentConfig

    selected_agent_data = None
    if request.target_agent_name and request.target_agent_name in fast.agents:
        selected_agent_data = fast.agents.get(request.target_agent_name)
    elif request.agent_name and request.agent_name in fast.agents:
        selected_agent_data = fast.agents.get(request.agent_name)

    if selected_agent_data is None:
        for agent_data in fast.agents.values():
            config = agent_data.get("config")
            if isinstance(config, AgentConfig) and config.default:
                selected_agent_data = agent_data
                break

    if selected_agent_data is None and fast.agents:
        for agent_data in fast.agents.values():
            if not agent_data.get("tool_only", False):
                selected_agent_data = agent_data
                break

    if selected_agent_data:
        config = selected_agent_data.get("config")
        if isinstance(config, AgentConfig):
            existing = list(config.servers) if config.servers else []
            config.servers = existing + [
                server for server in request.server_list if server not in existing
            ]


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
        suffix = sanitize_filename_suffix(agent_name)
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if suffix_counts[suffix] > 1:
            suffix = f"{suffix}-{suffix_counts[suffix]}"
        exports.append((agent_name, _build_result_file_with_suffix(base_path, suffix)))

    return exports


def _build_transient_result_messages(
    request_messages: str | "PromptMessageExtended" | list["PromptMessageExtended"],
    response: "PromptMessageExtended",
) -> list["PromptMessageExtended"]:
    from fast_agent.types import normalize_to_extended_list

    export_messages = [
        message.model_copy(deep=True) for message in normalize_to_extended_list(request_messages)
    ]
    export_messages.append(response.model_copy(deep=True))
    return export_messages


def _cli_attachment_token(source: str) -> str:
    from fast_agent.ui.prompt.attachment_tokens import (
        build_local_attachment_token,
        build_remote_attachment_token,
    )

    parsed = urlparse(source)
    if strip_casefold(parsed.scheme) in {"http", "https"}:
        return build_remote_attachment_token(source)
    return build_local_attachment_token(source)


async def _resolve_cli_attachment_blocks(
    resolver_agent: Any,
    attachments: list[str] | None,
) -> list["ContentBlock"]:
    if not attachments:
        return []

    from fast_agent.ui.prompt.resource_mentions import parse_mentions, resolve_mentions

    parsed = parse_mentions(" ".join(_cli_attachment_token(source) for source in attachments))
    try:
        resolved = await resolve_mentions(resolver_agent, parsed)
    except Exception as exc:
        typer.echo(f"Error resolving --attach: {exc}", err=True)
        raise typer.Exit(1) from exc
    return resolved.resources


async def _build_cli_message_payload(
    agent_obj: Any,
    message: str,
    attachments: list[str] | None,
) -> str | "PromptMessageExtended":
    blocks = await _resolve_cli_attachment_blocks(agent_obj, attachments)
    if not blocks:
        return message

    from mcp.types import TextContent

    from fast_agent.types import PromptMessageExtended

    return PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text=message), *blocks],
    )


async def _build_cli_prompt_file_payload(
    agent_obj: Any,
    prompt: list["PromptMessageExtended"],
    attachments: list[str] | None,
) -> list["PromptMessageExtended"]:
    blocks = await _resolve_cli_attachment_blocks(agent_obj, attachments)
    if not blocks:
        return prompt

    prompt_with_attachments = [message.model_copy(deep=True) for message in prompt]
    for message in reversed(prompt_with_attachments):
        if message.role == "user":
            message.content.extend(blocks)
            return prompt_with_attachments

    typer.echo(
        "Error: --attach requires the prompt file to contain at least one user message.",
        err=True,
    )
    raise typer.Exit(1)


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
    except Exception as exc:
        typer.echo(f"Error exporting result file: {exc}", err=True)
        raise typer.Exit(1) from exc


def _enable_atif_child_capture(agent_app: Any, request: AgentRunRequest) -> None:
    if request.trajectory_output is None:
        return
    from fast_agent.agents.llm_agent import LlmAgent

    for agent in agent_app.registered_agents().values():
        if isinstance(agent, LlmAgent) and not agent.config.use_history:
            agent.config.save_trajectory = True


def _live_atif_session_id(
    session_manager: SessionManager | None,
    harness_session: HarnessSession | None,
) -> str:
    if harness_session is not None:
        return harness_session.id
    if session_manager is not None and session_manager.current_session is not None:
        return session_manager.current_session.info.name
    return f"run_{uuid.uuid4().hex}"


def _live_atif_model_metadata(agent_obj: Any, request: AgentRunRequest) -> tuple[str | None, str | None]:
    llm = agent_obj.llm
    if llm is None:
        return request.model, None
    return llm.model_name or request.model, llm.provider.config_name


def _live_child_trajectory_dir(
    session_manager: SessionManager | None,
    harness_session: HarnessSession | None,
) -> Path | None:
    if harness_session is not None and harness_session.session_manager is not None:
        session = harness_session.session_manager.current_session
        return None if session is None else session.directory / "trajectories"
    if session_manager is None or session_manager.current_session is None:
        return None
    return session_manager.current_session.directory / "trajectories"


async def _live_atif_tool_definitions(agent_obj: Any) -> list[dict[str, object]] | None:
    listed = await agent_obj.list_tools()
    definitions: list[dict[str, object]] = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in listed.tools
    ]
    return definitions or None


async def _export_live_atif_trajectory(
    agent_app: Any,
    request: AgentRunRequest,
    *,
    transient_messages_by_agent: Mapping[str, list[PromptMessageExtended]] | None,
    session_manager: SessionManager | None,
    harness_session: HarnessSession | None,
    termination_error: BaseException | None = None,
) -> None:
    if request.trajectory_output is None:
        return

    from fast_agent.session.trace_export_atif import (
        AtifRunSource,
        build_atif_trajectory,
        write_atif_trajectory,
    )

    agent_obj = agent_app._agent(request.target_agent_name)
    messages = (
        transient_messages_by_agent.get(agent_obj.name)
        if transient_messages_by_agent is not None
        else None
    ) or [message.model_copy(deep=True) for message in agent_obj.message_history]
    if not messages:
        return
    model_name, provider = _live_atif_model_metadata(agent_obj, request)
    trajectory = build_atif_trajectory(
        AtifRunSource(
            session_id=_live_atif_session_id(session_manager, harness_session),
            agent_name=agent_obj.name,
            model_name=model_name,
            provider=provider,
            history=messages,
            message_timestamps=tuple(message.timestamp for message in messages),
            child_trajectory_dir=_live_child_trajectory_dir(
                session_manager, harness_session
            ),
            tool_definitions=await _live_atif_tool_definitions(agent_obj),
            extra=(
                {
                    "termination": {
                        "status": (
                            "cancelled"
                            if isinstance(
                                termination_error,
                                (asyncio.CancelledError, KeyboardInterrupt),
                            )
                            else "error"
                        ),
                        "error_type": type(termination_error).__name__,
                        "message": str(termination_error),
                    }
                }
                if termination_error is not None
                else None
            ),
            system_prompt=agent_obj.instruction,
        )
    )
    write_atif_trajectory(trajectory, request.trajectory_output.expanduser().resolve())


async def _export_parallel_atif_trajectory(
    agent_app: Any,
    request: AgentRunRequest,
    fan_out_agent_names: list[str],
    *,
    transient_messages_by_agent: Mapping[str, list[PromptMessageExtended]] | None,
    session_manager: SessionManager | None,
    harness_session: HarnessSession | None,
) -> None:
    if request.trajectory_output is None:
        return
    from fast_agent.session.trace_export_atif import (
        AtifRunSource,
        build_atif_fanout_trajectory,
        write_atif_trajectory,
    )

    session_id = _live_atif_session_id(session_manager, harness_session)
    sources: list[AtifRunSource] = []
    for agent_name in fan_out_agent_names:
        agent_obj = agent_app._agent(agent_name)
        messages = (
            transient_messages_by_agent.get(agent_name)
            if transient_messages_by_agent is not None
            else None
        ) or [message.model_copy(deep=True) for message in agent_obj.message_history]
        if not messages:
            continue
        model_name, provider = _live_atif_model_metadata(agent_obj, request)
        sources.append(
            AtifRunSource(
                session_id=session_id,
                agent_name=agent_name,
                model_name=model_name,
                provider=provider,
                history=messages,
                message_timestamps=tuple(message.timestamp for message in messages),
                child_trajectory_dir=_live_child_trajectory_dir(
                    session_manager, harness_session
                ),
                tool_definitions=await _live_atif_tool_definitions(agent_obj),
                system_prompt=agent_obj.instruction,
            )
        )
    if not sources:
        return
    trajectory = build_atif_fanout_trajectory(session_id=session_id, sources=sources)
    write_atif_trajectory(trajectory, request.trajectory_output.expanduser().resolve())


async def _export_failed_one_shot_atif(
    agent_app: Any,
    agent_obj: Any,
    prompt_payload: str | PromptMessageExtended | list[PromptMessageExtended],
    request: AgentRunRequest,
    *,
    history_before: list[PromptMessageExtended],
    session_manager: SessionManager | None,
    harness_session: HarnessSession | None,
    error: BaseException,
) -> None:
    if request.trajectory_output is None:
        return
    from fast_agent.agents.tool_agent import ToolAgent
    from fast_agent.types import normalize_to_extended_list

    new_history = agent_obj.message_history[len(history_before) :]
    if isinstance(agent_obj, ToolAgent) and agent_obj.last_turn_messages:
        messages = [
            message.model_copy(deep=True) for message in agent_obj.last_turn_messages
        ]
    else:
        messages = [
            *(
                message.model_copy(deep=True)
                for message in normalize_to_extended_list(prompt_payload)
            ),
            *(message.model_copy(deep=True) for message in new_history),
        ]
    await _export_live_atif_trajectory(
        agent_app,
        request,
        transient_messages_by_agent={agent_obj.name: messages},
        session_manager=session_manager,
        harness_session=harness_session,
        termination_error=error,
    )


async def _run_cli_flow(
    agent_app: Any,
    request: AgentRunRequest,
    *,
    session_manager: "SessionManager | None" = None,
    harness_session: "HarnessSession | None" = None,
) -> None:
    from fast_agent.mcp.prompts.prompt_load import load_prompt

    _enable_atif_child_capture(agent_app, request)

    # Allow interactive prompt startup checks to honor per-run CLI override policy.
    agent_app.missing_shell_cwd_policy_override = request.missing_shell_cwd_policy
    try:
        from fast_agent.llm.structured_schema import load_structured_schema_source

        structured_source = (
            load_structured_schema_source(
                json_schema=request.json_schema,
                schema_model=request.schema_model,
            )
            if request.json_schema is not None or request.schema_model is not None
            else None
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if harness_session is None:
        await _resume_session_if_requested(agent_app, request)
    transient_messages_by_agent: dict[str, list[PromptMessageExtended]] | None = None
    one_shot_response: PromptMessageExtended | None = None
    if request.execution_mode == "one_shot_message":
        assert request.message is not None
        agent_obj = agent_app._agent(request.target_agent_name)
        history_before = [message.model_copy(deep=True) for message in agent_obj.message_history]
        prompt_payload = await _build_cli_message_payload(
            agent_obj,
            request.message,
            request.attachments,
        )
        try:
            response = await _run_one_shot_payload(
                agent_obj,
                prompt_payload,
                request,
                structured_source,
                harness_session=harness_session,
            )
        except BaseException as exc:
            await _export_failed_one_shot_atif(
                agent_app,
                agent_obj,
                prompt_payload,
                request,
                history_before=history_before,
                session_manager=session_manager,
                harness_session=harness_session,
                error=exc,
            )
            raise
        one_shot_response = response
        transient_messages_by_agent = _transient_result_messages_if_needed(
            agent_obj,
            request,
            history_before,
            prompt_payload,
            response,
        )
    elif request.execution_mode == "one_shot_prompt_file":
        assert request.prompt_file is not None
        prompt = load_prompt(request.prompt_file)
        agent_obj = agent_app._agent(request.target_agent_name)
        history_before = [message.model_copy(deep=True) for message in agent_obj.message_history]
        prompt_payload = await _build_cli_prompt_file_payload(
            agent_obj,
            prompt,
            request.attachments,
        )
        try:
            response = await _run_one_shot_payload(
                agent_obj,
                prompt_payload,
                request,
                structured_source,
                harness_session=harness_session,
            )
        except BaseException as exc:
            await _export_failed_one_shot_atif(
                agent_app,
                agent_obj,
                prompt_payload,
                request,
                history_before=history_before,
                session_manager=session_manager,
                harness_session=harness_session,
                error=exc,
            )
            raise
        one_shot_response = response
        transient_messages_by_agent = _transient_result_messages_if_needed(
            agent_obj,
            request,
            history_before,
            prompt_payload,
            response,
        )
    else:
        await _run_interactive_with_interrupt_recovery(
            agent_app,
            request,
            session_manager=session_manager,
            harness_session=harness_session,
        )

    await _export_result_histories(
        agent_app,
        request,
        transient_messages_by_agent=transient_messages_by_agent,
    )
    await _export_live_atif_trajectory(
        agent_app,
        request,
        transient_messages_by_agent=transient_messages_by_agent,
        session_manager=session_manager,
        harness_session=harness_session,
    )
    if (
        one_shot_response is not None
        and one_shot_response.stop_reason == LlmStopReason.ERROR
    ):
        raise typer.Exit(1)


async def _run_interactive_with_interrupt_recovery(
    agent_app: Any,
    request: AgentRunRequest,
    *,
    session_manager: "SessionManager | None" = None,
    harness_session: "HarnessSession | None" = None,
) -> None:
    ctrl_c_exit_window_seconds = 2.0
    ctrl_c_deadline: float | None = None

    while True:
        try:
            write_interactive_trace("cli.interactive.enter", agent=request.target_agent_name)
            await agent_app.interactive(
                agent_name=request.target_agent_name,
                session_manager=session_manager,
                harness_session=harness_session,
            )
            write_interactive_trace("cli.interactive.return", agent=request.target_agent_name)
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
        except KeyboardInterrupt:
            now = time.monotonic()
            exiting = ctrl_c_deadline is not None and now <= ctrl_c_deadline
            write_interactive_trace(
                "cli.interactive.keyboard_interrupt",
                agent=request.target_agent_name,
                had_deadline=ctrl_c_deadline is not None,
                exiting=exiting,
            )
            if exiting:
                typer.echo("Second Ctrl+C received; exiting fast-agent.", err=True)
                raise

            ctrl_c_deadline = now + ctrl_c_exit_window_seconds
            typer.echo(
                "Interrupted operation; returning to fast-agent prompt. "
                "Press Ctrl+C again within 2 seconds to exit.",
                err=True,
            )


def _transient_result_messages_if_needed(
    agent_obj: Any,
    request: AgentRunRequest,
    history_before: list[PromptMessageExtended],
    prompt_payload: str | PromptMessageExtended | list[PromptMessageExtended],
    response: PromptMessageExtended,
) -> dict[str, list[PromptMessageExtended]] | None:
    if not request.result_file and request.trajectory_output is None:
        return None
    if _response_was_persisted(history_before, agent_obj.message_history, response):
        return None
    from fast_agent.agents.tool_agent import ToolAgent

    if isinstance(agent_obj, ToolAgent) and agent_obj.last_turn_messages:
        return {
            agent_obj.name: [
                message.model_copy(deep=True) for message in agent_obj.last_turn_messages
            ]
        }
    return {agent_obj.name: _build_transient_result_messages(prompt_payload, response)}


async def _select_startup_model_if_needed(request: AgentRunRequest) -> str | None:
    if request.model is not None:
        return None

    if request.resume:
        # Resuming a session: the persisted session snapshot owns the model
        # (restored by session hydration during run). Showing the startup
        # picker here is both misleading and ineffective -- its selection is
        # set on ``request.model`` before the agents are initialized, yet
        # hydration overrides it with the snapshot model afterward, so the
        # "Model selected via model picker" notice would fire while the
        # resumed model stays active. Surface a distinctive source instead so
        # the startup status notice names session resumption as the source,
        # mirroring the notices shown for other model sources.
        return "session resumption"

    settings = _load_request_settings(request)
    can_prompt_for_model = _should_prompt_for_model_picker(
        request,
        stdin_is_tty=sys.stdin.isatty(),
        stdout_is_tty=sys.stdout.isatty(),
    )
    startup_model_defined_by_card = _explicit_agent_cards_define_startup_model(
        request,
        model_references=settings.model_references,
        system_default_requires_explicit=(
            can_prompt_for_model
            and _system_default_reference_is_missing(_settings_model_references(settings))
        ),
    )
    resolved_model = _resolve_model_without_hardcoded_default(
        model=request.model,
        config_default_model=settings.default_model,
        model_references=settings.model_references,
    )

    if (
        not startup_model_defined_by_card
        and _should_prompt_for_unpinned_system_default(settings, can_prompt=can_prompt_for_model)
    ):
        initial_selection = _resolve_model_picker_initial_selection(settings=settings)
        request.model = await _select_model_from_picker(
            request,
            config_payload=settings.model_dump(),
            initial_provider=initial_selection.provider,
            initial_model_spec=initial_selection.model_spec,
        )
        _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec=request.model,
        )
        return "model picker"

    if resolved_model.source is not None or startup_model_defined_by_card:
        return None

    if can_prompt_for_model:
        initial_selection = _resolve_model_picker_initial_selection(settings=settings)
        request.model = await _select_model_from_picker(
            request,
            config_payload=settings.model_dump(),
            initial_provider=initial_selection.provider,
            initial_model_spec=initial_selection.model_spec,
        )
        _persist_model_picker_last_used_selection(
            request,
            settings=settings,
            model_spec=request.model,
        )
        return "model picker"

    initial_selection = _resolve_model_picker_initial_selection(settings=settings)
    if initial_selection.model_spec:
        request.model = initial_selection.model_spec
        return "last used model"
    return None


def _serve_permissions_enabled(request: AgentRunRequest) -> bool:
    return request.permissions_enabled and not (request.no_home and request.mode == "serve")


def _request_instruction(request: AgentRunRequest) -> str | None:
    if request.instruction is not None:
        return request.instruction
    return resolve_default_instruction(
        request.model,
        request.mode,
        force_smart=request.force_smart,
    )


def _smart_unavailable_warning() -> str:
    return (
        "Warning: --smart requested, but smart defaults are unavailable when using "
        "multiple models. Continuing with non-smart defaults."
    )


def _configure_stdio_server_console(request: AgentRunRequest) -> None:
    if request.mode == "serve" and uses_protocol_stdio(request.transport):
        from fast_agent.ui.console import configure_console_stream

        configure_console_stream("stderr")


def _build_fast_agent(request: AgentRunRequest):
    from fast_agent import FastAgent

    return FastAgent(
        name=request.name,
        config_path=request.config_path,
        ignore_unknown_args=True,
        parse_cli_args=False,
        quiet=request.mode == "serve" or request.quiet,
        skills_directory=request.skills_directory,
        home=request.home,
        no_home=request.no_home,
    )


def _apply_fast_args(
    fast: Any,
    request: AgentRunRequest,
    *,
    model_source_override: str | None = None,
) -> None:
    _validate_resume_request(request)
    if request.model:
        fast.args.model = request.model
    fast.args.resume_requested = request.resume is not None
    fast.args.resume_session_id = _resume_session_id(request)
    if model_source_override:
        fast.args.model_source_override = model_source_override
    fast.args.no_home = request.no_home
    fast.args.reload = request.reload
    fast.args.watch = request.watch
    fast.args.card_tools = request.card_tools
    fast.args.agent = request.target_agent_name or request.agent_name or "agent"


async def _apply_runtime_context_overrides(fast: Any, request: AgentRunRequest) -> None:
    if not (
        request.no_home or request.shell_runtime or request.no_shell or request.prefer_local_shell
    ):
        return

    await fast.app.initialize()
    config = fast.app.context.config
    if request.no_home and config is not None:
        config.session_history = False
    context = fast.app.context
    if request.shell_runtime:
        context.shell_runtime = True
    if request.no_shell:
        context.no_shell = True
    if request.prefer_local_shell and config is not None:
        config.shell_execution.prefer_local_shell = True


async def _add_cli_servers(fast: Any, request: AgentRunRequest) -> None:
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


def _card_defined_default_type(fast: Any) -> str | None:
    from fast_agent.agents.agent_types import AgentConfig

    for agent_data in fast.agents.values():
        config = agent_data.get("config")
        if isinstance(config, AgentConfig) and config.default:
            agent_type = agent_data.get("type")
            return str(agent_type) if agent_type is not None else None
    return None


def _warn_if_card_default_overrides_smart(
    request: AgentRunRequest,
    explicit_default_type: str | None,
    smart_agent_enabled: bool,
    smart_unavailable_warning: str,
) -> None:
    if explicit_default_type is not None and request.force_smart:
        from fast_agent.agents.agent_types import AgentType

        if explicit_default_type != AgentType.SMART.value:
            typer.echo(
                "Warning: --smart requested, but loaded AgentCards already define a "
                "non-smart default agent. Keeping the card-defined default.",
                err=True,
            )
    elif request.force_smart and not smart_agent_enabled:
        typer.echo(smart_unavailable_warning, err=True)


def _define_card_fallback_agent(
    fast: Any,
    request: AgentRunRequest,
    instruction: str | None,
    smart_agent_enabled: bool,
) -> None:
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


def _configure_card_agents(
    fast: Any,
    request: AgentRunRequest,
    instruction: str | None,
    smart_agent_enabled: bool,
    smart_unavailable_warning: str,
) -> None:
    try:
        loaded_agent_names: list[str] = []
        if request.agent_cards:
            for card_source in request.agent_cards:
                loaded_agent_names.extend(fast.load_agents(card_source))
        if request.mode == "serve" and loaded_agent_names:
            request.managed_mcp_agent_names = list(dict.fromkeys(loaded_agent_names))

        explicit_default_type = _card_defined_default_type(fast)
        _warn_if_card_default_overrides_smart(
            request,
            explicit_default_type,
            smart_agent_enabled,
            smart_unavailable_warning,
        )

        selected_loaded_agent = _select_loaded_card_agent(
            fast,
            request,
            loaded_agent_names,
        )
        if selected_loaded_agent:
            fast.args.agent = selected_loaded_agent

        if explicit_default_type is None and selected_loaded_agent is None:
            _define_card_fallback_agent(
                fast,
                request,
                instruction,
                smart_agent_enabled,
            )

        load_and_attach_card_tool_agents(
            fast,
            request.card_tools,
            preferred_agent_names=[request.target_agent_name, request.agent_name],
        )

        _validate_target_agent_name(fast, request)
        _apply_shell_cwd_policy_preflight(fast, request)
    except AgentConfigError as exc:
        fast._handle_error(exc)
        raise typer.Exit(1) from exc

    _attach_cli_servers_to_selected_agent(fast, request)


def _default_managed_mcp_agent_names(fast: Any) -> list[str]:
    names = [
        name
        for name, agent_data in fast.agents.items()
        if not bool(agent_data.get("tool_only", False))
    ]
    return names or list(fast.agents.keys())


def _build_card_cli_agent(
    fast: Any,
    request: AgentRunRequest,
    instruction: str | None,
    smart_agent_enabled: bool,
    smart_unavailable_warning: str,
) -> Callable[[], Awaitable[None]]:
    _configure_card_agents(
        fast,
        request,
        instruction,
        smart_agent_enabled,
        smart_unavailable_warning,
    )

    async def cli_agent() -> None:
        await run_cli_flow(
            fast,
            request,
            flow=_run_cli_flow,
            prepare=lambda: _attach_cli_servers_to_selected_agent(fast, request),
        )

    return cli_agent


def _split_requested_models(model: str) -> list[str]:
    return [
        model_name
        for raw_model_name in model.split(",")
        if (model_name := strip_to_none(raw_model_name)) is not None
    ]


def _define_model_fanout_agents(
    fast: Any,
    models: list[str],
    request: AgentRunRequest,
    instruction: str | None,
) -> list[str]:
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
    return fan_out_agents


def _define_parallel_fan_in_agent(fast: Any) -> None:
    from fast_agent.agents.llm_agent import LlmAgent

    class SilentFanInAgent(LlmAgent):
        async def show_assistant_message(self, *args, **kwargs):
            del args, kwargs

        def show_user_message(self, *args, **kwargs):
            del args, kwargs

    @fast.custom(
        SilentFanInAgent,
        name="aggregate",
        model="passthrough",
        instruction="You aggregate parallel outputs without displaying intermediate messages.",
    )
    async def aggregate() -> None:
        pass


def _define_parallel_agent(fast: Any, fan_out_agents: list[str]) -> None:
    @fast.parallel(
        name="parallel",
        fan_out=fan_out_agents,
        fan_in="aggregate",
        include_request=True,
        default=True,
    )
    async def parallel() -> None:
        pass


async def _run_parallel_target_one_shot(
    agent_obj: Any,
    request: AgentRunRequest,
    prompt_payload: Any,
) -> dict[str, list["PromptMessageExtended"]] | None:
    history_before = [message.model_copy(deep=True) for message in agent_obj.message_history]
    response = await agent_obj.generate(prompt_payload)
    print(response.last_text() or "")
    if request.result_file and not _response_was_persisted(
        history_before,
        agent_obj.message_history,
        response,
    ):
        return {
            agent_obj.name: _build_transient_result_messages(
                prompt_payload,
                response,
            )
        }
    return None


async def _run_parallel_message(
    agent_app: Any,
    request: AgentRunRequest,
) -> dict[str, list["PromptMessageExtended"]] | None:
    assert request.message is not None
    if request.target_agent_name:
        agent_obj = agent_app._agent(request.target_agent_name)
        prompt_payload = await _build_cli_message_payload(
            agent_obj,
            request.message,
            request.attachments,
        )
        return await _run_parallel_target_one_shot(agent_obj, request, prompt_payload)

    from fast_agent.ui.console_display import ConsoleDisplay

    prompt_payload = await _build_cli_message_payload(
        agent_app.parallel,
        request.message,
        request.attachments,
    )
    await agent_app.parallel.send(prompt_payload)
    ConsoleDisplay(config=None).show_parallel_results(agent_app.parallel)
    return None


async def _run_parallel_prompt_file(
    agent_app: Any,
    request: AgentRunRequest,
) -> dict[str, list["PromptMessageExtended"]] | None:
    assert request.prompt_file is not None
    from fast_agent.mcp.prompts.prompt_load import load_prompt

    prompt = load_prompt(request.prompt_file)
    if request.target_agent_name:
        agent_obj = agent_app._agent(request.target_agent_name)
        prompt_payload = await _build_cli_prompt_file_payload(
            agent_obj,
            prompt,
            request.attachments,
        )
        return await _run_parallel_target_one_shot(agent_obj, request, prompt_payload)

    from fast_agent.ui.console_display import ConsoleDisplay

    prompt_payload = await _build_cli_prompt_file_payload(
        agent_app.parallel,
        prompt,
        request.attachments,
    )
    await agent_app.parallel.generate(prompt_payload)
    ConsoleDisplay(config=None).show_parallel_results(agent_app.parallel)
    return None


async def _run_parallel_cli_flow(
    agent_app: Any,
    request: AgentRunRequest,
    fan_out_agent_names: list[str],
    *,
    session_manager: "SessionManager | None" = None,
    harness_session: "HarnessSession | None" = None,
) -> None:
    _enable_atif_child_capture(agent_app, request)
    if harness_session is None:
        await _resume_session_if_requested(agent_app, request)
    transient_messages_by_agent: dict[str, list["PromptMessageExtended"]] | None = None
    if request.execution_mode == "one_shot_message":
        transient_messages_by_agent = await _run_parallel_message(agent_app, request)
    elif request.execution_mode == "one_shot_prompt_file":
        transient_messages_by_agent = await _run_parallel_prompt_file(agent_app, request)
    else:
        await agent_app.interactive(
            agent_name=request.target_agent_name,
            pretty_print_parallel=True,
            session_manager=session_manager,
            harness_session=harness_session,
        )

    await _export_result_histories(
        agent_app,
        request,
        fan_out_agent_names=fan_out_agent_names,
        transient_messages_by_agent=transient_messages_by_agent,
    )
    await _export_parallel_atif_trajectory(
        agent_app,
        request,
        fan_out_agent_names,
        transient_messages_by_agent=transient_messages_by_agent,
        session_manager=session_manager,
        harness_session=harness_session,
    )


def _build_multi_model_cli_agent(
    fast: Any,
    request: AgentRunRequest,
    instruction: str | None,
    smart_agent_enabled: bool,
    smart_unavailable_warning: str,
) -> Callable[[], Awaitable[None]]:
    if request.force_smart and not smart_agent_enabled:
        typer.echo(smart_unavailable_warning, err=True)

    assert request.model is not None
    fan_out_agents = _define_model_fanout_agents(
        fast,
        _split_requested_models(request.model),
        request,
        instruction,
    )
    _validate_target_agent_name(fast, request)
    _define_parallel_fan_in_agent(fast)
    _define_parallel_agent(fast, fan_out_agents)

    async def cli_agent() -> None:
        await run_parallel_cli_flow(
            fast,
            request,
            fan_out_agents,
            flow=_run_parallel_cli_flow,
        )

    return cli_agent


async def run_agent_request(request: AgentRunRequest) -> None:
    """Run the normalized CLI request."""
    startup_model_source_override = await _select_startup_model_if_needed(request)
    serve_permissions_enabled = _serve_permissions_enabled(request)
    instruction = _request_instruction(request)
    smart_agent_enabled = resolve_smart_agent_enabled(
        request.model,
        request.mode,
        force_smart=request.force_smart,
    )
    smart_unavailable_warning = _smart_unavailable_warning()
    _configure_stdio_server_console(request)

    fast = _build_fast_agent(request)
    _apply_fast_args(
        fast,
        request,
        model_source_override=startup_model_source_override,
    )
    await _apply_runtime_context_overrides(fast, request)
    await _add_cli_servers(fast, request)

    if request.agent_cards or request.card_tools:
        cli_agent = _build_card_cli_agent(
            fast,
            request,
            instruction,
            smart_agent_enabled,
            smart_unavailable_warning,
        )

    elif request.model and "," in request.model:
        cli_agent = _build_multi_model_cli_agent(
            fast,
            request,
            instruction,
            smart_agent_enabled,
            smart_unavailable_warning,
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
            await run_cli_flow(
                fast,
                request,
                flow=_run_cli_flow,
            )

        _validate_target_agent_name(fast, request)

    if request.mode == "serve":
        if request.managed_mcp_agent_names is None:
            request.managed_mcp_agent_names = _default_managed_mcp_agent_names(fast)
        await fast.start_server(
            transport=request.transport,
            host=request.host,
            port=request.port,
            instance_scope=request.instance_scope,
            permissions_enabled=serve_permissions_enabled,
            managed_mcp_agent_names=request.managed_mcp_agent_names,
        )
    else:
        await cli_agent()
