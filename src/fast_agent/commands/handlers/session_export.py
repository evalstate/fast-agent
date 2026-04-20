"""Shared session trace export handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.commands.handlers.sessions import NOENV_SESSION_MESSAGE
from fast_agent.commands.results import CommandOutcome
from fast_agent.session.trace_export_errors import TraceExportError
from fast_agent.session.trace_export_models import ExportRequest
from fast_agent.session.trace_exporter import SessionTraceExporter

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


async def handle_session_export(
    ctx: CommandContext,
    *,
    target: str | None,
    agent_name: str | None,
    output_path: str | None,
    hf_dataset: str | None,
    hf_dataset_path: str | None,
    current_session_id: str | None = None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    if ctx.noenv:
        outcome.add_message(NOENV_SESSION_MESSAGE, channel="warning", right_info="session")
        return outcome

    if error is not None:
        outcome.add_message(error, channel="error", right_info="session")
        return outcome

    if hf_dataset_path is not None and hf_dataset is None:
        outcome.add_message(
            "--hf-dataset-path requires --hf-dataset.",
            channel="error",
            right_info="session",
        )
        return outcome

    request = ExportRequest(
        target=target,
        agent_name=agent_name,
        output_path=Path(output_path) if output_path is not None else None,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        current_session_id=current_session_id,
    )
    exporter = SessionTraceExporter(session_manager=ctx.resolve_session_manager())
    try:
        result = exporter.export(request)
    except TraceExportError as exc:
        outcome.add_message(str(exc), channel="error", right_info="session")
        return outcome

    outcome.add_message(
        (
            f"Exported {result.format} trace for agent '{result.agent_name}' "
            f"from session '{result.session_id}' to {result.output_path}"
        ),
        channel="info",
        right_info="session",
        agent_name=result.agent_name,
    )
    outcome.add_message(
        f"Wrote {result.record_count} trace records.",
        channel="info",
        right_info="session",
        agent_name=result.agent_name,
    )
    if result.upload is not None:
        outcome.add_message(
            (
                f"Uploaded trace to Hugging Face dataset '{result.upload.repo_id}' "
                f"as {result.upload.path_in_repo}"
            ),
            channel="info",
            right_info="session",
            agent_name=result.agent_name,
        )
        outcome.add_message(
            result.upload.file_url,
            channel="info",
            right_info="session",
            agent_name=result.agent_name,
        )
    return outcome
