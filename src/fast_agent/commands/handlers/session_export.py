"""Shared session trace export handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from fast_agent.commands.handlers.sessions import NOENV_SESSION_MESSAGE
from fast_agent.commands.results import CommandOutcome
from fast_agent.privacy.dependencies import (
    format_missing_privacy_dependencies,
    missing_privacy_dependencies,
)
from fast_agent.privacy.model_resolver import resolve_privacy_filter_model_dir
from fast_agent.privacy.privacy_filter_onnx import OpenAIPrivacyFilterOnnxSanitizer
from fast_agent.session.trace_export_errors import TraceExportError
from fast_agent.session.trace_export_models import ExportRequest
from fast_agent.session.trace_exporter import SessionTraceExporter

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.privacy.sanitizer import RedactionSummary


def _redaction_summary_text(summary: "RedactionSummary") -> str:
    elapsed = _format_elapsed(summary.elapsed.total_seconds()) if summary.elapsed else None
    if summary.total == 0:
        suffix = f" in {elapsed}" if elapsed else ""
        return f"Privacy filter redacted 0 text span(s){suffix}."
    suffix = f" in {elapsed}" if elapsed else ""
    lines = [f"Privacy filter redacted {summary.total} text span(s){suffix}:"]
    for label, count in summary.by_label.items():
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remaining:.0f}s"


async def handle_session_export(
    ctx: CommandContext,
    *,
    target: str | None,
    agent_name: str | None,
    output_path: str | None,
    hf_dataset: str | None,
    hf_dataset_path: str | None,
    privacy_filter: bool = False,
    privacy_filter_path: str | None = None,
    download_privacy_filter: bool = False,
    show_redactions: bool = False,
    progress_callback: Callable[[str], None] | None = None,
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
    if not privacy_filter and (privacy_filter_path is not None or download_privacy_filter):
        outcome.add_message(
            "--privacy-filter-path and --download-privacy-filter require --privacy-filter.",
            channel="error",
            right_info="session",
        )
        return outcome

    privacy_sanitizer = None
    if privacy_filter:
        if show_redactions:
            _emit_export_progress(
                progress_callback,
                "Privacy filter: warning: --show-redactions prints detected sensitive text.",
            )
        _emit_export_progress(progress_callback, "Privacy filter: checking dependencies...")
        missing = missing_privacy_dependencies()
        if missing:
            outcome.add_message(
                format_missing_privacy_dependencies(missing),
                channel="error",
                right_info="session",
            )
            return outcome
        try:
            _emit_export_progress(progress_callback, "Privacy filter: resolving model...")
            model_dir = resolve_privacy_filter_model_dir(
                model_path=Path(privacy_filter_path) if privacy_filter_path else None,
                allow_download=download_privacy_filter,
            )
            _emit_export_progress(progress_callback, f"Privacy filter: loading model from {model_dir}...")
            privacy_sanitizer = OpenAIPrivacyFilterOnnxSanitizer(
                model_dir,
                progress_callback=progress_callback,
                show_redactions=show_redactions,
            )
            _emit_export_progress(progress_callback, "Privacy filter: model loaded.")
        except TraceExportError as exc:
            outcome.add_message(str(exc), channel="error", right_info="session")
            return outcome

    request = ExportRequest(
        target=target,
        agent_name=agent_name,
        output_path=Path(output_path) if output_path is not None else None,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        current_session_id=current_session_id,
        privacy_filter=privacy_filter,
        privacy_filter_path=Path(privacy_filter_path) if privacy_filter_path else None,
        download_privacy_filter=download_privacy_filter,
    )
    exporter = SessionTraceExporter(
        session_manager=ctx.resolve_session_manager(),
        privacy_sanitizer=privacy_sanitizer,
        progress_callback=progress_callback,
    )
    try:
        _emit_export_progress(progress_callback, "Export: starting session trace export...")
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
    if result.redaction is not None:
        if result.upload is None:
            warning = (
                "Warning: privacy filtering is best-effort and applies to exported text "
                "content only. It can miss private data and can redact benign text. "
                "Review sanitized exports before sharing."
            )
        else:
            warning = (
                "Warning: privacy filtering is best-effort and applies to exported text "
                "content only. Review sanitized exports before uploading. Upload used "
                "the sanitized JSONL file only."
            )
        outcome.add_message(
            warning,
            channel="warning",
            right_info="session",
            agent_name=result.agent_name,
        )
        outcome.add_message(
            _redaction_summary_text(result.redaction),
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
        if result.redaction is not None:
            outcome.add_message(
                (
                    "Uploaded privacy-filtered trace. Privacy filtering is best-effort; "
                    "review shared traces for remaining sensitive data."
                ),
                channel="info",
                right_info="session",
                agent_name=result.agent_name,
            )
    return outcome


def _emit_export_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)
