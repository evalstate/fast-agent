"""Shared session trace export handler."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.commands.handlers.sessions import NOENV_SESSION_MESSAGE
from fast_agent.commands.results import CommandOutcome
from fast_agent.privacy.dependencies import (
    format_missing_privacy_dependencies,
    missing_privacy_dependencies,
)
from fast_agent.privacy.model_resolver import (
    DEFAULT_PRIVACY_FILTER_VARIANT,
    PRIVACY_FILTER_VARIANTS,
    resolve_privacy_filter_model_dir,
)
from fast_agent.privacy.privacy_filter_onnx import OpenAIPrivacyFilterOnnxSanitizer
from fast_agent.session.trace_export_errors import TraceExportError
from fast_agent.session.trace_export_models import ExportRequest, ExportResult
from fast_agent.session.trace_exporter import SessionTraceExporter
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import format_english_list, strip_to_none
from fast_agent.utils.time import format_duration

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.commands.context import CommandContext
    from fast_agent.privacy.sanitizer import RedactionSummary, TraceSanitizer

_PRIVACY_FILTER_DEVICES = ("auto", "cpu", "cuda")
_PRIVACY_FILTER_OPTION_FLAGS = (
    "--privacy-filter-path",
    "--download-privacy-filter",
    "--privacy-filter-device",
    "--privacy-filter-variant",
)


def _redaction_summary_text(summary: "RedactionSummary") -> str:
    elapsed = _format_elapsed(summary.elapsed.total_seconds()) if summary.elapsed else None
    count_text = format_count(summary.total, "text span")
    suffix = f" in {elapsed}" if elapsed else ""
    if summary.total == 0:
        return f"Privacy filter redacted {count_text}{suffix}."
    lines = [f"Privacy filter redacted {count_text}{suffix}:"]
    for label, count in summary.by_label.items():
        lines.append(f"  {label}: {count}")
    return "\n".join(lines)


def _format_elapsed(seconds: float) -> str:
    return format_duration(seconds)


def _unsupported_choice_error(
    *,
    flag: str,
    value: str,
    supported: tuple[str, ...],
    supported_label: str,
) -> str:
    return f"Unsupported {flag} '{value}'. Supported {supported_label}: {', '.join(supported)}."


def _privacy_filter_options_requested(
    *,
    privacy_filter_path: str | None,
    download_privacy_filter: bool,
    privacy_filter_device: str | None,
    privacy_filter_variant: str | None,
) -> bool:
    return any(
        (
            privacy_filter_path is not None,
            download_privacy_filter,
            privacy_filter_device is not None,
            privacy_filter_variant is not None,
        )
    )


def _path_option(value: str | None) -> Path | None:
    text = strip_to_none(value)
    return Path(text) if text is not None else None


def _session_export_option_error(
    *,
    hf_url: str | None = None,
    hf_dataset: str | None = None,
    hf_dataset_path: str | None = None,
    privacy_filter: bool,
    privacy_filter_path: str | None,
    download_privacy_filter: bool,
    privacy_filter_device: str | None,
    privacy_filter_variant: str | None,
) -> str | None:
    if hf_url is not None and hf_dataset is not None:
        return "--hf-url cannot be combined with --hf-dataset."
    if hf_url is not None and not hf_url.strip().startswith("hf://"):
        return "--hf-url must be an hf:// URL."
    if hf_dataset_path is not None and hf_dataset is None:
        return "--hf-dataset-path requires --hf-dataset."
    if not privacy_filter and _privacy_filter_options_requested(
        privacy_filter_path=privacy_filter_path,
        download_privacy_filter=download_privacy_filter,
        privacy_filter_device=privacy_filter_device,
        privacy_filter_variant=privacy_filter_variant,
    ):
        return f"{format_english_list(_PRIVACY_FILTER_OPTION_FLAGS)} require --privacy-filter."
    if (
        privacy_filter_device is not None
        and normalize_action_token(privacy_filter_device) not in _PRIVACY_FILTER_DEVICES
    ):
        return _unsupported_choice_error(
            flag="--privacy-filter-device",
            value=privacy_filter_device,
            supported=_PRIVACY_FILTER_DEVICES,
            supported_label="devices",
        )
    if (
        privacy_filter_variant is not None
        and normalize_action_token(privacy_filter_variant) not in PRIVACY_FILTER_VARIANTS
    ):
        return _unsupported_choice_error(
            flag="--privacy-filter-variant",
            value=privacy_filter_variant,
            supported=PRIVACY_FILTER_VARIANTS,
            supported_label="variants",
        )
    return None


def _add_export_result_messages(outcome: CommandOutcome, result: ExportResult) -> None:
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
        f"Wrote {format_count(result.record_count, 'trace record')}.",
        channel="info",
        right_info="session",
        agent_name=result.agent_name,
    )

    if result.redaction is not None:
        _add_redaction_result_messages(outcome, result)
    if result.upload is not None:
        _add_upload_result_messages(outcome, result)


def _add_redaction_result_messages(outcome: CommandOutcome, result: ExportResult) -> None:
    redaction = result.redaction
    if redaction is None:
        return
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
        _redaction_summary_text(redaction),
        channel="info",
        right_info="session",
        agent_name=result.agent_name,
    )


def _add_upload_result_messages(outcome: CommandOutcome, result: ExportResult) -> None:
    if result.upload is None:
        return
    if result.upload.destination_label == "url":
        destination = result.upload.destination_url or result.upload.file_url
        outcome.add_message(
            f"Uploaded trace to Hugging Face URL {destination}",
            channel="info",
            right_info="session",
            agent_name=result.agent_name,
        )
    else:
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


def _add_session_export_preflight_error(
    outcome: CommandOutcome,
    ctx: CommandContext,
    *,
    hf_url: str | None = None,
    hf_dataset: str | None = None,
    hf_dataset_path: str | None = None,
    privacy_filter: bool,
    privacy_filter_path: str | None,
    download_privacy_filter: bool,
    privacy_filter_device: str | None,
    privacy_filter_variant: str | None,
    error: str | None,
) -> bool:
    if ctx.noenv:
        outcome.add_message(NOENV_SESSION_MESSAGE, channel="warning", right_info="session")
        return True

    if error is not None:
        outcome.add_message(error, channel="error", right_info="session")
        return True

    option_error = _session_export_option_error(
        hf_url=hf_url,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        privacy_filter=privacy_filter,
        privacy_filter_path=privacy_filter_path,
        download_privacy_filter=download_privacy_filter,
        privacy_filter_device=privacy_filter_device,
        privacy_filter_variant=privacy_filter_variant,
    )
    if option_error is not None:
        outcome.add_message(option_error, channel="error", right_info="session")
        return True

    return False


def _build_privacy_sanitizer(
    outcome: CommandOutcome,
    *,
    privacy_filter: bool,
    privacy_filter_path: str | None,
    download_privacy_filter: bool,
    privacy_filter_device: str | None,
    privacy_filter_variant: str | None,
    show_redactions: bool,
    progress_callback: Callable[[str], None] | None,
) -> "TraceSanitizer | None":
    if not privacy_filter:
        return None

    variant = (
        DEFAULT_PRIVACY_FILTER_VARIANT
        if privacy_filter_variant is None
        else normalize_action_token(privacy_filter_variant)
    )
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
        return None

    try:
        _emit_export_progress(progress_callback, "Privacy filter: resolving model...")
        model_dir, resolved_variant = resolve_privacy_filter_model_dir(
            model_path=Path(privacy_filter_path) if privacy_filter_path else None,
            variant=variant,
            allow_download=download_privacy_filter,
            variant_explicit=privacy_filter_variant is not None,
        )
        if resolved_variant != variant:
            _emit_export_progress(
                progress_callback,
                (
                    f"Privacy filter: variant '{variant}' not cached; "
                    f"using cached variant '{resolved_variant}'."
                ),
            )
        _emit_export_progress(
            progress_callback, f"Privacy filter: loading model from {model_dir}..."
        )
        sanitizer = OpenAIPrivacyFilterOnnxSanitizer(
            model_dir,
            variant=resolved_variant,
            device=privacy_filter_device,
            progress_callback=progress_callback,
            show_redactions=show_redactions,
        )
        _emit_export_progress(progress_callback, "Privacy filter: model loaded.")
        return sanitizer
    except TraceExportError as exc:
        outcome.add_message(str(exc), channel="error", right_info="session")
        return None


async def handle_session_export(
    ctx: CommandContext,
    *,
    target: str | None,
    agent_name: str | None,
    output_path: str | None,
    hf_url: str | None = None,
    hf_dataset: str | None = None,
    hf_dataset_path: str | None = None,
    privacy_filter: bool = False,
    privacy_filter_path: str | None = None,
    download_privacy_filter: bool = False,
    privacy_filter_device: str | None = None,
    privacy_filter_variant: str | None = None,
    show_redactions: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    current_session_id: str | None = None,
    error: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    output_path = strip_to_none(output_path)
    hf_url = strip_to_none(hf_url)
    hf_dataset = strip_to_none(hf_dataset)
    hf_dataset_path = strip_to_none(hf_dataset_path)
    privacy_filter_path = strip_to_none(privacy_filter_path)
    privacy_filter_device = strip_to_none(privacy_filter_device)
    privacy_filter_variant = strip_to_none(privacy_filter_variant)

    if _add_session_export_preflight_error(
        outcome,
        ctx,
        hf_url=hf_url,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        privacy_filter=privacy_filter,
        privacy_filter_path=privacy_filter_path,
        download_privacy_filter=download_privacy_filter,
        privacy_filter_device=privacy_filter_device,
        privacy_filter_variant=privacy_filter_variant,
        error=error,
    ):
        return outcome

    privacy_sanitizer = _build_privacy_sanitizer(
        outcome,
        privacy_filter=privacy_filter,
        privacy_filter_path=privacy_filter_path,
        download_privacy_filter=download_privacy_filter,
        privacy_filter_device=privacy_filter_device,
        privacy_filter_variant=privacy_filter_variant,
        show_redactions=show_redactions,
        progress_callback=progress_callback,
    )
    if privacy_filter and privacy_sanitizer is None:
        return outcome

    request = ExportRequest(
        target=target,
        agent_name=agent_name,
        output_path=_path_option(output_path),
        hf_url=hf_url,
        hf_dataset=hf_dataset,
        hf_dataset_path=hf_dataset_path,
        current_session_id=current_session_id,
        privacy_filter=privacy_filter,
        privacy_filter_path=_path_option(privacy_filter_path),
        download_privacy_filter=download_privacy_filter,
        privacy_filter_variant=privacy_filter_variant,
    )
    if ctx.session_runtime is None:
        outcome.add_message(
            "Session commands are unavailable in this context.",
            channel="warning",
            right_info="session",
        )
        return outcome
    exporter = SessionTraceExporter(
        session_manager=ctx.session_runtime.resolve_manager(),
        privacy_sanitizer=privacy_sanitizer,
        progress_callback=progress_callback,
    )
    try:
        _emit_export_progress(progress_callback, "Export: starting session trace export...")
        result = exporter.export(request)
    except TraceExportError as exc:
        outcome.add_message(str(exc), channel="error", right_info="session")
        return outcome

    _add_export_result_messages(outcome, result)
    return outcome


def _emit_export_progress(
    progress_callback: Callable[[str], None] | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)
