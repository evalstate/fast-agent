"""Shared session export help metadata for CLI and slash-command discovery."""

from __future__ import annotations

from typing import TypedDict

from fast_agent.commands.metadata_labels import (
    metadata_argument_label,
    metadata_option_label,
)
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span
from fast_agent.utils.text import strip_to_none


class SessionExportArgumentDetail(TypedDict):
    name: str
    required: bool
    value_name: str | None
    summary: str


class SessionExportOptionDetail(TypedDict):
    name: str
    aliases: list[str]
    value_name: str | None
    summary: str


class SessionExportActionDetail(TypedDict):
    name: str
    summary: str
    usage: str
    examples: list[str]
    arguments: list[SessionExportArgumentDetail]
    options: list[SessionExportOptionDetail]
    notes: list[str]


SESSION_EXPORT_USAGE = (
    "/session export [latest|id|path] [--agent name] [--output path] "
    "[--hf-url hf://...] [--hf-dataset owner/name] [--hf-dataset-path path] [--privacy-filter] "
    "[--privacy-filter-path path] [--download-privacy-filter] "
    "[--privacy-filter-device auto|cpu|cuda] "
    "[--privacy-filter-variant q4|q4f16|q8|fp16] [--show-redactions]"
)

SESSION_EXPORT_TARGET_HELP = (
    "Session target: latest, session id, session dir, or session.json path."
)
SESSION_EXPORT_AGENT_HELP = "Agent name to export."
SESSION_EXPORT_OUTPUT_HELP = (
    "Write trace to this file path. Relative paths resolve from the current "
    "working directory. Parent directories are created as needed."
)
SESSION_EXPORT_HF_DATASET_HELP = "Compatibility option: upload the exported trace to this Hugging Face dataset repo (owner/name). Prefer --hf-url for new workflows."
SESSION_EXPORT_HF_URL_HELP = "Upload the exported trace to this Hugging Face URL. Supports hf://buckets/... and hf://datasets/...."
SESSION_EXPORT_HF_DATASET_PATH_HELP = (
    "Path in the dataset repo. Defaults to the root using the local filename. "
    "If the value ends with '/', it is treated as a folder. Requires --hf-dataset."
)
SESSION_EXPORT_PRIVACY_FILTER_HELP = "Redact exported text content with the local privacy filter."
SESSION_EXPORT_PRIVACY_PATH_HELP = "Local OpenAI Privacy Filter model directory."
SESSION_EXPORT_PRIVACY_DOWNLOAD_HELP = (
    "Download the default privacy-filter model if it is not already cached."
)
SESSION_EXPORT_PRIVACY_DEVICE_HELP = "Privacy filter device: auto, cpu, or cuda. Defaults to auto."
SESSION_EXPORT_PRIVACY_VARIANT_HELP = (
    "Privacy filter model variant: q4, q4f16, q8, or fp16. Defaults to q8."
)
SESSION_EXPORT_SHOW_REDACTIONS_HELP = (
    "Print detected redaction labels and original text to stderr. Use only for local review."
)

SESSION_EXPORT_EXAMPLES: tuple[str, ...] = (
    "/session export latest --output trace.jsonl",
    "/session export latest --hf-url hf://buckets/owner/traces/",
    "/session export latest --hf-url hf://datasets/owner/name/trace.jsonl",
    "/session export latest --privacy-filter",
    "/session export latest --help",
)

SESSION_EXPORT_NOTES: tuple[str, ...] = (
    "Default format: codex.",
    "If --output is omitted, the exporter writes "
    "`{session_id}__{agent_name}__codex.jsonl` in the current working directory.",
    "--output is a file path, not a directory path.",
    "If --agent is omitted, the current agent is used only for the current or latest session target.",
    "Privacy filtering is best-effort and requires the optional `privacy` extra.",
)


def build_session_export_action_detail() -> SessionExportActionDetail:
    """Return structured discovery metadata for `/session export`."""

    return {
        "name": "export",
        "summary": "export a session trace, optionally to HF",
        "usage": SESSION_EXPORT_USAGE,
        "examples": list(SESSION_EXPORT_EXAMPLES),
        "arguments": [
            {
                "name": "target",
                "required": False,
                "value_name": "latest|id|path",
                "summary": SESSION_EXPORT_TARGET_HELP,
            }
        ],
        "options": [
            {
                "name": "--agent",
                "aliases": ["-a"],
                "value_name": "name",
                "summary": SESSION_EXPORT_AGENT_HELP,
            },
            {
                "name": "--output",
                "aliases": ["-o"],
                "value_name": "path",
                "summary": SESSION_EXPORT_OUTPUT_HELP,
            },
            {
                "name": "--hf-url",
                "aliases": [],
                "value_name": "hf://...",
                "summary": SESSION_EXPORT_HF_URL_HELP,
            },
            {
                "name": "--hf-dataset",
                "aliases": [],
                "value_name": "owner/name",
                "summary": SESSION_EXPORT_HF_DATASET_HELP,
            },
            {
                "name": "--hf-dataset-path",
                "aliases": [],
                "value_name": "path",
                "summary": SESSION_EXPORT_HF_DATASET_PATH_HELP,
            },
            {
                "name": "--privacy-filter",
                "aliases": [],
                "value_name": None,
                "summary": SESSION_EXPORT_PRIVACY_FILTER_HELP,
            },
            {
                "name": "--privacy-filter-path",
                "aliases": [],
                "value_name": "path",
                "summary": SESSION_EXPORT_PRIVACY_PATH_HELP,
            },
            {
                "name": "--download-privacy-filter",
                "aliases": [],
                "value_name": None,
                "summary": SESSION_EXPORT_PRIVACY_DOWNLOAD_HELP,
            },
            {
                "name": "--privacy-filter-device",
                "aliases": [],
                "value_name": "auto|cpu|cuda",
                "summary": SESSION_EXPORT_PRIVACY_DEVICE_HELP,
            },
            {
                "name": "--privacy-filter-variant",
                "aliases": ["--privacy-filter-quant"],
                "value_name": "q4|q4f16|q8|fp16",
                "summary": SESSION_EXPORT_PRIVACY_VARIANT_HELP,
            },
            {
                "name": "--show-redactions",
                "aliases": [],
                "value_name": None,
                "summary": SESSION_EXPORT_SHOW_REDACTIONS_HELP,
            },
            {
                "name": "--help",
                "aliases": ["-h"],
                "value_name": None,
                "summary": "Show export-specific help.",
            },
        ],
        "notes": list(SESSION_EXPORT_NOTES),
    }


def _argument_label(argument: SessionExportArgumentDetail) -> str | None:
    return metadata_argument_label(argument)


def _option_label(option: SessionExportOptionDetail) -> str | None:
    return metadata_option_label(option)


def _append_labeled_help_line(
    lines: list[str],
    *,
    label: str | None,
    summary: str,
) -> None:
    if label is None:
        return
    summary_text = strip_to_none(summary)
    if summary_text is None:
        lines.append(f"- {label}")
        return
    lines.append(f"- {label} — {escape_markdown_text(summary_text)}")


def render_session_export_help_markdown() -> str:
    """Render markdown help for `/session export`."""

    detail = build_session_export_action_detail()
    lines = [
        "# session export",
        "",
        "Export a persisted session trace.",
        "",
        f"Usage: {markdown_code_span(detail['usage'])}",
        "",
        "Arguments:",
    ]

    for argument in detail["arguments"]:
        _append_labeled_help_line(
            lines,
            label=_argument_label(argument),
            summary=argument["summary"],
        )

    lines.extend(["", "Options:"])
    for option in detail["options"]:
        _append_labeled_help_line(
            lines,
            label=_option_label(option),
            summary=option["summary"],
        )

    lines.extend(["", "Behavior:"])
    lines.extend(
        f"- {escape_markdown_text(note)}"
        for value in detail["notes"]
        if (note := strip_to_none(value)) is not None
    )

    lines.extend(["", "Examples:"])
    lines.extend(
        f"- {markdown_code_span(example)}"
        for value in detail["examples"]
        if (example := strip_to_none(value)) is not None
    )

    return "\n".join(lines)
