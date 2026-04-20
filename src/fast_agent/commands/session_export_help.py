"""Shared session export help metadata for CLI and slash-command discovery."""

from __future__ import annotations

from typing import cast

SESSION_EXPORT_USAGE = (
    "/session export [latest|id|path] [--agent name] [--output path] "
    "[--hf-dataset owner/name] [--hf-dataset-path path]"
)

SESSION_EXPORT_TARGET_HELP = (
    "Session target: latest, session id, session dir, or session.json path."
)
SESSION_EXPORT_AGENT_HELP = "Agent name to export."
SESSION_EXPORT_OUTPUT_HELP = (
    "Write trace to this file path. Relative paths resolve from the current "
    "working directory. Parent directories are created as needed."
)
SESSION_EXPORT_HF_DATASET_HELP = (
    "Upload the exported trace to this Hugging Face dataset repo (owner/name)."
)
SESSION_EXPORT_HF_DATASET_PATH_HELP = (
    "Path in the dataset repo. Defaults to the root using the local filename. "
    "If the value ends with '/', it is treated as a folder. Requires --hf-dataset."
)

SESSION_EXPORT_EXAMPLES: tuple[str, ...] = (
    "/session export latest --output trace.jsonl",
    "/session export latest --hf-dataset owner/name",
    "/session export latest --help",
)

SESSION_EXPORT_NOTES: tuple[str, ...] = (
    "Default format: codex.",
    "If --output is omitted, the exporter writes "
    "`{session_id}__{agent_name}__codex.jsonl` in the current working directory.",
    "--output is a file path, not a directory path.",
    "If --agent is omitted, the current agent is used only for the current or latest session target.",
)


def build_session_export_action_detail() -> dict[str, object]:
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
                "name": "--help",
                "aliases": ["-h"],
                "value_name": None,
                "summary": "Show export-specific help.",
            },
        ],
        "notes": list(SESSION_EXPORT_NOTES),
    }


def render_session_export_help_markdown() -> str:
    """Render markdown help for `/session export`."""

    detail = build_session_export_action_detail()
    lines = [
        "# session export",
        "",
        "Export a persisted session trace.",
        "",
        f"Usage: `{detail['usage']}`",
        "",
        "Arguments:",
    ]

    arguments = detail.get("arguments")
    if isinstance(arguments, list):
        for argument in arguments:
            if not isinstance(argument, dict):
                continue
            argument_map = cast("dict[str, object]", argument)
            name = str(argument_map.get("name", "")).strip()
            if not name:
                continue
            value_name = argument_map.get("value_name")
            label = f"`{name}`"
            if isinstance(value_name, str) and value_name:
                label = f"`{name}` (`{value_name}`)"
            summary = str(argument_map.get("summary", "")).strip()
            lines.append(f"- {label} — {summary}")

    lines.extend(["", "Options:"])
    options = detail.get("options")
    if isinstance(options, list):
        for option in options:
            if not isinstance(option, dict):
                continue
            option_map = cast("dict[str, object]", option)
            name = str(option_map.get("name", "")).strip()
            if not name:
                continue
            labels = [f"`{name}`"]
            aliases = option_map.get("aliases")
            if isinstance(aliases, list):
                labels.extend(f"`{alias}`" for alias in aliases if isinstance(alias, str) and alias)
            value_name = option_map.get("value_name")
            if isinstance(value_name, str) and value_name:
                labels[0] = f"`{name} {value_name}`"
            summary = str(option_map.get("summary", "")).strip()
            lines.append(f"- {', '.join(labels)} — {summary}")

    lines.extend(["", "Behavior:"])
    notes = detail.get("notes")
    if isinstance(notes, list):
        for note in notes:
            if isinstance(note, str) and note:
                lines.append(f"- {note}")

    lines.extend(["", "Examples:"])
    examples = detail.get("examples")
    if isinstance(examples, list):
        for example in examples:
            if isinstance(example, str) and example:
                lines.append(f"- `{example}`")

    return "\n".join(lines)
