"""Shared /model management command handlers."""

from __future__ import annotations

import os
import shlex
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from rich.text import Text

from fast_agent.commands.command_catalog import suggest_command_action
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import parse_model_reference_token, resolve_model_reference
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_reference_config import (
    ModelReferenceConfigService,
    ModelReferenceMutationResult,
    ModelReferenceWriteTarget,
    resolve_model_reference_start_path,
)
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.a3_headers import build_a3_section_header
from fast_agent.ui.model_picker_common import infer_initial_picker_provider

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.commands.context import CommandContext

_PROVIDER_NAME_ALIASES: dict[str, str] = {
    "hf": "huggingface",
    "codex-responses": "codexresponses",
    "codex_responses": "codexresponses",
}

_NO_MODEL_REFERENCES_NOTE = (
    "No model_references are configured. Add a model_references section in fastagent.config.yaml."
)
_REFERENCES_USAGE = (
    "Usage: /model references "
    "[list|set [<token> [<model-spec>]] [--target env|project] [--dry-run]|"
    "unset [<token>] [--target env|project] [--dry-run]]"
)
_MODELS_USAGE = "Usage: /model [doctor|references|catalog|help] [args]"


@dataclass(frozen=True)
class _ReferencesMutationArgs:
    operation: Literal["set", "unset"]
    token: str | None
    model_spec: str | None
    target: ModelReferenceWriteTarget
    dry_run: bool


@dataclass(frozen=True)
class _AgentModelDoctorRow:
    name: str
    specified_model: str
    resolved_model: str
    status_symbol: str
    status_style: str
    resolution_note: str | None = None


@dataclass(frozen=True)
class _ModelsDoctorReport:
    readiness_ready: bool
    env_dir_env: str | None
    effective_env_dir: object
    fast_agent_model_env: str | None
    loaded_config_file: object
    unresolved: list[tuple[str, str, str]]
    configured_providers: set[Provider]
    agent_rows: list[_AgentModelDoctorRow]
    default_provider: Provider | None
    default_provider_ready: bool


def _append_line(content: Text, line: str | Text = "") -> None:
    if isinstance(line, Text):
        content.append_text(line)
    else:
        content.append(line)
    content.append("\n")


def _a3_header(title: str, *, color: str = "blue") -> Text:
    return build_a3_section_header(title, color=color, include_dot=False)


def _a3_section(title: str) -> Text:
    return build_a3_section_header(title.rstrip(":"), color="blue", include_dot=False)


def _a3_bullet(text: str, *, style: str = "white") -> Text:
    line = Text()
    line.append("• ", style="dim")
    line.append(text, style=style)
    return line


def _a3_status_line(label: str, value: str, *, value_style: str) -> Text:
    line = Text()
    line.append(f"{label}: ", style="dim")
    line.append(value, style=value_style)
    return line


def _a3_error_block(title: str, message: str) -> Text:
    content = Text()
    _append_line(content, _a3_header(title, color="red"))
    _append_line(content)
    _append_line(content, _a3_bullet(message, style="red"))
    return content


def _is_help_flag(value: str | None) -> bool:
    token = (value or "").strip().lower()
    return token in {"help", "--help", "-h"}


def _all_agent_names(ctx: "CommandContext") -> list[str]:
    return sorted(str(name) for name in ctx.agent_provider.registered_agent_names())


def _safe_stripped(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _canonical_model_name(model_spec: str) -> str:
    normalized = model_spec.strip()
    if not normalized:
        return normalized

    try:
        parsed = ModelFactory.parse_model_string(
            normalized,
            presets=ModelFactory.MODEL_PRESETS,
        )
    except Exception:
        return normalized

    return parsed.model_name


def _models_equivalent(expected: str, runtime: str) -> bool:
    if expected == runtime:
        return True

    try:
        expected_normalized = ModelDatabase.normalize_model_name(expected)
        runtime_normalized = ModelDatabase.normalize_model_name(runtime)
        if expected_normalized and runtime_normalized and expected_normalized == runtime_normalized:
            return True
    except Exception:
        pass

    return _canonical_model_name(expected) == _canonical_model_name(runtime)


def _build_agent_model_rows(
    ctx: "CommandContext",
    *,
    references: dict[str, dict[str, str]],
    default_model: str | None,
) -> list[_AgentModelDoctorRow]:
    rows: list[_AgentModelDoctorRow] = []

    for agent_name in _all_agent_names(ctx):
        try:
            agent = ctx.agent_provider._agent(agent_name)
        except Exception:
            continue

        config = getattr(agent, "config", None)

        specified = _safe_stripped(getattr(config, "model", None))
        effective_spec = specified or _safe_stripped(default_model)
        specified_display = specified or "<default>"

        resolved_from_spec: str | None = None
        status_symbol = "…"
        status_style = "dim"
        resolution_note: str | None = None
        reference_error: str | None = None

        if effective_spec and effective_spec.startswith("$"):
            try:
                resolved_from_spec = resolve_model_reference(effective_spec, references)
            except ModelConfigError as exc:
                reference_error = exc.details
        elif effective_spec:
            resolved_from_spec = effective_spec

        llm = getattr(agent, "llm", None) or getattr(agent, "_llm", None)
        llm_model = _safe_stripped(getattr(llm, "model_name", None)) if llm is not None else None

        if reference_error:
            if llm_model:
                resolved_display = llm_model
                status_symbol = "◐"
                status_style = "yellow"
                resolution_note = reference_error
            else:
                resolved_display = "<unresolved>"
                status_symbol = "✗"
                status_style = "red"
                resolution_note = reference_error
        elif resolved_from_spec and llm_model:
            if _models_equivalent(resolved_from_spec, llm_model):
                resolved_display = llm_model
                status_symbol = "✓"
                status_style = "green"
            else:
                resolved_display = llm_model
                status_symbol = "◐"
                status_style = "cyan"
                resolution_note = (
                    f"Resolved spec suggests '{resolved_from_spec}' but runtime uses '{llm_model}'."
                )
        elif resolved_from_spec:
            resolved_display = resolved_from_spec
            status_symbol = "✓"
            status_style = "green"
        elif llm_model:
            resolved_display = llm_model
            status_symbol = "✓"
            status_style = "cyan"
        else:
            resolved_display = "<unknown>"

        rows.append(
            _AgentModelDoctorRow(
                name=agent_name,
                specified_model=specified_display,
                resolved_model=resolved_display,
                status_symbol=status_symbol,
                status_style=status_style,
                resolution_note=resolution_note,
            )
        )

    return rows


def _truncate_cell(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


def _render_agent_model_table(rows: list[_AgentModelDoctorRow]) -> Text:
    content = Text()
    _append_line(content, _a3_section("Agent model resolution:"))

    if not rows:
        _append_line(content, _a3_bullet("No agents are currently registered.", style="dim"))
        return content

    status_labels = {
        "✓": "✓ resolved",
        "◐": "◐ fallback/override",
        "✗": "✗ unresolved",
    }
    table_values = [
        [
            row.name,
            row.specified_model,
            row.resolved_model,
            status_labels.get(row.status_symbol, row.status_symbol),
        ]
        for row in rows
    ]

    headers = ["Agent", "Specified", "Resolved", "Resolution"]
    max_limits = [24, 34, 34, 20]
    widths = [len(header) for header in headers]
    for row_values in table_values:
        for index, value in enumerate(row_values):
            widths[index] = min(max(widths[index], len(value)), max_limits[index])

    def _build_row(
        values: list[tuple[str, str]],
        *,
        indent: str = "  ",
    ) -> Text:
        line = Text(indent, style="dim")
        for index, (value, style) in enumerate(values):
            cell = _truncate_cell(value, limit=widths[index]).ljust(widths[index])
            line.append(cell, style=style)
            if index < len(values) - 1:
                line.append("  ", style="dim")
        return line

    _append_line(
        content,
        _build_row(
            [
                (headers[0], "bold bright_white"),
                (headers[1], "bold bright_white"),
                (headers[2], "bold bright_white"),
                (headers[3], "bold bright_white"),
            ]
        ),
    )

    note_counts = Counter(note for note in (row.resolution_note for row in rows) if note)
    repeated_note_lines: list[str] = []
    repeated_seen: set[str] = set()

    for row, values in zip(rows, table_values, strict=False):
        resolved_style = "red" if row.status_symbol == "✗" else "green"
        _append_line(
            content,
            _build_row(
                [
                    (values[0], "cyan"),
                    (values[1], "white"),
                    (values[2], resolved_style),
                    (values[3], row.status_style),
                ]
            ),
        )
        if row.resolution_note and note_counts[row.resolution_note] <= 1:
            _append_line(content, Text(f"    note: {row.resolution_note}", style="dim"))
        elif row.resolution_note and row.resolution_note not in repeated_seen:
            repeated_seen.add(row.resolution_note)
            repeated_note_lines.append(row.resolution_note)

    if repeated_note_lines:
        _append_line(content)
        _append_line(content, _a3_section("Notes:"))
        for note in repeated_note_lines:
            _append_line(content, Text(f"  • {note}", style="dim"))

    return content


def _render_agent_model_summary(rows: list[_AgentModelDoctorRow]) -> Text:
    total = len(rows)
    resolved = sum(1 for row in rows if row.status_symbol == "✓")
    attention = sum(1 for row in rows if row.status_symbol == "◐")
    unresolved = sum(1 for row in rows if row.status_symbol == "✗")
    unknown = max(total - resolved - attention - unresolved, 0)

    line = Text()
    line.append("Agent summary: ", style="bold")
    line.append(str(total), style="bold cyan")
    line.append(" total  ", style="dim")
    line.append("✓ ", style="green")
    line.append(str(resolved), style="green")
    line.append(" resolved  ", style="dim")
    line.append("◐ ", style="yellow")
    line.append(str(attention), style="yellow")
    line.append(" attention  ", style="dim")
    line.append("✗ ", style="red")
    line.append(str(unresolved), style="red")
    line.append(" unresolved", style="dim")
    if unknown:
        line.append("  … ", style="dim")
        line.append(str(unknown), style="dim")
        line.append(" unknown", style="dim")

    content = Text()
    _append_line(content, line)
    return content


def _catalog_providers() -> list[Provider]:
    providers = list(ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER.keys())
    return sorted(providers, key=lambda provider: provider.display_name.lower())


def _normalize_provider_name(value: str) -> str:
    return value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _resolve_catalog_provider(name: str) -> Provider | None:
    normalized = _normalize_provider_name(_PROVIDER_NAME_ALIASES.get(name.strip().lower(), name))
    for provider in _catalog_providers():
        variants = {
            _normalize_provider_name(provider.config_name),
            _normalize_provider_name(provider.name),
            _normalize_provider_name(provider.display_name),
        }
        if normalized in variants:
            return provider
    return None


def _provider_display_choices() -> str:
    return ", ".join(provider.config_name for provider in _catalog_providers())


def _resolve_reference_service(ctx: "CommandContext") -> ModelReferenceConfigService:
    settings = ctx.resolve_settings()
    env_dir = getattr(settings, "environment_dir", None)
    start_path = resolve_model_reference_start_path(settings=settings)
    return ModelReferenceConfigService(start_path=start_path, env_dir=env_dir)


def _flatten_references(references: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for namespace, entries in sorted(references.items(), key=lambda item: str(item[0])):
        for key, model_spec in sorted(entries.items(), key=lambda item: str(item[0])):
            rows.append((f"${namespace}.{key}", model_spec))
    return rows


def _collect_unresolved_references(
    references: dict[str, dict[str, str]],
    *,
    default_model: str | None,
) -> list[tuple[str, str, str]]:
    unresolved: dict[tuple[str, str], str] = {}

    for token, _ in _flatten_references(references):
        try:
            resolve_model_reference(token, references)
        except ModelConfigError as exc:
            unresolved[(token, f"reference: {token}")] = exc.details

    if default_model and default_model.strip().startswith("$"):
        token = default_model.strip()
        try:
            resolve_model_reference(token, references)
        except ModelConfigError as exc:
            unresolved[(token, "default_model")] = exc.details

    rows = [
        (token, source, details)
        for (token, source), details in sorted(unresolved.items(), key=lambda item: item[0])
    ]
    return rows


def _resolve_config_payload(settings: Any) -> dict[str, Any]:
    try:
        dumped = settings.model_dump() if hasattr(settings, "model_dump") else {}
    except Exception:
        return {}
    return dumped if isinstance(dumped, dict) else {}


def _default_model_provider(
    *,
    default_model: str | None,
    references: dict[str, dict[str, str]],
) -> Provider | None:
    if not default_model:
        return None

    try:
        resolved_model = resolve_model_reference(default_model, references)
    except ModelConfigError:
        return None

    try:
        parsed = ModelFactory.parse_model_string(
            resolved_model,
            presets=ModelFactory.MODEL_PRESETS,
        )
    except Exception:
        return None

    provider = parsed.provider
    return provider if isinstance(provider, Provider) else None


def _provider_is_ready(provider: Provider, configured: set[Provider]) -> bool:
    if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
        return True
    return provider in configured


def _build_models_doctor_report(ctx: "CommandContext") -> _ModelsDoctorReport:
    settings = ctx.resolve_settings()
    env_dir_env = os.getenv("ENVIRONMENT_DIR")
    fast_agent_model_env = os.getenv("FAST_AGENT_MODEL")
    effective_env_dir = getattr(settings, "environment_dir", None)
    loaded_config_file = getattr(settings, "_config_file", None)

    service = _resolve_reference_service(ctx)
    references = service.list_references_tolerant()
    config_payload = _resolve_config_payload(settings)
    configured_providers = set(ModelSelectionCatalog.configured_providers(config_payload))
    default_model = getattr(settings, "default_model", None)
    unresolved = _collect_unresolved_references(references, default_model=default_model)
    agent_rows = _build_agent_model_rows(
        ctx,
        references=references,
        default_model=_safe_stripped(default_model),
    )

    default_provider = _default_model_provider(
        default_model=default_model,
        references=references,
    )
    default_provider_ready = True
    if default_provider is not None:
        default_provider_ready = _provider_is_ready(default_provider, configured_providers)

    readiness_ready = not unresolved and default_provider_ready and bool(configured_providers)
    return _ModelsDoctorReport(
        readiness_ready=readiness_ready,
        env_dir_env=env_dir_env,
        effective_env_dir=effective_env_dir,
        fast_agent_model_env=fast_agent_model_env,
        loaded_config_file=loaded_config_file,
        unresolved=unresolved,
        configured_providers=configured_providers,
        agent_rows=agent_rows,
        default_provider=default_provider,
        default_provider_ready=default_provider_ready,
    )


def _status_label_for_row(row: _AgentModelDoctorRow) -> str:
    return {
        "✓": "✓ Resolved",
        "◐": "◐ Fallback",
        "✗": "✗ Unresolved",
    }.get(row.status_symbol, row.status_symbol)


def _markdown_code_or_dash(value: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {"<default>"}:
        return "—"
    return f"`{normalized}`"


def render_models_doctor_markdown(ctx: "CommandContext") -> str:
    """Render `/model doctor` as markdown, optimized for ACP clients."""
    try:
        report = _build_models_doctor_report(ctx)
    except Exception as exc:  # noqa: BLE001
        return f"# model.doctor\n\n**Error:** Failed to load model references: {exc}"

    lines: list[str] = ["# model.doctor", ""]
    lines.append("## Readiness")
    lines.append(
        f"- **Status**: {'ready' if report.readiness_ready else 'action required'}"
    )

    lines.extend(["", "## Runtime config context"])
    lines.append(f"- **ENVIRONMENT_DIR**: `{report.env_dir_env or '<unset>'}`")
    lines.append(f"- **Effective environment_dir**: `{report.effective_env_dir or '<unset>'}`")
    lines.append(f"- **FAST_AGENT_MODEL**: `{report.fast_agent_model_env or '<unset>'}`")
    lines.append(f"- **Loaded config file**: `{report.loaded_config_file or '<none>'}`")

    lines.extend(["", "## Unresolved references"])
    if report.unresolved:
        for token, source, details in report.unresolved:
            lines.append(f"- `{token}` ({source})")
            if details:
                lines.append(f"  - {details}")
    else:
        lines.append("- none")

    lines.extend(["", "## Provider readiness"])
    for provider in _catalog_providers():
        state = "configured" if provider in report.configured_providers else "not configured"
        lines.append(f"- **{provider.display_name}**: {state}")

    lines.extend(["", "## Agent model resolution", ""])
    if report.agent_rows:
        lines.extend(
            [
                "| Agent | Specified | Resolved | Status |",
                "| --- | --- | --- | --- |",
            ]
        )
        repeated_note_counts = Counter(
            note for note in (row.resolution_note for row in report.agent_rows) if note
        )
        repeated_notes: list[str] = []
        repeated_seen: set[str] = set()
        for row in report.agent_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{row.name}`",
                        _markdown_code_or_dash(row.specified_model),
                        _markdown_code_or_dash(row.resolved_model),
                        _status_label_for_row(row),
                    ]
                )
                + " |"
            )
            if row.resolution_note and repeated_note_counts[row.resolution_note] > 1:
                if row.resolution_note not in repeated_seen:
                    repeated_seen.add(row.resolution_note)
                    repeated_notes.append(row.resolution_note)
        total = len(report.agent_rows)
        resolved = sum(1 for row in report.agent_rows if row.status_symbol == "✓")
        attention = sum(1 for row in report.agent_rows if row.status_symbol == "◐")
        unresolved_count = sum(1 for row in report.agent_rows if row.status_symbol == "✗")
        lines.extend(
            [
                "",
                "## Agent summary",
                f"- **Total**: {total}",
                f"- **Resolved**: {resolved}",
                f"- **Attention**: {attention}",
                f"- **Unresolved**: {unresolved_count}",
            ]
        )
        single_notes = [
            f"`{row.name}`: {row.resolution_note}"
            for row in report.agent_rows
            if row.resolution_note and repeated_note_counts[row.resolution_note] <= 1
        ]
        all_notes = [*single_notes, *repeated_notes]
        if all_notes:
            lines.extend(["", "## Notes"])
            lines.extend(f"- {note}" for note in all_notes)
    else:
        lines.append("_No agents are currently registered._")

    if report.default_provider is not None and not report.default_provider_ready:
        lines.append("")
        lines.append(
            f"- Default model provider `{report.default_provider.display_name}` is not configured for current settings."
        )

    if not report.configured_providers:
        lines.append("")
        lines.append("- No provider credentials detected.")

    if not report.readiness_ready:
        lines.extend(["", "## Next steps"])
        if report.unresolved:
            lines.append("- `/model references`")
        lines.append("- `/model catalog <provider>`")

    return "\n".join(lines)


async def handle_models_command(
    ctx: "CommandContext",
    *,
    agent_name: str,
    action: str,
    argument: str | None,
) -> CommandOutcome:
    del agent_name

    if _is_help_flag(action) or _is_help_flag(argument):
        outcome = CommandOutcome()
        outcome.add_message(_a3_header("model help"), right_info="model")
        outcome.add_message(_MODELS_USAGE, right_info="model")
        outcome.add_message(
            "Examples: /model doctor, /model references, /model catalog openai",
            right_info="model",
        )
        return outcome

    normalized_action = (action or "doctor").strip().lower()
    if normalized_action in {"", "list"}:
        normalized_action = "doctor"

    if normalized_action == "doctor":
        return await _handle_models_doctor(ctx)
    if normalized_action == "references":
        return await _handle_models_references(ctx, argument=argument)
    if normalized_action == "catalog":
        return await _handle_models_catalog(ctx, argument=argument)
    if normalized_action == "help":
        outcome = CommandOutcome()
        outcome.add_message(_MODELS_USAGE, right_info="model")
        return outcome

    outcome = CommandOutcome()
    suggestions = suggest_command_action("model", normalized_action)
    suggestion_text = ""
    if suggestions:
        suggestion_text = " Did you mean: " + ", ".join(f"`{name}`" for name in suggestions)
    outcome.add_message(
        _a3_error_block(
            "model",
            (
                "Unknown /model action. "
                "Use /model, /model doctor, /model references, "
                f"/model catalog <provider> [--all], or /model help.{suggestion_text}"
            ),
        ),
        channel="error",
        right_info="model",
    )
    return outcome


async def _handle_models_doctor(ctx: "CommandContext") -> CommandOutcome:
    outcome = CommandOutcome()
    try:
        report = _build_models_doctor_report(ctx)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(
            _a3_error_block("model doctor", f"Failed to load model references: {exc}"),
            channel="error",
            right_info="model",
        )
        return outcome

    content = Text()
    _append_line(content, _a3_header("model doctor"))
    _append_line(content)
    if report.readiness_ready:
        _append_line(content, _a3_status_line("Readiness", "ready", value_style="bold green"))
    else:
        _append_line(
            content,
            _a3_status_line("Readiness", "action required", value_style="bold yellow"),
        )

    _append_line(content)
    _append_line(content, _a3_section("Runtime config context:"))
    _append_line(
        content,
        _a3_bullet(f"ENVIRONMENT_DIR: {report.env_dir_env or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(
            f"Effective environment_dir: {report.effective_env_dir or '<unset>'}",
            style="dim",
        ),
    )
    _append_line(
        content,
        _a3_bullet(f"FAST_AGENT_MODEL: {report.fast_agent_model_env or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(f"Loaded config file: {report.loaded_config_file or '<none>'}", style="dim"),
    )

    _append_line(content)
    _append_line(content, _a3_section("Unresolved references:"))
    if report.unresolved:
        for token, source, details in report.unresolved:
            _append_line(content, _a3_bullet(f"{token} ({source})", style="yellow"))
            if details:
                _append_line(content, Text(f"  {details}", style="dim"))
    else:
        _append_line(content, _a3_bullet("none", style="dim"))

    _append_line(content)
    _append_line(content, _a3_section("Provider readiness:"))
    for provider in _catalog_providers():
        state = "configured" if provider in report.configured_providers else "not configured"
        state_style = "green" if state == "configured" else "dim"
        _append_line(
            content,
            _a3_bullet(f"{provider.display_name}: {state}", style=state_style),
        )

    _append_line(content)
    content.append_text(_render_agent_model_table(report.agent_rows))
    _append_line(content)
    content.append_text(_render_agent_model_summary(report.agent_rows))

    if report.default_provider is not None and not report.default_provider_ready:
        _append_line(content)
        _append_line(
            content,
            _a3_bullet(
                f"Default model provider '{report.default_provider.display_name}' is not configured for current settings.",
                style="yellow",
            ),
        )

    if not report.configured_providers:
        _append_line(content)
        _append_line(content, _a3_bullet("No provider credentials detected.", style="yellow"))

    if not report.readiness_ready:
        _append_line(content)
        _append_line(content, _a3_section("Next steps:"))
        if report.unresolved:
            _append_line(content, _a3_bullet("/model references", style="cyan"))
        _append_line(content, _a3_bullet("/model catalog <provider>", style="cyan"))

    outcome.add_message(content, right_info="model")
    return outcome


def _parse_references_arguments(
    argument: str | None,
) -> tuple[Literal["list", "mutate"], _ReferencesMutationArgs | None, str | None]:
    if not argument:
        return "list", None, None

    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        return "list", None, f"Invalid references arguments: {exc}"

    if not tokens:
        return "list", None, None

    subcmd = tokens[0].lower()
    if subcmd == "list":
        if len(tokens) > 1:
            return "list", None, "Unexpected arguments after 'list'."
        return "list", None, None

    if subcmd not in {"set", "unset"}:
        return "list", None, f"Unknown references action '{tokens[0]}'."

    target: ModelReferenceWriteTarget = "env"
    dry_run = False
    positional: list[str] = []

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--dry-run":
            dry_run = True
            index += 1
            continue
        if token == "--target":
            index += 1
            if index >= len(tokens):
                return "list", None, "Missing value for --target (expected env or project)."
            target_value = tokens[index].strip().lower()
            if target_value == "env":
                target = "env"
            elif target_value == "project":
                target = "project"
            else:
                return "list", None, "--target must be either 'env' or 'project'."
            index += 1
            continue
        if token.startswith("--target="):
            target_value = token.split("=", 1)[1].strip().lower()
            if target_value == "env":
                target = "env"
            elif target_value == "project":
                target = "project"
            else:
                return "list", None, "--target must be either 'env' or 'project'."
            index += 1
            continue
        if token.startswith("--"):
            return "list", None, f"Unknown option: {token}"

        positional.append(token)
        index += 1

    if subcmd == "set":
        if len(positional) > 2:
            return "list", None, "Too many positional arguments for references set."

        return (
            "mutate",
            _ReferencesMutationArgs(
                operation="set",
                token=positional[0] if positional else None,
                model_spec=positional[1] if len(positional) > 1 else None,
                target=target,
                dry_run=dry_run,
            ),
            None,
        )

    if len(positional) > 1:
        return "list", None, "Too many positional arguments for references unset."

    return (
        "mutate",
        _ReferencesMutationArgs(
            operation="unset",
            token=positional[0] if positional else None,
            model_spec=None,
            target=target,
            dry_run=dry_run,
        ),
        None,
    )


def _canonicalize_reference_token(token: str) -> str:
    namespace, key = parse_model_reference_token(token)
    return f"${namespace}.{key}"


def _normalize_interactive_reference_token(token: str) -> str:
    stripped = token.strip()
    if not stripped or stripped.startswith("$"):
        return stripped
    return f"${stripped}"


def _infer_initial_provider_name(model_spec: str | None) -> str | None:
    return infer_initial_picker_provider(model_spec)


async def _prompt_for_reference_token(
    ctx: "CommandContext",
    *,
    references: dict[str, dict[str, str]],
    operation: Literal["set", "unset"],
    target_path: Path,
) -> str | None:
    rows = _flatten_references(references)
    if operation == "unset" and not rows:
        return None

    if operation == "set":
        if rows:
            selection_content = Text()
            _append_line(selection_content, _a3_section("Reference setup target:"))
            _append_line(
                selection_content,
                _a3_bullet(str(target_path.resolve()), style="cyan"),
            )
            _append_line(selection_content)
            _append_line(selection_content, _a3_section("Available references:"))
            for index, (token, model_spec) in enumerate(rows, start=1):
                _append_line(
                    selection_content,
                    _a3_bullet(f"{index}. {token} → {model_spec}"),
                )
            _append_line(selection_content, _a3_bullet("new. Create a new reference", style="cyan"))
            await ctx.io.emit(
                CommandMessage(
                    text=selection_content,
                    right_info="model",
                )
            )

            option_labels = {str(index): token for index, (token, _) in enumerate(rows, start=1)}
            selection = await ctx.io.prompt_selection(
                "Reference to update (number or 'new'):",
                options=[*option_labels.keys(), "new"],
                allow_cancel=True,
            )
            if selection is None:
                return None
            normalized_selection = selection.strip().lower()
            if normalized_selection != "new":
                return option_labels.get(normalized_selection)

        prompt_default = "$system.default" if not rows else None
        entered = await ctx.io.prompt_text(
            "Reference token ($namespace.key):",
            default=prompt_default,
            allow_empty=False,
        )
        if entered is None:
            return None
        try:
            return _canonicalize_reference_token(_normalize_interactive_reference_token(entered))
        except ModelConfigError as exc:
            raise ValueError(exc.details) from exc

    selection_content = Text()
    _append_line(selection_content, _a3_section("Reference setup target:"))
    _append_line(
        selection_content,
        _a3_bullet(str(target_path.resolve()), style="cyan"),
    )
    _append_line(selection_content)
    _append_line(selection_content, _a3_section("Available references:"))
    option_labels = {}
    for index, (token, model_spec) in enumerate(rows, start=1):
        _append_line(
            selection_content,
            _a3_bullet(f"{index}. {token} → {model_spec}"),
        )
        option_labels[str(index)] = token
    await ctx.io.emit(
        CommandMessage(
            text=selection_content,
            right_info="model",
        )
    )
    selection = await ctx.io.prompt_selection(
        "Reference to remove (number):",
        options=list(option_labels.keys()),
        allow_cancel=True,
    )
    if selection is None:
        return None
    return option_labels.get(selection.strip().lower())


async def _resolve_reference_mutation_args(
    ctx: "CommandContext",
    *,
    service: ModelReferenceConfigService,
    mutation_args: _ReferencesMutationArgs,
) -> tuple[_ReferencesMutationArgs | None, str | None]:
    references = service.list_references_tolerant()
    target_path = (
        service.paths.env_path
        if mutation_args.target == "env"
        else service.paths.project_write_path
    )
    token = mutation_args.token
    try:
        if token is None:
            token = await _prompt_for_reference_token(
                ctx,
                references=references,
                operation=mutation_args.operation,
                target_path=target_path,
            )
            if token is None:
                return None, "Reference update cancelled."
        else:
            token = _canonicalize_reference_token(token)
    except ValueError as exc:
        return None, str(exc)
    except ModelConfigError as exc:
        return None, exc.details

    if mutation_args.operation == "unset":
        return (
            _ReferencesMutationArgs(
                operation="unset",
                token=token,
                model_spec=None,
                target=mutation_args.target,
                dry_run=mutation_args.dry_run,
            ),
            None,
        )

    model_spec = mutation_args.model_spec
    if model_spec is None:
        current_model = next(
            (
                value
                for reference_token, value in _flatten_references(references)
                if reference_token == token
            ),
            None,
        )
        model_spec = await ctx.io.prompt_model_selection(
            initial_provider=_infer_initial_provider_name(current_model),
            default_model=current_model,
        )
        if model_spec is None:
            return None, "Reference update cancelled."

    return (
        _ReferencesMutationArgs(
            operation="set",
            token=token,
            model_spec=model_spec,
            target=mutation_args.target,
            dry_run=mutation_args.dry_run,
        ),
        None,
    )


def _render_reference_mutation(
    *,
    title: str,
    result: ModelReferenceMutationResult,
) -> Text:
    content = Text()
    _append_line(content, _a3_header(title))
    _append_line(content)

    if result.dry_run:
        _append_line(content, _a3_status_line("Mode", "dry-run", value_style="bold yellow"))
    elif result.applied:
        _append_line(content, _a3_status_line("Result", "applied", value_style="bold green"))
    else:
        _append_line(content, _a3_status_line("Result", "no changes", value_style="bold dim"))

    _append_line(
        content,
        _a3_status_line("Target", str(result.target_path.resolve()), value_style="cyan"),
    )
    _append_line(content)
    _append_line(content, _a3_section("Changes:"))

    for change in result.changes:
        old_value = change.old if change.old is not None else "<unset>"
        new_value = change.new if change.new is not None else "<unset>"
        _append_line(content, Text(f"{change.key_path}:", style="bold"))
        _append_line(content, Text(f"  old: {old_value}", style="dim"))
        _append_line(content, Text(f"  new: {new_value}", style="dim"))

    if result.dry_run:
        _append_line(content)
        _append_line(content, _a3_bullet("Dry run only (no files changed)", style="yellow"))

    return content


async def _handle_models_references(ctx: "CommandContext", *, argument: str | None) -> CommandOutcome:
    outcome = CommandOutcome()

    mode, mutation_args, parse_error = _parse_references_arguments(argument)
    if parse_error is not None:
        error = Text()
        _append_line(error, _a3_header("model references", color="red"))
        _append_line(error)
        _append_line(error, _a3_bullet(parse_error, style="red"))
        _append_line(error, Text(_REFERENCES_USAGE, style="dim"))
        outcome.add_message(error, channel="error", right_info="model")
        return outcome

    try:
        service = _resolve_reference_service(ctx)
        if mode == "mutate":
            assert mutation_args is not None
            resolved_args, interactive_error = await _resolve_reference_mutation_args(
                ctx,
                service=service,
                mutation_args=mutation_args,
            )
            if interactive_error is not None:
                content = _a3_error_block("model references", interactive_error)
                is_cancelled = interactive_error.endswith("cancelled.")
                if not is_cancelled:
                    _append_line(content, Text(_REFERENCES_USAGE, style="dim"))
                outcome.add_message(
                    content,
                    channel="warning" if is_cancelled else "error",
                    right_info="model",
                )
                return outcome

            assert resolved_args is not None
            if mutation_args.operation == "set":
                assert resolved_args.token is not None
                assert resolved_args.model_spec is not None
                mutation_result = service.set_reference(
                    resolved_args.token,
                    resolved_args.model_spec,
                    target=resolved_args.target,
                    dry_run=resolved_args.dry_run,
                )
                outcome.add_message(
                    _render_reference_mutation(
                        title="model references set",
                        result=mutation_result,
                    ),
                    right_info="model",
                )
                return outcome

            assert resolved_args.token is not None
            mutation_result = service.unset_reference(
                resolved_args.token,
                target=resolved_args.target,
                dry_run=resolved_args.dry_run,
            )
            outcome.add_message(
                _render_reference_mutation(
                    title="model references unset",
                    result=mutation_result,
                ),
                right_info="model",
            )
            return outcome

        references = service.list_references()
    except ValueError as exc:
        error = Text()
        _append_line(error, _a3_header("model references", color="red"))
        _append_line(error)
        _append_line(error, _a3_bullet(str(exc), style="red"))
        _append_line(error, Text(_REFERENCES_USAGE, style="dim"))
        outcome.add_message(error, channel="error", right_info="model")
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(
            _a3_error_block("model references", f"Failed to load model references: {exc}"),
            channel="error",
            right_info="model",
        )
        return outcome

    rows = _flatten_references(references)
    if not rows:
        empty = Text()
        _append_line(empty, _a3_header("model references"))
        _append_line(empty)
        _append_line(empty, _a3_bullet("No model references configured.", style="yellow"))
        outcome.add_message(empty, channel="warning", right_info="model")
        return outcome

    content = Text()
    _append_line(content, _a3_header("model references"))
    _append_line(content)
    _append_line(content, _a3_section("Model references:"))
    for token, model_spec in rows:
        try:
            resolved = resolve_model_reference(token, references)
        except ModelConfigError as exc:
            _append_line(content, _a3_bullet(f"{token} = {model_spec}", style="yellow"))
            _append_line(content, Text(f"  unresolved: {exc.details}", style="dim"))
            continue

        if resolved != model_spec:
            _append_line(
                content,
                _a3_bullet(f"{token} = {model_spec} -> {resolved}", style="green"),
            )
        else:
            _append_line(content, _a3_bullet(f"{token} = {model_spec}"))

    _append_line(content)
    _append_line(content, _a3_section("Manage references:"))
    _append_line(content, _a3_bullet("/model references set", style="cyan"))
    _append_line(content, _a3_bullet("/model references set <token>", style="cyan"))
    _append_line(content, _a3_bullet("/model references unset", style="cyan"))

    outcome.add_message(content, right_info="model")
    return outcome


def _parse_catalog_arguments(argument: str | None) -> tuple[str | None, bool, str | None]:
    if not argument:
        return None, False, "Usage: /model catalog <provider> [--all]"

    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        return None, False, f"Invalid catalog arguments: {exc}"

    provider_name: str | None = None
    show_all = False

    for token in tokens:
        if token == "--all":
            show_all = True
            continue
        if token.startswith("--"):
            return None, False, f"Unknown option: {token}"
        if provider_name is not None:
            return None, False, "Only one provider may be specified."
        provider_name = token

    if not provider_name:
        return None, False, "Usage: /model catalog <provider> [--all]"

    return provider_name, show_all, None


async def _handle_models_catalog(ctx: "CommandContext", *, argument: str | None) -> CommandOutcome:
    outcome = CommandOutcome()
    provider_name, show_all, parse_error = _parse_catalog_arguments(argument)
    if parse_error is not None:
        outcome.add_message(
            _a3_error_block("model catalog", parse_error),
            channel="error",
            right_info="model",
        )
        return outcome

    assert provider_name is not None
    provider = _resolve_catalog_provider(provider_name)
    if provider is None:
        outcome.add_message(
            _a3_error_block(
                "model catalog",
                f"Unknown provider '{provider_name}'. Choose one of: {_provider_display_choices()}.",
            ),
            channel="error",
            right_info="model",
        )
        return outcome

    settings = ctx.resolve_settings()
    config_payload = _resolve_config_payload(settings)

    curated_entries = ModelSelectionCatalog.list_current_entries(provider)

    content = Text()
    _append_line(content, _a3_header("model catalog"))
    _append_line(content)
    _append_line(
        content,
        _a3_status_line("Provider", provider.display_name, value_style="bold cyan"),
    )
    _append_line(content)
    _append_line(content, _a3_section("Curated models:"))
    if curated_entries:
        for entry in curated_entries:
            fast_tag = " [fast]" if entry.fast else ""
            preset_token = entry.alias or "-"
            style = "green" if entry.fast else "white"
            _append_line(
                content,
                _a3_bullet(f"{preset_token} -> {entry.model}{fast_tag}", style=style),
            )
    else:
        _append_line(content, _a3_bullet("none", style="dim"))

    if show_all:
        curated_models = {entry.model for entry in curated_entries}
        all_models = ModelSelectionCatalog.list_all_models(provider, config=config_payload)
        _append_line(content)
        _append_line(content, _a3_section("All known models:"))
        if not all_models:
            _append_line(content, _a3_bullet("none", style="dim"))
        else:
            for model in all_models:
                tags: list[str] = []
                if model in curated_models:
                    tags.append("catalog")
                if ModelSelectionCatalog.is_fast_model(model):
                    tags.append("fast")
                suffix = f" [{', '.join(tags)}]" if tags else ""
                style = "green" if "fast" in tags else "white"
                _append_line(content, _a3_bullet(f"{model}{suffix}", style=style))

    outcome.add_message(content, right_info="model")
    return outcome
