"""Shared /model management command handlers."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, cast

from rich.text import Text

from fast_agent.commands.command_catalog import (
    format_unknown_command_action,
    get_command_action_spec,
    get_command_spec,
    normalize_command_action,
)
from fast_agent.commands.option_parsing import ValueOption, is_long_option_token, read_value_option
from fast_agent.commands.results import CommandChannel, CommandMessage, CommandOutcome
from fast_agent.commands.summary_utils import optional_string
from fast_agent.constants import DEFAULT_HOME_DIR
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import parse_model_reference_token, resolve_model_reference
from fast_agent.interfaces import LlmCapableProtocol
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_reference_config import (
    ModelReferenceConfigService,
    ModelReferenceMutationResult,
    ModelReferenceWriteTarget,
    resolve_model_reference_start_path,
)
from fast_agent.llm.model_selection import CatalogModelEntry, ModelSelectionCatalog
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.a3_headers import build_a3_section_header
from fast_agent.ui.model_picker_common import infer_initial_picker_provider
from fast_agent.utils.action_normalization import is_help_flag, normalize_action_token
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.markdown import escape_markdown_table_cell, markdown_code_span
from fast_agent.utils.name_normalization import normalize_provider_key
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.commands.context import CommandContext
    from fast_agent.config import Settings
    from fast_agent.interfaces import AgentProtocol

_PROVIDER_NAME_ALIASES: dict[str, str] = {
    "hf": "huggingface",
    "codex-responses": "codexresponses",
    "codex_responses": "codexresponses",
}

_NO_MODEL_REFERENCES_NOTE = (
    "No model_references are configured. Add a model_references section in fast-agent.yaml."
)
_REFERENCES_USAGE = (
    "Usage: /model references "
    "[list|set [<token> [<model-spec>]] [--target env|project] [--dry-run]|"
    "unset [<token>] [--target env|project] [--dry-run]]"
)
_MODELS_USAGE = "Usage: /model [doctor|references|catalog|help] [args]"
_REFERENCE_WRITE_TARGETS: dict[str, ModelReferenceWriteTarget] = {
    "env": "env",
    "project": "project",
}
_REFERENCE_TARGET_OPTIONS: tuple[ValueOption[Literal["target"]], ...] = (
    ValueOption("target", ("--target",), error_name="--target (expected env or project)"),
)

ModelsActionHandler = Callable[["CommandContext", str | None], Awaitable[CommandOutcome]]
ReferenceMutationOperation = Literal["set", "unset"]


def _model_action_usage(action: str, *, command_name: str = "model") -> str:
    action_spec = get_command_action_spec(command_name, action)
    usage = action_spec.usage if action_spec is not None else None
    return f"Usage: {usage}" if usage else _models_command_usage(command_name)


def _models_command_usage(command_name: str) -> str:
    command_spec = get_command_spec(command_name)
    if command_spec is None:
        return _MODELS_USAGE
    return f"Usage: {command_spec.usage}"


def _models_command_examples(command_name: str) -> str:
    command_spec = get_command_spec(command_name)
    if command_spec is None or not command_spec.examples:
        return "Examples: /model doctor, /model references, /model catalog openai"
    return f"Examples: {', '.join(command_spec.examples)}"


@dataclass(frozen=True)
class _ReferencesMutationArgs:
    operation: ReferenceMutationOperation
    token: str | None
    model_spec: str | None
    target: ModelReferenceWriteTarget
    dry_run: bool


@dataclass(frozen=True)
class _ReferencesArguments:
    mode: Literal["list", "mutate"] = "list"
    mutation: _ReferencesMutationArgs | None = None
    error: str | None = None


@dataclass(frozen=True)
class _SplitReferencesArgument:
    tokens: list[str]
    error: str | None = None


@dataclass(frozen=True)
class _ResolvedReferenceMutationArgs:
    mutation: _ReferencesMutationArgs | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ReferenceWriteTarget:
    target: ModelReferenceWriteTarget = "env"
    error: str | None = None


@dataclass(frozen=True)
class _ReferencesParseState:
    target: ModelReferenceWriteTarget = "env"
    dry_run: bool = False
    positional: tuple[str, ...] = ()
    error: str | None = None


ReferenceArgumentBuilder = Callable[[_ReferencesParseState], _ReferencesArguments]


@dataclass(frozen=True)
class _CatalogArguments:
    provider_name: str | None = None
    show_all: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _CatalogParseState:
    provider_name: str | None = None
    show_all: bool = False
    error: str | None = None


@dataclass(frozen=True)
class _CatalogProviderAssignment:
    provider_name: str | None
    error: str | None = None


class _AgentModelStatus(Enum):
    RESOLVED = ("✓", "resolved", "Resolved")
    ATTENTION = ("◐", "fallback/override", "Fallback")
    UNRESOLVED = ("✗", "unresolved", "Unresolved")
    UNKNOWN = ("…", "unknown", "Unknown")

    def __init__(self, symbol: str, table_label: str, markdown_label: str) -> None:
        self.symbol = symbol
        self.table_label = table_label
        self.markdown_label = markdown_label

    @property
    def table_status(self) -> str:
        return f"{self.symbol} {self.table_label}"

    @property
    def markdown_status(self) -> str:
        return f"{self.symbol} {self.markdown_label}"


@dataclass(frozen=True)
class _AgentModelDoctorRow:
    name: str
    specified_model: str
    resolved_model: str
    status: _AgentModelStatus
    status_style: str
    resolution_note: str | None = None


@dataclass(frozen=True)
class _AgentModelSpecResolution:
    resolved_from_spec: str | None = None
    reference_error: str | None = None


@dataclass(frozen=True)
class _AgentModelRuntimeResolution:
    resolved_model: str
    status: _AgentModelStatus
    status_style: str
    resolution_note: str | None = None


@dataclass(frozen=True)
class _ModelsDoctorReport:
    readiness_ready: bool
    home_env: str | None
    effective_home: object
    fast_agent_model_env: str | None
    loaded_config_file: object
    unresolved: list[tuple[str, str, str]]
    configured_providers: set[Provider]
    agent_rows: list[_AgentModelDoctorRow]
    default_provider: Provider | None
    default_provider_ready: bool


@dataclass(frozen=True)
class _AgentModelNoteGroups:
    single_notes: tuple[tuple[_AgentModelDoctorRow, str], ...]
    repeated_notes: tuple[str, ...]


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


def _all_agent_names(ctx: "CommandContext") -> list[str]:
    return sorted(str(name) for name in ctx.agent_provider.registered_agent_names())


def _canonical_model_name(model_spec: str) -> str:
    normalized = strip_to_none(model_spec)
    if normalized is None:
        return ""

    with suppress(Exception):
        parsed = ModelFactory.parse_model_string(
            normalized,
            presets=ModelFactory.MODEL_PRESETS,
        )
        return parsed.model_name

    return normalized


def _models_equivalent(expected: str, runtime: str) -> bool:
    if expected == runtime:
        return True

    with suppress(Exception):
        expected_normalized = ModelDatabase.normalize_model_name(expected)
        runtime_normalized = ModelDatabase.normalize_model_name(runtime)
        if expected_normalized and runtime_normalized and expected_normalized == runtime_normalized:
            return True

    return _canonical_model_name(expected) == _canonical_model_name(runtime)


def _resolve_agent_model_spec(
    model_spec: str | None,
    references: dict[str, dict[str, str]],
) -> _AgentModelSpecResolution:
    if not model_spec:
        return _AgentModelSpecResolution()
    if not model_spec.startswith("$"):
        return _AgentModelSpecResolution(resolved_from_spec=model_spec)

    try:
        return _AgentModelSpecResolution(
            resolved_from_spec=resolve_model_reference(model_spec, references)
        )
    except ModelConfigError as exc:
        return _AgentModelSpecResolution(reference_error=exc.details)


def _agent_runtime_reference_error_resolution(
    *,
    reference_error: str,
    llm_model: str | None,
) -> _AgentModelRuntimeResolution:
    if llm_model:
        return _AgentModelRuntimeResolution(
            resolved_model=llm_model,
            status=_AgentModelStatus.ATTENTION,
            status_style="yellow",
            resolution_note=reference_error,
        )
    return _AgentModelRuntimeResolution(
        resolved_model="<unresolved>",
        status=_AgentModelStatus.UNRESOLVED,
        status_style="red",
        resolution_note=reference_error,
    )


def _agent_runtime_spec_resolution(
    *,
    resolved_from_spec: str,
    llm_model: str | None,
) -> _AgentModelRuntimeResolution:
    if llm_model and _models_equivalent(resolved_from_spec, llm_model):
        return _AgentModelRuntimeResolution(
            resolved_model=llm_model,
            status=_AgentModelStatus.RESOLVED,
            status_style="green",
        )
    if llm_model:
        return _AgentModelRuntimeResolution(
            resolved_model=llm_model,
            status=_AgentModelStatus.ATTENTION,
            status_style="cyan",
            resolution_note=(
                f"Resolved spec suggests '{resolved_from_spec}' but runtime uses '{llm_model}'."
            ),
        )
    return _AgentModelRuntimeResolution(
        resolved_model=resolved_from_spec,
        status=_AgentModelStatus.RESOLVED,
        status_style="green",
    )


def _resolve_agent_runtime_model(
    *,
    spec_resolution: _AgentModelSpecResolution,
    llm_model: str | None,
) -> _AgentModelRuntimeResolution:
    reference_error = spec_resolution.reference_error
    resolved_from_spec = spec_resolution.resolved_from_spec

    if reference_error:
        return _agent_runtime_reference_error_resolution(
            reference_error=reference_error,
            llm_model=llm_model,
        )
    if resolved_from_spec:
        return _agent_runtime_spec_resolution(
            resolved_from_spec=resolved_from_spec,
            llm_model=llm_model,
        )
    if llm_model:
        return _AgentModelRuntimeResolution(
            resolved_model=llm_model,
            status=_AgentModelStatus.RESOLVED,
            status_style="cyan",
        )
    return _AgentModelRuntimeResolution(
        resolved_model="<unknown>",
        status=_AgentModelStatus.UNKNOWN,
        status_style="dim",
    )


def _agent_llm_model(agent: AgentProtocol) -> str | None:
    if not isinstance(agent, LlmCapableProtocol):
        return None
    llm = agent.llm
    return optional_string(llm.model_name) if llm is not None else None


def _build_agent_model_row(
    agent_name: str,
    agent: AgentProtocol,
    *,
    references: dict[str, dict[str, str]],
    default_model: str | None,
) -> _AgentModelDoctorRow:
    specified = optional_string(agent.config.model)
    effective_spec = specified or optional_string(default_model)
    spec_resolution = _resolve_agent_model_spec(effective_spec, references)
    runtime_resolution = _resolve_agent_runtime_model(
        spec_resolution=spec_resolution,
        llm_model=_agent_llm_model(agent),
    )
    return _AgentModelDoctorRow(
        name=agent_name,
        specified_model=specified or "<default>",
        resolved_model=runtime_resolution.resolved_model,
        status=runtime_resolution.status,
        status_style=runtime_resolution.status_style,
        resolution_note=runtime_resolution.resolution_note,
    )


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

        agent = cast("AgentProtocol", agent)
        rows.append(
            _build_agent_model_row(
                agent_name,
                agent,
                references=references,
                default_model=default_model,
            )
        )

    return rows


def _truncate_cell(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


def _agent_model_table_values(rows: list[_AgentModelDoctorRow]) -> list[list[str]]:
    return [
        [
            row.name,
            row.specified_model,
            row.resolved_model,
            row.status.table_status,
        ]
        for row in rows
    ]


def _agent_model_table_widths(
    table_values: list[list[str]],
    *,
    headers: list[str],
    max_limits: list[int],
) -> list[int]:
    widths = [len(header) for header in headers]
    for row_values in table_values:
        for index, value in enumerate(row_values):
            widths[index] = min(max(widths[index], len(value)), max_limits[index])
    return widths


def _build_agent_model_table_row(
    values: list[tuple[str, str]],
    *,
    widths: list[int],
    indent: str = "  ",
) -> Text:
    line = Text(indent, style="dim")
    for index, (value, style) in enumerate(values):
        cell = _truncate_cell(value, limit=widths[index]).ljust(widths[index])
        line.append(cell, style=style)
        if index < len(values) - 1:
            line.append("  ", style="dim")
    return line


def _agent_model_note_groups(rows: list[_AgentModelDoctorRow]) -> _AgentModelNoteGroups:
    note_counts = Counter(note for note in (row.resolution_note for row in rows) if note)
    single_notes: list[tuple[_AgentModelDoctorRow, str]] = []
    repeated_notes: list[str] = []
    repeated_seen: set[str] = set()

    for row in rows:
        note = row.resolution_note
        if not note:
            continue
        if note_counts[note] <= 1:
            single_notes.append((row, note))
        elif note not in repeated_seen:
            repeated_seen.add(note)
            repeated_notes.append(note)

    return _AgentModelNoteGroups(
        single_notes=tuple(single_notes),
        repeated_notes=tuple(repeated_notes),
    )


def _render_agent_model_table(rows: list[_AgentModelDoctorRow]) -> Text:
    content = Text()
    _append_line(content, _a3_section("Agent model resolution:"))

    if not rows:
        _append_line(content, _a3_bullet("No agents are currently registered.", style="dim"))
        return content

    headers = ["Agent", "Specified", "Resolved", "Resolution"]
    max_limits = [24, 34, 34, 20]
    table_values = _agent_model_table_values(rows)
    widths = _agent_model_table_widths(
        table_values,
        headers=headers,
        max_limits=max_limits,
    )

    _append_line(
        content,
        _build_agent_model_table_row(
            [
                (headers[0], "bold bright_white"),
                (headers[1], "bold bright_white"),
                (headers[2], "bold bright_white"),
                (headers[3], "bold bright_white"),
            ],
            widths=widths,
        ),
    )

    note_groups = _agent_model_note_groups(rows)
    single_notes = dict(note_groups.single_notes)

    for row, values in zip(rows, table_values, strict=False):
        resolved_style = "red" if row.status is _AgentModelStatus.UNRESOLVED else "green"
        _append_line(
            content,
            _build_agent_model_table_row(
                [
                    (values[0], "cyan"),
                    (values[1], "white"),
                    (values[2], resolved_style),
                    (values[3], row.status_style),
                ],
                widths=widths,
            ),
        )
        if note := single_notes.get(row):
            _append_line(content, Text(f"    note: {note}", style="dim"))

    if note_groups.repeated_notes:
        _append_line(content)
        _append_line(content, _a3_section("Notes:"))
        for note in note_groups.repeated_notes:
            _append_line(content, Text(f"  • {note}", style="dim"))

    return content


def _render_agent_model_summary(rows: list[_AgentModelDoctorRow]) -> Text:
    total = len(rows)
    status_counts = Counter(row.status for row in rows)
    resolved = status_counts[_AgentModelStatus.RESOLVED]
    attention = status_counts[_AgentModelStatus.ATTENTION]
    unresolved = status_counts[_AgentModelStatus.UNRESOLVED]
    unknown = status_counts[_AgentModelStatus.UNKNOWN]

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
    return sorted(providers, key=lambda provider: strip_casefold(provider.display_name))


def _resolve_catalog_provider(name: str) -> Provider | None:
    alias_key = normalize_action_token(name)
    normalized = normalize_provider_key(_PROVIDER_NAME_ALIASES.get(alias_key, name))
    for provider in _catalog_providers():
        variants = {
            normalize_provider_key(provider.config_name),
            normalize_provider_key(provider.name),
            normalize_provider_key(provider.display_name),
        }
        if normalized in variants:
            return provider
    return None


def _provider_display_choices() -> str:
    return ", ".join(provider.config_name for provider in _catalog_providers())


def _resolve_reference_service(ctx: "CommandContext") -> ModelReferenceConfigService:
    settings = ctx.resolve_settings()
    home = settings.home
    start_path = resolve_model_reference_start_path(settings=settings)
    if home is None and settings._config_file:
        home = start_path / DEFAULT_HOME_DIR
    return ModelReferenceConfigService(start_path=start_path, home=home)


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

    token = strip_to_none(default_model)
    if token is not None and token.startswith("$"):
        try:
            resolve_model_reference(token, references)
        except ModelConfigError as exc:
            unresolved[(token, "default_model")] = exc.details

    return [
        (token, source, details)
        for (token, source), details in sorted(unresolved.items(), key=lambda item: item[0])
    ]


def _resolve_config_payload(settings: "Settings") -> dict[str, object]:
    with suppress(Exception):
        dumped = settings.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {}


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

    with suppress(Exception):
        parsed = ModelFactory.parse_model_string(
            resolved_model,
            presets=ModelFactory.MODEL_PRESETS,
        )
        provider = parsed.provider
        return provider if isinstance(provider, Provider) else None
    return None


def _provider_is_ready(provider: Provider, configured: set[Provider]) -> bool:
    if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
        return True
    return provider in configured


def _build_models_doctor_report(ctx: "CommandContext") -> _ModelsDoctorReport:
    settings = ctx.resolve_settings()
    home_env = os.getenv("FAST_AGENT_HOME")
    fast_agent_model_env = os.getenv("FAST_AGENT_MODEL")
    effective_home = settings.home
    loaded_config_file = settings._config_file

    service = _resolve_reference_service(ctx)
    references = service.list_references_tolerant()
    config_payload = _resolve_config_payload(settings)
    configured_providers = set(ModelSelectionCatalog.configured_providers(config_payload))
    default_model = settings.default_model
    unresolved = _collect_unresolved_references(references, default_model=default_model)
    agent_rows = _build_agent_model_rows(
        ctx,
        references=references,
        default_model=optional_string(default_model),
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
        home_env=home_env,
        effective_home=effective_home,
        fast_agent_model_env=fast_agent_model_env,
        loaded_config_file=loaded_config_file,
        unresolved=unresolved,
        configured_providers=configured_providers,
        agent_rows=agent_rows,
        default_provider=default_provider,
        default_provider_ready=default_provider_ready,
    )


def _status_label_for_row(row: _AgentModelDoctorRow) -> str:
    return row.status.markdown_status


def _markdown_code_or_dash(value: str) -> str:
    normalized = optional_string(value)
    if not normalized or normalized in {"<default>"}:
        return "—"
    return _markdown_table_code_span(normalized)


def _markdown_code_or_placeholder(value: object | None, placeholder: str) -> str:
    return markdown_code_span(str(value) if value is not None else placeholder)


def _markdown_table_code_span(value: str) -> str:
    return markdown_code_span(escape_markdown_table_cell(value))


def _extend_markdown_doctor_overview(lines: list[str], report: _ModelsDoctorReport) -> None:
    lines.append("## Readiness")
    lines.append(f"- **Status**: {'ready' if report.readiness_ready else 'action required'}")

    lines.extend(["", "## Runtime config context"])
    lines.append(
        f"- **FAST_AGENT_HOME**: {_markdown_code_or_placeholder(report.home_env, '<unset>')}"
    )
    lines.append(
        "- **Effective home**: "
        f"{_markdown_code_or_placeholder(report.effective_home, '<unset>')}"
    )
    lines.append(
        f"- **FAST_AGENT_MODEL**: "
        f"{_markdown_code_or_placeholder(report.fast_agent_model_env, '<unset>')}"
    )
    lines.append(
        f"- **Loaded config file**: "
        f"{_markdown_code_or_placeholder(report.loaded_config_file, '<none>')}"
    )


def _extend_markdown_unresolved_references(
    lines: list[str],
    unresolved: list[tuple[str, str, str]],
) -> None:
    lines.extend(["", "## Unresolved references"])
    if not unresolved:
        lines.append("- none")
        return

    for token, source, details in unresolved:
        lines.append(f"- {markdown_code_span(token)} ({source})")
        if details:
            lines.append(f"  - {details}")


def _extend_markdown_provider_readiness(
    lines: list[str],
    configured_providers: set[Provider],
) -> None:
    lines.extend(["", "## Provider readiness"])
    for provider in _catalog_providers():
        state = "configured" if provider in configured_providers else "not configured"
        lines.append(f"- **{provider.display_name}**: {state}")


def _agent_model_markdown_row(row: _AgentModelDoctorRow) -> str:
    return (
        "| "
        + " | ".join(
            [
                _markdown_table_code_span(row.name),
                _markdown_code_or_dash(row.specified_model),
                _markdown_code_or_dash(row.resolved_model),
                _status_label_for_row(row),
            ]
        )
        + " |"
    )


def _agent_model_markdown_notes(rows: list[_AgentModelDoctorRow]) -> list[str]:
    note_groups = _agent_model_note_groups(rows)
    single_notes = [
        f"{markdown_code_span(row.name)}: {note}" for row, note in note_groups.single_notes
    ]
    return [*single_notes, *note_groups.repeated_notes]


def _extend_markdown_agent_summary(
    lines: list[str],
    rows: list[_AgentModelDoctorRow],
) -> None:
    status_counts = Counter(row.status for row in rows)
    lines.extend(
        [
            "",
            "## Agent summary",
            f"- **Total**: {len(rows)}",
            f"- **Resolved**: {status_counts[_AgentModelStatus.RESOLVED]}",
            f"- **Attention**: {status_counts[_AgentModelStatus.ATTENTION]}",
            f"- **Unresolved**: {status_counts[_AgentModelStatus.UNRESOLVED]}",
        ]
    )

    notes = _agent_model_markdown_notes(rows)
    if notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {note}" for note in notes)


def _extend_markdown_agent_resolution(
    lines: list[str],
    rows: list[_AgentModelDoctorRow],
) -> None:
    lines.extend(["", "## Agent model resolution", ""])
    if not rows:
        lines.append("_No agents are currently registered._")
        return

    lines.extend(
        [
            "| Agent | Specified | Resolved | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(_agent_model_markdown_row(row) for row in rows)
    _extend_markdown_agent_summary(lines, rows)


def _extend_markdown_doctor_followups(lines: list[str], report: _ModelsDoctorReport) -> None:
    if report.default_provider is not None and not report.default_provider_ready:
        lines.append("")
        lines.append(
            "- Default model provider "
            f"{markdown_code_span(report.default_provider.display_name)} "
            "is not configured for current settings."
        )

    if not report.configured_providers:
        lines.append("")
        lines.append("- No provider credentials detected.")

    if report.readiness_ready:
        return

    lines.extend(["", "## Next steps"])
    if report.unresolved:
        lines.append("- `/model references`")
    lines.append("- `/model catalog <provider>`")


def render_models_doctor_markdown(ctx: "CommandContext") -> str:
    """Render `/model doctor` as markdown, optimized for ACP clients."""
    try:
        report = _build_models_doctor_report(ctx)
    except Exception as exc:
        return f"# model.doctor\n\n**Error:** Failed to load model references: {exc}"

    lines: list[str] = ["# model.doctor", ""]
    _extend_markdown_doctor_overview(lines, report)
    _extend_markdown_unresolved_references(lines, report.unresolved)
    _extend_markdown_provider_readiness(lines, report.configured_providers)
    _extend_markdown_agent_resolution(lines, report.agent_rows)
    _extend_markdown_doctor_followups(lines, report)

    return "\n".join(lines)


async def handle_models_command(
    ctx: "CommandContext",
    *,
    agent_name: str,
    action: str,
    argument: str | None,
    command_name: str = "model",
) -> CommandOutcome:
    del agent_name

    if is_help_flag(action) or is_help_flag(argument):
        outcome = CommandOutcome()
        outcome.add_message(_a3_header(f"{command_name} help"), right_info=command_name)
        outcome.add_message(_models_command_usage(command_name), right_info=command_name)
        outcome.add_message(
            _models_command_examples(command_name),
            right_info=command_name,
        )
        return outcome

    normalized_action = normalize_command_action(command_name, action or "doctor")
    if normalized_action in {"", "list"}:
        normalized_action = "doctor"

    if normalized_action == "catalog":
        return await _handle_models_catalog(ctx, argument, command_name=command_name)

    action_handler = _MODELS_ACTION_HANDLERS.get(normalized_action)
    if action_handler is not None:
        return await action_handler(ctx, argument)

    outcome = CommandOutcome()
    outcome.add_message(
        _a3_error_block(
            command_name,
            format_unknown_command_action(command_name, normalized_action),
        ),
        channel="error",
        right_info=command_name,
    )
    return outcome


def _append_doctor_readiness(content: Text, *, ready: bool) -> None:
    if ready:
        _append_line(content, _a3_status_line("Readiness", "ready", value_style="bold green"))
        return
    _append_line(
        content,
        _a3_status_line("Readiness", "action required", value_style="bold yellow"),
    )


def _append_doctor_runtime_context(content: Text, report: _ModelsDoctorReport) -> None:
    _append_line(content)
    _append_line(content, _a3_section("Runtime config context:"))
    _append_line(
        content,
        _a3_bullet(f"FAST_AGENT_HOME: {report.home_env or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(
            f"Effective home: {report.effective_home or '<unset>'}",
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


def _append_doctor_unresolved_references(
    content: Text, unresolved: list[tuple[str, str, str]]
) -> None:
    _append_line(content)
    _append_line(content, _a3_section("Unresolved references:"))
    if not unresolved:
        _append_line(content, _a3_bullet("none", style="dim"))
        return

    for token, source, details in unresolved:
        _append_line(content, _a3_bullet(f"{token} ({source})", style="yellow"))
        if details:
            _append_line(content, Text(f"  {details}", style="dim"))


def _append_doctor_provider_readiness(content: Text, configured_providers: set[Provider]) -> None:
    _append_line(content)
    _append_line(content, _a3_section("Provider readiness:"))
    for provider in _catalog_providers():
        state = "configured" if provider in configured_providers else "not configured"
        state_style = "green" if state == "configured" else "dim"
        _append_line(
            content,
            _a3_bullet(f"{provider.display_name}: {state}", style=state_style),
        )


def _append_doctor_followups(content: Text, report: _ModelsDoctorReport) -> None:
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

    if report.readiness_ready:
        return

    _append_line(content)
    _append_line(content, _a3_section("Next steps:"))
    if report.unresolved:
        _append_line(content, _a3_bullet("/model references", style="cyan"))
    _append_line(content, _a3_bullet("/model catalog <provider>", style="cyan"))


async def _handle_models_doctor(
    ctx: "CommandContext", argument: str | None = None
) -> CommandOutcome:
    del argument
    outcome = CommandOutcome()
    try:
        report = _build_models_doctor_report(ctx)
    except Exception as exc:
        outcome.add_message(
            _a3_error_block("model doctor", f"Failed to load model references: {exc}"),
            channel="error",
            right_info="model",
        )
        return outcome

    content = Text()
    _append_line(content, _a3_header("model doctor"))
    _append_line(content)
    _append_doctor_readiness(content, ready=report.readiness_ready)
    _append_doctor_runtime_context(content, report)
    _append_doctor_unresolved_references(content, report.unresolved)
    _append_doctor_provider_readiness(content, report.configured_providers)

    _append_line(content)
    content.append_text(_render_agent_model_table(report.agent_rows))
    _append_line(content)
    content.append_text(_render_agent_model_summary(report.agent_rows))
    _append_doctor_followups(content, report)

    outcome.add_message(content, right_info="model")
    return outcome


def _split_references_argument(argument: str | None) -> _SplitReferencesArgument:
    try:
        return _SplitReferencesArgument(tokens=split_commandline(argument or "", syntax="posix"))
    except ValueError as exc:
        return _SplitReferencesArgument(tokens=[], error=f"Invalid references arguments: {exc}")


def _parse_references_option_state(tokens: list[str]) -> _ReferencesParseState:
    target: ModelReferenceWriteTarget = "env"
    target_seen = False
    dry_run = False
    positional: list[str] = []

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--dry-run":
            dry_run = True
            index += 1
            continue

        parsed_target_option = read_value_option(
            tokens,
            index,
            _REFERENCE_TARGET_OPTIONS,
        )
        if parsed_target_option.matched:
            if parsed_target_option.error is not None:
                return _ReferencesParseState(error=parsed_target_option.error)
            if target_seen:
                return _ReferencesParseState(error="Duplicate option: --target")
            parsed_target = _parse_reference_write_target(parsed_target_option.require_value())
            if parsed_target.error is not None:
                return _ReferencesParseState(error=parsed_target.error)
            target = parsed_target.target
            target_seen = True
            index = parsed_target_option.next_index
            continue

        if is_long_option_token(token):
            return _ReferencesParseState(error=f"Unknown option: {token}")

        positional_error = _append_reference_positional(positional, token)
        if positional_error is not None:
            return _ReferencesParseState(error=positional_error)
        index += 1

    return _ReferencesParseState(
        target=target,
        dry_run=dry_run,
        positional=tuple(positional),
    )


def _build_reference_set_arguments(state: _ReferencesParseState) -> _ReferencesArguments:
    if len(state.positional) > 2:
        return _ReferencesArguments(error="Too many positional arguments for references set.")

    return _ReferencesArguments(
        mode="mutate",
        mutation=_ReferencesMutationArgs(
            operation="set",
            token=state.positional[0] if state.positional else None,
            model_spec=state.positional[1] if len(state.positional) > 1 else None,
            target=state.target,
            dry_run=state.dry_run,
        ),
    )


def _build_reference_unset_arguments(state: _ReferencesParseState) -> _ReferencesArguments:
    if len(state.positional) > 1:
        return _ReferencesArguments(error="Too many positional arguments for references unset.")

    return _ReferencesArguments(
        mode="mutate",
        mutation=_ReferencesMutationArgs(
            operation="unset",
            token=state.positional[0] if state.positional else None,
            model_spec=None,
            target=state.target,
            dry_run=state.dry_run,
        ),
    )


_REFERENCE_MUTATION_BUILDERS: dict[str, ReferenceArgumentBuilder] = {
    "set": _build_reference_set_arguments,
    "unset": _build_reference_unset_arguments,
}


def _parse_references_subcommand(tokens: list[str]) -> _ReferencesArguments:
    subcmd = normalize_action_token(tokens[0])
    if subcmd == "list":
        if len(tokens) > 1:
            return _ReferencesArguments(error="Unexpected arguments after 'list'.")
        return _ReferencesArguments()

    mutation_builder = _REFERENCE_MUTATION_BUILDERS.get(subcmd)
    if mutation_builder is None:
        return _ReferencesArguments(error=f"Unknown references action '{tokens[0]}'.")

    state = _parse_references_option_state(tokens)
    if state.error is not None:
        return _ReferencesArguments(error=state.error)

    return mutation_builder(state)


def _parse_references_arguments(argument: str | None) -> _ReferencesArguments:
    split_result = _split_references_argument(argument)
    if split_result.error is not None:
        return _ReferencesArguments(error=split_result.error)

    if not split_result.tokens:
        return _ReferencesArguments()

    return _parse_references_subcommand(split_result.tokens)


def _append_reference_positional(positional: list[str], token: str) -> str | None:
    normalized = strip_to_none(token)
    if normalized is None:
        return "Reference positional arguments cannot be empty."
    positional.append(normalized)
    return None


def _parse_reference_write_target(value: str) -> _ReferenceWriteTarget:
    target_value = normalize_action_token(value)
    target = _REFERENCE_WRITE_TARGETS.get(target_value)
    if target is not None:
        return _ReferenceWriteTarget(target=target)
    return _ReferenceWriteTarget(error="--target must be either 'env' or 'project'.")


def _canonicalize_reference_token(token: str) -> str:
    namespace, key = parse_model_reference_token(token)
    return f"${namespace}.{key}"


def _normalize_interactive_reference_token(token: str) -> str:
    stripped = strip_to_none(token)
    if stripped is None:
        return ""
    if stripped.startswith("$"):
        return stripped
    return f"${stripped}"


def _reference_selection_content(
    *,
    target_path: Path,
    rows: list[tuple[str, str]],
    include_new: bool,
) -> Text:
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
    if include_new:
        _append_line(selection_content, _a3_bullet("new. Create a new reference", style="cyan"))
    return selection_content


def _reference_selection_options(rows: list[tuple[str, str]]) -> dict[str, str]:
    return {str(index): token for index, (token, _) in enumerate(rows, start=1)}


async def _emit_reference_selection(
    ctx: "CommandContext",
    *,
    target_path: Path,
    rows: list[tuple[str, str]],
    include_new: bool,
) -> None:
    await ctx.io.emit(
        CommandMessage(
            text=_reference_selection_content(
                target_path=target_path,
                rows=rows,
                include_new=include_new,
            ),
            right_info="model",
        )
    )


async def _prompt_for_existing_reference_selection(
    ctx: "CommandContext",
    *,
    rows: list[tuple[str, str]],
    target_path: Path,
    prompt: str,
    include_new: bool,
) -> str | None:
    await _emit_reference_selection(
        ctx,
        target_path=target_path,
        rows=rows,
        include_new=include_new,
    )
    option_labels = _reference_selection_options(rows)
    options = [*option_labels.keys(), "new"] if include_new else list(option_labels.keys())
    selection = await ctx.io.prompt_selection(
        prompt,
        options=options,
        allow_cancel=True,
    )
    if selection is None:
        return None

    normalized_selection = normalize_action_token(selection)
    if include_new and normalized_selection == "new":
        return "new"
    return option_labels.get(normalized_selection)


async def _prompt_for_new_reference_token(
    ctx: "CommandContext",
    *,
    default: str | None,
) -> str | None:
    entered = await ctx.io.prompt_text(
        "Reference token ($namespace.key):",
        default=default,
        allow_empty=False,
    )
    if entered is None:
        return None
    try:
        return _canonicalize_reference_token(_normalize_interactive_reference_token(entered))
    except ModelConfigError as exc:
        raise ValueError(exc.details) from exc


async def _prompt_for_set_reference_token(
    ctx: "CommandContext",
    *,
    rows: list[tuple[str, str]],
    target_path: Path,
) -> str | None:
    if rows:
        selection = await _prompt_for_existing_reference_selection(
            ctx,
            rows=rows,
            target_path=target_path,
            prompt="Reference to update (number or 'new'):",
            include_new=True,
        )
        if selection is None:
            return None
        if selection != "new":
            return selection

    prompt_default = "$system.default" if not rows else None
    return await _prompt_for_new_reference_token(ctx, default=prompt_default)


async def _prompt_for_unset_reference_token(
    ctx: "CommandContext",
    *,
    rows: list[tuple[str, str]],
    target_path: Path,
) -> str | None:
    return await _prompt_for_existing_reference_selection(
        ctx,
        rows=rows,
        target_path=target_path,
        prompt="Reference to remove (number):",
        include_new=False,
    )


async def _prompt_for_reference_token(
    ctx: "CommandContext",
    *,
    references: dict[str, dict[str, str]],
    operation: ReferenceMutationOperation,
    target_path: Path,
) -> str | None:
    rows = _flatten_references(references)
    if operation == "unset":
        if not rows:
            return None
        return await _prompt_for_unset_reference_token(
            ctx,
            rows=rows,
            target_path=target_path,
        )
    return await _prompt_for_set_reference_token(
        ctx,
        rows=rows,
        target_path=target_path,
    )


async def _resolve_reference_mutation_args(
    ctx: "CommandContext",
    *,
    service: ModelReferenceConfigService,
    mutation_args: _ReferencesMutationArgs,
) -> _ResolvedReferenceMutationArgs:
    references = service.list_references_tolerant()
    target_path = (
        service.paths.home_path
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
                return _ResolvedReferenceMutationArgs(error="Reference update cancelled.")
        else:
            token = _canonicalize_reference_token(token)
    except ValueError as exc:
        return _ResolvedReferenceMutationArgs(error=str(exc))
    except ModelConfigError as exc:
        return _ResolvedReferenceMutationArgs(error=exc.details)

    if mutation_args.operation == "unset":
        return _ResolvedReferenceMutationArgs(
            mutation=_ReferencesMutationArgs(
                operation="unset",
                token=token,
                model_spec=None,
                target=mutation_args.target,
                dry_run=mutation_args.dry_run,
            ),
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
            initial_provider=infer_initial_picker_provider(current_model),
            default_model=current_model,
        )
        if model_spec is None:
            return _ResolvedReferenceMutationArgs(error="Reference update cancelled.")

    return _ResolvedReferenceMutationArgs(
        mutation=_ReferencesMutationArgs(
            operation="set",
            token=token,
            model_spec=model_spec,
            target=mutation_args.target,
            dry_run=mutation_args.dry_run,
        ),
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


def _reference_argument_error(message: str) -> Text:
    error = Text()
    _append_line(error, _a3_header("model references", color="red"))
    _append_line(error)
    _append_line(error, _a3_bullet(message, style="red"))
    _append_line(error, Text(_REFERENCES_USAGE, style="dim"))
    return error


def _add_reference_resolution_error(
    outcome: CommandOutcome,
    *,
    message: str,
) -> None:
    content = _a3_error_block("model references", message)
    is_cancelled = message.endswith("cancelled.")
    if not is_cancelled:
        _append_line(content, Text(_REFERENCES_USAGE, style="dim"))
    outcome.add_message(
        content,
        channel="warning" if is_cancelled else "error",
        right_info="model",
    )


def _add_reference_mutation_result(
    outcome: CommandOutcome,
    *,
    service: ModelReferenceConfigService,
    mutation_args: _ReferencesMutationArgs,
) -> None:
    token = mutation_args.token
    if token is None:
        outcome.add_message(
            _a3_error_block("model references", "Missing resolved reference token."),
            channel="error",
            right_info="model",
        )
        return

    if mutation_args.operation == "set":
        model_spec = mutation_args.model_spec
        if model_spec is None:
            outcome.add_message(
                _a3_error_block("model references", "Missing resolved model spec."),
                channel="error",
                right_info="model",
            )
            return
        mutation_result = service.set_reference(
            token,
            model_spec,
            target=mutation_args.target,
            dry_run=mutation_args.dry_run,
        )
        title = "model references set"
    else:
        mutation_result = service.unset_reference(
            token,
            target=mutation_args.target,
            dry_run=mutation_args.dry_run,
        )
        title = "model references unset"

    outcome.add_message(
        _render_reference_mutation(title=title, result=mutation_result),
        right_info="model",
    )


async def _handle_reference_mutation(
    ctx: "CommandContext",
    outcome: CommandOutcome,
    *,
    service: ModelReferenceConfigService,
    mutation_args: _ReferencesMutationArgs | None,
) -> CommandOutcome:
    if mutation_args is None:
        outcome.add_message(
            _a3_error_block("model references", "Missing reference mutation arguments."),
            channel="error",
            right_info="model",
        )
        return outcome

    resolved = await _resolve_reference_mutation_args(
        ctx,
        service=service,
        mutation_args=mutation_args,
    )
    if resolved.error is not None:
        _add_reference_resolution_error(outcome, message=resolved.error)
        return outcome

    if resolved.mutation is None:
        outcome.add_message(
            _a3_error_block("model references", "Missing resolved reference token."),
            channel="error",
            right_info="model",
        )
        return outcome

    _add_reference_mutation_result(
        outcome,
        service=service,
        mutation_args=resolved.mutation,
    )
    return outcome


def _render_reference_list(references: dict[str, dict[str, str]]) -> tuple[Text, CommandChannel]:
    rows = _flatten_references(references)
    if not rows:
        empty = Text()
        _append_line(empty, _a3_header("model references"))
        _append_line(empty)
        _append_line(empty, _a3_bullet("No model references configured.", style="yellow"))
        return empty, "warning"

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
    return content, "system"


async def _handle_models_references(ctx: "CommandContext", argument: str | None) -> CommandOutcome:
    outcome = CommandOutcome()

    parsed = _parse_references_arguments(argument)
    if parsed.error is not None:
        outcome.add_message(
            _reference_argument_error(parsed.error),
            channel="error",
            right_info="model",
        )
        return outcome

    try:
        service = _resolve_reference_service(ctx)
        if parsed.mode == "mutate":
            return await _handle_reference_mutation(
                ctx,
                outcome,
                service=service,
                mutation_args=parsed.mutation,
            )

        references = service.list_references()
    except ValueError as exc:
        outcome.add_message(
            _reference_argument_error(str(exc)),
            channel="error",
            right_info="model",
        )
        return outcome
    except Exception as exc:
        outcome.add_message(
            _a3_error_block("model references", f"Failed to load model references: {exc}"),
            channel="error",
            right_info="model",
        )
        return outcome

    content, channel = _render_reference_list(references)
    outcome.add_message(content, channel=channel, right_info="model")
    return outcome


def _parse_catalog_arguments(
    argument: str | None,
    *,
    command_name: str = "model",
) -> _CatalogArguments:
    usage = _model_action_usage("catalog", command_name=command_name)
    argument_text = strip_to_none(argument)
    if argument_text is None:
        return _CatalogArguments(error=usage)

    try:
        tokens = split_commandline(argument_text, syntax="posix")
    except ValueError as exc:
        return _CatalogArguments(error=f"Invalid catalog arguments: {exc}")

    state = _CatalogParseState()

    for token in tokens:
        state = _parse_catalog_token(state, token, usage=usage)
        if state.error is not None:
            return _CatalogArguments(error=state.error)

    if not state.provider_name:
        return _CatalogArguments(error=usage)

    return _CatalogArguments(provider_name=state.provider_name, show_all=state.show_all)


def _parse_catalog_token(
    state: _CatalogParseState,
    token: str,
    *,
    usage: str,
) -> _CatalogParseState:
    if token == "--all":
        return _CatalogParseState(provider_name=state.provider_name, show_all=True)
    if is_long_option_token(token):
        return _CatalogParseState(
            provider_name=state.provider_name,
            show_all=state.show_all,
            error=f"Unknown option: {token}",
        )

    assignment = _assign_catalog_provider(state.provider_name, token, usage=usage)
    return _CatalogParseState(
        provider_name=assignment.provider_name,
        show_all=state.show_all,
        error=assignment.error,
    )


def _assign_catalog_provider(
    current: str | None,
    token: str,
    *,
    usage: str,
) -> _CatalogProviderAssignment:
    if not token:
        return _CatalogProviderAssignment(provider_name=current, error=usage)
    if current is not None:
        return _CatalogProviderAssignment(
            provider_name=current,
            error="Only one provider may be specified.",
        )
    return _CatalogProviderAssignment(provider_name=token)


def _append_catalog_curated_models(
    content: Text,
    curated_entries: Sequence[CatalogModelEntry],
) -> None:
    _append_line(content)
    _append_line(content, _a3_section("Curated models:"))
    if not curated_entries:
        _append_line(content, _a3_bullet("none", style="dim"))
        return

    for entry in curated_entries:
        fast_tag = " [fast]" if entry.fast else ""
        preset_token = entry.alias or "-"
        style = "green" if entry.fast else "white"
        _append_line(
            content,
            _a3_bullet(f"{preset_token} -> {entry.model}{fast_tag}", style=style),
        )


def _catalog_model_suffix(model: str, curated_models: set[str]) -> tuple[str, str]:
    tags: list[str] = []
    if model in curated_models:
        tags.append("catalog")
    if ModelSelectionCatalog.is_fast_model(model):
        tags.append("fast")
    suffix = f" [{', '.join(tags)}]" if tags else ""
    style = "green" if "fast" in tags else "white"
    return suffix, style


def _append_catalog_all_models(
    content: Text,
    *,
    provider: Provider,
    config_payload: dict[str, object],
    curated_entries: Sequence[CatalogModelEntry],
) -> None:
    curated_models = {entry.model for entry in curated_entries}
    all_models = ModelSelectionCatalog.list_all_models(provider, config=config_payload)
    _append_line(content)
    _append_line(content, _a3_section("All known models:"))
    if not all_models:
        _append_line(content, _a3_bullet("none", style="dim"))
        return

    for model in all_models:
        suffix, style = _catalog_model_suffix(model, curated_models)
        _append_line(content, _a3_bullet(f"{model}{suffix}", style=style))


def _render_catalog_content(
    *,
    provider: Provider,
    config_payload: dict[str, object],
    show_all: bool,
) -> Text:
    curated_entries = ModelSelectionCatalog.list_current_entries(provider)
    content = Text()
    _append_line(content, _a3_header("model catalog"))
    _append_line(content)
    _append_line(
        content,
        _a3_status_line("Provider", provider.display_name, value_style="bold cyan"),
    )
    _append_catalog_curated_models(content, curated_entries)
    if show_all:
        _append_catalog_all_models(
            content,
            provider=provider,
            config_payload=config_payload,
            curated_entries=curated_entries,
        )
    return content


async def _handle_models_catalog(
    ctx: "CommandContext",
    argument: str | None,
    *,
    command_name: str = "model",
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = _parse_catalog_arguments(argument, command_name=command_name)
    if parsed.error is not None:
        outcome.add_message(
            _a3_error_block("model catalog", parsed.error),
            channel="error",
            right_info=command_name,
        )
        return outcome

    provider_name = parsed.provider_name
    if provider_name is None:
        outcome.add_message(
            _a3_error_block(
                "model catalog",
                _model_action_usage("catalog", command_name=command_name),
            ),
            channel="error",
            right_info=command_name,
        )
        return outcome

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
    content = _render_catalog_content(
        provider=provider,
        config_payload=config_payload,
        show_all=parsed.show_all,
    )
    outcome.add_message(content, right_info="model")
    return outcome


async def _handle_models_help(ctx: "CommandContext", argument: str | None = None) -> CommandOutcome:
    del ctx, argument
    outcome = CommandOutcome()
    outcome.add_message(_MODELS_USAGE, right_info="model")
    return outcome


_MODELS_ACTION_HANDLERS: dict[str, ModelsActionHandler] = {
    "doctor": _handle_models_doctor,
    "references": _handle_models_references,
    "catalog": _handle_models_catalog,
    "help": _handle_models_help,
}


def is_model_manager_action(action: str) -> bool:
    """Return whether action is handled by the shared model manager."""
    return normalize_action_token(action) in _MODELS_ACTION_HANDLERS
