"""Shared /models command handlers."""

from __future__ import annotations

import os
import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rich.text import Text

from fast_agent.commands.command_catalog import suggest_command_action
from fast_agent.commands.results import CommandOutcome
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.model_resolution import resolve_model_alias
from fast_agent.llm.model_alias_config import (
    ModelAliasConfigService,
    ModelAliasMutationResult,
    ModelAliasWriteTarget,
)
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext

_PROVIDER_NAME_ALIASES: dict[str, str] = {
    "hf": "huggingface",
    "codex-responses": "codexresponses",
    "codex_responses": "codexresponses",
}

_NO_MODEL_ALIASES_NOTE = (
    "No model_aliases are configured. Add a model_aliases section in fastagent.config.yaml."
)
_ALIASES_USAGE = (
    "Usage: /models aliases "
    "[list|set <token> <model-spec> [--target env|project] [--dry-run]|"
    "unset <token> [--target env|project] [--dry-run]]"
)
_MODELS_USAGE = "Usage: /models [doctor|aliases|catalog|help] [args]"


@dataclass(frozen=True)
class _AliasesMutationArgs:
    operation: Literal["set", "unset"]
    token: str
    model_spec: str | None
    target: ModelAliasWriteTarget
    dry_run: bool


@dataclass(frozen=True)
class _AgentModelDoctorRow:
    name: str
    specified_model: str
    resolved_model: str
    status_symbol: str
    status_style: str
    resolution_note: str | None = None


def _append_line(content: Text, line: str | Text = "") -> None:
    if isinstance(line, Text):
        content.append_text(line)
    else:
        content.append(line)
    content.append("\n")


def _a3_header(title: str, *, color: str = "blue") -> Text:
    header = Text()
    header.append("▎", style=color)
    header.append("• ", style=f"dim {color}")
    header.append(title, style=color)
    return header


def _a3_section(title: str) -> Text:
    line = Text()
    line.append(title, style="bold")
    return line


def _a3_bullet(text: str, *, style: str = "white") -> Text:
    line = Text()
    line.append("▎", style="dim")
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
    provider = ctx.agent_provider
    provider_agents = getattr(provider, "_agents", None)
    if isinstance(provider_agents, dict):
        return sorted(str(name) for name in provider_agents.keys())

    try:
        names = provider.agent_names()
    except Exception:
        return []
    return sorted(str(name) for name in names)


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
            aliases=ModelFactory.MODEL_ALIASES,
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
    aliases: dict[str, dict[str, str]],
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
        alias_error: str | None = None

        if effective_spec and effective_spec.startswith("$"):
            try:
                resolved_from_spec = resolve_model_alias(effective_spec, aliases)
            except ModelConfigError as exc:
                alias_error = exc.details
        elif effective_spec:
            resolved_from_spec = effective_spec

        llm = getattr(agent, "llm", None) or getattr(agent, "_llm", None)
        llm_model = _safe_stripped(getattr(llm, "model_name", None)) if llm is not None else None

        if alias_error:
            if llm_model:
                resolved_display = llm_model
                status_symbol = "◐"
                status_style = "yellow"
                resolution_note = alias_error
            else:
                resolved_display = "<unresolved>"
                status_symbol = "✗"
                status_style = "red"
                resolution_note = alias_error
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

    headers = [
        "Agent",
        "Specified",
        "Resolved",
        "Status",
    ]
    table_values = [
        [
            row.name,
            row.specified_model,
            row.resolved_model,
            row.status_symbol,
        ]
        for row in rows
    ]

    max_limits = [24, 34, 34, 6]
    widths = [len(header) for header in headers]
    for row_values in table_values:
        for index, value in enumerate(row_values):
            widths[index] = min(max(widths[index], len(value)), max_limits[index])

    def _fmt_row(values: list[str]) -> str:
        cells = [
            _truncate_cell(value, limit=widths[index]).ljust(widths[index])
            for index, value in enumerate(values)
        ]
        return " | ".join(cells)

    header_line = _fmt_row(headers)
    separator_line = "-+-".join("-" * width for width in widths)
    _append_line(content, Text(header_line, style="bold"))
    _append_line(content, Text(separator_line, style="dim"))

    note_counts = Counter(note for note in (row.resolution_note for row in rows) if note)
    repeated_note_lines: list[str] = []
    repeated_seen: set[str] = set()

    for row, values in zip(rows, table_values, strict=False):
        _append_line(content, Text(_fmt_row(values), style=row.status_style))
        if row.resolution_note and note_counts[row.resolution_note] <= 1:
            _append_line(content, Text(f"  note: {row.resolution_note}", style="dim"))
        elif row.resolution_note and row.resolution_note not in repeated_seen:
            repeated_seen.add(row.resolution_note)
            repeated_note_lines.append(row.resolution_note)

    if repeated_note_lines:
        _append_line(content)
        _append_line(content, Text("Notes:", style="bold dim"))
        for note in repeated_note_lines:
            _append_line(content, Text(f"  note: {note}", style="dim"))

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


def _resolve_alias_service(ctx: "CommandContext") -> ModelAliasConfigService:
    settings = ctx.resolve_settings()
    env_dir = getattr(settings, "environment_dir", None)
    return ModelAliasConfigService(cwd=Path.cwd(), env_dir=env_dir)


def _flatten_aliases(aliases: dict[str, dict[str, str]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for namespace, entries in sorted(aliases.items(), key=lambda item: str(item[0])):
        for key, model_spec in sorted(entries.items(), key=lambda item: str(item[0])):
            rows.append((f"${namespace}.{key}", model_spec))
    return rows


def _collect_unresolved_aliases(
    aliases: dict[str, dict[str, str]],
    *,
    default_model: str | None,
) -> list[tuple[str, str, str]]:
    unresolved: dict[tuple[str, str], str] = {}

    for token, _ in _flatten_aliases(aliases):
        try:
            resolve_model_alias(token, aliases)
        except ModelConfigError as exc:
            unresolved[(token, f"alias: {token}")] = exc.details

    if default_model and default_model.strip().startswith("$"):
        token = default_model.strip()
        try:
            resolve_model_alias(token, aliases)
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
    aliases: dict[str, dict[str, str]],
) -> Provider | None:
    if not default_model:
        return None

    try:
        resolved_model = resolve_model_alias(default_model, aliases)
    except ModelConfigError:
        return None

    try:
        parsed = ModelFactory.parse_model_string(
            resolved_model,
            aliases=ModelFactory.MODEL_ALIASES,
        )
    except Exception:
        return None

    provider = parsed.provider
    return provider if isinstance(provider, Provider) else None


def _provider_is_ready(provider: Provider, configured: set[Provider]) -> bool:
    if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
        return True
    return provider in configured


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
        outcome.add_message(_a3_header("models help"), right_info="models")
        outcome.add_message(_MODELS_USAGE, right_info="models")
        outcome.add_message("Examples: /models doctor, /models aliases list, /models catalog openai", right_info="models")
        return outcome

    normalized_action = (action or "doctor").strip().lower()
    if normalized_action in {"", "list"}:
        normalized_action = "doctor"

    if normalized_action == "doctor":
        return await _handle_models_doctor(ctx)
    if normalized_action in {"aliases", "alias"}:
        return await _handle_models_aliases(ctx, argument=argument)
    if normalized_action == "catalog":
        return await _handle_models_catalog(ctx, argument=argument)
    if normalized_action == "help":
        outcome = CommandOutcome()
        outcome.add_message(_MODELS_USAGE, right_info="models")
        return outcome

    outcome = CommandOutcome()
    suggestions = suggest_command_action("models", normalized_action)
    suggestion_text = ""
    if suggestions:
        suggestion_text = " Did you mean: " + ", ".join(f"`{name}`" for name in suggestions)
    outcome.add_message(
        _a3_error_block(
            "models",
            (
                "Unknown /models action. "
                "Use /models, /models doctor, /models aliases, "
                f"/models catalog <provider> [--all], or /models help.{suggestion_text}"
            ),
        ),
        channel="error",
        right_info="models",
    )
    return outcome


async def _handle_models_doctor(ctx: "CommandContext") -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    env_dir_env = os.getenv("ENVIRONMENT_DIR")
    fast_agent_model_env = os.getenv("FAST_AGENT_MODEL")
    effective_env_dir = getattr(settings, "environment_dir", None)
    loaded_config_file = getattr(settings, "_config_file", None)

    try:
        service = _resolve_alias_service(ctx)
        aliases = service.list_aliases()
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(
            _a3_error_block("models doctor", f"Failed to load model aliases: {exc}"),
            channel="error",
            right_info="models",
        )
        return outcome

    config_payload = _resolve_config_payload(settings)
    configured_providers = set(ModelSelectionCatalog.configured_providers(config_payload))
    default_model = getattr(settings, "default_model", None)
    unresolved = _collect_unresolved_aliases(aliases, default_model=default_model)
    agent_rows = _build_agent_model_rows(
        ctx,
        aliases=aliases,
        default_model=_safe_stripped(default_model),
    )

    default_provider = _default_model_provider(default_model=default_model, aliases=aliases)
    default_provider_ready = True
    if default_provider is not None:
        default_provider_ready = _provider_is_ready(default_provider, configured_providers)

    readiness_ready = not unresolved and default_provider_ready and bool(configured_providers)

    content = Text()
    _append_line(content, _a3_header("models doctor"))
    _append_line(content)
    if readiness_ready:
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
        _a3_bullet(f"ENVIRONMENT_DIR: {env_dir_env or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(f"Effective environment_dir: {effective_env_dir or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(f"FAST_AGENT_MODEL: {fast_agent_model_env or '<unset>'}", style="dim"),
    )
    _append_line(
        content,
        _a3_bullet(f"Loaded config file: {loaded_config_file or '<none>'}", style="dim"),
    )

    _append_line(content)
    _append_line(content, _a3_section("Unresolved aliases:"))
    if unresolved:
        for token, source, details in unresolved:
            _append_line(content, _a3_bullet(f"{token} ({source})", style="yellow"))
            if details:
                _append_line(content, Text(f"  {details}", style="dim"))
    else:
        _append_line(content, _a3_bullet("none", style="dim"))

    _append_line(content)
    _append_line(content, _a3_section("Provider readiness:"))
    for provider in _catalog_providers():
        state = "configured" if provider in configured_providers else "not configured"
        state_style = "green" if state == "configured" else "dim"
        _append_line(
            content,
            _a3_bullet(f"{provider.display_name}: {state}", style=state_style),
        )

    _append_line(content)
    content.append_text(_render_agent_model_table(agent_rows))
    _append_line(content)
    content.append_text(_render_agent_model_summary(agent_rows))
    _append_line(content)
    _append_line(content, _a3_bullet("✓ resolved/config-default/runtime-default", style="green"))
    _append_line(content, _a3_bullet("◐ runtime fallback or runtime override", style="yellow"))
    _append_line(content, _a3_bullet("✗ unresolved", style="red"))

    if default_provider is not None and not default_provider_ready:
        _append_line(content)
        _append_line(
            content,
            _a3_bullet(
                f"Default model provider '{default_provider.display_name}' is not configured for current settings.",
                style="yellow",
            ),
        )

    if not configured_providers:
        _append_line(content)
        _append_line(content, _a3_bullet("No provider credentials detected.", style="yellow"))

    if not readiness_ready:
        _append_line(content)
        _append_line(content, _a3_section("Next steps:"))
        if unresolved:
            _append_line(content, _a3_bullet("/models aliases", style="cyan"))
        _append_line(content, _a3_bullet("/models catalog <provider>", style="cyan"))

    outcome.add_message(content, right_info="models")
    return outcome


def _parse_aliases_arguments(
    argument: str | None,
) -> tuple[Literal["list", "mutate"], _AliasesMutationArgs | None, str | None]:
    if not argument:
        return "list", None, None

    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        return "list", None, f"Invalid aliases arguments: {exc}"

    if not tokens:
        return "list", None, None

    subcmd = tokens[0].lower()
    if subcmd == "list":
        if len(tokens) > 1:
            return "list", None, "Unexpected arguments after 'list'."
        return "list", None, None

    if subcmd not in {"set", "unset"}:
        return "list", None, f"Unknown aliases action '{tokens[0]}'."

    target: ModelAliasWriteTarget = "env"
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
        if len(positional) < 2:
            return "list", None, "Usage: /models aliases set <token> <model-spec> [--target env|project] [--dry-run]"
        if len(positional) > 2:
            return "list", None, "Too many positional arguments for aliases set."

        return (
            "mutate",
            _AliasesMutationArgs(
                operation="set",
                token=positional[0],
                model_spec=positional[1],
                target=target,
                dry_run=dry_run,
            ),
            None,
        )

    if len(positional) < 1:
        return "list", None, "Usage: /models aliases unset <token> [--target env|project] [--dry-run]"
    if len(positional) > 1:
        return "list", None, "Too many positional arguments for aliases unset."

    return (
        "mutate",
        _AliasesMutationArgs(
            operation="unset",
            token=positional[0],
            model_spec=None,
            target=target,
            dry_run=dry_run,
        ),
        None,
    )


def _render_alias_mutation(
    *,
    title: str,
    result: ModelAliasMutationResult,
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


async def _handle_models_aliases(ctx: "CommandContext", *, argument: str | None) -> CommandOutcome:
    outcome = CommandOutcome()

    mode, mutation_args, parse_error = _parse_aliases_arguments(argument)
    if parse_error is not None:
        error = Text()
        _append_line(error, _a3_header("models aliases", color="red"))
        _append_line(error)
        _append_line(error, _a3_bullet(parse_error, style="red"))
        _append_line(error, Text(_ALIASES_USAGE, style="dim"))
        outcome.add_message(error, channel="error", right_info="models")
        return outcome

    try:
        service = _resolve_alias_service(ctx)
        if mode == "mutate":
            assert mutation_args is not None
            if mutation_args.operation == "set":
                assert mutation_args.model_spec is not None
                mutation_result = service.set_alias(
                    mutation_args.token,
                    mutation_args.model_spec,
                    target=mutation_args.target,
                    dry_run=mutation_args.dry_run,
                )
                outcome.add_message(
                    _render_alias_mutation(
                        title="models aliases set",
                        result=mutation_result,
                    ),
                    right_info="models",
                )
                return outcome

            mutation_result = service.unset_alias(
                mutation_args.token,
                target=mutation_args.target,
                dry_run=mutation_args.dry_run,
            )
            outcome.add_message(
                _render_alias_mutation(
                    title="models aliases unset",
                    result=mutation_result,
                ),
                right_info="models",
            )
            return outcome

        aliases = service.list_aliases()
    except ValueError as exc:
        error = Text()
        _append_line(error, _a3_header("models aliases", color="red"))
        _append_line(error)
        _append_line(error, _a3_bullet(str(exc), style="red"))
        _append_line(error, Text(_ALIASES_USAGE, style="dim"))
        outcome.add_message(error, channel="error", right_info="models")
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(
            _a3_error_block("models aliases", f"Failed to load model aliases: {exc}"),
            channel="error",
            right_info="models",
        )
        return outcome

    rows = _flatten_aliases(aliases)
    if not rows:
        empty = Text()
        _append_line(empty, _a3_header("models aliases"))
        _append_line(empty)
        _append_line(empty, _a3_bullet("No model aliases configured.", style="yellow"))
        outcome.add_message(empty, channel="warning", right_info="models")
        return outcome

    content = Text()
    _append_line(content, _a3_header("models aliases"))
    _append_line(content)
    _append_line(content, _a3_section("Model aliases:"))
    for token, model_spec in rows:
        try:
            resolved = resolve_model_alias(token, aliases)
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

    outcome.add_message(content, right_info="models")
    return outcome


def _parse_catalog_arguments(argument: str | None) -> tuple[str | None, bool, str | None]:
    if not argument:
        return None, False, "Usage: /models catalog <provider> [--all]"

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
        return None, False, "Usage: /models catalog <provider> [--all]"

    return provider_name, show_all, None


async def _handle_models_catalog(ctx: "CommandContext", *, argument: str | None) -> CommandOutcome:
    outcome = CommandOutcome()
    provider_name, show_all, parse_error = _parse_catalog_arguments(argument)
    if parse_error is not None:
        outcome.add_message(
            _a3_error_block("models catalog", parse_error),
            channel="error",
            right_info="models",
        )
        return outcome

    assert provider_name is not None
    provider = _resolve_catalog_provider(provider_name)
    if provider is None:
        outcome.add_message(
            _a3_error_block(
                "models catalog",
                f"Unknown provider '{provider_name}'. Choose one of: {_provider_display_choices()}.",
            ),
            channel="error",
            right_info="models",
        )
        return outcome

    settings = ctx.resolve_settings()
    config_payload = _resolve_config_payload(settings)

    curated_entries = ModelSelectionCatalog.list_current_entries(provider)

    content = Text()
    _append_line(content, _a3_header("models catalog"))
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
            alias = entry.alias or "-"
            style = "green" if entry.fast else "white"
            _append_line(content, _a3_bullet(f"{alias} -> {entry.model}{fast_tag}", style=style))
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

    outcome.add_message(content, right_info="models")
    return outcome
