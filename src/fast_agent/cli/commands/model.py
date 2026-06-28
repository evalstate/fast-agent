"""Interactive CLI helpers for model reference setup."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import typer
from pydantic import ValidationError
from rich.table import Table
from rich.text import Text

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.commands.context import CommandContext, CommandIO, StaticAgentProvider
from fast_agent.commands.handlers import models_manager
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.config import (
    Settings,
    deep_merge,
    load_implicit_settings,
    load_yaml_mapping,
)
from fast_agent.llm.llamacpp_discovery import (
    DEFAULT_LLAMA_CPP_URL,
    LlamaCppDiscoveredModel,
    LlamaCppDiscoveryCatalog,
    LlamaCppDiscoveryError,
    LlamaCppModelListing,
    build_llamacpp_overlay_manifest,
    default_overlay_name_for_model,
    discover_llamacpp_models,
    interrogate_llamacpp_model,
    uniquify_overlay_name,
)
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import (
    LoadedModelOverlay,
    build_model_overlay_manifest_from_database,
    load_model_overlay_registry,
    load_model_overlay_secret_entries,
    serialize_model_overlay_manifest,
    write_model_overlay_manifest,
)
from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
from fast_agent.llm.model_reference_diagnostics import (
    ModelReferenceSetupDiagnostics,
    ModelReferenceSetupItem,
    collect_model_reference_setup_diagnostics,
)
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.console import console
from fast_agent.utils.async_utils import run_coroutine

if TYPE_CHECKING:
    from collections.abc import Callable
from fast_agent.ui.llamacpp_model_picker import (
    LlamaCppModelPickerContext,
    run_llamacpp_model_picker_async,
)
from fast_agent.ui.model_picker import run_model_picker_async
from fast_agent.ui.model_reference_picker import (
    ModelReferencePickerItem,
    ModelReferencePickerResult,
    run_model_reference_picker_async,
)
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.commandline import join_commandline, quote_commandline_token
from fast_agent.utils.count_display import format_count
from fast_agent.utils.text import strip_casefold, strip_to_none

type WriteTarget = Literal["env", "project"]
type LlamaCppAuthMode = Literal["none", "env", "secret_ref"]
type _LlamaCppImportAction = Literal[
    "start_now",
    "start_now_with_shell",
    "start_now_smart",
    "generate_overlay",
]
type _LlamaCppStringGroupOptionName = Literal[
    "env",
    "url",
    "auth",
    "api_key_env",
    "secret_ref",
    "name",
]
type _LlamaCppBoolGroupOptionName = Literal["include_sampling_defaults"]


@runtime_checkable
class ModelReferencePickerIO(Protocol):
    async def pick_model_reference_token(
        self,
        *,
        items: tuple[ModelReferenceSetupItem, ...],
    ) -> str | None: ...


app = typer.Typer(help="Interactive model reference setup.", add_completion=False)
llamacpp_app = typer.Typer(
    help="Discover llama.cpp models, preview overlays, and import local runtime overlays.",
    add_completion=False,
)
app.add_typer(llamacpp_app, name="llamacpp")


def _llamacpp_env_option() -> str | None:
    return cast(
        "str | None",
        typer.Option(
            None,
            "--env",
            help="Override the base fast-agent environment directory",
        ),
    )


def _llamacpp_url_option() -> str:
    return cast(
        "str",
        typer.Option(
            DEFAULT_LLAMA_CPP_URL,
            "--url",
            "--base-url",
            help="llama.cpp server URL to interrogate. Root URLs are normalized to /v1 for runtime use.",
        ),
    )


def _llamacpp_auth_option() -> str | None:
    return cast(
        "str | None",
        typer.Option(
            None,
            "--auth",
            help="Persisted overlay auth mode.",
        ),
    )


def _llamacpp_api_key_env_option(*, discovery_only: bool = False) -> str | None:
    return cast(
        "str | None",
        typer.Option(
            None,
            "--api-key-env",
            help=(
                "Environment variable to use for llama.cpp discovery."
                if discovery_only
                else "Environment variable to use for interrogation and/or persisted overlay auth."
            ),
        ),
    )


def _llamacpp_secret_ref_option(*, discovery_only: bool = False) -> str | None:
    return cast(
        "str | None",
        typer.Option(
            None,
            "--secret-ref",
            help=(
                "Secret ref to use for discovery if no --api-key-env is supplied."
                if discovery_only
                else "Secret ref to persist in the overlay. If no --api-key-env is supplied, "
                "an existing secret is used for interrogation."
            ),
        ),
    )


def _llamacpp_name_option() -> str | None:
    return cast(
        "str | None",
        typer.Option(
            None,
            "--name",
            help="Optional overlay name.",
        ),
    )


def _build_reference_setup_argument(
    *,
    token: str | None,
    target: WriteTarget,
    dry_run: bool,
) -> str:
    parts = ["set"]
    normalized_token = strip_to_none(token)
    if normalized_token is not None:
        parts.append(quote_commandline_token(normalized_token, syntax="posix"))
    parts.extend(["--target", target])
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def _normalize_write_target(value: str) -> WriteTarget:
    normalized = normalize_action_token(value)
    if normalized == "env":
        return "env"
    if normalized == "project":
        return "project"
    raise typer.BadParameter("--target must be either 'env' or 'project'.")


def _normalize_interactive_reference_token(token: str | None) -> str | None:
    stripped = strip_to_none(token)
    if stripped is None:
        return None
    if stripped.startswith("$"):
        return stripped
    return f"${stripped}"


def _bootstrap_settings_start_path(env_dir: str | Path | None) -> Path:
    env_dir_string = strip_to_none(env_dir) if isinstance(env_dir, str) else None
    if env_dir_string is not None:
        env_root = Path(env_dir_string).expanduser()
        if env_root.is_absolute():
            return env_root.resolve().parent
    elif isinstance(env_dir, Path):
        env_root = env_dir.expanduser()
        if env_root.is_absolute():
            return env_root.resolve().parent
    return Path.cwd()


async def _prompt_manual_reference_token(io: CommandIO) -> str | None:
    return _normalize_interactive_reference_token(
        await io.prompt_text(
            "Reference token ($namespace.key):",
            allow_empty=False,
        )
    )


async def run_model_setup(
    *,
    io: CommandIO,
    settings: Settings,
    token: str | None,
    target: WriteTarget = "env",
    dry_run: bool = False,
) -> CommandOutcome:
    """Execute the shared interactive reference-setup flow."""
    resolved_token = token
    start_path = resolve_model_reference_start_path(settings=settings)
    if resolved_token is None:
        diagnostics = collect_model_reference_setup_diagnostics(
            cwd=start_path,
            env_dir=settings.environment_dir,
        )
        common_items = _build_common_setup_items(diagnostics.valid_references)
        has_guided_choices = bool(diagnostics.items) or (
            bool(common_items) and isinstance(io, ModelReferencePickerIO)
        )
        resolved_token = await _select_model_setup_token(
            io,
            diagnostics=diagnostics,
            common_items=common_items,
        )
        if has_guided_choices and resolved_token is None:
            outcome = CommandOutcome()
            outcome.add_message("Model setup cancelled.", channel="warning", right_info="model")
            return outcome

    provider = StaticAgentProvider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="cli",
        io=io,
        settings=settings,
    )
    argument = _build_reference_setup_argument(
        token=resolved_token,
        target=target,
        dry_run=dry_run,
    )
    return await models_manager.handle_models_command(
        ctx,
        agent_name="cli",
        action="references",
        argument=argument,
    )


async def run_model_doctor(
    *,
    io: CommandIO,
    settings: Settings,
) -> CommandOutcome:
    """Execute the shared model doctor flow."""
    effective_settings = settings
    if (
        getattr(settings, "_config_file", None) is None
        and settings.default_model is None
        and not settings.model_references
    ):
        start_path = _bootstrap_settings_start_path(settings.environment_dir)
        effective_settings = _load_cli_settings(
            cwd=start_path,
            env_dir=settings.environment_dir,
        )

    provider = StaticAgentProvider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="cli",
        io=io,
        settings=effective_settings,
    )
    return await models_manager.handle_models_command(
        ctx,
        agent_name="cli",
        action="doctor",
        argument=None,
    )


async def _select_model_setup_token(
    io: CommandIO,
    *,
    diagnostics: ModelReferenceSetupDiagnostics,
    common_items: tuple[ModelReferenceSetupItem, ...] | None = None,
) -> str | None:
    items = diagnostics.items
    resolved_common_items = (
        common_items
        if common_items is not None
        else _build_common_setup_items(diagnostics.valid_references)
    )
    if isinstance(io, ModelReferencePickerIO):
        return await _select_model_setup_token_from_picker(
            io,
            items=items,
            common_items=resolved_common_items,
        )

    if not items:
        return None

    return await _select_model_setup_token_from_prompt(io, items=items)


async def _select_model_setup_token_from_picker(
    io: ModelReferencePickerIO,
    *,
    items: tuple[ModelReferenceSetupItem, ...],
    common_items: tuple[ModelReferenceSetupItem, ...],
) -> str | None:
    picker_items = _merge_setup_items(items, common_items) if items else common_items
    if not picker_items:
        return None
    return await _pick_or_prompt_reference_token(io, items=picker_items)


async def _select_model_setup_token_from_prompt(
    io: CommandIO,
    *,
    items: tuple[ModelReferenceSetupItem, ...],
) -> str | None:
    if len(items) == 1:
        item = items[0]
        await io.emit(
            CommandMessage(
                text=_render_setup_item_summary(
                    item,
                    title="Detected one reference that needs setup",
                ),
                right_info="model",
            )
        )
        return item.token

    await io.emit(
        CommandMessage(
            text=_render_setup_item_list(items),
            right_info="model",
        )
    )
    option_labels = {str(index): item.token for index, item in enumerate(items, start=1)}
    selection = await io.prompt_selection(
        "Reference to configure (number or 'custom'):",
        options=[*option_labels.keys(), "custom"],
        allow_cancel=True,
    )
    if selection is None:
        return None

    normalized_selection = normalize_action_token(selection)
    if normalized_selection == "custom":
        return await _prompt_manual_reference_token(io)
    return option_labels.get(normalized_selection)


async def _pick_or_prompt_reference_token(
    io: ModelReferencePickerIO,
    *,
    items: tuple[ModelReferenceSetupItem, ...],
) -> str | None:
    return await io.pick_model_reference_token(items=items)


def _render_setup_item_summary(item: ModelReferenceSetupItem, *, title: str) -> Text:
    content = Text()
    content.append(f"{title}\n", style="bold")
    content.append(f"• {item.token}\n", style="cyan")
    content.append(f"  {item.priority}/{item.status}: {item.summary}\n", style="yellow")
    if item.references:
        content.append(
            f"  used by: {', '.join(item.references)}",
            style="dim",
        )
    return content


def _render_setup_item_list(items: tuple[ModelReferenceSetupItem, ...]) -> Text:
    content = Text()
    content.append("References that need setup\n", style="bold")
    for index, item in enumerate(items, start=1):
        content.append(
            f"{index}. {item.token}  [{item.priority}/{item.status}]\n",
            style="cyan" if item.priority == "recommended" else "yellow",
        )
        content.append(f"   {item.summary}\n", style="white")
        if item.references:
            content.append(
                f"   used by: {', '.join(item.references)}\n",
                style="dim",
            )
        if item.current_value is not None:
            current_value = item.current_value if item.current_value else "<empty>"
            content.append(f"   current: {current_value}\n", style="dim")
    content.append("\nType 'custom' to enter a different reference token.", style="dim")
    return content


def _build_common_setup_items(
    valid_references: dict[str, dict[str, str]],
    *,
    suppressed_tokens: set[str] | None = None,
) -> tuple[ModelReferenceSetupItem, ...]:
    items: list[ModelReferenceSetupItem] = []
    hidden_tokens = suppressed_tokens or set()
    system_references = valid_references.get("system", {})
    if (
        "default" not in system_references
        and "last_used" not in system_references
        and "$system.default" not in hidden_tokens
    ):
        items.append(
            ModelReferenceSetupItem(
                token="$system.default",
                priority="required",
                status="missing",
                current_value=None,
                summary="Recommended starter reference for your main default model.",
                references=("starter setup",),
            )
        )
    if "fast" not in system_references and "$system.fast" not in hidden_tokens:
        items.append(
            ModelReferenceSetupItem(
                token="$system.fast",
                priority="recommended",
                status="missing",
                current_value=None,
                summary="Optional starter reference for a faster or cheaper model.",
                references=("starter setup",),
            )
        )
    return tuple(items)


def _merge_setup_items(
    primary_items: tuple[ModelReferenceSetupItem, ...],
    extra_items: tuple[ModelReferenceSetupItem, ...],
) -> tuple[ModelReferenceSetupItem, ...]:
    merged: list[ModelReferenceSetupItem] = list(primary_items)
    seen_tokens = {item.token for item in primary_items}
    for item in extra_items:
        if item.token in seen_tokens:
            continue
        seen_tokens.add(item.token)
        merged.append(item)
    return tuple(merged)


def _build_picker_items(
    diagnostics: ModelReferenceSetupDiagnostics,
    *,
    suppressed_tokens: set[str] | None = None,
) -> tuple[ModelReferencePickerItem, ...]:
    items: list[ModelReferencePickerItem] = []
    seen_tokens: set[str] = set()
    hidden_tokens = suppressed_tokens or set()

    def _add_item(item: ModelReferencePickerItem) -> None:
        if item.token in seen_tokens:
            return
        seen_tokens.add(item.token)
        items.append(item)

    for item in diagnostics.items:
        _add_item(
            ModelReferencePickerItem(
                token=item.token,
                priority=item.priority,
                status=f"{item.priority}/{item.status}",
                summary=item.summary,
                current_value=item.current_value,
                references=item.references,
                removable=False,
            )
        )

    for item in _build_common_setup_items(
        diagnostics.valid_references,
        suppressed_tokens=hidden_tokens,
    ):
        _add_item(
            ModelReferencePickerItem(
                token=item.token,
                priority=item.priority,
                status=f"{item.priority}/{item.status}",
                summary=item.summary,
                current_value=item.current_value,
                references=item.references,
                removable=False,
            )
        )

    for namespace, entries in sorted(diagnostics.valid_references.items()):
        for key, model_spec in sorted(entries.items()):
            token = f"${namespace}.{key}"
            _add_item(
                ModelReferencePickerItem(
                    token=token,
                    priority="configured",
                    status="configured",
                    summary="Existing reference mapping.",
                    current_value=model_spec,
                    references=(),
                    removable=True,
                )
            )

    return tuple(items)


async def _run_model_reference_unset(
    *,
    io: CommandIO,
    settings: Settings,
    token: str,
    target: WriteTarget,
    dry_run: bool,
) -> CommandOutcome:
    provider = StaticAgentProvider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="cli",
        io=io,
        settings=settings,
    )
    argument = f"unset {quote_commandline_token(token, syntax='posix')} --target {target}"
    if dry_run:
        argument += " --dry-run"
    return await models_manager.handle_models_command(
        ctx,
        agent_name="cli",
        action="references",
        argument=argument,
    )


async def _run_model_setup_command(
    *,
    settings: Settings,
    token: str | None,
    target: WriteTarget,
    dry_run: bool,
) -> None:
    start_path = resolve_model_reference_start_path(settings=settings)
    io = _model_reference_command_io(settings=settings, start_path=start_path)
    if token is not None:
        outcome = await run_model_setup(
            io=io,
            settings=settings,
            token=token,
            target=target,
            dry_run=dry_run,
        )
        for message in outcome.messages:
            await io.emit(message)
        return

    suppressed_tokens: set[str] = set()
    while True:
        picker_result = await _run_reference_picker(
            settings=settings,
            start_path=start_path,
            suppressed_tokens=suppressed_tokens,
        )
        if picker_result is None:
            return
        if picker_result.action == "done":
            return

        outcome = await _handle_model_setup_picker_result(
            picker_result,
            io=io,
            settings=settings,
            target=target,
            dry_run=dry_run,
            suppressed_tokens=suppressed_tokens,
        )
        if outcome is None:
            return
        if picker_result.action == "unset" and not dry_run:
            continue

        for message in outcome.messages:
            await io.emit(message)
        if dry_run:
            return


def _model_reference_command_io(*, settings: Settings, start_path: Path) -> TuiCommandIO:
    return TuiCommandIO(
        prompt_provider=StaticAgentProvider(),
        agent_name="cli",
        settings=settings,
        config_payload=_load_tolerant_config_payload(
            cwd=start_path,
            env_dir=settings.environment_dir,
        ),
    )


async def _run_reference_picker(
    *,
    settings: Settings,
    start_path: Path,
    suppressed_tokens: set[str],
) -> ModelReferencePickerResult | None:
    diagnostics = collect_model_reference_setup_diagnostics(
        cwd=start_path,
        env_dir=settings.environment_dir,
    )
    picker_items = _build_picker_items(
        diagnostics,
        suppressed_tokens=suppressed_tokens,
    )
    return await run_model_reference_picker_async(picker_items)


async def _handle_model_setup_picker_result(
    picker_result: ModelReferencePickerResult,
    *,
    io: CommandIO,
    settings: Settings,
    target: WriteTarget,
    dry_run: bool,
    suppressed_tokens: set[str],
) -> CommandOutcome | None:
    if picker_result.action == "custom":
        selected_token = await _prompt_manual_reference_token(io)
        if selected_token is None:
            return None
        return await run_model_setup(
            io=io,
            settings=settings,
            token=selected_token,
            target=target,
            dry_run=dry_run,
        )

    if picker_result.action == "unset":
        assert picker_result.token is not None
        outcome = await _run_model_reference_unset(
            io=io,
            settings=settings,
            token=picker_result.token,
            target=target,
            dry_run=dry_run,
        )
        if dry_run:
            return outcome

        suppressed_tokens.add(picker_result.token)
        await _emit_warning_and_error_messages(io, outcome)
        return outcome

    assert picker_result.token is not None
    suppressed_tokens.discard(picker_result.token)
    return await run_model_setup(
        io=io,
        settings=settings,
        token=picker_result.token,
        target=target,
        dry_run=dry_run,
    )


async def _emit_warning_and_error_messages(io: CommandIO, outcome: CommandOutcome) -> None:
    for message in outcome.messages:
        if message.channel in {"warning", "error"}:
            await io.emit(message)


async def _run_model_doctor_command(*, settings: Settings) -> None:
    start_path = resolve_model_reference_start_path(settings=settings)
    provider = StaticAgentProvider()
    io = TuiCommandIO(
        prompt_provider=provider,
        agent_name="cli",
        settings=settings,
        config_payload=_load_tolerant_config_payload(
            cwd=start_path,
            env_dir=settings.environment_dir,
        ),
    )
    outcome = await run_model_doctor(
        io=io,
        settings=settings,
    )
    for message in outcome.messages:
        await io.emit(message)


def _load_cli_settings(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> Settings:
    merged_settings, discovery = load_implicit_settings(start_path=cwd, env_dir=env_dir)
    config_file = discovery.config_path
    secrets_path = discovery.secrets_path
    if secrets_path and secrets_path.exists():
        merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))

    settings = Settings(**merged_settings)
    settings._config_file = str(config_file) if config_file else None
    return settings


def _load_tolerant_config_payload(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> dict[str, object] | None:
    try:
        merged_settings, discovery = load_implicit_settings(start_path=cwd, env_dir=env_dir)
        secrets_path = discovery.secrets_path
        if secrets_path and secrets_path.exists():
            merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))
    except Exception:
        return None
    return merged_settings or None


def _print_validation_error(exc: ValidationError) -> None:
    typer.echo("fast-agent model setup could not load the current configuration.", err=True)
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        if location:
            typer.echo(f"  - {location}: {message}", err=True)
        else:
            typer.echo(f"  - {message}", err=True)
    typer.echo("Hint: run `fast-agent check` for a broader config report.", err=True)


@dataclass(frozen=True, slots=True)
class _LlamaCppImportResult:
    catalog: LlamaCppDiscoveryCatalog
    discovered_model: LlamaCppDiscoveredModel
    action: _LlamaCppImportAction
    overlay_name: str
    manifest_payload: dict[str, object]
    overlay_yaml: str
    output_path: Path | None


@dataclass(frozen=True, slots=True)
class _LlamaCppSelection:
    model_id: str
    action: _LlamaCppImportAction


@dataclass(frozen=True, slots=True)
class _LlamaCppLaunchOptions:
    with_shell: bool = False
    smart: bool = False


_LLAMACPP_LAUNCH_OPTIONS_BY_ACTION: dict[
    _LlamaCppImportAction,
    _LlamaCppLaunchOptions,
] = {
    "start_now": _LlamaCppLaunchOptions(),
    "start_now_with_shell": _LlamaCppLaunchOptions(with_shell=True),
    "start_now_smart": _LlamaCppLaunchOptions(with_shell=True, smart=True),
}


@dataclass(frozen=True, slots=True)
class _LlamaCppCommandContext:
    resolved_env_dir: Path | None
    start_path: Path
    interrogation_api_key: str | None


@dataclass(frozen=True, slots=True)
class _LlamaCppPersistedAuth:
    auth: LlamaCppAuthMode | None
    api_key_env: str | None
    secret_ref: str | None
    default_headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class _LlamaCppPickerImportDefaults:
    url: str
    auth: LlamaCppAuthMode | None
    interrogation_api_key: str | None


@dataclass(frozen=True, slots=True)
class _LlamaCppGroupOptions:
    env: str | None
    url: str
    auth: str | None
    api_key_env: str | None
    secret_ref: str | None
    name: str | None
    include_sampling_defaults: bool


_LLAMACPP_STRING_GROUP_OPTION_READERS: dict[
    _LlamaCppStringGroupOptionName,
    "Callable[[_LlamaCppGroupOptions], str | None]",
] = {
    "env": lambda options: options.env,
    "url": lambda options: options.url,
    "auth": lambda options: options.auth,
    "api_key_env": lambda options: options.api_key_env,
    "secret_ref": lambda options: options.secret_ref,
    "name": lambda options: options.name,
}
_LLAMACPP_BOOL_GROUP_OPTION_READERS: dict[
    _LlamaCppBoolGroupOptionName,
    "Callable[[_LlamaCppGroupOptions], bool]",
] = {
    "include_sampling_defaults": lambda options: options.include_sampling_defaults,
}


def _store_llamacpp_group_options(
    ctx: typer.Context,
    *,
    env: str | None,
    url: str,
    auth: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
    name: str | None,
    include_sampling_defaults: bool,
) -> None:
    payload = _LlamaCppGroupOptions(
        env=env,
        url=url,
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        name=name,
        include_sampling_defaults=include_sampling_defaults,
    )
    if isinstance(ctx.obj, dict):
        ctx.obj["llamacpp_group_options"] = payload
    else:
        ctx.obj = {"llamacpp_group_options": payload}


def _inherit_llamacpp_group_option(
    ctx: typer.Context,
    *,
    option_name: _LlamaCppStringGroupOptionName,
    value: str | None,
) -> str | None:
    parameter_source = ctx.get_parameter_source(option_name)
    if parameter_source is None or parameter_source.name != "DEFAULT":
        return value

    if not isinstance(ctx.obj, dict):
        return value
    payload = ctx.obj.get("llamacpp_group_options")
    if not isinstance(payload, _LlamaCppGroupOptions):
        return value
    return _LLAMACPP_STRING_GROUP_OPTION_READERS[option_name](payload)


def _inherit_llamacpp_group_url(ctx: typer.Context, *, url: str) -> str:
    inherited = _inherit_llamacpp_group_option(ctx, option_name="url", value=url)
    return inherited if inherited is not None else url


def _inherit_llamacpp_group_bool_option(
    ctx: typer.Context,
    *,
    option_name: _LlamaCppBoolGroupOptionName,
    value: bool,
) -> bool:
    parameter_source = ctx.get_parameter_source(option_name)
    if parameter_source is None or parameter_source.name != "DEFAULT":
        return value

    if not isinstance(ctx.obj, dict):
        return value
    payload = ctx.obj.get("llamacpp_group_options")
    if not isinstance(payload, _LlamaCppGroupOptions):
        return value
    return _LLAMACPP_BOOL_GROUP_OPTION_READERS[option_name](payload)


def _llamacpp_include_sampling_defaults_option() -> bool:
    return cast(
        "bool",
        typer.Option(
            False,
            "--include-sampling-defaults",
            help="Persist current llama.cpp sampling defaults into the generated overlay.",
        ),
    )


def _llamacpp_option_was_explicit(ctx: typer.Context, *, option_name: str) -> bool:
    current_source = ctx.get_parameter_source(option_name)
    if current_source is not None and current_source.name != "DEFAULT":
        return True

    parent_ctx = ctx.parent
    if parent_ctx is None:
        return False
    parent_source = parent_ctx.get_parameter_source(option_name)
    return parent_source is not None and parent_source.name != "DEFAULT"


def _optional_cli_text(value: str | None) -> str | None:
    return strip_to_none(value)


def _normalize_llamacpp_auth(
    *,
    auth: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
) -> LlamaCppAuthMode:
    normalized_env = _optional_cli_text(api_key_env)
    normalized_secret_ref = _optional_cli_text(secret_ref)

    if auth is not None:
        resolved_auth = normalize_action_token(auth)
        if resolved_auth == "none":
            resolved = "none"
        elif resolved_auth == "env":
            resolved = "env"
        elif resolved_auth == "secret_ref":
            resolved = "secret_ref"
        else:
            raise typer.BadParameter("--auth must be one of: none, env, secret_ref.")
    elif normalized_secret_ref is not None:
        resolved = "secret_ref"
    elif normalized_env is not None:
        resolved = "env"
    else:
        resolved = "none"

    if resolved == "env" and not normalized_env:
        raise typer.BadParameter("--api-key-env is required when --auth env is used.")
    if resolved == "secret_ref" and not normalized_secret_ref:
        raise typer.BadParameter("--secret-ref is required when --auth secret_ref is used.")
    return resolved


def _resolve_llamacpp_command_context(
    *,
    ctx: typer.Context,
    env: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
) -> _LlamaCppCommandContext:
    resolved_env_dir = resolve_environment_dir_option(
        ctx,
        Path(env) if env is not None else None,
    )
    start_path = _bootstrap_settings_start_path(resolved_env_dir)
    interrogation_api_key = _resolve_llamacpp_interrogation_api_key(
        start_path=start_path,
        env_dir=resolved_env_dir,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
    )
    return _LlamaCppCommandContext(
        resolved_env_dir=resolved_env_dir,
        start_path=start_path,
        interrogation_api_key=interrogation_api_key,
    )


def _resolve_llamacpp_interrogation_api_key(
    *,
    start_path: Path,
    env_dir: str | Path | None,
    api_key_env: str | None,
    secret_ref: str | None,
) -> str | None:
    normalized_env = _optional_cli_text(api_key_env)
    if normalized_env:
        value = os.getenv(normalized_env)
        if value is None:
            raise typer.BadParameter(
                f"Environment variable {normalized_env!r} is required for llama.cpp interrogation."
            )
        return value

    normalized_secret_ref = _optional_cli_text(secret_ref)
    if not normalized_secret_ref:
        return None

    secret_entries = load_model_overlay_secret_entries(start_path=start_path, env_dir=env_dir)
    secret_entry = secret_entries.get(normalized_secret_ref)
    if secret_entry is None or secret_entry.api_key is None:
        raise typer.BadParameter(
            "Could not resolve --secret-ref for llama.cpp interrogation. "
            "Set --api-key-env for first-time imports or add the secret to "
            "model-overlays.secrets.yaml."
        )
    return secret_entry.api_key


async def _select_llamacpp_model(
    *,
    catalog: LlamaCppDiscoveryCatalog,
    interrogation_api_key: str | None,
    selected_model: str | None,
    interactive: bool,
    requested_action: _LlamaCppImportAction | None,
) -> _LlamaCppSelection | None:
    requested = strip_to_none(selected_model)
    if requested is not None:
        if any(model.model_id == requested for model in catalog.models):
            return _LlamaCppSelection(
                model_id=requested,
                action=requested_action or "generate_overlay",
            )
        raise typer.BadParameter(
            f"Model {requested!r} was not found in the discovered llama.cpp catalog."
        )

    if not interactive:
        return None

    runtime_context_cache: dict[str, LlamaCppModelPickerContext] = {}

    async def _load_runtime_context(model_id: str) -> LlamaCppModelPickerContext:
        if model_id in runtime_context_cache:
            return runtime_context_cache[model_id]
        discovered_model = await interrogate_llamacpp_model(
            catalog=catalog,
            model_id=model_id,
            api_key=interrogation_api_key,
        )
        loaded_context = LlamaCppModelPickerContext(
            runtime_context_window=discovered_model.runtime_context_window,
            training_context_window=discovered_model.listing.training_context_window,
        )
        runtime_context_cache[model_id] = loaded_context
        return loaded_context

    picker_result = await run_llamacpp_model_picker_async(
        catalog.models,
        runtime_context_loader=_load_runtime_context,
    )
    if picker_result is None:
        return None
    return _LlamaCppSelection(
        model_id=picker_result.model_id,
        action=picker_result.action,
    )


def _build_llamacpp_overlay_name(
    *,
    requested_name: str | None,
    model_id: str,
    base_url: str,
    start_path: Path,
    env_dir: str | Path | None,
) -> tuple[str, bool, LoadedModelOverlay | None]:
    registry = load_model_overlay_registry(start_path=start_path, env_dir=env_dir)
    existing_names = set(registry.by_name())

    candidate = strip_to_none(requested_name)
    if candidate is not None:
        return uniquify_overlay_name(candidate, existing_names=existing_names), False, None

    generated_candidate = default_overlay_name_for_model(model_id)

    for overlay in registry.overlays:
        manifest = overlay.manifest
        if overlay.name != generated_candidate and not overlay.name.startswith(
            f"{generated_candidate}-"
        ):
            continue
        if manifest.provider != Provider.OPENRESPONSES:
            continue
        if manifest.model != model_id:
            continue
        if manifest.connection.base_url != base_url:
            continue
        if overlay.description != "Imported from llama.cpp":
            continue
        return overlay.name, True, overlay

    return uniquify_overlay_name(generated_candidate, existing_names=existing_names), False, None


def _resolve_llamacpp_persisted_auth(
    *,
    auth: LlamaCppAuthMode | None,
    api_key_env: str | None,
    secret_ref: str | None,
    reused_overlay: LoadedModelOverlay | None,
    preserve_existing_auth: bool,
) -> _LlamaCppPersistedAuth:
    normalized_api_key_env = _optional_cli_text(api_key_env)
    normalized_secret_ref = _optional_cli_text(secret_ref)

    if not preserve_existing_auth or reused_overlay is None:
        return _LlamaCppPersistedAuth(
            auth=auth,
            api_key_env=normalized_api_key_env,
            secret_ref=normalized_secret_ref,
            default_headers={},
        )

    connection = reused_overlay.manifest.connection
    existing_auth = connection.auth_mode()
    return _LlamaCppPersistedAuth(
        auth=existing_auth if existing_auth is not None else auth,
        api_key_env=connection.api_key_env,
        secret_ref=connection.secret_ref,
        default_headers=dict(connection.default_headers),
    )


def _llamacpp_catalog_json_payload(catalog: LlamaCppDiscoveryCatalog) -> dict[str, object]:
    return {
        "requested_url": catalog.endpoints.requested_url,
        "server_url": catalog.endpoints.server_url,
        "request_base_url": catalog.endpoints.request_base_url,
        "models_url": catalog.models_url,
        "models": [
            {
                "id": model.model_id,
                "owned_by": model.owned_by,
                "training_context_window": model.training_context_window,
            }
            for model in catalog.models
        ],
    }


def _format_llamacpp_model_listing(model: LlamaCppModelListing) -> str:
    context_label = (
        f"ctx {model.training_context_window}"
        if model.training_context_window is not None
        else "ctx ?"
    )
    return f"{model.model_id} ({context_label})"


def _emit_llamacpp_catalog_listing(catalog: LlamaCppDiscoveryCatalog) -> None:
    typer.echo(f"Discovered {format_count(len(catalog.models), 'llama.cpp model')}:")
    for model in catalog.models:
        typer.echo(f"  - {_format_llamacpp_model_listing(model)}")


def _llamacpp_import_json_payload(result: _LlamaCppImportResult) -> dict[str, object]:
    return {
        **_llamacpp_catalog_json_payload(result.catalog),
        "selected_model": result.discovered_model.listing.model_id,
        "props_url": result.discovered_model.props_url,
        "overlay_name": result.overlay_name,
        "overlay_path": str(result.output_path) if result.output_path is not None else None,
        "manifest": result.manifest_payload,
    }


async def _run_llamacpp_import(
    *,
    start_path: Path,
    env_dir: str | Path | None,
    url: str,
    auth: LlamaCppAuthMode | None,
    api_key_env: str | None,
    secret_ref: str | None,
    selected_model: str | None,
    requested_name: str | None,
    dry_run: bool,
    interactive: bool,
    interrogation_api_key: str | None = None,
    include_sampling_defaults: bool = False,
    preserve_existing_auth: bool = False,
    requested_action: _LlamaCppImportAction | None = None,
) -> _LlamaCppImportResult | None:
    if interrogation_api_key is None:
        interrogation_api_key = _resolve_llamacpp_interrogation_api_key(
            start_path=start_path,
            env_dir=env_dir,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
    catalog = await discover_llamacpp_models(
        url=url,
        api_key=interrogation_api_key,
    )
    selection = await _select_llamacpp_model(
        catalog=catalog,
        interrogation_api_key=interrogation_api_key,
        selected_model=selected_model,
        interactive=interactive,
        requested_action=requested_action,
    )
    if selection is None:
        return None
    if dry_run and selection.action != "generate_overlay":
        raise typer.BadParameter("Start-now actions are not available together with --dry-run.")
    model_id = selection.model_id

    discovered_model = await interrogate_llamacpp_model(
        catalog=catalog,
        model_id=model_id,
        api_key=interrogation_api_key,
    )
    overlay_name, replace_existing, reused_overlay = _build_llamacpp_overlay_name(
        requested_name=requested_name,
        model_id=model_id,
        base_url=catalog.endpoints.request_base_url,
        start_path=start_path,
        env_dir=env_dir,
    )
    persisted_auth = _resolve_llamacpp_persisted_auth(
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        reused_overlay=reused_overlay,
        preserve_existing_auth=preserve_existing_auth,
    )
    manifest = build_llamacpp_overlay_manifest(
        overlay_name=overlay_name,
        discovered_model=discovered_model,
        base_url=catalog.endpoints.request_base_url,
        auth=persisted_auth.auth,
        api_key_env=persisted_auth.api_key_env,
        secret_ref=persisted_auth.secret_ref,
        current=True,
        include_sampling_defaults=include_sampling_defaults,
    )
    manifest.connection.default_headers = dict(persisted_auth.default_headers)
    overlay_yaml = serialize_model_overlay_manifest(manifest)
    output_path = None
    if not dry_run:
        output_path = write_model_overlay_manifest(
            manifest,
            start_path=start_path,
            env_dir=env_dir,
            replace=replace_existing,
        )

    return _LlamaCppImportResult(
        catalog=catalog,
        discovered_model=discovered_model,
        action=selection.action,
        overlay_name=overlay_name,
        manifest_payload=manifest.model_dump(mode="json", exclude_none=True),
        overlay_yaml=overlay_yaml,
        output_path=output_path,
    )


def _emit_llamacpp_import_summary(
    result: _LlamaCppImportResult,
    *,
    include_sampling_defaults: bool,
    print_overlay_yaml: bool,
) -> None:
    context_window = result.discovered_model.effective_context_window
    context_source = result.discovered_model.context_window_source

    typer.echo(
        "Discovered llama.cpp model "
        f"{result.discovered_model.listing.model_id!r} from {result.catalog.models_url}."
    )
    if result.output_path is not None:
        typer.echo(f"Wrote overlay: {result.output_path}")
    else:
        typer.echo("Dry run only; no overlay files were written.")
    typer.echo(f"Overlay token: {result.overlay_name}")
    if context_window is not None:
        if context_source == "runtime":
            typer.echo(f"Context window: {context_window} (runtime /props)")
        elif context_source == "catalog":
            typer.echo(f"Context window: {context_window} (catalog fallback; /props reported none)")
    typer.echo(f'Use it now: fast-agent go --model {result.overlay_name} --message "hello"')
    if include_sampling_defaults and any(
        value is not None
        for value in (
            result.discovered_model.temperature,
            result.discovered_model.top_k,
            result.discovered_model.top_p,
            result.discovered_model.min_p,
        )
    ):
        typer.echo(
            "Note: the overlay copied the server's current sampling defaults. "
            "Review and edit the defaults block if you want different behavior."
        )
    if print_overlay_yaml:
        typer.echo()
        typer.echo(result.overlay_yaml.rstrip())


def _resolve_llamacpp_picker_import_defaults() -> _LlamaCppPickerImportDefaults:
    return _LlamaCppPickerImportDefaults(
        url=DEFAULT_LLAMA_CPP_URL,
        auth="none",
        interrogation_api_key=None,
    )


async def import_llamacpp_overlay_from_default_url(
    *,
    start_path: Path,
    env_dir: str | Path | None,
) -> str | None:
    picker_defaults = _resolve_llamacpp_picker_import_defaults()
    result = await _run_llamacpp_import(
        start_path=start_path,
        env_dir=env_dir,
        url=picker_defaults.url,
        auth=picker_defaults.auth,
        api_key_env=None,
        secret_ref=None,
        selected_model=None,
        requested_name=None,
        dry_run=False,
        interactive=True,
        interrogation_api_key=picker_defaults.interrogation_api_key,
        preserve_existing_auth=True,
        requested_action="generate_overlay",
    )
    if result is None:
        return None
    return result.overlay_name


def _finalize_llamacpp_import(
    *,
    result: _LlamaCppImportResult | None,
    resolved_env_dir: Path | None,
    include_sampling_defaults: bool = False,
    json_output: bool = False,
    print_overlay_yaml: bool = False,
) -> None:
    if result is None:
        typer.echo("llama.cpp import cancelled.")
        raise typer.Exit(0)

    if json_output:
        typer.echo(json.dumps(_llamacpp_import_json_payload(result), indent=2))
    else:
        _emit_llamacpp_import_summary(
            result,
            include_sampling_defaults=include_sampling_defaults,
            print_overlay_yaml=print_overlay_yaml,
        )

    launch_options = _LLAMACPP_LAUNCH_OPTIONS_BY_ACTION.get(result.action)
    if launch_options is not None:
        _launch_llamacpp_overlay_now(
            overlay_name=result.overlay_name,
            env_dir=resolved_env_dir,
            with_shell=launch_options.with_shell,
            smart=launch_options.smart,
            announce=not json_output,
        )


def _run_llamacpp_noninteractive_command(
    *,
    ctx: typer.Context,
    env: str | None,
    url: str,
    auth: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
    model_id: str,
    name: str | None,
    dry_run: bool,
    requested_action: _LlamaCppImportAction,
    include_sampling_defaults: bool = False,
    preserve_existing_auth: bool = False,
    json_output: bool = False,
    print_overlay_yaml: bool = False,
) -> None:
    try:
        command_context = _resolve_llamacpp_command_context(
            ctx=ctx,
            env=env,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
        resolved_auth = _normalize_llamacpp_auth(
            auth=auth,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
        result = run_coroutine(
            _run_llamacpp_import(
                start_path=command_context.start_path,
                env_dir=command_context.resolved_env_dir,
                url=url,
                auth=resolved_auth,
                api_key_env=api_key_env,
                secret_ref=secret_ref,
                selected_model=model_id,
                requested_name=name,
                dry_run=dry_run,
                interactive=False,
                include_sampling_defaults=include_sampling_defaults,
                preserve_existing_auth=preserve_existing_auth,
                requested_action=requested_action,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc
    except (LlamaCppDiscoveryError, ValueError, FileExistsError, typer.BadParameter) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    _finalize_llamacpp_import(
        result=result,
        resolved_env_dir=command_context.resolved_env_dir,
        include_sampling_defaults=include_sampling_defaults,
        json_output=json_output,
        print_overlay_yaml=print_overlay_yaml,
    )


def _build_llamacpp_start_now_argv(
    *,
    overlay_name: str,
    env_dir: Path | None,
    with_shell: bool,
    smart: bool,
) -> list[str]:
    argv = [sys.executable, "-m", "fast_agent.cli", "go", "--model", overlay_name]
    if smart:
        argv.append("--smart")
    if with_shell:
        argv.append("-x")
    if env_dir is not None:
        argv.extend(["--env", str(env_dir)])
    return argv


def _launch_llamacpp_overlay_now(
    *,
    overlay_name: str,
    env_dir: Path | None,
    with_shell: bool = False,
    smart: bool = False,
    announce: bool = True,
    execvpe_fn: Callable[[str, list[str], dict[str, str]], object] = os.execvpe,
) -> None:
    argv = _build_llamacpp_start_now_argv(
        overlay_name=overlay_name,
        env_dir=env_dir,
        with_shell=with_shell,
        smart=smart,
    )
    if announce:
        typer.echo(f"Launching: {join_commandline(argv, syntax='posix')}")
        sys.stdout.flush()
    execvpe_fn(sys.executable, argv, os.environ.copy())


@app.callback(invoke_without_command=True)
def model_main(ctx: typer.Context) -> None:
    """Manage interactive model setup flows."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("export", help="Export a model-database entry as a local overlay manifest.")
def model_export(
    model: str | None = typer.Argument(
        None,
        help="Model name from the catalog (e.g. claude-4-sonnet-20250514). "
        "Omit to choose interactively from the model picker.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        "-n",
        help="Name for the generated overlay (defaults to a sanitized model name).",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Override the provider for the overlay.",
    ),
    env: str | None = CommonAgentOptions.env_dir(),
    replace: bool = typer.Option(
        False,
        "--replace",
        "-r",
        help="Overwrite an existing overlay with the same name.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Print the generated manifest without writing it.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Emit a machine-readable JSON result.",
    ),
) -> None:
    """Export a ModelDatabase entry to a user-writable overlay manifest.

    The generated overlay pre-populates model_specific, modalities (tokenizes),
    context window, and other parameters so users can customize them locally.
    """

    run_coroutine(
        _run_model_export_command(
            model=model,
            name=name,
            provider=provider,
            env=env,
            replace=replace,
            dry_run=dry_run,
            json_output=json_output,
        )
    )


def _resolve_model_export_provider(provider: str | None) -> Provider | None:
    if not provider:
        return None
    try:
        return Provider(strip_casefold(provider))
    except ValueError as exc:
        typer.echo(f"Unknown provider: {provider}", err=True)
        raise typer.Exit(1) from exc


async def _select_model_for_export(model: str | None) -> str:
    if model:
        return model

    picker_result = await run_model_picker_async()
    if picker_result is None:
        typer.echo("No model selected.", err=True)
        raise typer.Exit(1)

    selected_model = picker_result.resolved_model or picker_result.selected_model
    if selected_model:
        return selected_model

    typer.echo("No model selected.", err=True)
    raise typer.Exit(1)


def _build_model_export_manifest(
    *,
    selected_model: str,
    provider: Provider | None,
    overlay_name: str | None,
) -> Any:
    try:
        return build_model_overlay_manifest_from_database(
            selected_model,
            provider=provider,
            overlay_name=overlay_name,
        )
    except Exception as exc:
        typer.echo(f"Failed to build overlay for '{selected_model}': {exc}", err=True)
        raise typer.Exit(1) from exc


def _emit_model_export_dry_run(*, manifest: Any, yaml_text: str, json_output: bool) -> None:
    if json_output:
        payload = {
            "overlay_name": manifest.name,
            "dry_run": True,
            "manifest": manifest.model_dump(mode="json", exclude_none=True),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"# Dry-run: would write overlay '{manifest.name}'")
    typer.echo(yaml_text)


def _write_model_export_manifest(
    *,
    manifest: Any,
    env: str | None,
    replace: bool,
) -> Path:
    try:
        return write_model_overlay_manifest(
            manifest,
            env_dir=env,
            replace=replace,
        )
    except FileExistsError as exc:
        typer.echo(str(exc), err=True)
        typer.echo("Use --replace to overwrite.", err=True)
        raise typer.Exit(1) from exc


def _emit_model_export_result(*, manifest: Any, out_path: Path, json_output: bool) -> None:
    if json_output:
        payload = {
            "overlay_name": manifest.name,
            "path": str(out_path),
            "dry_run": False,
            "manifest": manifest.model_dump(mode="json", exclude_none=True),
        }
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Wrote overlay: {out_path}")
    typer.echo(f"Use: fast-agent go --model {manifest.name}")


async def _run_model_export_command(
    *,
    model: str | None,
    name: str | None,
    provider: str | None,
    env: str | None,
    replace: bool,
    dry_run: bool,
    json_output: bool,
) -> None:
    resolved_provider = _resolve_model_export_provider(provider)
    selected_model = await _select_model_for_export(model)
    manifest = _build_model_export_manifest(
        selected_model=selected_model,
        provider=resolved_provider,
        overlay_name=name,
    )
    yaml_text = serialize_model_overlay_manifest(manifest)

    if dry_run:
        _emit_model_export_dry_run(
            manifest=manifest,
            yaml_text=yaml_text,
            json_output=json_output,
        )
        return

    out_path = _write_model_export_manifest(
        manifest=manifest,
        env=env,
        replace=replace,
    )
    _emit_model_export_result(manifest=manifest, out_path=out_path, json_output=json_output)


def _model_presets_provider_filter(raw_provider: str | None) -> Provider | None:
    if raw_provider is None:
        return None

    normalized = normalize_action_token(raw_provider)
    aliases = {
        "codex-responses": "codexresponses",
        "codex_responses": "codexresponses",
        "huggingface": "hf",
        "anthropicvertex": "anthropic-vertex",
    }
    provider_name = aliases.get(normalized, normalized)
    try:
        return Provider(provider_name)
    except ValueError as exc:
        raise typer.BadParameter(
            f"Unknown provider: {raw_provider}", param_hint="--provider"
        ) from exc


def _model_presets_key_status(
    provider: Provider,
    *,
    api_keys: dict[str, dict[str, str]],
) -> tuple[str, str]:
    provider_name = provider.config_name
    if provider in {Provider.FAST_AGENT, Provider.GENERIC, Provider.ANTHROPIC_VERTEX}:
        return ("available", "[green]keyless[/green]")

    provider_status = api_keys.get(provider_name, {})
    if provider_status.get("env"):
        return ("available", "[green]env[/green]")
    if provider_status.get("config"):
        label = provider_status["config"]
        return ("available", f"[green]{label}[/green]")

    env_var = ProviderKeyManager.get_env_key_name(provider_name)
    missing = f"missing {env_var}" if env_var else "missing"
    return ("missing", f"[red]{missing}[/red]")


def _model_preset_rows(
    *,
    provider_filter: Provider | None,
    env_dir: Path | None,
) -> list[dict[str, Any]]:
    from fast_agent.cli.commands.check_config import (
        check_api_keys,
        find_config_files,
        get_config_summary,
        get_secrets_summary,
    )

    config_files = find_config_files(Path.cwd(), env_dir=env_dir)
    config_summary = get_config_summary(config_files["config"])
    secrets_summary = get_secrets_summary(config_files["secrets"])
    api_keys = check_api_keys(secrets_summary, config_summary)
    presets = ModelFactory.get_runtime_presets()
    rows: list[dict[str, Any]] = []

    for alias, model_spec in sorted(presets.items()):
        try:
            parsed = ModelFactory.parse_model_spec(alias, presets=presets)
        except Exception as exc:
            if provider_filter is not None:
                continue
            rows.append(
                {
                    "alias": alias,
                    "provider": None,
                    "provider_display": "unresolved",
                    "model": model_spec,
                    "expanded": model_spec,
                    "key_status": "error",
                    "key_status_text": f"[red]{exc}[/red]",
                }
            )
            continue

        if provider_filter is not None and parsed.provider != provider_filter:
            continue

        key_status, key_status_text = _model_presets_key_status(
            parsed.provider,
            api_keys=api_keys,
        )
        rows.append(
            {
                "alias": alias,
                "provider": parsed.provider.config_name,
                "provider_display": parsed.provider.display_name,
                "model": parsed.model_name,
                "expanded": model_spec,
                "key_status": key_status,
                "key_status_text": key_status_text,
            }
        )

    return rows


def _render_model_presets(rows: list[dict[str, Any]], *, provider: str | None) -> None:
    title = "Model presets" if provider is None else f"Model presets ({provider})"
    console.print(f"\n[bold blue]{title}[/bold blue]\n")

    table = Table(show_header=True, box=None)
    table.add_column("Preset", style="cyan", header_style="bold bright_white")
    table.add_column("Provider", style="white", header_style="bold bright_white")
    table.add_column("Key", header_style="bold bright_white")
    table.add_column(
        "Maps To",
        style="green",
        header_style="bold bright_white",
        overflow="fold",
    )

    for row in rows:
        table.add_row(
            row["alias"],
            row["provider_display"],
            row["key_status_text"],
            row["expanded"],
        )

    console.print(table)


@app.command("presets")
def model_presets(
    ctx: typer.Context,
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Only show presets that resolve to this provider.",
    ),
    env: str | None = CommonAgentOptions.env_dir(),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON result.",
    ),
) -> None:
    """List built-in and runtime model presets with provider/key readiness."""
    resolved_env_dir = resolve_environment_dir_option(
        ctx,
        Path(env) if env is not None else None,
    )
    provider_filter = _model_presets_provider_filter(provider)
    rows = _model_preset_rows(provider_filter=provider_filter, env_dir=resolved_env_dir)

    if json_output:
        console.print(json.dumps(rows, indent=2, sort_keys=True))
        return

    if not rows:
        provider_text = f" for provider '{provider}'" if provider is not None else ""
        console.print(f"[yellow]No model presets found{provider_text}.[/yellow]")
        return

    _render_model_presets(rows, provider=provider)


@app.command("setup")
def model_setup(
    ctx: typer.Context,
    token: str | None = typer.Argument(
        None,
        help="Reference token to update, such as $system.fast. Omit to choose or create one interactively.",
    ),
    env: str | None = CommonAgentOptions.env_dir(),
    target: str = typer.Option(
        "env",
        "--target",
        help="Where to save reference changes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview changes without writing files.",
    ),
) -> None:
    """Interactively create or update a model reference using the model selector."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo("fast-agent model setup requires an interactive terminal.", err=True)
        raise typer.Exit(1)

    resolved_env_dir = resolve_environment_dir_option(
        ctx,
        Path(env) if env is not None else None,
    )
    resolved_target = _normalize_write_target(target)
    settings = (
        Settings(environment_dir=str(resolved_env_dir))
        if resolved_env_dir is not None
        else Settings()
    )

    try:
        run_coroutine(
            _run_model_setup_command(
                settings=settings,
                token=token,
                target=resolved_target,
                dry_run=dry_run,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc


@app.command("doctor")
def model_doctor(
    ctx: typer.Context,
    env: str | None = CommonAgentOptions.env_dir(),
) -> None:
    """Inspect model onboarding readiness and reference resolution."""
    resolved_env_dir = resolve_environment_dir_option(
        ctx,
        Path(env) if env is not None else None,
    )
    settings = _load_cli_settings(
        cwd=_bootstrap_settings_start_path(resolved_env_dir),
        env_dir=resolved_env_dir,
    )

    try:
        run_coroutine(
            _run_model_doctor_command(
                settings=settings,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc


@llamacpp_app.callback(invoke_without_command=True)
def model_llamacpp(
    ctx: typer.Context,
    env: str | None = _llamacpp_env_option(),
    url: str = _llamacpp_url_option(),
    auth: str | None = _llamacpp_auth_option(),
    api_key_env: str | None = _llamacpp_api_key_env_option(),
    secret_ref: str | None = _llamacpp_secret_ref_option(),
    name: str | None = _llamacpp_name_option(),
    include_sampling_defaults: bool = _llamacpp_include_sampling_defaults_option(),
) -> None:
    """Interactively choose a llama.cpp model and import it as a local overlay."""

    _store_llamacpp_group_options(
        ctx,
        env=env,
        url=url,
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        name=name,
        include_sampling_defaults=include_sampling_defaults,
    )
    if ctx.invoked_subcommand is not None:
        return
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo("fast-agent model llamacpp requires an interactive terminal.", err=True)
        raise typer.Exit(1)

    try:
        command_context = _resolve_llamacpp_command_context(
            ctx=ctx,
            env=env,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
        resolved_auth = _normalize_llamacpp_auth(
            auth=auth,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )

        result = run_coroutine(
            _run_llamacpp_import(
                start_path=command_context.start_path,
                env_dir=command_context.resolved_env_dir,
                url=url,
                auth=resolved_auth,
                api_key_env=api_key_env,
                secret_ref=secret_ref,
                selected_model=None,
                requested_name=name,
                dry_run=False,
                interactive=True,
                include_sampling_defaults=include_sampling_defaults,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc
    except (LlamaCppDiscoveryError, ValueError, FileExistsError, typer.BadParameter) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    _finalize_llamacpp_import(
        result=result,
        resolved_env_dir=command_context.resolved_env_dir,
        include_sampling_defaults=include_sampling_defaults,
    )


@llamacpp_app.command("list")
def model_llamacpp_list(
    ctx: typer.Context,
    env: str | None = _llamacpp_env_option(),
    url: str = _llamacpp_url_option(),
    api_key_env: str | None = _llamacpp_api_key_env_option(discovery_only=True),
    secret_ref: str | None = _llamacpp_secret_ref_option(discovery_only=True),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable discovery output.",
    ),
) -> None:
    """List models discovered from a llama.cpp server."""

    env = _inherit_llamacpp_group_option(ctx, option_name="env", value=env)
    url = _inherit_llamacpp_group_url(ctx, url=url)
    api_key_env = _inherit_llamacpp_group_option(ctx, option_name="api_key_env", value=api_key_env)
    secret_ref = _inherit_llamacpp_group_option(ctx, option_name="secret_ref", value=secret_ref)
    try:
        command_context = _resolve_llamacpp_command_context(
            ctx=ctx,
            env=env,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
        catalog = run_coroutine(
            discover_llamacpp_models(
                url=url,
                api_key=command_context.interrogation_api_key,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc
    except (LlamaCppDiscoveryError, ValueError, typer.BadParameter) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(json.dumps(_llamacpp_catalog_json_payload(catalog), indent=2))
        return
    _emit_llamacpp_catalog_listing(catalog)


@llamacpp_app.command("preview")
def model_llamacpp_preview(
    ctx: typer.Context,
    model_id: str = typer.Argument(
        ...,
        help="Model ID to preview as an overlay.",
    ),
    env: str | None = _llamacpp_env_option(),
    url: str = _llamacpp_url_option(),
    auth: str | None = _llamacpp_auth_option(),
    api_key_env: str | None = _llamacpp_api_key_env_option(),
    secret_ref: str | None = _llamacpp_secret_ref_option(),
    name: str | None = _llamacpp_name_option(),
    include_sampling_defaults: bool = _llamacpp_include_sampling_defaults_option(),
) -> None:
    """Preview the generated overlay YAML for a discovered llama.cpp model."""

    env = _inherit_llamacpp_group_option(ctx, option_name="env", value=env)
    url = _inherit_llamacpp_group_url(ctx, url=url)
    auth = _inherit_llamacpp_group_option(ctx, option_name="auth", value=auth)
    api_key_env = _inherit_llamacpp_group_option(ctx, option_name="api_key_env", value=api_key_env)
    secret_ref = _inherit_llamacpp_group_option(ctx, option_name="secret_ref", value=secret_ref)
    name = _inherit_llamacpp_group_option(ctx, option_name="name", value=name)
    include_sampling_defaults = _inherit_llamacpp_group_bool_option(
        ctx,
        option_name="include_sampling_defaults",
        value=include_sampling_defaults,
    )
    _run_llamacpp_noninteractive_command(
        ctx=ctx,
        env=env,
        url=url,
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        model_id=model_id,
        name=name,
        dry_run=True,
        requested_action="generate_overlay",
        include_sampling_defaults=include_sampling_defaults,
        print_overlay_yaml=True,
    )


@llamacpp_app.command("import")
def model_llamacpp_import(
    ctx: typer.Context,
    model_id: str = typer.Argument(
        ...,
        help="Model ID to import as an overlay.",
    ),
    env: str | None = _llamacpp_env_option(),
    url: str = _llamacpp_url_option(),
    auth: str | None = _llamacpp_auth_option(),
    api_key_env: str | None = _llamacpp_api_key_env_option(),
    secret_ref: str | None = _llamacpp_secret_ref_option(),
    name: str | None = _llamacpp_name_option(),
    include_sampling_defaults: bool = _llamacpp_include_sampling_defaults_option(),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable import output.",
    ),
    start_now: bool = typer.Option(
        False,
        "--start-now",
        help="Immediately launch fast-agent go with the imported overlay.",
    ),
    with_shell: bool = typer.Option(
        False,
        "--with-shell",
        help="Use fast-agent go -x when launching with --start-now.",
    ),
    smart: bool = typer.Option(
        False,
        "--smart",
        help="Use fast-agent go --smart -x when launching with --start-now.",
    ),
) -> None:
    """Import a discovered llama.cpp model as a local overlay."""

    env = _inherit_llamacpp_group_option(ctx, option_name="env", value=env)
    url = _inherit_llamacpp_group_url(ctx, url=url)
    auth = _inherit_llamacpp_group_option(ctx, option_name="auth", value=auth)
    api_key_env = _inherit_llamacpp_group_option(ctx, option_name="api_key_env", value=api_key_env)
    secret_ref = _inherit_llamacpp_group_option(ctx, option_name="secret_ref", value=secret_ref)
    name = _inherit_llamacpp_group_option(ctx, option_name="name", value=name)
    include_sampling_defaults = _inherit_llamacpp_group_bool_option(
        ctx,
        option_name="include_sampling_defaults",
        value=include_sampling_defaults,
    )
    if with_shell and not start_now:
        raise typer.BadParameter("--with-shell requires --start-now.")
    if smart and not start_now:
        raise typer.BadParameter("--smart requires --start-now.")

    preserve_existing_auth = not any(
        _llamacpp_option_was_explicit(ctx, option_name=option_name)
        for option_name in ("auth", "api_key_env", "secret_ref")
    )
    requested_action: _LlamaCppImportAction
    if smart:
        requested_action = "start_now_smart"
    elif with_shell:
        requested_action = "start_now_with_shell"
    elif start_now:
        requested_action = "start_now"
    else:
        requested_action = "generate_overlay"

    _run_llamacpp_noninteractive_command(
        ctx=ctx,
        env=env,
        url=url,
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        model_id=model_id,
        name=name,
        dry_run=False,
        requested_action=requested_action,
        include_sampling_defaults=include_sampling_defaults,
        preserve_existing_auth=preserve_existing_auth,
        json_output=json_output,
    )
