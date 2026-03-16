"""Interactive CLI helpers for model reference setup."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

import typer
from pydantic import ValidationError
from rich.text import Text

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.commands.context import CommandContext, CommandIO, StaticAgentProvider
from fast_agent.commands.handlers import models_manager
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.config import (
    Settings,
    deep_merge,
    find_fastagent_config_files,
    load_layered_settings,
    load_yaml_mapping,
    resolve_config_search_root,
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
from fast_agent.llm.model_overlays import (
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
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.llamacpp_model_picker import run_llamacpp_model_picker_async
from fast_agent.ui.model_reference_picker import (
    ModelReferencePickerItem,
    run_model_reference_picker_async,
)

type WriteTarget = Literal["env", "project"]
type LlamaCppAuthMode = Literal["none", "env", "secret_ref"]

app = typer.Typer(help="Interactive model reference setup.")
llamacpp_app = typer.Typer(
    help="Discover llama.cpp models, preview overlays, and import local runtime overlays."
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
    if token is not None and token.strip():
        parts.append(shlex.quote(token.strip()))
    parts.extend(["--target", target])
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def _normalize_write_target(value: str) -> WriteTarget:
    normalized = value.strip().lower()
    if normalized == "env":
        return "env"
    if normalized == "project":
        return "project"
    raise typer.BadParameter("--target must be either 'env' or 'project'.")


def _normalize_interactive_reference_token(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip()
    if not stripped:
        return stripped
    if stripped.startswith("$"):
        return stripped
    return f"${stripped}"


def _bootstrap_settings_start_path(env_dir: str | Path | None) -> Path:
    if isinstance(env_dir, str) and env_dir.strip():
        env_root = Path(env_dir).expanduser()
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
            env_dir=getattr(settings, "environment_dir", None),
        )
        has_guided_choices = bool(diagnostics.items) or (
            isinstance(io, TuiCommandIO)
            and bool(_build_common_setup_items(diagnostics.valid_references))
        )
        resolved_token = await _select_model_setup_token(
            io,
            diagnostics=diagnostics,
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
        start_path = _bootstrap_settings_start_path(getattr(settings, "environment_dir", None))
        effective_settings = _load_cli_settings(
            cwd=start_path,
            env_dir=getattr(settings, "environment_dir", None),
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
) -> str | None:
    items = diagnostics.items
    common_items = _build_common_setup_items(diagnostics.valid_references)
    if not items:
        if isinstance(io, TuiCommandIO) and common_items:
            return await _pick_or_prompt_reference_token(
                io,
                items=common_items,
            )
        return None

    if isinstance(io, TuiCommandIO):
        return await _pick_or_prompt_reference_token(
            io,
            items=_merge_setup_items(items, common_items),
        )

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
    option_labels = {
        str(index): item.token
        for index, item in enumerate(items, start=1)
    }
    selection = await io.prompt_selection(
        "Reference to configure (number or 'custom'):",
        options=[*option_labels.keys(), "custom"],
        allow_cancel=True,
    )
    if selection is None:
        return None

    normalized_selection = selection.strip().lower()
    if normalized_selection == "custom":
        return await _prompt_manual_reference_token(io)
    return option_labels.get(normalized_selection)


async def _pick_or_prompt_reference_token(
    io: TuiCommandIO,
    *,
    items: tuple[ModelReferenceSetupItem, ...],
) -> str | None:
    picker_items = tuple(
        ModelReferencePickerItem(
            token=item.token,
            priority=item.priority,
            status=f"{item.priority}/{item.status}",
            summary=item.summary,
            current_value=item.current_value,
            references=item.references,
            removable=False,
        )
        for item in items
    )
    result = await run_model_reference_picker_async(picker_items)
    if result is None:
        return None
    if result.action == "custom":
        return await _prompt_manual_reference_token(io)
    return result.token


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
    if "default" not in system_references and "$system.default" not in hidden_tokens:
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
    argument = f"unset {shlex.quote(token)} --target {target}"
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
    config_payload = _load_tolerant_config_payload(
        cwd=start_path,
        env_dir=getattr(settings, "environment_dir", None),
    )
    provider = StaticAgentProvider()
    io = TuiCommandIO(
        prompt_provider=provider,
        agent_name="cli",
        settings=settings,
        config_payload=config_payload,
    )
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
        diagnostics = collect_model_reference_setup_diagnostics(
            cwd=start_path,
            env_dir=getattr(settings, "environment_dir", None),
        )
        picker_items = _build_picker_items(
            diagnostics,
            suppressed_tokens=suppressed_tokens,
        )
        picker_result = await run_model_reference_picker_async(picker_items)
        if picker_result is None:
            return
        if picker_result.action == "done":
            return

        if picker_result.action == "custom":
            selected_token = await _prompt_manual_reference_token(io)
            if selected_token is None:
                return
            outcome = await run_model_setup(
                io=io,
                settings=settings,
                token=selected_token,
                target=target,
                dry_run=dry_run,
            )
        elif picker_result.action == "unset":
            assert picker_result.token is not None
            outcome = await _run_model_reference_unset(
                io=io,
                settings=settings,
                token=picker_result.token,
                target=target,
                dry_run=dry_run,
            )
            if dry_run:
                for message in outcome.messages:
                    await io.emit(message)
                return

            suppressed_tokens.add(picker_result.token)
            for message in outcome.messages:
                if message.channel in {"warning", "error"}:
                    await io.emit(message)
            continue
        else:
            assert picker_result.token is not None
            suppressed_tokens.discard(picker_result.token)
            outcome = await run_model_setup(
                io=io,
                settings=settings,
                token=picker_result.token,
                target=target,
                dry_run=dry_run,
            )

        for message in outcome.messages:
            await io.emit(message)
        if dry_run:
            return


async def _run_model_doctor_command(*, settings: Settings) -> None:
    start_path = resolve_model_reference_start_path(settings=settings)
    provider = StaticAgentProvider()
    io = TuiCommandIO(
        prompt_provider=provider,
        agent_name="cli",
        settings=settings,
        config_payload=_load_tolerant_config_payload(
            cwd=start_path,
            env_dir=getattr(settings, "environment_dir", None),
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
    merged_settings, config_file = load_layered_settings(start_path=cwd, env_dir=env_dir)
    search_root = resolve_config_search_root(cwd, env_dir=env_dir)
    _, secrets_path = find_fastagent_config_files(search_root)
    if secrets_path and secrets_path.exists():
        merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))

    settings = Settings(**merged_settings)
    settings._config_file = str(config_file) if config_file else None
    settings._secrets_file = str(secrets_path) if secrets_path and secrets_path.exists() else None
    return settings


def _load_tolerant_config_payload(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> dict[str, object] | None:
    try:
        merged_settings, _ = load_layered_settings(start_path=cwd, env_dir=env_dir)
        search_root = resolve_config_search_root(cwd, env_dir=env_dir)
        _, secrets_path = find_fastagent_config_files(search_root)
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
    action: Literal["start_now", "start_now_with_shell", "generate_overlay"]
    overlay_name: str
    manifest_payload: dict[str, object]
    overlay_yaml: str
    output_path: Path | None


@dataclass(frozen=True, slots=True)
class _LlamaCppSelection:
    model_id: str
    action: Literal["start_now", "start_now_with_shell", "generate_overlay"]


@dataclass(frozen=True, slots=True)
class _LlamaCppCommandContext:
    resolved_env_dir: Path | None
    start_path: Path
    interrogation_api_key: str | None


@dataclass(frozen=True, slots=True)
class _LlamaCppGroupOptions:
    env: str | None
    url: str
    auth: str | None
    api_key_env: str | None
    secret_ref: str | None
    name: str | None


def _store_llamacpp_group_options(
    ctx: typer.Context,
    *,
    env: str | None,
    url: str,
    auth: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
    name: str | None,
) -> None:
    payload = _LlamaCppGroupOptions(
        env=env,
        url=url,
        auth=auth,
        api_key_env=api_key_env,
        secret_ref=secret_ref,
        name=name,
    )
    if isinstance(ctx.obj, dict):
        ctx.obj["llamacpp_group_options"] = payload
    else:
        ctx.obj = {"llamacpp_group_options": payload}


def _inherit_llamacpp_group_option(
    ctx: typer.Context,
    *,
    option_name: str,
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
    return cast("str | None", getattr(payload, option_name))


def _inherit_llamacpp_group_url(ctx: typer.Context, *, url: str) -> str:
    inherited = _inherit_llamacpp_group_option(ctx, option_name="url", value=url)
    return inherited if inherited is not None else url


def _normalize_llamacpp_auth(
    *,
    auth: str | None,
    api_key_env: str | None,
    secret_ref: str | None,
) -> LlamaCppAuthMode:
    normalized_env = api_key_env.strip() if api_key_env else None
    normalized_secret_ref = secret_ref.strip() if secret_ref else None

    if auth is not None:
        resolved_auth = auth.strip().lower()
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
    normalized_env = api_key_env.strip() if api_key_env else None
    if normalized_env:
        value = os.getenv(normalized_env)
        if value is None:
            raise typer.BadParameter(
                f"Environment variable {normalized_env!r} is required for llama.cpp interrogation."
            )
        return value

    normalized_secret_ref = secret_ref.strip() if secret_ref else None
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
    selected_model: str | None,
    interactive: bool,
    requested_action: Literal["start_now", "start_now_with_shell", "generate_overlay"] | None,
) -> _LlamaCppSelection | None:
    if selected_model is not None and selected_model.strip():
        requested = selected_model.strip()
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

    picker_result = await run_llamacpp_model_picker_async(catalog.models)
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
    start_path: Path,
    env_dir: str | Path | None,
) -> str:
    registry = load_model_overlay_registry(start_path=start_path, env_dir=env_dir)
    existing_names = set(registry.by_name())

    if requested_name is not None and requested_name.strip():
        candidate = requested_name.strip()
    else:
        candidate = default_overlay_name_for_model(model_id)

    return uniquify_overlay_name(candidate, existing_names=existing_names)


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
    typer.echo(f"Discovered {len(catalog.models)} llama.cpp model(s):")
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
    auth: LlamaCppAuthMode,
    api_key_env: str | None,
    secret_ref: str | None,
    selected_model: str | None,
    requested_name: str | None,
    dry_run: bool,
    interactive: bool,
    requested_action: Literal["start_now", "start_now_with_shell", "generate_overlay"] | None = None,
) -> _LlamaCppImportResult | None:
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
        selected_model=selected_model,
        interactive=interactive,
        requested_action=requested_action,
    )
    if selection is None:
        return None
    if dry_run and selection.action != "generate_overlay":
        raise typer.BadParameter(
            "Start-now actions are not available together with --dry-run."
        )
    model_id = selection.model_id

    discovered_model = await interrogate_llamacpp_model(
        catalog=catalog,
        model_id=model_id,
        api_key=interrogation_api_key,
    )
    overlay_name = _build_llamacpp_overlay_name(
        requested_name=requested_name,
        model_id=model_id,
        start_path=start_path,
        env_dir=env_dir,
    )
    manifest = build_llamacpp_overlay_manifest(
        overlay_name=overlay_name,
        discovered_model=discovered_model,
        base_url=catalog.endpoints.request_base_url,
        auth=auth,
        api_key_env=api_key_env.strip() if api_key_env else None,
        secret_ref=secret_ref.strip() if secret_ref else None,
        current=True,
    )
    overlay_yaml = serialize_model_overlay_manifest(manifest)
    output_path = None
    if not dry_run:
        output_path = write_model_overlay_manifest(
            manifest,
            start_path=start_path,
            env_dir=env_dir,
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
    print_overlay_yaml: bool,
) -> None:
    typer.echo(
        "Discovered llama.cpp model "
        f"{result.discovered_model.listing.model_id!r} from {result.catalog.models_url}."
    )
    if result.output_path is not None:
        typer.echo(f"Wrote overlay: {result.output_path}")
    else:
        typer.echo("Dry run only; no overlay files were written.")
    typer.echo(f"Overlay token: {result.overlay_name}")
    typer.echo(
        f'Use it now: fast-agent go --model {result.overlay_name} --message "hello"'
    )
    if print_overlay_yaml:
        typer.echo()
        typer.echo(result.overlay_yaml.rstrip())


def _finalize_llamacpp_import(
    *,
    result: _LlamaCppImportResult | None,
    resolved_env_dir: Path | None,
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
            print_overlay_yaml=print_overlay_yaml,
        )
    if result.action == "start_now":
        _launch_llamacpp_overlay_now(
            overlay_name=result.overlay_name,
            env_dir=resolved_env_dir,
            announce=not json_output,
        )
    if result.action == "start_now_with_shell":
        _launch_llamacpp_overlay_now(
            overlay_name=result.overlay_name,
            env_dir=resolved_env_dir,
            with_shell=True,
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
    requested_action: Literal["start_now", "start_now_with_shell", "generate_overlay"],
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
        result = asyncio.run(
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
        json_output=json_output,
        print_overlay_yaml=print_overlay_yaml,
    )


def _build_llamacpp_start_now_argv(
    *,
    overlay_name: str,
    env_dir: Path | None,
    with_shell: bool,
) -> list[str]:
    argv = [sys.executable, "-m", "fast_agent.cli", "go", "--model", overlay_name]
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
    announce: bool = True,
    execvpe_fn: Callable[[str, list[str], dict[str, str]], object] = os.execvpe,
) -> None:
    argv = _build_llamacpp_start_now_argv(
        overlay_name=overlay_name,
        env_dir=env_dir,
        with_shell=with_shell,
    )
    if announce:
        typer.echo(f"Launching: {' '.join(shlex.quote(part) for part in argv)}")
        sys.stdout.flush()
    execvpe_fn(sys.executable, argv, os.environ.copy())


@app.callback(invoke_without_command=True)
def model_main(ctx: typer.Context) -> None:
    """Manage interactive model setup flows."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


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
        asyncio.run(
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
        asyncio.run(
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

        result = asyncio.run(
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
    api_key_env = _inherit_llamacpp_group_option(
        ctx, option_name="api_key_env", value=api_key_env
    )
    secret_ref = _inherit_llamacpp_group_option(
        ctx, option_name="secret_ref", value=secret_ref
    )
    try:
        command_context = _resolve_llamacpp_command_context(
            ctx=ctx,
            env=env,
            api_key_env=api_key_env,
            secret_ref=secret_ref,
        )
        catalog = asyncio.run(
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
) -> None:
    """Preview the generated overlay YAML for a discovered llama.cpp model."""

    env = _inherit_llamacpp_group_option(ctx, option_name="env", value=env)
    url = _inherit_llamacpp_group_url(ctx, url=url)
    auth = _inherit_llamacpp_group_option(ctx, option_name="auth", value=auth)
    api_key_env = _inherit_llamacpp_group_option(
        ctx, option_name="api_key_env", value=api_key_env
    )
    secret_ref = _inherit_llamacpp_group_option(
        ctx, option_name="secret_ref", value=secret_ref
    )
    name = _inherit_llamacpp_group_option(ctx, option_name="name", value=name)
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
) -> None:
    """Import a discovered llama.cpp model as a local overlay."""

    env = _inherit_llamacpp_group_option(ctx, option_name="env", value=env)
    url = _inherit_llamacpp_group_url(ctx, url=url)
    auth = _inherit_llamacpp_group_option(ctx, option_name="auth", value=auth)
    api_key_env = _inherit_llamacpp_group_option(
        ctx, option_name="api_key_env", value=api_key_env
    )
    secret_ref = _inherit_llamacpp_group_option(
        ctx, option_name="secret_ref", value=secret_ref
    )
    name = _inherit_llamacpp_group_option(ctx, option_name="name", value=name)
    if with_shell and not start_now:
        raise typer.BadParameter("--with-shell requires --start-now.")

    requested_action: Literal["start_now", "start_now_with_shell", "generate_overlay"]
    if with_shell:
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
        json_output=json_output,
    )
