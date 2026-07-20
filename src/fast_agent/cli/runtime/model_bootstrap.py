"""Model bootstrap and picker helpers for CLI runtime startup."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import typer

from fast_agent.cli.command_support import get_settings_or_exit
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from fast_agent.config import Settings
    from fast_agent.core.model_resolution import ResolvedModelSpec
    from fast_agent.ui.model_picker_common import ProviderActivation

    from .run_request import AgentRunRequest

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelPickerInitialSelection:
    provider: str | None = None
    model_spec: str | None = None


def should_prompt_for_model_picker(
    request: AgentRunRequest,
    *,
    stdin_is_tty: bool,
    stdout_is_tty: bool,
) -> bool:
    """Return True when interactive startup can safely prompt for model selection."""
    if request.mode != "interactive" or not request.is_repl:
        return False
    return stdin_is_tty and stdout_is_tty


def explicit_agent_cards_define_startup_model(
    request: AgentRunRequest,
    *,
    model_references: Mapping[str, Mapping[str, str]] | None = None,
    system_default_requires_explicit: bool = False,
) -> bool:
    if not request.agent_cards or request.target_agent_name:
        return False

    try:
        from fast_agent.core.agent_card_loader import load_agent_cards
    except Exception:
        return False

    loaded_cards = []
    temp_paths: list[Path] = []
    try:
        for path, is_temporary in materialized_agent_card_paths(request.agent_cards):
            if is_temporary:
                temp_paths.append(path)
            loaded_cards.extend(load_agent_cards(path))
    except Exception:
        return False
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)

    runnable_configs = runnable_agent_card_configs(loaded_cards)
    if len(runnable_configs) != 1:
        return False

    return agent_config_defines_startup_model(
        runnable_configs[0],
        model_references=model_references,
        system_default_requires_explicit=system_default_requires_explicit,
    )


def materialized_agent_card_paths(sources: list[str]) -> list[tuple[Path, bool]]:
    from fast_agent.io.source_resolver import REMOTE_TEXT_SCHEMES, materialize_text_source

    paths: list[tuple[Path, bool]] = []
    for source in sources:
        parsed = urlparse(source)
        if parsed.scheme in REMOTE_TEXT_SCHEMES:
            suffix = Path(parsed.path).suffix or ".md"
            paths.append(
                (
                    materialize_text_source(source, label="AgentCard URL", suffix=suffix),
                    True,
                )
            )
        else:
            paths.append((materialize_text_source(source, label="AgentCard source"), False))
    return paths


def runnable_agent_card_configs(loaded_cards: list[Any]) -> list[Any]:
    runnable_configs = []
    for card in loaded_cards:
        if card.agent_data.get("tool_only", False):
            continue
        config = card.agent_data.get("config")
        if config is not None:
            runnable_configs.append(config)
    return runnable_configs


def agent_config_defines_startup_model(
    agent_config: Any,
    *,
    model_references: Mapping[str, Mapping[str, str]] | None,
    system_default_requires_explicit: bool = False,
) -> bool:
    model = agent_config.model
    if not isinstance(model, str):
        return False

    model_spec = strip_to_none(model)
    if model_spec is None:
        return False
    if not model_spec.startswith("$"):
        return True
    if (
        system_default_requires_explicit
        and model_spec == "$system.default"
        and system_default_reference_is_missing(model_references)
    ):
        return False

    try:
        from fast_agent.core.model_resolution import resolve_model_reference

        resolved_model = resolve_model_reference(model_spec, model_references)
    except ModelConfigError:
        return False

    return strip_to_none(resolved_model) is not None


def resolve_model_without_hardcoded_default(
    *,
    model: str | None,
    config_default_model: str | None,
    model_references: Mapping[str, Mapping[str, str]] | None,
) -> ResolvedModelSpec:
    """Resolve model precedence without falling back to the hardcoded system default."""
    from fast_agent.core.model_resolution import resolve_model_spec

    return resolve_model_spec(
        context=None,
        model=model,
        default_model=config_default_model,
        cli_model=model,
        fallback_to_hardcoded=False,
        model_references=model_references,
    )


def load_request_settings(request: AgentRunRequest) -> Settings:
    from fast_agent import config as config_module

    if request.config_path is None:
        config_module._settings = None

    return get_settings_or_exit(
        request.config_path,
        home=request.home,
        no_home=request.no_home,
    )


def resolve_model_picker_initial_selection(
    *,
    settings: Settings,
) -> ModelPickerInitialSelection:
    from fast_agent.core.exceptions import ModelConfigError
    from fast_agent.core.model_resolution import resolve_model_reference
    from fast_agent.llm.model_overlays import load_model_overlay_registry
    from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
    from fast_agent.ui.model_picker_common import model_identity

    references = settings_model_references(settings)
    initial_model_spec = last_used_model_reference(references)
    if initial_model_spec is None:
        return ModelPickerInitialSelection()

    overlay_registry = load_model_overlay_registry(
        start_path=resolve_model_reference_start_path(
            settings=settings,
            fallback_path=Path.cwd(),
        ),
        home=settings.home,
    )
    overlay_selection = overlay_model_picker_selection(
        overlay_registry,
        initial_model_spec,
    )
    if overlay_selection is not None:
        return overlay_selection

    identity_selection = identity_model_picker_selection(
        initial_model_spec,
        model_identity=model_identity,
    )
    if identity_selection is not None:
        return identity_selection

    try:
        resolved_model_spec = resolve_model_reference(initial_model_spec, references)
    except ModelConfigError:
        return ModelPickerInitialSelection(model_spec=initial_model_spec)

    overlay_selection = overlay_model_picker_selection(overlay_registry, resolved_model_spec)
    if overlay_selection is not None:
        return overlay_selection

    identity_selection = identity_model_picker_selection(
        resolved_model_spec,
        model_identity=model_identity,
    )
    return identity_selection or ModelPickerInitialSelection(model_spec=initial_model_spec)


def settings_model_references(settings: Settings) -> Mapping[str, Mapping[str, str]] | None:
    references = settings.model_references
    if not isinstance(references, Mapping):
        return None
    return references


def last_used_model_reference(
    references: Mapping[str, Mapping[str, str]] | None,
) -> str | None:
    if references is None:
        return None
    system_references = references.get("system")
    if not isinstance(system_references, Mapping):
        return None

    raw_last_used = system_references.get("last_used")
    if not isinstance(raw_last_used, str):
        return None

    return strip_to_none(raw_last_used)


def system_default_reference_is_missing(
    references: Mapping[str, Mapping[str, str]] | None,
) -> bool:
    if references is None:
        return True
    system_references = references.get("system")
    if not isinstance(system_references, Mapping):
        return True

    raw_default = system_references.get("default")
    if not isinstance(raw_default, str):
        return True
    return strip_to_none(raw_default) is None


def should_prompt_for_unpinned_system_default(
    settings: Settings,
    *,
    can_prompt: bool,
) -> bool:
    if not can_prompt:
        return False
    if strip_to_none(settings.default_model) != "$system.default":
        return False
    if os.getenv("FAST_AGENT_MODEL"):
        return False
    return system_default_reference_is_missing(settings_model_references(settings))


def overlay_model_picker_selection(
    overlay_registry: Any,
    model_spec: str,
) -> ModelPickerInitialSelection | None:
    if overlay_registry.resolve_model_string(model_spec) is None:
        return None
    return ModelPickerInitialSelection(provider="overlays", model_spec=model_spec)


def identity_model_picker_selection(
    model_spec: str,
    *,
    model_identity: Callable[[str], object | None],
) -> ModelPickerInitialSelection | None:
    from fast_agent.ui.model_picker_common import infer_initial_picker_provider

    if model_identity(model_spec) is None:
        return None
    return ModelPickerInitialSelection(
        provider=infer_initial_picker_provider(model_spec),
        model_spec=model_spec,
    )


def persist_model_picker_last_used_selection(
    request: AgentRunRequest,
    *,
    settings: Settings,
    model_spec: str,
) -> bool:
    from fast_agent.llm.model_reference_config import (
        ModelReferenceConfigService,
        resolve_model_reference_start_path,
    )
    from fast_agent.paths import resolve_home_dir

    normalized_model = strip_to_none(model_spec)
    if request.no_home or normalized_model is None:
        return False

    start_path = resolve_model_reference_start_path(settings=settings, fallback_path=Path.cwd())
    explicit_config_path = None
    if request.config_path is not None:
        loaded_config_file = getattr(settings, "_config_file", None)
        loaded_config_file_path = (
            strip_to_none(loaded_config_file) if isinstance(loaded_config_file, str) else None
        )
        if loaded_config_file_path is not None:
            explicit_config_path = Path(loaded_config_file_path).expanduser().resolve()
        else:
            explicit_config_path = Path(request.config_path).expanduser().resolve()

    home = resolve_home_dir(
        settings=settings,
        cwd=Path.cwd(),
        override=request.home or settings.home,
    )
    write_target = "project" if explicit_config_path is not None else "env"

    try:
        ModelReferenceConfigService(
            start_path=start_path,
            home=home,
            project_write_path=explicit_config_path,
        ).set_reference(
            "$system.last_used",
            normalized_model,
            target=write_target,
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist model picker last-used selection",
            home=str(home) if home is not None else None,
            config_path=str(explicit_config_path) if explicit_config_path is not None else None,
            target=write_target,
            model_spec=normalized_model,
            error=str(exc),
        )
        return False

    references = settings.model_references
    if isinstance(references, dict):
        system_references = references.get("system")
        if isinstance(system_references, dict):
            system_references["last_used"] = normalized_model
        else:
            references["system"] = {"last_used": normalized_model}
    else:
        settings.model_references = {"system": {"last_used": normalized_model}}

    return True


def generic_model_prompt_default(initial_model_spec: str | None) -> str:
    from fast_agent.ui.model_picker_common import has_explicit_provider_prefix

    candidate = strip_to_none(initial_model_spec) or ""
    if candidate.startswith("generic."):
        candidate = candidate.removeprefix("generic.")
        return candidate or "llama3.2"
    if has_explicit_provider_prefix(candidate):
        return "llama3.2"
    return candidate or "llama3.2"


async def prompt_for_generic_model_spec(*, default_model: str = "llama3.2") -> str:
    from prompt_toolkit import PromptSession

    from fast_agent.ui.model_picker_common import normalize_generic_model_spec

    prompt_session = PromptSession()
    while True:
        try:
            entered = await prompt_session.prompt_async(
                "Local model (e.g. llama3.2): ",
                default=default_model,
            )
        except (EOFError, KeyboardInterrupt):
            typer.echo("Model selection cancelled.", err=True)
            raise typer.Exit(1) from None

        normalized = normalize_generic_model_spec(entered)
        if normalized:
            return normalized

        typer.echo("Please enter a non-empty model string.", err=True)


async def import_llamacpp_overlay_from_picker(
    *,
    request: AgentRunRequest,
    start_path: Path,
) -> str | None:
    from fast_agent.cli.commands.model import import_llamacpp_overlay_from_default_url
    from fast_agent.paths import resolve_home_dir

    settings = load_request_settings(request)
    home = (
        None
        if request.no_home
        else resolve_home_dir(
            settings=settings,
            cwd=Path.cwd(),
            override=request.home or settings.home,
        )
    )

    try:
        return await import_llamacpp_overlay_from_default_url(
            start_path=start_path,
            home=home,
        )
    except (EOFError, KeyboardInterrupt):
        return None
    except Exception as exc:
        typer.echo(f"llama.cpp import failed: {exc}", err=True)
        return None


def activate_model_picker_provider(action: ProviderActivation) -> bool:
    from fast_agent.auth.providers import get_oauth_provider
    from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
    from fast_agent.ui import console

    handler = get_oauth_provider(action.provider.config_name)
    status = handler.status()
    if status.get("present") and not status.get("expired"):
        return True

    typer.echo(f"Starting {handler.display_name} OAuth login...", err=True)
    try:
        console.ensure_blocking_console()
        handler.login()
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        return False
    except (EOFError, KeyboardInterrupt):
        typer.echo(f"{handler.display_name} OAuth login cancelled.", err=True)
        return False

    typer.echo(f"{handler.display_name} OAuth login complete.", err=True)
    return True


async def select_model_from_picker(
    request: AgentRunRequest,
    *,
    config_payload: dict[str, Any] | None = None,
    initial_provider: str | None = None,
    initial_model_spec: str | None = None,
) -> str:
    """Prompt user for model selection and return a resolved model string."""
    from fast_agent.llm.model_reference_config import resolve_model_reference_start_path
    from fast_agent.llm.provider_types import Provider
    from fast_agent.ui.model_picker import run_model_picker_async
    from fast_agent.ui.model_picker_common import LLAMACPP_PROVIDER_KEY

    config_path = Path(request.config_path) if request.config_path else None
    picker_start_path = (
        config_path.parent
        if config_path is not None
        else resolve_model_reference_start_path(settings=load_request_settings(request))
    )
    while True:
        picker_result = await run_model_picker_async(
            config_path=config_path,
            config_payload=config_payload,
            start_path=picker_start_path,
            initial_provider=initial_provider,
            initial_model_spec=initial_model_spec,
        )
        if picker_result is None:
            typer.echo("Model selection cancelled.", err=True)
            raise typer.Exit(1)

        initial_provider = picker_result.provider

        if picker_result.activation_action is not None:
            if activate_model_picker_provider(picker_result.activation_action):
                if picker_result.selected_model:
                    return picker_result.selected_model
            continue

        if picker_result.provider == LLAMACPP_PROVIDER_KEY:
            imported_overlay = await import_llamacpp_overlay_from_picker(
                request=request,
                start_path=picker_start_path,
            )
            if imported_overlay is not None:
                typer.echo(f"Imported llama.cpp overlay '{imported_overlay}'.", err=True)
                return imported_overlay
            continue

        if (
            picker_result.provider == Provider.GENERIC.config_name
            and picker_result.resolved_model is None
        ):
            return await prompt_for_generic_model_spec(
                default_model=generic_model_prompt_default(initial_model_spec),
            )

        if picker_result.refer_to_docs or not picker_result.resolved_model:
            typer.echo(
                "Selected provider requires manual model IDs/options. "
                "Please choose a concrete model (or press q to cancel).",
                err=True,
            )
            continue

        selected_model = picker_result.resolved_model or picker_result.selected_model
        assert selected_model is not None
        return selected_model


__all__ = [
    "ModelPickerInitialSelection",
    "agent_config_defines_startup_model",
    "explicit_agent_cards_define_startup_model",
    "generic_model_prompt_default",
    "last_used_model_reference",
    "load_request_settings",
    "persist_model_picker_last_used_selection",
    "resolve_model_picker_initial_selection",
    "resolve_model_without_hardcoded_default",
    "select_model_from_picker",
    "settings_model_references",
    "should_prompt_for_model_picker",
    "should_prompt_for_unpinned_system_default",
    "system_default_reference_is_missing",
]
