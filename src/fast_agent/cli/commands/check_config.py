"""Command to check FastAgent configuration."""

import json
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.table import Table

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.config import resolve_config_search_root
from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.core.agent_card_validation import AgentCardScanResult, scan_agent_card_directory
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.keyring_utils import KeyringStatus, get_keyring_status
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import ModelOverlayRegistry, load_model_overlay_registry
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.openai.openresponses import DEFAULT_OPENRESPONSES_BASE_URL
from fast_agent.llm.provider_key_manager import API_KEY_HINT_TEXT, ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.paths import EnvironmentPaths, default_skill_paths, resolve_environment_paths
from fast_agent.skills import SkillManifest, SkillRegistry
from fast_agent.ui.a3_headers import build_a3_section_header
from fast_agent.ui.console import console
from fast_agent.utils.huggingface_hub import get_huggingface_hub_token

app = typer.Typer(
    help="Check and diagnose FastAgent configuration",
    no_args_is_help=False,  # Allow showing our custom help instead
)
logger = get_logger(__name__)


@dataclass(frozen=True)
class ProviderCatalogScope:
    """A CLI provider scope for model catalog inspection."""

    display_name: str
    providers: tuple[Provider, ...]


_PROVIDER_CATALOG_SCOPES_BY_KEY: dict[str, ProviderCatalogScope] = {
    "openai": ProviderCatalogScope(
        display_name="OpenAI",
        providers=(
            Provider.OPENAI,
            Provider.RESPONSES,
            Provider.CODEX_RESPONSES,
        ),
    ),
    "responses": ProviderCatalogScope(
        display_name="Responses",
        providers=(Provider.RESPONSES,),
    ),
    "codexresponses": ProviderCatalogScope(
        display_name="Codex Responses",
        providers=(Provider.CODEX_RESPONSES,),
    ),
    "anthropic": ProviderCatalogScope(
        display_name="Anthropic",
        providers=(Provider.ANTHROPIC,),
    ),
    "anthropic-vertex": ProviderCatalogScope(
        display_name="Anthropic (Vertex)",
        providers=(Provider.ANTHROPIC_VERTEX,),
    ),
    "google": ProviderCatalogScope(
        display_name="Google",
        providers=(Provider.GOOGLE,),
    ),
    "deepseek": ProviderCatalogScope(
        display_name="Deepseek",
        providers=(Provider.DEEPSEEK,),
    ),
    "aliyun": ProviderCatalogScope(
        display_name="Aliyun",
        providers=(Provider.ALIYUN,),
    ),
    "huggingface": ProviderCatalogScope(
        display_name="HuggingFace",
        providers=(Provider.HUGGINGFACE,),
    ),
    "xai": ProviderCatalogScope(
        display_name="XAI",
        providers=(Provider.XAI,),
    ),
    "openrouter": ProviderCatalogScope(
        display_name="OpenRouter",
        providers=(Provider.OPENROUTER,),
    ),
}

_PROVIDER_CATALOG_SCOPE_ALIASES: dict[str, str] = {
    "hf": "huggingface",
    "codex-responses": "codexresponses",
    "codex_responses": "codexresponses",
    "anthropicvertex": "anthropic-vertex",
}

_PROVIDER_CATALOG_VISIBLE_CHOICES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "anthropic-vertex",
    "google",
    "deepseek",
    "aliyun",
    "huggingface",
    "xai",
    "openrouter",
    "responses",
    "codexresponses",
)


def _normalize_provider_catalog_scope_name(value: str) -> str:
    return value.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


def _build_provider_catalog_scope_lookup() -> dict[str, ProviderCatalogScope]:
    lookup: dict[str, ProviderCatalogScope] = {}
    for name, scope in _PROVIDER_CATALOG_SCOPES_BY_KEY.items():
        lookup[_normalize_provider_catalog_scope_name(name)] = scope

    for alias, canonical_name in _PROVIDER_CATALOG_SCOPE_ALIASES.items():
        normalized_alias = _normalize_provider_catalog_scope_name(alias)
        canonical_scope = _PROVIDER_CATALOG_SCOPES_BY_KEY.get(canonical_name)
        if canonical_scope is None:
            continue
        lookup[normalized_alias] = canonical_scope

    return lookup


_PROVIDER_CATALOG_SCOPE_LOOKUP = _build_provider_catalog_scope_lookup()


def _resolve_provider_catalog_scope(provider_name: str) -> ProviderCatalogScope:
    normalized_name = _normalize_provider_catalog_scope_name(provider_name)
    scope = _PROVIDER_CATALOG_SCOPE_LOOKUP.get(normalized_name)
    if scope is not None:
        return scope

    choices = ", ".join(_PROVIDER_CATALOG_VISIBLE_CHOICES)
    raise ValueError(f"Unknown provider '{provider_name}'. Choose one of: {choices}")


def _print_section_header(title: str, color: str = "blue") -> None:
    """Render section headers with compact a3 styling."""
    header = build_a3_section_header(title, color=color, include_dot=False)
    console.print()
    console.print(header)
    console.print()


def _get_named_alias_rows(config_payload: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(config_payload, dict):
        return []

    references_payload = config_payload.get("model_references")
    if not isinstance(references_payload, dict):
        return []

    rows: list[tuple[str, str]] = []
    for namespace, entries in sorted(references_payload.items(), key=lambda item: str(item[0])):
        if not isinstance(namespace, str) or not isinstance(entries, dict):
            continue
        for alias_name, model in sorted(entries.items(), key=lambda item: str(item[0])):
            if not isinstance(alias_name, str) or not isinstance(model, str):
                continue
            alias_token = f"${namespace}.{alias_name}"
            rows.append((alias_token, model))
    return rows


def _resolve_active_model_providers(
    *,
    api_keys: dict[str, dict[str, str]],
    config_payload: dict[str, Any] | None,
    start_path: Path,
    env_dir: Path | None,
) -> set[Provider]:
    active_providers: set[Provider] = set()

    for provider_name, status in api_keys.items():
        if not status.get("env") and not status.get("config"):
            continue
        try:
            active_providers.add(Provider(provider_name))
        except ValueError:
            continue

    config_mapping: dict[str, Any] = config_payload if isinstance(config_payload, dict) else {}
    active_providers.update(
        ModelSelectionCatalog.configured_providers(
            config_mapping,
            start_path=start_path,
            env_dir=env_dir,
        )
    )
    return active_providers


def find_config_files(start_path: Path, env_dir: Path | None = None) -> dict[str, Path | None]:
    """Find FastAgent configuration files, preferring secrets file next to config file."""
    from fast_agent.config import (
        find_fastagent_config_files,
        resolve_config_search_root,
        resolve_layered_config_file,
    )

    search_root = resolve_config_search_root(start_path, env_dir=env_dir)
    config_path = resolve_layered_config_file(start_path, env_dir=env_dir)
    _, secrets_path = find_fastagent_config_files(search_root)
    return {
        "config": config_path,
        "secrets": secrets_path,
    }


def get_system_info() -> dict:
    """Get system information including Python version, OS, etc."""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "python_path": sys.executable,
    }


def get_secrets_summary(secrets_path: Path | None) -> dict:
    """Extract information from the secrets file."""
    result = {
        "status": "not_found",  # Default status: not found
        "error": None,
        "secrets": {},
    }

    if not secrets_path:
        return result

    if not secrets_path.exists():
        result["status"] = "not_found"
        return result

    # File exists, attempt to parse
    try:
        with open(secrets_path, "r") as f:
            secrets = yaml.safe_load(f)

        # Mark as successfully parsed
        result["status"] = "parsed"
        result["secrets"] = secrets or {}

    except Exception as e:
        # File exists but has parse errors
        result["status"] = "error"
        result["error"] = str(e)
        console.print(f"[yellow]Warning:[/yellow] Error parsing secrets file: {e}")

    return result


def _empty_api_key_results() -> dict[str, dict[str, str]]:
    return {
        provider.config_name: {"env": "", "config": ""}
        for provider in Provider
        if provider not in {Provider.FAST_AGENT, Provider.ANTHROPIC_VERTEX}
    }


def _mask_configured_secret(secret_value: str) -> str:
    if len(secret_value) > 5:
        return f"...{secret_value[-5:]}"
    return "...***"


def _parsed_main_config(config_summary: dict[str, Any]) -> dict[str, Any]:
    if config_summary.get("status") != "parsed":
        return {}
    config_payload = config_summary.get("config")
    return config_payload if isinstance(config_payload, dict) else {}


def _resolve_azure_key_source(
    *,
    secrets_status: object,
    secrets: dict[str, Any],
    config_azure: dict[str, Any],
) -> dict[str, Any]:
    if secrets_status == "parsed" and "azure" in secrets:
        azure_cfg = secrets.get("azure", {})
        return azure_cfg if isinstance(azure_cfg, dict) else {}
    return config_azure


def _resolve_azure_default_credential_label(
    provider_name: str,
    *,
    secrets_status: object,
    secrets: dict[str, Any],
    config_azure: dict[str, Any],
) -> str | None:
    if provider_name != "azure":
        return None

    azure_cfg = _resolve_azure_key_source(
        secrets_status=secrets_status,
        secrets=secrets,
        config_azure=config_azure,
    )
    use_default_cred = azure_cfg.get("use_default_azure_credential", False)
    base_url = azure_cfg.get("base_url")
    if use_default_cred and base_url:
        return "DefaultAzureCredential"
    return None


def _resolve_provider_config_key(
    provider_name: str,
    *,
    secrets_status: object,
    secrets: dict[str, Any],
    main_config: dict[str, Any],
) -> str | None:
    config_key: str | None = None
    if secrets_status == "parsed":
        config_key = ProviderKeyManager.get_config_file_key(provider_name, secrets)

    if not config_key or config_key == API_KEY_HINT_TEXT:
        if main_config:
            config_key = ProviderKeyManager.get_config_file_key(provider_name, main_config)

    if not config_key or config_key == API_KEY_HINT_TEXT:
        return None
    return _mask_configured_secret(config_key)


def _resolve_huggingface_login_label(provider_name: str) -> str | None:
    if provider_name not in {Provider.HUGGINGFACE.config_name, "huggingface"}:
        return None

    hub_token = get_huggingface_hub_token()

    return "Hub login" if hub_token else None


def _resolve_codex_oauth_label(provider_name: str) -> str | None:
    if provider_name != Provider.CODEX_RESPONSES.config_name:
        return None

    try:
        from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

        codex_status = get_codex_token_status()
    except Exception:
        codex_status = {"present": False, "source": None}

    if not codex_status.get("present"):
        return None

    source = codex_status.get("source")
    if source == "keyring":
        source_label = "Keyring OAuth"
    elif source == "auth.json":
        source_label = "Codex auth.json"
    else:
        source_label = "OAuth token"

    if codex_status.get("expired"):
        return f"Expired {source_label}"
    return source_label


def check_api_keys(secrets_summary: dict, config_summary: dict) -> dict:
    """Check if API keys are configured in secrets file or environment, including Azure DefaultAzureCredential.
    Now also checks Azure config in main config file for retrocompatibility.
    """

    results = _empty_api_key_results()
    secrets_payload = secrets_summary.get("secrets", {})
    secrets = secrets_payload if isinstance(secrets_payload, dict) else {}
    secrets_status = secrets_summary.get("status", "not_found")
    main_config = _parsed_main_config(config_summary)
    config_azure_payload = main_config.get("azure", {})
    config_azure = config_azure_payload if isinstance(config_azure_payload, dict) else {}

    for provider_name, status in results.items():
        env_key_name = ProviderKeyManager.get_env_key_name(provider_name)
        env_key_value = os.environ.get(env_key_name) if env_key_name else None
        if env_key_value:
            status["env"] = _mask_configured_secret(env_key_value)

        azure_label = _resolve_azure_default_credential_label(
            provider_name,
            secrets_status=secrets_status,
            secrets=secrets,
            config_azure=config_azure,
        )
        if azure_label is not None:
            status["config"] = azure_label
            continue

        config_key = _resolve_provider_config_key(
            provider_name,
            secrets_status=secrets_status,
            secrets=secrets,
            main_config=main_config,
        )
        if config_key is not None:
            status["config"] = config_key

        if status["env"] or status["config"]:
            continue

        huggingface_label = _resolve_huggingface_login_label(provider_name)
        if huggingface_label is not None:
            status["config"] = huggingface_label
            continue

        codex_label = _resolve_codex_oauth_label(provider_name)
        if codex_label is not None:
            status["config"] = codex_label

    return results


def get_fastagent_version() -> str:
    """Get the installed version of FastAgent."""
    try:
        return version("fast-agent-mcp")
    except:  # noqa: E722
        return "unknown"


def _default_logger_summary(default_settings: Any) -> dict[str, Any]:
    return {
        "level": default_settings.logger.level,
        "type": default_settings.logger.type,
        "streaming": default_settings.logger.streaming,
        "progress_display": default_settings.logger.progress_display,
        "show_chat": default_settings.logger.show_chat,
        "show_tools": default_settings.logger.show_tools,
        "truncate_tools": default_settings.logger.truncate_tools,
        "enable_markup": default_settings.logger.enable_markup,
        "enable_prompt_marks": default_settings.logger.enable_prompt_marks,
    }


def _build_default_config_summary(default_settings: Any) -> dict[str, Any]:
    return {
        "status": "not_found",
        "error": None,
        "default_model": default_settings.default_model,
        "logger": _default_logger_summary(default_settings),
        "mcp_ui_mode": default_settings.mcp_ui_mode,
        "timeline": {
            "steps": default_settings.mcp_timeline.steps,
            "step_seconds": default_settings.mcp_timeline.step_seconds,
        },
        "mcp_servers": [],
        "skills_directories": None,
    }


def _build_logger_summary(
    logger_config: dict[str, Any],
    *,
    default_settings: Any,
) -> dict[str, Any]:
    return {
        "level": logger_config.get("level", default_settings.logger.level),
        "type": logger_config.get("type", default_settings.logger.type),
        "streaming": logger_config.get("streaming", default_settings.logger.streaming),
        "progress_display": logger_config.get(
            "progress_display",
            default_settings.logger.progress_display,
        ),
        "show_chat": logger_config.get("show_chat", default_settings.logger.show_chat),
        "show_tools": logger_config.get("show_tools", default_settings.logger.show_tools),
        "truncate_tools": logger_config.get(
            "truncate_tools",
            default_settings.logger.truncate_tools,
        ),
        "enable_markup": logger_config.get(
            "enable_markup",
            default_settings.logger.enable_markup,
        ),
        "enable_prompt_marks": logger_config.get(
            "enable_prompt_marks",
            default_settings.logger.enable_prompt_marks,
        ),
    }


def _resolve_timeline_summary(
    config: dict[str, Any],
    *,
    default_settings: Any,
) -> dict[str, int]:
    timeline = {
        "steps": default_settings.mcp_timeline.steps,
        "step_seconds": default_settings.mcp_timeline.step_seconds,
    }
    if "mcp_timeline" not in config:
        return timeline

    from fast_agent.config import MCPTimelineSettings

    try:
        timeline_override = MCPTimelineSettings(**(config.get("mcp_timeline") or {}))
    except Exception as exc:  # pragma: no cover - defensive
        console.print(
            "[yellow]Warning:[/yellow] Invalid mcp_timeline configuration; using defaults."
        )
        console.print(f"[yellow]Details:[/yellow] {exc}")
        return timeline

    return {
        "steps": timeline_override.steps,
        "step_seconds": timeline_override.step_seconds,
    }


def _truncate_server_display(value: str) -> str:
    if len(value) <= 60:
        return value
    return value[:57] + "..."


def _resolve_mcp_server_transport(server_config: dict[str, Any]) -> tuple[str, str]:
    url = str(server_config.get("url", "") or "")
    if not url:
        return "STDIO", ""

    try:
        from .url_parser import parse_server_url

        _, transport_type, _ = parse_server_url(url)
        transport = transport_type.upper()
    except Exception:
        transport = "HTTP"
    return transport, _truncate_server_display(url)


def _resolve_mcp_server_command(server_config: dict[str, Any]) -> str:
    command = str(server_config.get("command", "") or "")
    if not command:
        return ""

    args = server_config.get("args", [])
    if not args:
        return command

    args_str = " ".join(str(arg) for arg in args)
    return _truncate_server_display(f"{command} {args_str}")


def _build_mcp_server_summaries(config: dict[str, Any]) -> list[dict[str, str]]:
    mcp_config = config.get("mcp")
    if not isinstance(mcp_config, dict):
        return []

    servers_config = mcp_config.get("servers")
    if not isinstance(servers_config, dict):
        return []

    server_summaries: list[dict[str, str]] = []
    for server_name, server_config in servers_config.items():
        if not isinstance(server_name, str) or not isinstance(server_config, dict):
            continue
        transport, url = _resolve_mcp_server_transport(server_config)
        server_summaries.append(
            {
                "name": server_name,
                "transport": transport,
                "command": _resolve_mcp_server_command(server_config),
                "url": url,
            }
        )
    return server_summaries


def _extract_skills_directories(config: dict[str, Any]) -> list[str] | None:
    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        return None

    directory_value = skills_cfg.get("directories")
    if not isinstance(directory_value, list):
        return None

    cleaned = [str(value).strip() for value in directory_value if str(value).strip()]
    return cleaned or None


def get_config_summary(config_path: Path | None) -> dict:
    """Extract key information from the configuration file."""
    from fast_agent.config import Settings

    default_settings = Settings()
    result = _build_default_config_summary(default_settings)

    if not config_path:
        return result

    if not config_path.exists():
        result["status"] = "not_found"
        return result

    # File exists, attempt to parse
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        result["status"] = "parsed"
        result["config"] = config

        if not config:
            return result

        if not isinstance(config, dict):
            return result

        if "default_model" in config:
            result["default_model"] = config["default_model"]
        logger_config = config.get("logger")
        if isinstance(logger_config, dict):
            result["logger"] = _build_logger_summary(
                logger_config,
                default_settings=default_settings,
            )
        if "mcp_ui_mode" in config:
            result["mcp_ui_mode"] = config["mcp_ui_mode"]

        result["timeline"] = _resolve_timeline_summary(
            config,
            default_settings=default_settings,
        )
        result["mcp_servers"] = _build_mcp_server_summaries(config)
        result["skills_directories"] = _extract_skills_directories(config)

    except Exception as e:
        # File exists but has parse errors
        result["status"] = "error"
        result["error"] = str(e)
        console.print(f"[red]Error parsing configuration file:[/red] {e}")

    return result


def _load_catalog_config(env_dir: Path | None) -> dict[str, Any] | None:
    from fast_agent.config import load_layered_settings

    config_payload, _ = load_layered_settings(start_path=Path.cwd(), env_dir=env_dir)
    return config_payload or None


def show_models_overview(env_dir: Path | None = None) -> None:
    """Show providers accepted by `fast-agent check models <provider>` and alias status."""
    cwd = Path.cwd()
    config_files = find_config_files(cwd, env_dir=env_dir)
    config_summary = get_config_summary(config_files["config"])
    secrets_summary = get_secrets_summary(config_files["secrets"])
    api_keys = check_api_keys(secrets_summary, config_summary)
    config_payload = _load_catalog_config(env_dir)
    active_providers = _resolve_active_model_providers(
        api_keys=api_keys,
        config_payload=config_payload,
        start_path=cwd,
        env_dir=env_dir,
    )

    _print_section_header("Model Catalog", color="blue")

    provider_table = Table(show_header=True, box=None)
    provider_table.add_column("Provider Arg", style="cyan", header_style="bold bright_white")
    provider_table.add_column("Scope", style="white", header_style="bold bright_white")
    provider_table.add_column("Active", justify="center", header_style="bold bright_white")

    for scope_name in _PROVIDER_CATALOG_VISIBLE_CHOICES:
        scope = _PROVIDER_CATALOG_SCOPES_BY_KEY.get(scope_name)
        if scope is None:
            continue
        if len(scope.providers) > 1:
            scope_label = ", ".join(provider.display_name for provider in scope.providers)
            scope_text = f"{scope_label} (family)"
        elif scope_name in {"responses", "codexresponses"}:
            scope_text = f"{scope.providers[0].display_name} (direct)"
        else:
            scope_text = scope.providers[0].display_name
        is_active = any(provider in active_providers for provider in scope.providers)
        active_symbol = "[bold green]✓[/bold green]" if is_active else "[dim]✗[/dim]"
        provider_table.add_row(scope_name, scope_text, active_symbol)

    console.print(provider_table)

    alias_rows = sorted(_PROVIDER_CATALOG_SCOPE_ALIASES.items())
    if alias_rows:
        alias_table = Table(show_header=True, box=None)
        alias_table.add_column("Arg Alias", style="cyan", header_style="bold bright_white")
        alias_table.add_column("Resolves To", style="white", header_style="bold bright_white")
        for alias, target in alias_rows:
            alias_table.add_row(alias, target)
        console.print(alias_table)

    _print_section_header("Named Model Aliases", color="blue")
    alias_rows = _get_named_alias_rows(config_payload)
    if alias_rows:
        alias_table = Table(show_header=True, box=None)
        alias_table.add_column("Alias", style="magenta", header_style="bold bright_white")
        alias_table.add_column("Resolves To", style="green", header_style="bold bright_white")

        for alias_token, model in alias_rows:
            alias_table.add_row(alias_token, model)
        console.print(alias_table)
    else:
        console.print("[dim]No model_references configured in fastagent.config.yaml[/dim]")

    console.print()
    console.print(
        "Use [cyan]fast-agent check models <provider>[/cyan] to inspect provider models and aliases."
    )
    console.print(
        "Use [cyan]fast-agent check models <provider> --all[/cyan] to list every known model."
    )


def _load_all_models_by_provider(
    scope: ProviderCatalogScope,
    *,
    config_payload: dict[str, Any] | None,
    overlay_registry: ModelOverlayRegistry,
) -> dict[Provider, list[str]]:
    return {
        provider: ModelSelectionCatalog.list_all_models(
            provider,
            config=config_payload,
            overlay_registry=overlay_registry,
        )
        for provider in scope.providers
    }


def _build_curated_models_table(
    scope: ProviderCatalogScope,
    *,
    overlay_registry: ModelOverlayRegistry,
) -> tuple[Table, dict[Provider, set[str]], int]:
    curated_table = Table(show_header=True, box=None)
    curated_table.add_column("Provider", style="white", header_style="bold bright_white")
    curated_table.add_column("Alias", style="magenta", header_style="bold bright_white")
    curated_table.add_column("Tags", style="cyan", header_style="bold bright_white")
    curated_table.add_column(
        "Model",
        style="green",
        header_style="bold bright_white",
        overflow="fold",
    )

    row_count = 0
    curated_models_by_provider: dict[Provider, set[str]] = {}
    for provider in scope.providers:
        provider_entries = ModelSelectionCatalog.list_current_entries(
            provider,
            overlay_registry=overlay_registry,
        )
        curated_models_by_provider[provider] = {entry.model for entry in provider_entries}
        for entry in provider_entries:
            curated_table.add_row(
                provider.display_name,
                entry.alias if entry.alias else "[dim]-[/dim]",
                "fast" if entry.fast else "[dim]-[/dim]",
                entry.model,
            )
            row_count += 1

    return curated_table, curated_models_by_provider, row_count


def _has_additional_provider_models(
    *,
    all_models_by_provider: dict[Provider, list[str]],
    curated_models_by_provider: dict[Provider, set[str]],
) -> bool:
    return any(
        any(model not in curated_models_by_provider.get(provider, set()) for model in all_models)
        for provider, all_models in all_models_by_provider.items()
    )


def _build_all_models_table(
    scope: ProviderCatalogScope,
    *,
    all_models_by_provider: dict[Provider, list[str]],
    curated_models_by_provider: dict[Provider, set[str]],
) -> tuple[Table, int]:
    all_models_table = Table(show_header=True, box=None)
    all_models_table.add_column("Provider", style="white", header_style="bold bright_white")
    all_models_table.add_column("Tags", style="cyan", header_style="bold bright_white")
    all_models_table.add_column(
        "Model",
        style="green",
        header_style="bold bright_white",
        overflow="fold",
    )

    all_row_count = 0
    for provider in scope.providers:
        models = all_models_by_provider.get(provider, [])
        curated_models = curated_models_by_provider.get(provider, set())
        if not models:
            all_models_table.add_row(provider.display_name, "[dim]-[/dim]", "[dim]-[/dim]")
            all_row_count += 1
            continue

        for model in models:
            labels: list[str] = []
            if ModelSelectionCatalog.is_fast_model(model):
                labels.append("fast")
            if model in curated_models:
                labels.append("catalog")
            all_models_table.add_row(
                provider.display_name,
                " • ".join(labels) if labels else "[dim]-[/dim]",
                model,
            )
            all_row_count += 1

    return all_models_table, all_row_count


def show_provider_model_catalog(
    provider_name: str,
    *,
    show_all: bool = False,
    env_dir: Path | None = None,
) -> None:
    """Show provider model catalog with curated entries first."""
    scope = _resolve_provider_catalog_scope(provider_name)
    config_payload = _load_catalog_config(env_dir)
    overlay_registry = load_model_overlay_registry(start_path=Path.cwd(), env_dir=env_dir)
    all_models_by_provider = _load_all_models_by_provider(
        scope,
        config_payload=config_payload,
        overlay_registry=overlay_registry,
    )

    mode = "curated + all models" if show_all else "curated"
    _print_section_header(f"{scope.display_name} model catalog ({mode})", color="blue")

    curated_table, curated_models_by_provider, row_count = _build_curated_models_table(
        scope,
        overlay_registry=overlay_registry,
    )
    if row_count == 0:
        console.print("[yellow]No curated models found for this provider scope.[/yellow]")
    else:
        console.print(curated_table)

    has_additional_models = _has_additional_provider_models(
        all_models_by_provider=all_models_by_provider,
        curated_models_by_provider=curated_models_by_provider,
    )

    if not show_all:
        if has_additional_models:
            console.print(
                f"[dim]More models are available. Run [cyan]fast-agent check models {provider_name} --all[/cyan] "
                "for the complete catalog.[/dim]"
            )
        return

    _print_section_header("All known models", color="blue")
    all_models_table, all_row_count = _build_all_models_table(
        scope,
        all_models_by_provider=all_models_by_provider,
        curated_models_by_provider=curated_models_by_provider,
    )
    if all_row_count:
        console.print(all_models_table)


def _split_model_specs(raw_models: str) -> list[str]:
    return [chunk.strip() for chunk in raw_models.split(",") if chunk.strip()]


def _build_model_references(config_payload: dict[str, Any] | None) -> dict[str, str]:
    aliases = ModelFactory.get_runtime_presets()

    if not isinstance(config_payload, dict):
        return aliases

    alias_tree = config_payload.get("model_references")
    if not isinstance(alias_tree, dict):
        return aliases

    for namespace, entries in alias_tree.items():
        if not isinstance(namespace, str) or not isinstance(entries, dict):
            continue
        for alias_name, model_spec in entries.items():
            if not isinstance(alias_name, str) or not isinstance(model_spec, str):
                continue
            token = f"${namespace}.{alias_name}"
            aliases[token] = model_spec
            aliases[f"{namespace}.{alias_name}"] = model_spec

    return aliases


def _resolve_model_secret_entry(
    spec: str,
    *,
    aliases: dict[str, str],
    api_keys: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], str | None]:
    parsed = ModelFactory.parse_model_string(spec, presets=aliases)
    provider = parsed.provider
    provider_key = provider.config_name
    env_var_value = ProviderKeyManager.get_env_key_name(provider_key)
    env_var = env_var_value if isinstance(env_var_value, str) else None
    provider_status = api_keys.get(provider_key, {})
    return (
        {
            "input": spec,
            "resolved_model": parsed.model_name,
            "provider": provider_key,
            "provider_display": provider.display_name,
            "required_env": env_var,
            "local_env_present": bool(provider_status.get("env")),
            "local_config_present": bool(provider_status.get("config")),
        },
        env_var,
    )


def _resolve_model_secret_entries(
    specs: list[str],
    *,
    aliases: dict[str, str],
    api_keys: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved_entries: list[dict[str, Any]] = []
    unique_secret_envs: list[str] = []

    for spec in specs:
        try:
            entry, env_var = _resolve_model_secret_entry(
                spec,
                aliases=aliases,
                api_keys=api_keys,
            )
            resolved_entries.append(entry)
            if env_var is not None and env_var not in unique_secret_envs:
                unique_secret_envs.append(env_var)
        except Exception as exc:
            resolved_entries.append({"input": spec, "error": str(exc)})

    return resolved_entries, unique_secret_envs


def _build_model_secret_payload(
    *,
    specs: list[str],
    resolved_entries: list[dict[str, Any]],
    unique_secret_envs: list[str],
) -> dict[str, Any]:
    return {
        "models": specs,
        "resolved": resolved_entries,
        "candidate_secret_env_vars": unique_secret_envs,
        "safety_rule": (
            "Pass secret names only; never pass secret values via CLI arguments. "
            "Use secure job secret references (for example --secrets ENV_VAR_NAME)."
        ),
    }


def _render_model_secret_requirements_table(
    *,
    resolved_entries: list[dict[str, Any]],
    unique_secret_envs: list[str],
) -> None:
    _print_section_header("Model secret requirements", color="blue")

    results_table = Table(show_header=True, box=None)
    results_table.add_column("Input", style="cyan", header_style="bold bright_white")
    results_table.add_column("Provider", style="white", header_style="bold bright_white")
    results_table.add_column("Resolved Model", style="green", header_style="bold bright_white")
    results_table.add_column("Required Env", style="magenta", header_style="bold bright_white")
    results_table.add_column("Local Key Status", style="yellow", header_style="bold bright_white")

    for entry in resolved_entries:
        if entry.get("error"):
            results_table.add_row(
                entry.get("input", "?"),
                "[red]unresolved[/red]",
                "[red]-[/red]",
                "[red]-[/red]",
                f"[red]{entry['error']}[/red]",
            )
            continue

        local_bits: list[str] = []
        if entry.get("local_env_present"):
            local_bits.append("env")
        if entry.get("local_config_present"):
            local_bits.append("config")
        local_status = " + ".join(local_bits) if local_bits else "missing"

        results_table.add_row(
            entry["input"],
            entry["provider_display"],
            entry["resolved_model"],
            entry["required_env"],
            local_status,
        )

    console.print(results_table)

    if unique_secret_envs:
        console.print()
        console.print(
            "[bold]Candidate secret env var names:[/bold] " + ", ".join(unique_secret_envs)
        )

    console.print()
    console.print(
        "[bold yellow]IMPORTANT:[/bold yellow] Never pass secret values through command arguments. "
        "Forward secret [bold]names[/bold] only via secure secret stores (for example: "
        "[cyan]hf jobs ... --secrets OPENAI_API_KEY[/cyan])."
    )


def show_model_secret_requirements(
    models: str,
    *,
    env_dir: Path | None = None,
    json_output: bool = False,
) -> None:
    """Show provider + secret-env requirements for one or more model specs."""

    specs = _split_model_specs(models)
    if not specs:
        raise ValueError("No model values provided. Pass one or more model specs.")

    config_files = find_config_files(Path.cwd(), env_dir=env_dir)
    config_summary = get_config_summary(config_files["config"])
    secrets_summary = get_secrets_summary(config_files["secrets"])
    api_keys = check_api_keys(secrets_summary, config_summary)
    config_payload = _load_catalog_config(env_dir)
    aliases = _build_model_references(config_payload)
    resolved_entries, unique_secret_envs = _resolve_model_secret_entries(
        specs,
        aliases=aliases,
        api_keys=api_keys,
    )
    payload = _build_model_secret_payload(
        specs=specs,
        resolved_entries=resolved_entries,
        unique_secret_envs=unique_secret_envs,
    )

    if json_output:
        console.print(json.dumps(payload, indent=2, sort_keys=True))
        return

    _render_model_secret_requirements_table(
        resolved_entries=resolved_entries,
        unique_secret_envs=unique_secret_envs,
    )


def _effective_environment_override(
    *,
    env_dir: Path | None,
    config_summary: dict[str, Any],
) -> str | Path:
    if env_dir is not None:
        return env_dir

    env_override = os.getenv("ENVIRONMENT_DIR")
    if isinstance(env_override, str) and env_override.strip():
        return env_override.strip()

    config_payload = config_summary.get("config")
    if isinstance(config_payload, dict):
        configured_env_dir = config_payload.get("environment_dir")
        if isinstance(configured_env_dir, str) and configured_env_dir.strip():
            return configured_env_dir.strip()

    return DEFAULT_ENVIRONMENT_DIR


def _validate_effective_settings(
    *,
    cwd: Path,
    config_files: dict[str, Path | None],
    env_override: str | Path,
) -> str | None:
    from fast_agent.config import (
        Settings,
        deep_merge,
        load_layered_settings,
        load_yaml_mapping,
    )

    try:
        merged_settings, _ = load_layered_settings(start_path=cwd, env_dir=env_override)

        secrets_path = config_files.get("secrets")
        if isinstance(secrets_path, Path):
            merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))

        Settings(**merged_settings)
    except Exception as exc:
        return str(exc)

    return None


@dataclass(frozen=True)
class _CheckSummaryContext:
    cwd: Path
    search_root: Path
    config_files: dict[str, Path | None]
    system_info: dict[str, str]
    config_summary: dict[str, Any]
    secrets_summary: dict[str, Any]
    api_keys: dict[str, dict[str, str]]
    fastagent_version: str
    environment_override: str | Path
    effective_settings_error: str | None
    keyring: Any | None
    keyring_status: KeyringStatus
    skills_dirs: list[Path]
    skills_manifests: list[SkillManifest]
    skill_errors: list[dict[str, str]]
    env_paths: EnvironmentPaths
    server_names: set[str] | None
    overlay_preset_collision_messages: tuple[str, ...]


def _relative_summary_path(search_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(search_root))
    except ValueError:
        return str(path)


def _truncate_summary_text(text: str, length: int = 70) -> str:
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def _load_optional_keyring_module() -> Any | None:
    try:
        import keyring as keyring_module

        return keyring_module
    except Exception:
        return None


def _build_check_summary_context(env_dir: Path | None) -> _CheckSummaryContext:
    cwd = Path.cwd()
    search_root = resolve_config_search_root(cwd, env_dir=env_dir)
    config_files = find_config_files(cwd, env_dir=env_dir)
    system_info = get_system_info()
    config_summary = get_config_summary(config_files["config"])
    secrets_summary = get_secrets_summary(config_files["secrets"])
    api_keys = check_api_keys(secrets_summary, config_summary)
    fastagent_version = get_fastagent_version()
    environment_override = _effective_environment_override(
        env_dir=env_dir,
        config_summary=config_summary,
    )
    effective_settings_error = _validate_effective_settings(
        cwd=cwd,
        config_files=config_files,
        env_override=environment_override,
    )
    keyring = _load_optional_keyring_module()
    keyring_status = get_keyring_status()

    skills_override = config_summary.get("skills_directories")
    override_directories = (
        [Path(entry).expanduser() for entry in skills_override]
        if isinstance(skills_override, list)
        else None
    )
    default_directories = (
        default_skill_paths(cwd=search_root, override=environment_override)
        if override_directories is None
        else None
    )
    skills_registry = SkillRegistry(
        base_dir=search_root,
        directories=override_directories
        if override_directories is not None
        else default_directories,
    )
    skills_dirs = list(skills_registry.directories)
    skills_manifests, skill_errors = skills_registry.load_manifests_with_errors()
    env_paths = resolve_environment_paths(cwd=cwd, override=environment_override)
    server_names = _build_server_names_from_config(config_summary)
    overlay_preset_collision_messages = _collect_overlay_preset_collision_messages(
        start_path=cwd,
        env_dir=environment_override,
    )

    return _CheckSummaryContext(
        cwd=cwd,
        search_root=search_root,
        config_files=config_files,
        system_info=system_info,
        config_summary=config_summary,
        secrets_summary=secrets_summary,
        api_keys=api_keys,
        fastagent_version=fastagent_version,
        environment_override=environment_override,
        effective_settings_error=effective_settings_error,
        keyring=keyring,
        keyring_status=keyring_status,
        skills_dirs=skills_dirs,
        skills_manifests=skills_manifests,
        skill_errors=skill_errors,
        env_paths=env_paths,
        server_names=server_names,
        overlay_preset_collision_messages=overlay_preset_collision_messages,
    )


def _built_in_preset_sources() -> dict[str, str]:
    sources = {preset_name: "built-in model preset" for preset_name in ModelFactory.MODEL_PRESETS}
    for provider_entries in ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER.values():
        for entry in provider_entries:
            sources.setdefault(entry.alias, "curated model alias")
    return sources


def _collect_overlay_preset_collision_messages(
    *,
    start_path: Path,
    env_dir: str | Path | None,
) -> tuple[str, ...]:
    preset_sources = _built_in_preset_sources()
    overlay_registry = load_model_overlay_registry(start_path=start_path, env_dir=env_dir)

    messages: list[str] = []
    for overlay in overlay_registry.overlays:
        source = preset_sources.get(overlay.name)
        if source is None:
            continue
        message = (
            f'Local model overlay "{overlay.name}" overrides existing {source} "{overlay.name}".'
        )
        messages.append(message)
        logger.info(
            "Local model overlay overrides existing preset",
            overlay_name=overlay.name,
            source=source,
            overlay_path=str(overlay.manifest_path),
        )

    return tuple(messages)


def _render_environment_summary(context: _CheckSummaryContext) -> None:
    header_title = f"fast-agent v{context.fastagent_version} ({context.system_info['platform']})"
    _print_section_header(header_title, color="blue")

    config_path = context.config_files["config"]
    secrets_path = context.config_files["secrets"]
    env_table = Table(show_header=False, box=None)
    env_table.add_column("Setting", style="white")
    env_table.add_column("Value")

    python_version = ".".join(context.system_info["python_version"].split(".")[:3])
    env_table.add_row("Python Version", f"[green]{python_version}[/green]")
    env_table.add_row("Python Path", f"[green]{context.system_info['python_path']}[/green]")

    secrets_status = context.secrets_summary.get("status", "not_found")
    if secrets_status == "not_found":
        env_table.add_row("Secrets File", "[yellow]Not found[/yellow]")
    elif secrets_status == "error":
        env_table.add_row("Secrets File", f"[orange_red1]Errors[/orange_red1] ({secrets_path})")
        env_table.add_row(
            "Secrets Error",
            f"[orange_red1]{context.secrets_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:
        env_table.add_row("Secrets File", f"[green]Found[/green] ({secrets_path})")

    config_status = context.config_summary.get("status", "not_found")
    if config_status == "not_found":
        env_table.add_row("Config File", "[red]Not found[/red]")
    elif config_status == "error":
        env_table.add_row("Config File", f"[orange_red1]Errors[/orange_red1] ({config_path})")
        env_table.add_row(
            "Config Error",
            f"[orange_red1]{context.config_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:
        env_table.add_row("Config File", f"[green]Found[/green] ({config_path})")
        default_model_value = context.config_summary.get(
            "default_model",
            "gpt-5.4-mini?reasoning=low (system default)",
        )
        env_table.add_row("Default Model", f"[green]{default_model_value}[/green]")

    if context.effective_settings_error:
        env_table.add_row("Effective Config", "[orange_red1]Errors[/orange_red1]")
        env_table.add_row(
            "Effective Error",
            f"[orange_red1]{context.effective_settings_error}[/orange_red1]",
        )

    if context.keyring_status.available:
        if context.keyring_status.writable:
            keyring_display = f"[green]{context.keyring_status.name}[/green]"
        else:
            keyring_display = f"[yellow]{context.keyring_status.name} (not writable)[/yellow]"
    else:
        keyring_display = "[red]not available[/red]"
    env_table.add_row("Keyring Backend", keyring_display)

    console.print(env_table)


def _bool_to_symbol(value: object) -> str:
    return "[bold green]✓[/bold green]" if value else "[bold red]✗[/bold red]"


def _format_step_interval(seconds: int) -> str:
    try:
        total = int(seconds)
    except (TypeError, ValueError):
        return str(seconds)
    if total <= 0:
        return "0s"
    if total % 86400 == 0:
        return f"{total // 86400}d"
    if total % 3600 == 0:
        return f"{total // 3600}h"
    if total % 60 == 0:
        return f"{total // 60}m"
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _build_application_settings_rows(
    config_summary: dict[str, Any],
) -> list[tuple[str, str]]:
    logger = config_summary.get("logger", {})
    mcp_ui_mode = config_summary.get("mcp_ui_mode", "auto")
    mcp_ui_display = (
        "[dim]disabled[/dim]" if mcp_ui_mode == "disabled" else f"[green]{mcp_ui_mode}[/green]"
    )

    timeline_settings = config_summary.get("timeline", {})
    timeline_steps = timeline_settings.get("steps", 20)
    timeline_step_seconds = timeline_settings.get("step_seconds", 30)

    return [
        ("Log Level", logger.get("level", "warning (default)")),
        ("Log Type", logger.get("type", "file (default)")),
        ("MCP-UI", mcp_ui_display),
        ("Streaming Mode", f"[green]{logger.get('streaming', 'markdown')}[/green]"),
        ("Streaming Display", _bool_to_symbol(logger.get("streaming_display", True))),
        ("Progress Display", _bool_to_symbol(logger.get("progress_display", True))),
        ("Show Chat", _bool_to_symbol(logger.get("show_chat", True))),
        ("Show Tools", _bool_to_symbol(logger.get("show_tools", True))),
        ("Truncate Tools", _bool_to_symbol(logger.get("truncate_tools", True))),
        ("Enable Markup", _bool_to_symbol(logger.get("enable_markup", True))),
        ("Prompt Marks", _bool_to_symbol(logger.get("enable_prompt_marks", False))),
        ("Timeline Steps", f"[green]{timeline_steps}[/green]"),
        (
            "Timeline Interval",
            f"[green]{_format_step_interval(timeline_step_seconds)}[/green]",
        ),
    ]


def _render_application_settings(config_summary: dict[str, Any]) -> None:
    logger_table = Table(show_header=True, box=None)
    logger_table.add_column("Setting", style="white", header_style="bold bright_white")
    logger_table.add_column("Value", header_style="bold bright_white")
    logger_table.add_column("Setting", style="white", header_style="bold bright_white")
    logger_table.add_column("Value", header_style="bold bright_white")

    settings_data = _build_application_settings_rows(config_summary)
    for index in range(0, len(settings_data), 2):
        left_setting, left_value = settings_data[index]
        if left_setting in {"Log Level", "Log Type"}:
            left_value = f"[green]{left_value}[/green]"

        if index + 1 < len(settings_data):
            right_setting, right_value = settings_data[index + 1]
            if right_setting in {"Log Level", "Log Type"}:
                right_value = f"[green]{right_value}[/green]"
            logger_table.add_row(left_setting, left_value, right_setting, right_value)
        else:
            logger_table.add_row(left_setting, left_value, "", "")

    _print_section_header("Application Settings", color="blue")
    console.print(logger_table)


def _render_model_overlay_notices(context: _CheckSummaryContext) -> None:
    if not context.overlay_preset_collision_messages:
        return

    _print_section_header("Model Overlays", color="blue")
    for message in context.overlay_preset_collision_messages:
        console.print(f"[cyan]Info:[/cyan] {message}")


def _format_provider_row(provider: str, status: dict[str, str]) -> tuple[str, str, str, str]:
    if status["env"] and status["config"]:
        env_status = "[yellow]✓[/yellow]"
    elif status["env"]:
        env_status = "[bold green]✓[/bold green]"
    else:
        env_status = "[dim]✗[/dim]"

    config_status = "[bold green]✓[/bold green]" if status["config"] else "[dim]✗[/dim]"

    if status["config"]:
        active = f"[bold green]{status['config']}[/bold green]"
    elif status["env"]:
        active = f"[bold green]{status['env']}[/bold green]"
    elif provider == "generic":
        active = "[green]ollama (default)[/green]"
    elif provider == "openresponses":
        active = "[green]none (default)[/green]"
    else:
        active = "[dim]Not configured[/dim]"

    display_name = Provider(provider).display_name
    return display_name, env_status, config_status, active


def _render_api_keys_panel(api_keys: dict[str, dict[str, str]]) -> None:
    keys_table = Table(show_header=True, box=None)
    for title, style, justify in (
        ("Provider", "white", None),
        ("Env", None, "center"),
        ("Config", None, "center"),
        ("Active Key", "green", None),
        ("Provider", "white", None),
        ("Env", None, "center"),
        ("Config", None, "center"),
        ("Active Key", "green", None),
    ):
        if style is not None and justify is not None:
            keys_table.add_column(
                title,
                style=style,
                justify=justify,
                header_style="bold bright_white",
            )
        elif style is not None:
            keys_table.add_column(
                title,
                style=style,
                header_style="bold bright_white",
            )
        elif justify is not None:
            keys_table.add_column(
                title,
                justify=justify,
                header_style="bold bright_white",
            )
        else:
            keys_table.add_column(title, header_style="bold bright_white")

    providers_list = list(api_keys.items())
    mid_point = (len(providers_list) + 1) // 2
    for index in range(mid_point):
        left_provider, left_status = providers_list[index]
        left_data = _format_provider_row(left_provider, left_status)
        if index + mid_point < len(providers_list):
            right_provider, right_status = providers_list[index + mid_point]
            right_data = _format_provider_row(right_provider, right_status)
            keys_table.add_row(*left_data, *right_data)
        else:
            keys_table.add_row(*left_data, "", "", "", "")

    _print_section_header("API Keys", color="blue")
    console.print(keys_table)
    console.print("[dim]Use [cyan]fast-agent check models[/cyan] to see/configure models.[/dim]")


def _render_codex_oauth_panel(keyring_status: KeyringStatus) -> None:
    try:
        from datetime import datetime

        from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

        codex_status = get_codex_token_status()
    except Exception:
        return

    codex_table = Table(show_header=True, box=None)
    codex_table.add_column("Token", style="white", header_style="bold bright_white")
    codex_table.add_column("Source", style="white", header_style="bold bright_white")
    codex_table.add_column("Expires", style="white", header_style="bold bright_white")
    codex_table.add_column("Keyring", style="white", header_style="bold bright_white")

    if not keyring_status.available:
        keyring_display = "[red]not available[/red]"
    elif not keyring_status.writable:
        keyring_display = f"[yellow]{keyring_status.name} (not writable)[/yellow]"
    else:
        keyring_display = f"[green]{keyring_status.name}[/green]"

    if not codex_status["present"]:
        token_display = "[dim]Not configured[/dim]"
        source_display = "[dim]-[/dim]"
        expires_display = "[dim]-[/dim]"
    else:
        token_display = "[bold green]OAuth token[/bold green]"
        source = codex_status.get("source")
        if source == "keyring":
            source_display = "[green]Keyring OAuth[/green]"
        elif source == "auth.json":
            source_display = "[green]Codex auth.json[/green]"
        else:
            source_display = "[green]OAuth token[/green]"
        expires_at = codex_status.get("expires_at")
        if expires_at:
            expires_display = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M")
            if codex_status.get("expired"):
                expires_display = f"[red]expired {expires_display}[/red]"
            else:
                expires_display = f"[green]{expires_display}[/green]"
        else:
            expires_display = "[green]unknown[/green]"

    codex_table.add_row(token_display, source_display, expires_display, keyring_display)
    _print_section_header("Codex OAuth", color="blue")
    console.print(codex_table)


def _build_mcp_servers_table() -> Table:
    servers_table = Table(show_header=True, box=None)
    servers_table.add_column("Name", style="white", header_style="bold bright_white")
    servers_table.add_column("Transport", style="white", header_style="bold bright_white")
    servers_table.add_column("Command/URL", header_style="bold bright_white")
    servers_table.add_column("OAuth", header_style="bold bright_white")
    servers_table.add_column("Token", header_style="bold bright_white")
    return servers_table


def _display_mcp_server_target(server: dict[str, str]) -> str:
    transport = server["transport"]
    command_url = (
        server["command"] if transport == "STDIO" else server["url"]
    ) or "[dim]Not configured[/dim]"
    if "Not configured" in command_url:
        return command_url
    return f"[green]{command_url}[/green]"


def _build_mcp_server_settings(server: dict[str, str]) -> Any | None:
    from fast_agent.config import MCPServerSettings

    try:
        return MCPServerSettings(
            name=server["name"],
            transport="sse"
            if server["transport"] == "SSE"
            else ("stdio" if server["transport"] == "STDIO" else "http"),
            url=(server.get("url") or None),
        )
    except Exception:
        return None


def _resolve_oauth_enabled(cfg: Any) -> bool:
    if cfg.auth is None or not hasattr(cfg.auth, "oauth"):
        return True
    return bool(getattr(cfg.auth, "oauth"))


def _resolve_oauth_persist_mode(cfg: Any) -> str:
    if cfg.auth is None or not hasattr(cfg.auth, "persist"):
        return "keyring"
    return getattr(cfg.auth, "persist") or "keyring"


def _resolve_mcp_token_status(
    context: _CheckSummaryContext,
    cfg: Any,
    *,
    compute_server_identity: Any,
    oauth_enabled: bool,
    persist: str,
) -> str:
    if not oauth_enabled:
        return "[dim]n/a[/dim]"
    if context.keyring is not None and context.keyring_status.writable and persist == "keyring":
        identity = compute_server_identity(cfg)
        token_key = f"oauth:tokens:{identity}"
        try:
            has_token = context.keyring.get_password("fast-agent-mcp", token_key) is not None
        except Exception:
            has_token = False
        return "[bold green]✓[/bold green]" if has_token else "[dim]✗[/dim]"
    if persist == "keyring" and not context.keyring_status.available:
        return "[red]not available[/red]"
    if persist == "keyring" and not context.keyring_status.writable:
        return "[yellow]not writable[/yellow]"
    if persist == "memory":
        return "[yellow]memory[/yellow]"
    return "[dim]n/a[/dim]"


def _resolve_mcp_oauth_columns(
    context: _CheckSummaryContext,
    cfg: Any | None,
    *,
    compute_server_identity: Any,
) -> tuple[str, str]:
    if cfg is None or cfg.transport not in ("http", "sse"):
        return "[dim]-[/dim]", "[dim]n/a[/dim]"

    oauth_enabled = _resolve_oauth_enabled(cfg)
    oauth_status = "[green]on[/green]" if oauth_enabled else "[dim]off[/dim]"
    token_status = _resolve_mcp_token_status(
        context,
        cfg,
        compute_server_identity=compute_server_identity,
        oauth_enabled=oauth_enabled,
        persist=_resolve_oauth_persist_mode(cfg),
    )
    return oauth_status, token_status


def _build_mcp_server_row(
    context: _CheckSummaryContext,
    server: dict[str, str],
    *,
    compute_server_identity: Any,
) -> tuple[str, str, str, str, str]:
    cfg = _build_mcp_server_settings(server)
    oauth_status, token_status = _resolve_mcp_oauth_columns(
        context,
        cfg,
        compute_server_identity=compute_server_identity,
    )
    return (
        server["name"],
        server["transport"],
        _display_mcp_server_target(server),
        oauth_status,
        token_status,
    )


def _render_mcp_servers_panel(context: _CheckSummaryContext) -> None:
    if context.config_summary.get("status") != "parsed":
        return

    mcp_servers = context.config_summary.get("mcp_servers", [])
    if not mcp_servers:
        return

    from fast_agent.mcp.oauth_client import compute_server_identity

    servers_table = _build_mcp_servers_table()

    for server in mcp_servers:
        servers_table.add_row(
            *_build_mcp_server_row(
                context,
                server,
                compute_server_identity=compute_server_identity,
            )
        )

    _print_section_header("MCP Servers", color="blue")
    console.print(servers_table)


def _render_skills_panel(context: _CheckSummaryContext) -> None:
    _print_section_header("Agent Skills", color="blue")
    if not context.skills_dirs:
        console.print(
            "[dim]Agent Skills not configured. Go to https://fast-agent.ai/agents/skills/[/dim]"
        )
        return

    if len(context.skills_dirs) == 1:
        directory_display = _relative_summary_path(context.search_root, context.skills_dirs[0])
        console.print(f"Directory: [green]{directory_display}[/green]")
    else:
        console.print("Directories:")
        for directory in context.skills_dirs:
            relative = _relative_summary_path(context.search_root, directory)
            console.print(f"- [green]{relative}[/green]")

    if not context.skills_manifests and not context.skill_errors:
        console.print("[yellow]No skills found in the configured directories[/yellow]")
        return

    skills_table = Table(show_header=True, box=None)
    skills_table.add_column("Name", style="cyan", header_style="bold bright_white")
    skills_table.add_column("Description", style="white", header_style="bold bright_white")
    skills_table.add_column("Source", style="dim", header_style="bold bright_white")
    skills_table.add_column("Status", style="green", header_style="bold bright_white")

    for manifest in context.skills_manifests:
        source_display = _relative_summary_path(context.search_root, manifest.path.parent)
        skills_table.add_row(
            manifest.name,
            _truncate_summary_text(manifest.description or ""),
            source_display,
            "[green]ok[/green]",
        )

    for error in context.skill_errors:
        error_path_str = error.get("path", "")
        source_display = "[dim]n/a[/dim]"
        if error_path_str:
            error_path = Path(error_path_str)
            source_display = _relative_summary_path(context.search_root, error_path.parent)
        message = error.get("error", "Failed to parse skill manifest")
        skills_table.add_row(
            "[red]—[/red]",
            "[red]n/a[/red]",
            source_display,
            f"[red]{_truncate_summary_text(message, 60)}[/red]",
        )

    console.print(skills_table)


def _build_server_names_from_config(
    config_summary: dict[str, Any],
) -> set[str] | None:
    if config_summary.get("status") != "parsed":
        return None

    mcp_servers = config_summary.get("mcp_servers", [])
    if not isinstance(mcp_servers, list):
        return None

    return {
        server.get("name", "")
        for server in mcp_servers
        if isinstance(server, dict) and server.get("name")
    }


def _should_warn_for_provider(
    provider: Provider,
    config_summary: dict[str, Any],
) -> bool:
    if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
        return False
    if provider == Provider.OPENRESPONSES:
        cfg = config_summary.get("config") if config_summary.get("status") == "parsed" else {}
        openresponses_cfg = cfg.get("openresponses", {}) if isinstance(cfg, dict) else {}
        configured_base_url = None
        if isinstance(openresponses_cfg, dict):
            raw_base_url = openresponses_cfg.get("base_url")
            if isinstance(raw_base_url, str) and raw_base_url.strip():
                configured_base_url = raw_base_url.strip()
        if configured_base_url is None:
            env_base_url = os.getenv("OPENRESPONSES_BASE_URL")
            if env_base_url and env_base_url.strip():
                configured_base_url = env_base_url.strip()
        if configured_base_url is None:
            return False
        return configured_base_url.rstrip("/") != DEFAULT_OPENRESPONSES_BASE_URL.rstrip("/")
    if provider == Provider.GOOGLE:
        cfg = config_summary.get("config") if config_summary.get("status") == "parsed" else {}
        google_cfg = cfg.get("google", {}) if isinstance(cfg, dict) else {}
        vertex_cfg = google_cfg.get("vertex_ai", {}) if isinstance(google_cfg, dict) else {}
        if isinstance(vertex_cfg, dict) and vertex_cfg.get("enabled") is True:
            return False
    if provider == Provider.ANTHROPIC_VERTEX:
        return False
    return True


def _collect_card_directories(
    env_paths: EnvironmentPaths,
) -> list[tuple[str, Path]]:
    return [
        ("Agent Cards", env_paths.agent_cards),
        ("Tool Cards", env_paths.tool_cards),
    ]


def _collect_all_card_names(
    card_directories: list[tuple[str, Path]],
    *,
    server_names: set[str] | None,
) -> tuple[bool, set[str]]:
    found_card_dir = False
    all_card_names: set[str] = set()
    for _, directory in card_directories:
        if not directory.is_dir():
            continue
        found_card_dir = True
        entries = scan_agent_card_directory(directory, server_names=server_names)
        for entry in entries:
            if entry.name != "—" and entry.ignored_reason is None:
                all_card_names.add(entry.name)
    return found_card_dir, all_card_names


def _format_agent_card_error_status(entry: AgentCardScanResult) -> str:
    if entry.ignored_reason:
        return f"[dim]ignored - {entry.ignored_reason}[/dim]"
    if entry.errors:
        error_text = _truncate_summary_text(entry.errors[0], 60)
        if len(entry.errors) > 1:
            error_text = f"{error_text} (+{len(entry.errors) - 1} more)"
        return f"[red]{error_text}[/red]"
    return "[green]ok[/green]"


def _load_agent_cards_for_status(entry: AgentCardScanResult) -> list[Any]:
    try:
        from fast_agent.core.agent_card_loader import load_agent_cards

        return load_agent_cards(entry.path)
    except Exception:
        return []


def _update_default_agent_tracking(
    *,
    card_name: str,
    config: Any | None,
    default_agent_names: list[str],
    default_agent_seen: set[str],
) -> None:
    if not config or not getattr(config, "default", False):
        return
    if card_name in default_agent_seen:
        return
    default_agent_names.append(card_name)
    default_agent_seen.add(card_name)


def _runtime_mcp_entry_count(config: Any | None) -> int:
    if config is None:
        return 0
    return len(getattr(config, "mcp_connect", []) or [])


def _record_missing_api_key_warning(
    *,
    card_name: str,
    model: str | None,
    api_keys: dict[str, dict[str, str]],
    config_summary: dict[str, Any],
    warned_cards: set[str],
    api_warning_messages: list[str],
) -> None:
    if not model:
        return
    try:
        model_config = ModelFactory.parse_model_string(model)
    except ModelConfigError:
        return

    provider = model_config.provider
    if not _should_warn_for_provider(provider, config_summary):
        return

    key_status = api_keys.get(provider.config_name)
    if not key_status or key_status["env"] or key_status["config"]:
        return

    api_warning_messages.append(
        f'Warning: Card "{card_name}" uses model "{model}" '
        f"({provider.display_name}) but no API key configured."
    )
    warned_cards.add(card_name)


def _format_runtime_mcp_status(runtime_mcp_count: int) -> str:
    if runtime_mcp_count <= 0:
        return "[green]ok[/green]"
    plural = "entry" if runtime_mcp_count == 1 else "entries"
    return f"[green]ok[/green] [dim](mcp_connect: {runtime_mcp_count} {plural})[/dim]"


def _build_agent_card_status(
    entry: AgentCardScanResult,
    *,
    api_keys: dict[str, dict[str, str]],
    config_summary: dict[str, Any],
    warned_cards: set[str],
    api_warning_messages: list[str],
    default_agent_names: list[str],
    default_agent_seen: set[str],
) -> str:
    status = _format_agent_card_error_status(entry)
    if entry.errors or entry.ignored_reason is not None or entry.name in warned_cards:
        return status

    runtime_mcp_count = 0
    cards = _load_agent_cards_for_status(entry)
    for card in cards:
        config = card.agent_data.get("config")
        runtime_mcp_count += _runtime_mcp_entry_count(config)
        _update_default_agent_tracking(
            card_name=card.name,
            config=config,
            default_agent_names=default_agent_names,
            default_agent_seen=default_agent_seen,
        )
        _record_missing_api_key_warning(
            card_name=card.name,
            model=config.model if config else None,
            api_keys=api_keys,
            config_summary=config_summary,
            warned_cards=warned_cards,
            api_warning_messages=api_warning_messages,
        )

    return _format_runtime_mcp_status(runtime_mcp_count)


def _render_agent_card_panel(context: _CheckSummaryContext) -> None:
    _print_section_header("Agent Cards", color="blue")
    card_directories = _collect_card_directories(context.env_paths)
    found_card_dir, all_card_names = _collect_all_card_names(
        card_directories,
        server_names=context.server_names,
    )

    api_warning_messages: list[str] = []
    warned_cards: set[str] = set()
    default_agent_names: list[str] = []
    default_agent_seen: set[str] = set()

    for label, directory in card_directories:
        if not directory.is_dir():
            continue

        relative_directory = _relative_summary_path(context.search_root, directory)
        console.print(f"{label} Directory: [green]{relative_directory}[/green]")
        entries = scan_agent_card_directory(
            directory,
            server_names=context.server_names,
            extra_agent_names=all_card_names,
        )
        if not entries:
            console.print("[yellow]No AgentCards found in this directory[/yellow]")
            continue

        cards_table = Table(show_header=True, box=None)
        cards_table.add_column("Name", style="cyan", header_style="bold bright_white")
        cards_table.add_column("Type", style="white", header_style="bold bright_white")
        cards_table.add_column("Source", style="dim", header_style="bold bright_white")
        cards_table.add_column("Status", style="green", header_style="bold bright_white")

        for entry in entries:
            status = _build_agent_card_status(
                entry,
                api_keys=context.api_keys,
                config_summary=context.config_summary,
                warned_cards=warned_cards,
                api_warning_messages=api_warning_messages,
                default_agent_names=default_agent_names,
                default_agent_seen=default_agent_seen,
            )
            cards_table.add_row(
                entry.name,
                entry.type,
                _relative_summary_path(context.search_root, entry.path),
                status,
            )

        console.print(cards_table)

    if len(default_agent_names) > 1:
        joined = ", ".join(default_agent_names)
        console.print(f"[yellow]Warning:[/yellow] multiple agents are set as default: {joined}")

    for warning in api_warning_messages:
        console.print(f"[yellow]{warning}[/yellow]")

    if not found_card_dir:
        console.print(
            "[dim]No local AgentCard directories found in the fast-agent environment.[/dim]"
        )


def _render_check_summary_guidance(context: _CheckSummaryContext) -> None:
    config_status = context.config_summary.get("status", "not_found")
    secrets_status = context.secrets_summary.get("status", "not_found")

    if config_status == "error" or secrets_status == "error" or context.effective_settings_error:
        console.print("\n[bold]Config File Issues:[/bold]")
        if context.effective_settings_error:
            console.print(f"[orange_red1]{context.effective_settings_error}[/orange_red1]")
        console.print("Fix the YAML syntax errors in your configuration files")
    elif config_status == "not_found" or secrets_status == "not_found":
        console.print("\n[bold]Setup Tips:[/bold]")
        console.print(
            "Run [cyan]fast-agent scaffold[/cyan] to create configuration files. Visit [cyan][link=https://fast-agent.ai]fast-agent.ai[/link][/cyan] for configuration guides. "
        )

    if all(
        not context.api_keys[provider]["env"] and not context.api_keys[provider]["config"]
        for provider in context.api_keys
    ):
        console.print(
            "\n[yellow]No API keys configured. Set up API keys to use LLM services:[/yellow]"
        )
        console.print("1. Add keys to fastagent.secrets.yaml")
        env_vars = ", ".join(
            filter(
                None,
                (
                    ProviderKeyManager.get_env_key_name(p.config_name)
                    for p in Provider
                    if p != Provider.FAST_AGENT
                ),
            )
        )
        console.print(f"2. Or set environment variables ({env_vars})")


def show_check_summary(env_dir: Path | None = None) -> None:
    """Show a summary of checks with colorful styling."""
    context = _build_check_summary_context(env_dir)
    _render_environment_summary(context)
    _render_application_settings(context.config_summary)
    _render_model_overlay_notices(context)
    _render_api_keys_panel(context.api_keys)
    _render_codex_oauth_panel(context.keyring_status)
    _render_mcp_servers_panel(context)
    _render_skills_panel(context)
    _render_agent_card_panel(context)
    _render_check_summary_guidance(context)


@app.command()
def show(
    path: str | None = typer.Argument(None, help="Path to configuration file to display"),
    secrets: bool = typer.Option(
        False, "--secrets", "-s", help="Show secrets file instead of config"
    ),
) -> None:
    """Display the configuration file content or search for it."""
    file_type = "secrets" if secrets else "config"

    if path:
        config_path = Path(path).resolve()
        if not config_path.exists():
            console.print(
                f"[red]Error:[/red] {file_type.capitalize()} file not found at {config_path}"
            )
            raise typer.Exit(1)
    else:
        config_files = find_config_files(Path.cwd())
        config_path = config_files[file_type]
        if not config_path:
            console.print(
                f"[yellow]No {file_type} file found in current directory or parents[/yellow]"
            )
            console.print("Run [cyan]fast-agent scaffold[/cyan] to create configuration files")
            raise typer.Exit(1)

    console.print(f"\n[bold]{file_type.capitalize()} file:[/bold] {config_path}\n")

    try:
        with open(config_path, "r") as f:
            content = f.read()

        # Try to parse as YAML to check validity
        parsed = yaml.safe_load(content)

        # Show parsing success status
        console.print("[green]YAML syntax is valid[/green]")
        if parsed is None:
            console.print("[yellow]Warning: File is empty or contains only comments[/yellow]\n")
        else:
            console.print(
                f"[green]Successfully parsed {len(parsed) if isinstance(parsed, dict) else 0} root keys[/green]\n"
            )

        # Print the content
        console.print(content)

    except Exception as e:
        console.print(f"[red]Error parsing {file_type} file:[/red] {e}")


def _context_env_dir(ctx: typer.Context) -> Path | None:
    payload = ctx.obj
    if not isinstance(payload, dict):
        return None

    env_dir = payload.get("env_dir")
    if isinstance(env_dir, Path):
        return env_dir
    return None


@app.command("models")
def models(
    ctx: typer.Context,
    provider: str | None = typer.Argument(
        None,
        help=(
            "Provider scope to inspect. Omit to list available providers, key status, "
            "and configured named aliases."
        ),
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Show all known models after curated entries (requires a provider argument)",
    ),
    for_model: str | None = typer.Option(
        None,
        "--for-model",
        help="Resolve one or more model specs (comma-separated) to provider and secret env requirements.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit JSON output (supported with --for-model).",
    ),
) -> None:
    """Show model catalog provider guidance or provider-specific model entries."""
    env_dir = _context_env_dir(ctx)

    if for_model is not None:
        if provider is not None:
            raise typer.BadParameter("Do not pass a provider argument with --for-model.")
        if all_models:
            raise typer.BadParameter("Do not combine --all with --for-model.")
        try:
            show_model_secret_requirements(for_model, env_dir=env_dir, json_output=json_output)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--for-model") from exc
        return

    if json_output:
        raise typer.BadParameter("--json currently requires --for-model.")

    if provider is None:
        if all_models:
            console.print(
                "[yellow]Tip:[/yellow] Pass a provider name with [cyan]--all[/cyan], "
                "for example: [cyan]fast-agent check models openai --all[/cyan]"
            )
        show_models_overview(env_dir=env_dir)
        return

    try:
        show_provider_model_catalog(
            provider,
            show_all=all_models,
            env_dir=env_dir,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="provider") from exc


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    env_dir: Path | None = typer.Option(
        None, "--env", help="Override the base fast-agent environment directory"
    ),
) -> None:
    """Check and diagnose FastAgent configuration."""
    env_dir = resolve_environment_dir_option(ctx, env_dir)
    if isinstance(ctx.obj, dict):
        ctx.obj["env_dir"] = env_dir
    else:
        ctx.obj = {"env_dir": env_dir}

    if ctx.invoked_subcommand is None:
        show_check_summary(env_dir=env_dir)
