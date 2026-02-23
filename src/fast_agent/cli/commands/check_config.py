"""Command to check FastAgent configuration."""

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
from rich.text import Text

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.config import resolve_config_search_root
from fast_agent.constants import DEFAULT_ENVIRONMENT_DIR
from fast_agent.core.agent_card_validation import scan_agent_card_directory
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider_key_manager import API_KEY_HINT_TEXT, ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.paths import default_skill_paths, resolve_environment_paths
from fast_agent.skills import SkillRegistry
from fast_agent.ui.console import console

app = typer.Typer(
    help="Check and diagnose FastAgent configuration",
    no_args_is_help=False,  # Allow showing our custom help instead
)


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
}

_PROVIDER_CATALOG_VISIBLE_CHOICES: tuple[str, ...] = (
    "openai",
    "anthropic",
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
    header = Text.from_markup(
        f"[{color}]▎[/{color}][dim {color}]•[/dim {color}] [{color}]{title}[/{color}]"
    )
    console.print()
    console.print(header)
    console.print()


def _get_named_alias_rows(config_payload: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not isinstance(config_payload, dict):
        return []

    aliases_payload = config_payload.get("model_aliases")
    if not isinstance(aliases_payload, dict):
        return []

    rows: list[tuple[str, str]] = []
    for namespace, entries in sorted(aliases_payload.items(), key=lambda item: str(item[0])):
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
    active_providers.update(ModelSelectionCatalog.configured_providers(config_mapping))
    return active_providers


def find_config_files(start_path: Path, env_dir: Path | None = None) -> dict[str, Path | None]:
    """Find FastAgent configuration files, preferring secrets file next to config file."""
    from fast_agent.config import find_fastagent_config_files, resolve_config_search_root

    search_root = resolve_config_search_root(start_path, env_dir=env_dir)
    config_path, secrets_path = find_fastagent_config_files(search_root)
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


def check_api_keys(secrets_summary: dict, config_summary: dict) -> dict:
    """Check if API keys are configured in secrets file or environment, including Azure DefaultAzureCredential.
    Now also checks Azure config in main config file for retrocompatibility.
    """

    results = {
        provider.config_name: {"env": "", "config": ""}
        for provider in Provider
        if provider != Provider.FAST_AGENT
    }

    # Get secrets if available
    secrets = secrets_summary.get("secrets", {})
    secrets_status = secrets_summary.get("status", "not_found")
    # Get config if available
    config = config_summary if config_summary.get("status") == "parsed" else {}

    config_azure = {}
    if config and "azure" in config.get("config", {}):
        config_azure = config["config"]["azure"]

    for provider_name in results:
        # Always check environment variables first
        env_key_name = ProviderKeyManager.get_env_key_name(provider_name)
        env_key_value = os.environ.get(env_key_name)
        if env_key_value:
            if len(env_key_value) > 5:
                results[provider_name]["env"] = f"...{env_key_value[-5:]}"
            else:
                results[provider_name]["env"] = "...***"

        # Special handling for Azure: support api_key and DefaultAzureCredential
        if provider_name == "azure":
            # Prefer secrets if present, else fallback to config
            azure_cfg = {}
            if secrets_status == "parsed" and "azure" in secrets:
                azure_cfg = secrets.get("azure", {})
            elif config_azure:
                azure_cfg = config_azure

            use_default_cred = azure_cfg.get("use_default_azure_credential", False)
            base_url = azure_cfg.get("base_url")
            if use_default_cred and base_url:
                results[provider_name]["config"] = "DefaultAzureCredential"
                continue

        # Check secrets file first, then fall back to main config file
        config_key = None
        if secrets_status == "parsed":
            config_key = ProviderKeyManager.get_config_file_key(provider_name, secrets)

        # Fall back to main config file if not found in secrets
        if not config_key or config_key == API_KEY_HINT_TEXT:
            main_config = config.get("config", {})
            if main_config:
                config_key = ProviderKeyManager.get_config_file_key(provider_name, main_config)

        if config_key and config_key != API_KEY_HINT_TEXT:
            if len(config_key) > 5:
                results[provider_name]["config"] = f"...{config_key[-5:]}"
            else:
                results[provider_name]["config"] = "...***"

        if (
            provider_name in {Provider.HUGGINGFACE.config_name, "huggingface"}
            and not results[provider_name]["env"]
            and not results[provider_name]["config"]
        ):
            try:
                from huggingface_hub import get_token  # type: ignore

                hub_token = get_token()
            except Exception:
                hub_token = None

            if hub_token:
                results[provider_name]["config"] = "Hub login"

        if (
            provider_name == Provider.CODEX_RESPONSES.config_name
            and not results[provider_name]["env"]
            and not results[provider_name]["config"]
        ):
            try:
                from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

                codex_status = get_codex_token_status()
            except Exception:
                codex_status = {"present": False, "source": None}

            if codex_status.get("present"):
                source = codex_status.get("source")
                is_expired = bool(codex_status.get("expired"))
                if source == "keyring":
                    source_label = "Keyring OAuth"
                elif source == "auth.json":
                    source_label = "Codex auth.json"
                else:
                    source_label = "OAuth token"

                if is_expired:
                    results[provider_name]["config"] = f"Expired {source_label}"
                else:
                    results[provider_name]["config"] = source_label

    return results


def get_fastagent_version() -> str:
    """Get the installed version of FastAgent."""
    try:
        return version("fast-agent-mcp")
    except:  # noqa: E722
        return "unknown"


def get_config_summary(config_path: Path | None) -> dict:
    """Extract key information from the configuration file."""
    from fast_agent.config import MCPTimelineSettings, Settings

    # Get actual defaults from Settings class
    default_settings = Settings()

    result = {
        "status": "not_found",  # Default status: not found
        "error": None,
        "default_model": default_settings.default_model,
        "logger": {
            "level": default_settings.logger.level,
            "type": default_settings.logger.type,
            "streaming": default_settings.logger.streaming,
            "progress_display": default_settings.logger.progress_display,
            "show_chat": default_settings.logger.show_chat,
            "show_tools": default_settings.logger.show_tools,
            "truncate_tools": default_settings.logger.truncate_tools,
            "enable_markup": default_settings.logger.enable_markup,
            "enable_prompt_marks": default_settings.logger.enable_prompt_marks,
        },
        "mcp_ui_mode": default_settings.mcp_ui_mode,
        "timeline": {
            "steps": default_settings.mcp_timeline.steps,
            "step_seconds": default_settings.mcp_timeline.step_seconds,
        },
        "mcp_servers": [],
        "skills_directories": None,
    }

    if not config_path:
        return result

    if not config_path.exists():
        result["status"] = "not_found"
        return result

    # File exists, attempt to parse
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Mark as successfully parsed
        result["status"] = "parsed"
        result["config"] = config  # Store raw config for API key checking

        if not config:
            return result

        # Get default model
        if "default_model" in config:
            result["default_model"] = config["default_model"]

        # Get logger settings
        if "logger" in config:
            logger_config = config["logger"]
            result["logger"] = {
                "level": logger_config.get("level", default_settings.logger.level),
                "type": logger_config.get("type", default_settings.logger.type),
                "streaming": logger_config.get("streaming", default_settings.logger.streaming),
                "progress_display": logger_config.get(
                    "progress_display", default_settings.logger.progress_display
                ),
                "show_chat": logger_config.get("show_chat", default_settings.logger.show_chat),
                "show_tools": logger_config.get("show_tools", default_settings.logger.show_tools),
                "truncate_tools": logger_config.get(
                    "truncate_tools", default_settings.logger.truncate_tools
                ),
                "enable_markup": logger_config.get(
                    "enable_markup", default_settings.logger.enable_markup
                ),
                "enable_prompt_marks": logger_config.get(
                    "enable_prompt_marks", default_settings.logger.enable_prompt_marks
                ),
            }

        # Get MCP UI mode
        if "mcp_ui_mode" in config:
            result["mcp_ui_mode"] = config["mcp_ui_mode"]

        # Get timeline settings
        if "mcp_timeline" in config:
            try:
                timeline_override = MCPTimelineSettings(**(config.get("mcp_timeline") or {}))
            except Exception as exc:  # pragma: no cover - defensive
                console.print(
                    "[yellow]Warning:[/yellow] Invalid mcp_timeline configuration; using defaults."
                )
                console.print(f"[yellow]Details:[/yellow] {exc}")
            else:
                result["timeline"] = {
                    "steps": timeline_override.steps,
                    "step_seconds": timeline_override.step_seconds,
                }

        # Get MCP server info
        if "mcp" in config and "servers" in config["mcp"]:
            for server_name, server_config in config["mcp"]["servers"].items():
                server_info = {
                    "name": server_name,
                    "transport": "STDIO",  # Default transport type
                    "command": "",
                    "url": "",
                }

                # Determine transport type
                if "url" in server_config:
                    url = server_config.get("url", "")
                    server_info["url"] = url

                    # Use URL path to determine transport type
                    try:
                        from .url_parser import parse_server_url

                        _, transport_type, _ = parse_server_url(url)
                        server_info["transport"] = transport_type.upper()
                    except Exception:
                        # Fallback to HTTP if URL parsing fails
                        server_info["transport"] = "HTTP"

                # Get command and args
                command = server_config.get("command", "")
                args = server_config.get("args", [])

                if command:
                    if args:
                        args_str = " ".join([str(arg) for arg in args])
                        full_cmd = f"{command} {args_str}"
                        # Truncate if too long
                        if len(full_cmd) > 60:
                            full_cmd = full_cmd[:57] + "..."
                        server_info["command"] = full_cmd
                    else:
                        server_info["command"] = command

                # Truncate URL if too long
                if server_info["url"] and len(server_info["url"]) > 60:
                    server_info["url"] = server_info["url"][:57] + "..."

                mcp_servers = result["mcp_servers"]
                assert isinstance(mcp_servers, list)
                mcp_servers.append(server_info)

        # Skills directory override
        skills_cfg = config.get("skills") if isinstance(config, dict) else None
        if isinstance(skills_cfg, dict):
            directory_value = skills_cfg.get("directories")
            if isinstance(directory_value, list):
                cleaned = [str(value).strip() for value in directory_value if str(value).strip()]
                if cleaned:
                    result["skills_directories"] = cleaned

    except Exception as e:
        # File exists but has parse errors
        result["status"] = "error"
        result["error"] = str(e)
        console.print(f"[red]Error parsing configuration file:[/red] {e}")

    return result


def _load_catalog_config(env_dir: Path | None) -> dict[str, Any] | None:
    config_files = find_config_files(Path.cwd(), env_dir=env_dir)
    config_summary = get_config_summary(config_files["config"])
    if config_summary.get("status") != "parsed":
        return None

    config_payload = config_summary.get("config")
    if isinstance(config_payload, dict):
        return config_payload
    return None


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
        console.print("[dim]No model_aliases configured in fastagent.config.yaml[/dim]")

    console.print()
    console.print(
        "Use [cyan]fast-agent check models <provider>[/cyan] to inspect provider models and aliases."
    )
    console.print("Use [cyan]fast-agent check models <provider> --all[/cyan] to list every known model.")


def show_provider_model_catalog(
    provider_name: str,
    *,
    show_all: bool = False,
    env_dir: Path | None = None,
) -> None:
    """Show provider model catalog with curated entries first."""
    scope = _resolve_provider_catalog_scope(provider_name)
    config_payload = _load_catalog_config(env_dir)
    all_models_by_provider: dict[Provider, list[str]] = {
        provider: ModelSelectionCatalog.list_all_models(provider, config=config_payload)
        for provider in scope.providers
    }

    mode = "curated + all models" if show_all else "curated"
    _print_section_header(f"{scope.display_name} model catalog ({mode})", color="blue")

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
        provider_entries = ModelSelectionCatalog.list_current_entries(provider)
        curated_models_by_provider[provider] = {entry.model for entry in provider_entries}
        for entry in provider_entries:
            tags = "fast" if entry.fast else "[dim]-[/dim]"
            curated_table.add_row(
                provider.display_name,
                entry.alias if entry.alias else "[dim]-[/dim]",
                tags,
                entry.model,
            )
            row_count += 1

    if row_count == 0:
        console.print("[yellow]No curated models found for this provider scope.[/yellow]")
    else:
        console.print(curated_table)

    has_additional_models = any(
        any(model not in curated_models_by_provider.get(provider, set()) for model in all_models)
        for provider, all_models in all_models_by_provider.items()
    )

    if not show_all:
        if has_additional_models:
            console.print(
                f"[dim]More models are available. Run [cyan]fast-agent check models {provider_name} --all[/cyan] "
                "for the complete catalog.[/dim]"
            )
        return

    _print_section_header("All known models", color="blue")
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
            tags = " • ".join(labels) if labels else "[dim]-[/dim]"
            all_models_table.add_row(provider.display_name, tags, model)
            all_row_count += 1

    if all_row_count:
        console.print(all_models_table)


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
        load_layered_model_settings,
        load_yaml_mapping,
    )

    try:
        merged_settings: dict[str, Any] = {}

        config_path = config_files.get("config")
        if isinstance(config_path, Path):
            merged_settings = load_yaml_mapping(config_path)

        layered_model_settings = load_layered_model_settings(start_path=cwd, env_dir=env_override)
        merged_settings = deep_merge(merged_settings, layered_model_settings)

        secrets_path = config_files.get("secrets")
        if isinstance(secrets_path, Path):
            merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))

        Settings(**merged_settings)
    except Exception as exc:
        return str(exc)

    return None


def show_check_summary(env_dir: Path | None = None) -> None:
    """Show a summary of checks with colorful styling."""
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

    # Environment and configuration section (merged)
    # Header shows version and platform for a concise overview
    header_title = f"fast-agent v{fastagent_version} ({system_info['platform']})"
    _print_section_header(header_title, color="blue")

    config_path = config_files["config"]
    secrets_path = config_files["secrets"]

    env_table = Table(show_header=False, box=None)
    env_table.add_column("Setting", style="white")
    env_table.add_column("Value")

    # Determine keyring backend early so it can appear in the top section
    # Also detect whether the backend is actually writable (not just present)
    try:
        import keyring  # type: ignore
    except Exception:
        keyring = None  # type: ignore

    from fast_agent.core.keyring_utils import get_keyring_status

    keyring_status = get_keyring_status()
    keyring_name = keyring_status.name

    # Python info (highlight version and path in green)
    env_table.add_row(
        "Python Version", f"[green]{'.'.join(system_info['python_version'].split('.')[:3])}[/green]"
    )
    env_table.add_row("Python Path", f"[green]{system_info['python_path']}[/green]")

    # Secrets file status
    secrets_status = secrets_summary.get("status", "not_found")
    if secrets_status == "not_found":
        env_table.add_row("Secrets File", "[yellow]Not found[/yellow]")
    elif secrets_status == "error":
        env_table.add_row("Secrets File", f"[orange_red1]Errors[/orange_red1] ({secrets_path})")
        env_table.add_row(
            "Secrets Error",
            f"[orange_red1]{secrets_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:  # parsed successfully
        env_table.add_row("Secrets File", f"[green]Found[/green] ({secrets_path})")

    # Config file status
    config_status = config_summary.get("status", "not_found")
    if config_status == "not_found":
        env_table.add_row("Config File", "[red]Not found[/red]")
    elif config_status == "error":
        env_table.add_row("Config File", f"[orange_red1]Errors[/orange_red1] ({config_path})")
        env_table.add_row(
            "Config Error",
            f"[orange_red1]{config_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:  # parsed successfully
        env_table.add_row("Config File", f"[green]Found[/green] ({config_path})")
        default_model_value = config_summary.get(
            "default_model", "gpt-5-mini?reasoning=low (system default)"
        )
        env_table.add_row("Default Model", f"[green]{default_model_value}[/green]")

    if effective_settings_error:
        env_table.add_row("Effective Config", "[orange_red1]Errors[/orange_red1]")
        env_table.add_row("Effective Error", f"[orange_red1]{effective_settings_error}[/orange_red1]")

    # Keyring backend (always shown in application-level settings)
    if keyring_status.available:
        if keyring_status.writable:
            keyring_display = f"[green]{keyring_name}[/green]"
        else:
            keyring_display = f"[yellow]{keyring_name} (not writable)[/yellow]"
    else:
        keyring_display = "[red]not available[/red]"
    env_table.add_row("Keyring Backend", keyring_display)

    console.print(env_table)

    def _relative_path(path: Path) -> str:
        try:
            return str(path.relative_to(search_root))
        except ValueError:
            return str(path)

    def _truncate(text: str, length: int = 70) -> str:
        if len(text) <= length:
            return text
        return text[: length - 3] + "..."

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
        directories=override_directories if override_directories is not None else default_directories,
    )
    skills_dirs = skills_registry.directories
    skills_manifests, skill_errors = skills_registry.load_manifests_with_errors()

    # Logger Settings panel with two-column layout
    logger = config_summary.get("logger", {})
    logger_table = Table(show_header=True, box=None)
    logger_table.add_column("Setting", style="white", header_style="bold bright_white")
    logger_table.add_column("Value", header_style="bold bright_white")
    logger_table.add_column("Setting", style="white", header_style="bold bright_white")
    logger_table.add_column("Value", header_style="bold bright_white")

    def bool_to_symbol(value):
        return "[bold green]✓[/bold green]" if value else "[bold red]✗[/bold red]"

    # Format MCP-UI mode value
    mcp_ui_mode = config_summary.get("mcp_ui_mode", "auto")
    if mcp_ui_mode == "disabled":
        mcp_ui_display = "[dim]disabled[/dim]"
    else:
        mcp_ui_display = f"[green]{mcp_ui_mode}[/green]"

    timeline_settings = config_summary.get("timeline", {})
    timeline_steps = timeline_settings.get("steps", 20)
    timeline_step_seconds = timeline_settings.get("step_seconds", 30)

    def format_step_interval(seconds: int) -> str:
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

    # Prepare all settings as pairs
    settings_data = [
        ("Log Level", logger.get("level", "warning (default)")),
        ("Log Type", logger.get("type", "file (default)")),
        ("MCP-UI", mcp_ui_display),
        ("Streaming Mode", f"[green]{logger.get('streaming', 'markdown')}[/green]"),
        ("Streaming Display", bool_to_symbol(logger.get("streaming_display", True))),
        ("Progress Display", bool_to_symbol(logger.get("progress_display", True))),
        ("Show Chat", bool_to_symbol(logger.get("show_chat", True))),
        ("Show Tools", bool_to_symbol(logger.get("show_tools", True))),
        ("Truncate Tools", bool_to_symbol(logger.get("truncate_tools", True))),
        ("Enable Markup", bool_to_symbol(logger.get("enable_markup", True))),
        ("Prompt Marks", bool_to_symbol(logger.get("enable_prompt_marks", False))),
        ("Timeline Steps", f"[green]{timeline_steps}[/green]"),
        ("Timeline Interval", f"[green]{format_step_interval(timeline_step_seconds)}[/green]"),
    ]

    # Add rows in two-column layout, styling some values in green
    for i in range(0, len(settings_data), 2):
        left_setting, left_value = settings_data[i]
        # Style certain values in green (MCP-UI is already pre-styled)
        if left_setting in ("Log Level", "Log Type"):
            left_value = f"[green]{left_value}[/green]"
        if i + 1 < len(settings_data):
            right_setting, right_value = settings_data[i + 1]
            if right_setting in ("Log Level", "Log Type"):
                right_value = f"[green]{right_value}[/green]"
            logger_table.add_row(left_setting, left_value, right_setting, right_value)
        else:
            # Odd number of settings - fill right column with empty strings
            logger_table.add_row(left_setting, left_value, "", "")

    _print_section_header("Application Settings", color="blue")
    console.print(logger_table)

    # API keys panel with two-column layout
    keys_table = Table(show_header=True, box=None)
    keys_table.add_column("Provider", style="white", header_style="bold bright_white")
    keys_table.add_column("Env", justify="center", header_style="bold bright_white")
    keys_table.add_column("Config", justify="center", header_style="bold bright_white")
    keys_table.add_column("Active Key", style="green", header_style="bold bright_white")
    keys_table.add_column("Provider", style="white", header_style="bold bright_white")
    keys_table.add_column("Env", justify="center", header_style="bold bright_white")
    keys_table.add_column("Config", justify="center", header_style="bold bright_white")
    keys_table.add_column("Active Key", style="green", header_style="bold bright_white")

    def format_provider_row(provider, status):
        """Format a single provider's status for display."""
        # Environment key indicator
        if status["env"] and status["config"]:
            # Both exist but config takes precedence (env is present but not active)
            env_status = "[yellow]✓[/yellow]"
        elif status["env"]:
            # Only env exists and is active
            env_status = "[bold green]✓[/bold green]"
        else:
            # No env key
            env_status = "[dim]✗[/dim]"

        # Config file key indicator
        if status["config"]:
            # Config exists and takes precedence (is active)
            config_status = "[bold green]✓[/bold green]"
        else:
            # No config key
            config_status = "[dim]✗[/dim]"

        # Display active key
        if status["config"]:
            # Config key is active
            active = f"[bold green]{status['config']}[/bold green]"
        elif status["env"]:
            # Env key is active
            active = f"[bold green]{status['env']}[/bold green]"
        elif provider == "generic":
            # Generic provider uses "ollama" as a default when no key is set
            active = "[green]ollama (default)[/green]"
        else:
            # No key available for other providers
            active = "[dim]Not configured[/dim]"

        # Get the proper display name for the provider
        from fast_agent.llm.provider_types import Provider

        provider_enum = Provider(provider)
        display_name = provider_enum.display_name

        return display_name, env_status, config_status, active

    # Split providers into two columns
    providers_list = list(api_keys.items())
    mid_point = (len(providers_list) + 1) // 2  # Round up for odd numbers

    for i in range(mid_point):
        # Left column
        left_provider, left_status = providers_list[i]
        left_data = format_provider_row(left_provider, left_status)

        # Right column (if exists)
        if i + mid_point < len(providers_list):
            right_provider, right_status = providers_list[i + mid_point]
            right_data = format_provider_row(right_provider, right_status)
            # Add row with both columns
            keys_table.add_row(*left_data, *right_data)
        else:
            # Add row with only left column (right column empty)
            keys_table.add_row(*left_data, "", "", "", "")

    # API Keys section
    _print_section_header("API Keys", color="blue")
    console.print(keys_table)
    console.print("[dim]Use [cyan]fast-agent check models[/cyan] to see/configure models.[/dim]")

    # Codex OAuth panel (separate from API keys)
    try:
        from datetime import datetime

        from fast_agent.llm.provider.openai.codex_oauth import get_codex_token_status

        codex_status = get_codex_token_status()
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
    except Exception:
        pass

    # MCP Servers panel (shown after API Keys)
    if config_summary.get("status") == "parsed":
        mcp_servers = config_summary.get("mcp_servers", [])
        if mcp_servers:
            from fast_agent.config import MCPServerSettings
            from fast_agent.mcp.oauth_client import compute_server_identity

            servers_table = Table(show_header=True, box=None)
            servers_table.add_column("Name", style="white", header_style="bold bright_white")
            servers_table.add_column("Transport", style="white", header_style="bold bright_white")
            servers_table.add_column("Command/URL", header_style="bold bright_white")
            servers_table.add_column("OAuth", header_style="bold bright_white")
            servers_table.add_column("Token", header_style="bold bright_white")

            for server in mcp_servers:
                name = server["name"]
                transport = server["transport"]

                # Show either command or URL based on transport type
                if transport == "STDIO":
                    command_url = server["command"] or "[dim]Not configured[/dim]"
                else:  # SSE
                    command_url = server["url"] or "[dim]Not configured[/dim]"

                # Style configured command/url in green (keep "Not configured" dim)
                if "Not configured" not in command_url:
                    command_url = f"[green]{command_url}[/green]"

                # OAuth status and token presence
                # Default for unsupported transports (e.g., STDIO): show "-" rather than "off"
                oauth_status = "[dim]-[/dim]"
                token_status = "[dim]n/a[/dim]"
                # Attempt to reconstruct minimal server settings for identity check
                try:
                    cfg = MCPServerSettings(
                        name=name,
                        transport="sse"
                        if transport == "SSE"
                        else ("stdio" if transport == "STDIO" else "http"),
                        url=(server.get("url") or None),
                        auth=server.get("auth") if isinstance(server.get("auth"), dict) else None,
                    )
                except Exception:
                    cfg = None

                if cfg and cfg.transport in ("http", "sse"):
                    # Determine if OAuth is enabled for this server
                    oauth_enabled = True
                    if cfg.auth is not None and hasattr(cfg.auth, "oauth"):
                        oauth_enabled = bool(getattr(cfg.auth, "oauth"))
                    oauth_status = "[green]on[/green]" if oauth_enabled else "[dim]off[/dim]"

                    # Only check token presence when using keyring persist
                    persist = "keyring"
                    if cfg.auth is not None and hasattr(cfg.auth, "persist"):
                        persist = getattr(cfg.auth, "persist") or "keyring"
                    if (
                        keyring
                        and keyring_status.writable
                        and persist == "keyring"
                        and oauth_enabled
                    ):
                        identity = compute_server_identity(cfg)
                        tkey = f"oauth:tokens:{identity}"
                        try:
                            has = keyring.get_password("fast-agent-mcp", tkey) is not None
                        except Exception:
                            has = False
                        token_status = "[bold green]✓[/bold green]" if has else "[dim]✗[/dim]"
                    elif persist == "keyring" and not keyring_status.available and oauth_enabled:
                        token_status = "[red]not available[/red]"
                    elif persist == "keyring" and not keyring_status.writable and oauth_enabled:
                        token_status = "[yellow]not writable[/yellow]"
                    elif persist == "memory" and oauth_enabled:
                        token_status = "[yellow]memory[/yellow]"

                servers_table.add_row(name, transport, command_url, oauth_status, token_status)

            _print_section_header("MCP Servers", color="blue")
            console.print(servers_table)

    _print_section_header("Agent Skills", color="blue")
    if skills_dirs:
        if len(skills_dirs) == 1:
            console.print(f"Directory: [green]{_relative_path(skills_dirs[0])}[/green]")
        else:
            console.print("Directories:")
            for directory in skills_dirs:
                console.print(f"- [green]{_relative_path(directory)}[/green]")

        if skills_manifests or skill_errors:
            skills_table = Table(show_header=True, box=None)
            skills_table.add_column("Name", style="cyan", header_style="bold bright_white")
            skills_table.add_column("Description", style="white", header_style="bold bright_white")
            skills_table.add_column("Source", style="dim", header_style="bold bright_white")
            skills_table.add_column("Status", style="green", header_style="bold bright_white")

            for manifest in skills_manifests:
                source_display = _relative_path(manifest.path.parent)

                skills_table.add_row(
                    manifest.name,
                    _truncate(manifest.description or ""),
                    source_display,
                    "[green]ok[/green]",
                )

            for error in skill_errors:
                error_path_str = error.get("path", "")
                source_display = "[dim]n/a[/dim]"
                if error_path_str:
                    error_path = Path(error_path_str)
                    source_display = _relative_path(error_path.parent)
                message = error.get("error", "Failed to parse skill manifest")
                skills_table.add_row(
                    "[red]—[/red]",
                    "[red]n/a[/red]",
                    source_display,
                    f"[red]{_truncate(message, 60)}[/red]",
                )

            console.print(skills_table)
        else:
            console.print("[yellow]No skills found in the configured directories[/yellow]")
    else:
        console.print(
            "[dim]Agent Skills not configured. Go to https://fast-agent.ai/agents/skills/[/dim]"
        )

    server_names: set[str] | None = None
    if config_summary.get("status") == "parsed":
        mcp_servers = config_summary.get("mcp_servers", [])
        if isinstance(mcp_servers, list):
            server_names = {
                server.get("name", "")
                for server in mcp_servers
                if isinstance(server, dict) and server.get("name")
            }

    _print_section_header("Agent Cards", color="blue")
    env_paths = resolve_environment_paths(cwd=cwd, override=environment_override)
    card_directories = [
        ("Agent Cards", env_paths.agent_cards),
        ("Tool Cards", env_paths.tool_cards),
    ]
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

    api_warning_messages: list[str] = []
    warned_cards: set[str] = set()
    default_agent_names: list[str] = []
    default_agent_seen: set[str] = set()

    def _should_warn_for_provider(provider: Provider) -> bool:
        if provider in {Provider.FAST_AGENT, Provider.GENERIC}:
            return False
        if provider == Provider.GOOGLE:
            cfg = config_summary.get("config") if config_summary.get("status") == "parsed" else {}
            google_cfg = cfg.get("google", {}) if isinstance(cfg, dict) else {}
            vertex_cfg = google_cfg.get("vertex_ai", {}) if isinstance(google_cfg, dict) else {}
            if isinstance(vertex_cfg, dict) and vertex_cfg.get("enabled") is True:
                return False
        return True

    for label, directory in card_directories:
        if not directory.is_dir():
            continue
        console.print(f"{label} Directory: [green]{_relative_path(directory)}[/green]")

        entries = scan_agent_card_directory(
            directory,
            server_names=server_names,
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
            error_messages = entry.errors
            runtime_mcp_count = 0
            if entry.ignored_reason:
                status = f"[dim]ignored - {entry.ignored_reason}[/dim]"
            elif error_messages:
                error_text = _truncate(error_messages[0], 60)
                if len(error_messages) > 1:
                    error_text = f"{error_text} (+{len(error_messages) - 1} more)"
                status = f"[red]{error_text}[/red]"
            else:
                status = "[green]ok[/green]"

            if not entry.errors and entry.ignored_reason is None and entry.name not in warned_cards:
                try:
                    from fast_agent.core.agent_card_loader import load_agent_cards

                    cards = load_agent_cards(entry.path)
                except Exception:
                    cards = []

                for card in cards:
                    config = card.agent_data.get("config")
                    runtime_mcp_count += len(getattr(config, "mcp_connect", []) or []) if config else 0
                    if config and getattr(config, "default", False):
                        if card.name not in default_agent_seen:
                            default_agent_names.append(card.name)
                            default_agent_seen.add(card.name)
                    model = config.model if config else None
                    if not model:
                        continue
                    try:
                        model_config = ModelFactory.parse_model_string(model)
                    except ModelConfigError:
                        continue
                    provider = model_config.provider
                    if not _should_warn_for_provider(provider):
                        continue
                    key_status = api_keys.get(provider.config_name)
                    if key_status and not key_status["env"] and not key_status["config"]:
                        api_warning_messages.append(
                            f'Warning: Card "{card.name}" uses model "{model}" '
                            f"({provider.display_name}) but no API key configured."
                        )
                        warned_cards.add(card.name)

            if runtime_mcp_count > 0 and not entry.errors and entry.ignored_reason is None:
                plural = "entry" if runtime_mcp_count == 1 else "entries"
                status = f"[green]ok[/green] [dim](mcp_connect: {runtime_mcp_count} {plural})[/dim]"

            cards_table.add_row(
                entry.name,
                entry.type,
                _relative_path(entry.path),
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

    # Show help tips
    if config_status == "error" or secrets_status == "error" or effective_settings_error:
        console.print("\n[bold]Config File Issues:[/bold]")
        if effective_settings_error:
            console.print(f"[orange_red1]{effective_settings_error}[/orange_red1]")
        console.print("Fix the YAML syntax errors in your configuration files")
    elif config_status == "not_found" or secrets_status == "not_found":
        console.print("\n[bold]Setup Tips:[/bold]")
        console.print(
            "Run [cyan]fast-agent scaffold[/cyan] to create configuration files. Visit [cyan][link=https://fast-agent.ai]fast-agent.ai[/link][/cyan] for configuration guides. "
        )

    if all(
        not api_keys[provider]["env"] and not api_keys[provider]["config"] for provider in api_keys
    ):
        console.print(
            "\n[yellow]No API keys configured. Set up API keys to use LLM services:[/yellow]"
        )
        console.print("1. Add keys to fastagent.secrets.yaml")
        env_vars = ", ".join(
            [
                ProviderKeyManager.get_env_key_name(p.config_name)
                for p in Provider
                if p != Provider.FAST_AGENT
            ]
        )
        console.print(f"2. Or set environment variables ({env_vars})")


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
) -> None:
    """Show model catalog provider guidance or provider-specific model entries."""
    env_dir = _context_env_dir(ctx)

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
