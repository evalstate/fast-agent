"""Command to check FastAgent configuration."""

import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.table import Table
from rich.text import Text

from fast_agent.llm.provider_key_manager import API_KEY_HINT_TEXT, ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.skills import SkillRegistry
from fast_agent.ui.console import console

app = typer.Typer(
    help="Check and diagnose FastAgent configuration",
    no_args_is_help=False,  # Allow showing our custom help instead
)


def find_config_files(start_path: Path) -> dict[str, Optional[Path]]:
    """Find FastAgent configuration files, preferring secrets file next to config file."""
    from fast_agent.config import find_fastagent_config_files

    config_path, secrets_path = find_fastagent_config_files(start_path)
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


def get_secrets_summary(secrets_path: Optional[Path]) -> dict:
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
    import os

    results = {
        provider.value: {"env": "", "config": ""}
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

    for provider in results:
        # Always check environment variables first
        env_key_name = ProviderKeyManager.get_env_key_name(provider)
        env_key_value = os.environ.get(env_key_name)
        if env_key_value:
            if len(env_key_value) > 5:
                results[provider]["env"] = f"...{env_key_value[-5:]}"
            else:
                results[provider]["env"] = "...***"

        # Special handling for Azure: support api_key and DefaultAzureCredential
        if provider == "azure":
            # Prefer secrets if present, else fallback to config
            azure_cfg = {}
            if secrets_status == "parsed" and "azure" in secrets:
                azure_cfg = secrets.get("azure", {})
            elif config_azure:
                azure_cfg = config_azure

            use_default_cred = azure_cfg.get("use_default_azure_credential", False)
            base_url = azure_cfg.get("base_url")
            if use_default_cred and base_url:
                results[provider]["config"] = "DefaultAzureCredential"
                continue

        # Check secrets file if it was parsed successfully
        if secrets_status == "parsed":
            config_key = ProviderKeyManager.get_config_file_key(provider, secrets)
            if config_key and config_key != API_KEY_HINT_TEXT:
                if len(config_key) > 5:
                    results[provider]["config"] = f"...{config_key[-5:]}"
                else:
                    results[provider]["config"] = "...***"

    return results


def get_fastagent_version() -> str:
    """Get the installed version of FastAgent."""
    try:
        return version("fast-agent-mcp")
    except:  # noqa: E722
        return "unknown"


def get_config_summary(config_path: Optional[Path]) -> dict:
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
        },
        "mcp_ui_mode": default_settings.mcp_ui_mode,
        "timeline": {
            "steps": default_settings.mcp_timeline.steps,
            "step_seconds": default_settings.mcp_timeline.step_seconds,
        },
        "mcp_servers": [],
        "skills_directory": None,
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

                result["mcp_servers"].append(server_info)

        # Skills directory override
        skills_cfg = config.get("skills") if isinstance(config, dict) else None
        if isinstance(skills_cfg, dict):
            directory_value = skills_cfg.get("directory")
            if isinstance(directory_value, str) and directory_value.strip():
                result["skills_directory"] = directory_value.strip()

    except Exception as e:
        # File exists but has parse errors
        result["status"] = "error"
        result["error"] = str(e)
        console.print(f"[red]Error parsing configuration file:[/red] {e}")

    return result


def show_check_summary() -> None:
    """Show a summary of checks with colorful styling."""
    cwd = Path.cwd()
    config_files = find_config_files(cwd)
    system_info = get_system_info()
    config_summary = get_config_summary(config_files["config"])
    secrets_summary = get_secrets_summary(config_files["secrets"])
    api_keys = check_api_keys(secrets_summary, config_summary)
    fastagent_version = get_fastagent_version()

    # Helper to print section headers using the new console_display style
    def _print_section_header(title: str, color: str = "blue") -> None:
        width = console.size.width
        left = f"[{color}]▎[/{color}][dim {color}]▶[/dim {color}] [{color}]{title}[/{color}]"
        left_text = Text.from_markup(left)
        separator_count = max(1, width - left_text.cell_len - 1)

        combined = Text()
        combined.append_text(left_text)
        combined.append(" ")
        combined.append("─" * separator_count, style="dim")

        console.print()
        console.print(combined)
        console.print()

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
    # Also detect whether the backend is actually usable (not the fail backend)
    keyring_usable = False
    try:
        import keyring  # type: ignore

        keyring_backend = keyring.get_keyring()
        keyring_name = getattr(keyring_backend, "name", keyring_backend.__class__.__name__)
        try:
            # Detect the "fail" backend explicitly; it's present but unusable
            from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

            keyring_usable = not isinstance(keyring_backend, FailKeyring)
        except Exception:
            # If we can't import the fail backend marker, assume usable
            keyring_usable = True
    except Exception:
        keyring = None  # type: ignore
        keyring_name = "unavailable"
        keyring_usable = False

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
        default_model_value = config_summary.get("default_model", "gpt-5-mini.low (system default)")
        env_table.add_row("Default Model", f"[green]{default_model_value}[/green]")

    # Keyring backend (always shown in application-level settings)
    if keyring_usable and keyring_name != "unavailable":
        env_table.add_row("Keyring Backend", f"[green]{keyring_name}[/green]")
    else:
        env_table.add_row("Keyring Backend", "[red]not available[/red]")

    console.print(env_table)

    def _relative_path(path: Path) -> str:
        try:
            return str(path.relative_to(cwd))
        except ValueError:
            return str(path)

    skills_override = config_summary.get("skills_directory")
    override_directory = Path(skills_override).expanduser() if skills_override else None
    skills_registry = SkillRegistry(base_dir=cwd, override_directory=override_directory)
    skills_dir = skills_registry.directory
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
                    if keyring and keyring_usable and persist == "keyring" and oauth_enabled:
                        identity = compute_server_identity(cfg)
                        tkey = f"oauth:tokens:{identity}"
                        try:
                            has = keyring.get_password("fast-agent-mcp", tkey) is not None
                        except Exception:
                            has = False
                        token_status = "[bold green]✓[/bold green]" if has else "[dim]✗[/dim]"
                    elif persist == "keyring" and not keyring_usable and oauth_enabled:
                        token_status = "[red]not available[/red]"
                    elif persist == "memory" and oauth_enabled:
                        token_status = "[yellow]memory[/yellow]"

                servers_table.add_row(name, transport, command_url, oauth_status, token_status)

            _print_section_header("MCP Servers", color="blue")
            console.print(servers_table)

    _print_section_header("Agent Skills", color="blue")
    if skills_dir:
        console.print(f"Directory: [green]{_relative_path(skills_dir)}[/green]")

        if skills_manifests or skill_errors:
            skills_table = Table(show_header=True, box=None)
            skills_table.add_column("Name", style="cyan", header_style="bold bright_white")
            skills_table.add_column("Description", style="white", header_style="bold bright_white")
            skills_table.add_column("Source", style="dim", header_style="bold bright_white")
            skills_table.add_column("Status", style="green", header_style="bold bright_white")

            def _truncate(text: str, length: int = 70) -> str:
                if len(text) <= length:
                    return text
                return text[: length - 3] + "..."

            for manifest in skills_manifests:
                try:
                    relative_source = manifest.path.parent.relative_to(skills_dir)
                    source_display = str(relative_source) if relative_source != Path(".") else "."
                except ValueError:
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
                    try:
                        relative_error = error_path.parent.relative_to(skills_dir)
                        source_display = str(relative_error) if relative_error != Path(".") else "."
                    except ValueError:
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
            console.print("[yellow]No skills found in the directory[/yellow]")
    else:
        if skills_registry.override_failed and override_directory:
            console.print(
                f"[red]Override directory not found:[/red] {_relative_path(override_directory)}"
            )
            console.print(
                "[yellow]Default folders were not loaded because the override failed[/yellow]"
            )
        else:
            console.print(
                "[dim]Agent Skills not configured. Go to https://fast-agent.ai/agents/skills/[/dim]"
            )

    # Show help tips
    if config_status == "not_found" or secrets_status == "not_found":
        console.print("\n[bold]Setup Tips:[/bold]")
        console.print(
            "Run [cyan]fast-agent setup[/cyan] to create configuration files. Visit [cyan][link=https://fast-agent.ai]fast-agent.ai[/link][/cyan] for configuration guides. "
        )
    elif config_status == "error" or secrets_status == "error":
        console.print("\n[bold]Config File Issues:[/bold]")
        console.print("Fix the YAML syntax errors in your configuration files")

    if all(
        not api_keys[provider]["env"] and not api_keys[provider]["config"] for provider in api_keys
    ):
        console.print(
            "\n[yellow]No API keys configured. Set up API keys to use LLM services:[/yellow]"
        )
        console.print("1. Add keys to fastagent.secrets.yaml")
        env_vars = ", ".join(
            [
                ProviderKeyManager.get_env_key_name(p.value)
                for p in Provider
                if p != Provider.FAST_AGENT
            ]
        )
        console.print(f"2. Or set environment variables ({env_vars})")


@app.command()
def show(
    path: Optional[str] = typer.Argument(None, help="Path to configuration file to display"),
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
            console.print("Run [cyan]fast-agent setup[/cyan] to create configuration files")
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


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Check and diagnose FastAgent configuration."""
    if ctx.invoked_subcommand is None:
        show_check_summary()
