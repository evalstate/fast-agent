"""Command to check FastAgent configuration."""

import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mcp_agent.llm.provider_key_manager import API_KEY_HINT_TEXT, ProviderKeyManager
from mcp_agent.llm.provider_types import Provider

app = typer.Typer(
    help="Check and diagnose FastAgent configuration",
    no_args_is_help=False,  # Allow showing our custom help instead
)
console = Console()


def find_config_files(start_path: Path) -> dict[str, Optional[Path]]:
    """Find FastAgent configuration files, preferring secrets file next to config file."""
    from mcp_agent.config import find_fastagent_config_files

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
    from mcp_agent.config import Settings

    # Get actual defaults from Settings class
    default_settings = Settings()

    result = {
        "status": "not_found",  # Default status: not found
        "error": None,
        "default_model": default_settings.default_model,
        "logger": {
            "level": default_settings.logger.level,
            "type": default_settings.logger.type,
            "progress_display": default_settings.logger.progress_display,
            "show_chat": default_settings.logger.show_chat,
            "show_tools": default_settings.logger.show_tools,
            "truncate_tools": default_settings.logger.truncate_tools,
            "enable_markup": default_settings.logger.enable_markup,
        },
        "mcp_servers": [],
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

    # System info panel
    system_table = Table(show_header=False, box=None)
    system_table.add_column("Key", style="cyan")
    system_table.add_column("Value")

    system_table.add_row("FastAgent Version", fastagent_version)
    system_table.add_row("Platform", system_info["platform"])
    system_table.add_row("Python Version", ".".join(system_info["python_version"].split(".")[:3]))
    system_table.add_row("Python Path", system_info["python_path"])

    console.print(
        Panel(system_table, title="System Information", title_align="left", border_style="blue")
    )

    # Configuration files panel
    config_path = config_files["config"]
    secrets_path = config_files["secrets"]

    files_table = Table(show_header=False, box=None)
    files_table.add_column("Setting", style="cyan")
    files_table.add_column("Value")

    # Show secrets file status
    secrets_status = secrets_summary.get("status", "not_found")
    if secrets_status == "not_found":
        files_table.add_row("Secrets File", "[yellow]Not found[/yellow]")
    elif secrets_status == "error":
        files_table.add_row("Secrets File", f"[orange_red1]Errors[/orange_red1] ({secrets_path})")
        files_table.add_row(
            "Secrets Error",
            f"[orange_red1]{secrets_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:  # parsed successfully
        files_table.add_row("Secrets File", f"[green]Found[/green] ({secrets_path})")

    # Show config file status
    config_status = config_summary.get("status", "not_found")
    if config_status == "not_found":
        files_table.add_row("Config File", "[red]Not found[/red]")
    elif config_status == "error":
        files_table.add_row("Config File", f"[orange_red1]Errors[/orange_red1] ({config_path})")
        files_table.add_row(
            "Config Error",
            f"[orange_red1]{config_summary.get('error', 'Unknown error')}[/orange_red1]",
        )
    else:  # parsed successfully
        files_table.add_row("Config File", f"[green]Found[/green] ({config_path})")
        files_table.add_row(
            "Default Model", config_summary.get("default_model", "haiku (system default)")
        )

    console.print(
        Panel(files_table, title="Configuration Files", title_align="left", border_style="blue")
    )

    # Logger Settings panel with two-column layout
    logger = config_summary.get("logger", {})
    logger_table = Table(show_header=True, box=None)
    logger_table.add_column("Setting", style="cyan")
    logger_table.add_column("Value")
    logger_table.add_column("Setting", style="cyan")
    logger_table.add_column("Value")

    def bool_to_symbol(value):
        return "[bold green]✓[/bold green]" if value else "[bold red]✗[/bold red]"

    # Prepare all settings as pairs
    settings_data = [
        ("Logger Level", logger.get("level", "warning (default)")),
        ("Logger Type", logger.get("type", "file (default)")),
        ("Progress Display", bool_to_symbol(logger.get("progress_display", True))),
        ("Show Chat", bool_to_symbol(logger.get("show_chat", True))),
        ("Show Tools", bool_to_symbol(logger.get("show_tools", True))),
        ("Truncate Tools", bool_to_symbol(logger.get("truncate_tools", True))),
        ("Enable Markup", bool_to_symbol(logger.get("enable_markup", True))),
    ]

    # Add rows in two-column layout
    for i in range(0, len(settings_data), 2):
        left_setting, left_value = settings_data[i]
        if i + 1 < len(settings_data):
            right_setting, right_value = settings_data[i + 1]
            logger_table.add_row(left_setting, left_value, right_setting, right_value)
        else:
            # Odd number of settings - fill right column with empty strings
            logger_table.add_row(left_setting, left_value, "", "")

    console.print(
        Panel(logger_table, title="Logger Settings", title_align="left", border_style="blue")
    )

    # API keys panel with two-column layout
    keys_table = Table(show_header=True, box=None)
    keys_table.add_column("Provider", style="cyan")
    keys_table.add_column("Env", justify="center")
    keys_table.add_column("Config", justify="center")
    keys_table.add_column("Active Key", style="green")
    keys_table.add_column("Provider", style="cyan")
    keys_table.add_column("Env", justify="center")
    keys_table.add_column("Config", justify="center")
    keys_table.add_column("Active Key", style="green")

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
        from mcp_agent.llm.provider_types import Provider

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

    # Print the API Keys panel (fix: this was missing)
    keys_panel = Panel(keys_table, title="API Keys", title_align="left", border_style="blue")
    console.print(keys_panel)

    # MCP Servers panel (shown after API Keys)
    if config_summary.get("status") == "parsed":
        mcp_servers = config_summary.get("mcp_servers", [])
        if mcp_servers:
            servers_table = Table(show_header=True, box=None)
            servers_table.add_column("Name", style="cyan")
            servers_table.add_column("Transport", style="magenta")
            servers_table.add_column("Command/URL")

            for server in mcp_servers:
                name = server["name"]
                transport = server["transport"]

                # Show either command or URL based on transport type
                if transport == "STDIO":
                    command_url = server["command"] or "[dim]Not configured[/dim]"
                    servers_table.add_row(name, transport, command_url)
                else:  # SSE
                    command_url = server["url"] or "[dim]Not configured[/dim]"
                    servers_table.add_row(name, transport, command_url)

            console.print(
                Panel(servers_table, title="MCP Servers", title_align="left", border_style="blue")
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
