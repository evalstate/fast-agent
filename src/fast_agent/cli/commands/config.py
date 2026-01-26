"""Configuration command for fast-agent settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from ruamel.yaml import YAML

from fast_agent.config import Settings, ShellSettings
from fast_agent.human_input.form_fields import FormSchema, boolean, integer, string
from fast_agent.human_input.simple_form import form_sync

app = typer.Typer(help="Configure fast-agent settings interactively.")

# Use round-trip mode to preserve comments and formatting
_yaml = YAML()
_yaml.preserve_quotes = True

# Common option for specifying config file path
ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config file (default: auto-discover fastagent.config.yaml)",
        exists=False,  # Allow non-existent files (will be created)
    ),
]


def _find_config_file() -> Path | None:
    """Find the config file in the current directory or parent directories."""
    return Settings.find_config()


def _load_config(config_path: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load config file, creating if needed. Returns (config, path).

    Args:
        config_path: Optional explicit path to config file. If not provided,
                     auto-discovers fastagent.config.yaml in current or parent directories.
    """
    if config_path is not None:
        # Use explicit path
        resolved_path = config_path.resolve()
        if resolved_path.exists():
            with open(resolved_path) as f:
                config = _yaml.load(f) or {}
            return config, resolved_path
        # File doesn't exist yet - will be created
        return {}, resolved_path

    # Auto-discover config file
    found_path = _find_config_file()

    if found_path is None:
        # Create new config in current directory
        found_path = Path.cwd() / "fastagent.config.yaml"
        return {}, found_path

    if found_path.exists():
        with open(found_path) as f:
            config = _yaml.load(f) or {}
        return config, found_path

    return {}, found_path


def _save_config(config: dict[str, Any], config_path: Path) -> None:
    """Save config to file, preserving comments."""
    with open(config_path, "w") as f:
        _yaml.dump(config, f)


def _get_field_description(field_name: str) -> str:
    """Get description from ShellSettings model field."""
    field_info = ShellSettings.model_fields.get(field_name)
    return field_info.description if field_info and field_info.description else ""


def _build_shell_form(current: ShellSettings) -> FormSchema:
    """Build form schema for shell settings from ShellSettings model."""
    # Build form dynamically using descriptions from model fields
    fields: dict[str, Any] = {}

    for name, field_info in ShellSettings.model_fields.items():
        # Skip internal fields
        if name in ("model_config", "interactive_use_pty"):
            continue

        desc = field_info.description or ""
        current_value = getattr(current, name)
        annotation = field_info.annotation

        # Determine field type and build appropriate form field
        if annotation is bool:
            fields[name] = boolean(
                title=name.replace("_", " ").title(),
                description=desc,
                default=current_value,
            )
        elif annotation is int:
            fields[name] = integer(
                title=name.replace("_", " ").title(),
                description=desc,
                default=current_value,
                minimum=1,
                maximum=3600 if "timeout" in name or "interval" in name else 300,
            )
        elif annotation == int | None:
            # Handle optional integers (like output_display_lines, output_byte_limit)
            max_val = 1000 if "lines" in name else 1048576
            fields[name] = integer(
                title=name.replace("_", " ").title(),
                description=f"{desc} (0 = auto/unlimited)",
                default=current_value if current_value is not None else 0,
                minimum=0,
                maximum=max_val,
            )

    return FormSchema(**fields)


def _build_model_form(current_model: str | None) -> FormSchema:
    """Build form schema for model settings."""
    return FormSchema(
        default_model=string(
            title="Default Model",
            description="Format: provider.model_name (e.g., anthropic.claude-sonnet-4-20250514)",
            default=current_model or "",
            max_length=100,
        ),
    )


@app.command("shell")
def config_shell(config: ConfigOption = None) -> None:
    """Configure shell execution settings interactively."""
    from rich import print as rprint

    config_data, config_path = _load_config(config)

    # Load current settings
    shell_config = config_data.get("shell_execution", {}) or {}
    current = ShellSettings(**shell_config)

    # Build and show form
    schema = _build_shell_form(current)
    result = form_sync(
        schema,
        message="Configure shell execution behavior",
        title="Shell Settings",
    )

    if result is None:
        rprint("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    # Process results - handle special cases
    shell_updates: dict[str, Any] = {}

    if result.get("timeout_seconds"):
        shell_updates["timeout_seconds"] = result["timeout_seconds"]

    if result.get("warning_interval_seconds"):
        shell_updates["warning_interval_seconds"] = result["warning_interval_seconds"]

    # Handle output_display_lines: 0 means None (unlimited)
    output_lines = result.get("output_display_lines", 0)
    if output_lines == 0:
        shell_updates["output_display_lines"] = None
    else:
        shell_updates["output_display_lines"] = output_lines

    # Handle output_byte_limit: 0 means None (auto)
    byte_limit = result.get("output_byte_limit", 0)
    if byte_limit == 0:
        shell_updates["output_byte_limit"] = None
    else:
        shell_updates["output_byte_limit"] = byte_limit

    shell_updates["show_bash"] = result.get("show_bash", True)

    # Update config
    if "shell_execution" not in config_data or config_data["shell_execution"] is None:
        config_data["shell_execution"] = {}
    config_data["shell_execution"].update(shell_updates)

    # Save
    _save_config(config_data, config_path)
    rprint(f"[green]Shell settings saved to {config_path}[/green]")


@app.command("model")
def config_model(config: ConfigOption = None) -> None:
    """Configure default model settings interactively."""
    from rich import print as rprint

    config_data, config_path = _load_config(config)

    # Load current settings
    current_model = config_data.get("default_model")

    # Build and show form
    schema = _build_model_form(current_model)
    result = form_sync(
        schema,
        message="Configure the default model for agents",
        title="Model Settings",
    )

    if result is None:
        rprint("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    # Update config
    model_value = result.get("default_model", "").strip()
    if model_value:
        config_data["default_model"] = model_value
    elif "default_model" in config_data:
        # Clear if empty
        del config_data["default_model"]

    # Save
    _save_config(config_data, config_path)
    rprint(f"[green]Model settings saved to {config_path}[/green]")


@app.callback(invoke_without_command=True)
def config_main(ctx: typer.Context) -> None:
    """Configure fast-agent settings interactively.

    Use subcommands to configure specific areas:
      - shell: Shell execution settings (timeout, output limits, etc.)
      - model: Default model configuration
    """
    if ctx.invoked_subcommand is None:
        # Show help if no subcommand
        from rich import print as rprint
        from rich.table import Table

        rprint("\n[bold]fast-agent config[/bold] - Interactive configuration\n")

        table = Table(show_header=True, box=None)
        table.add_column("Subcommand", style="green")
        table.add_column("Description")

        table.add_row("shell", "Configure shell execution settings")
        table.add_row("model", "Configure default model")

        rprint(table)
        rprint("\nExample: [cyan]fast-agent config shell[/cyan]")
