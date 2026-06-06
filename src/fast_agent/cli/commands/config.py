"""Configuration command for fast-agent settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from ruamel.yaml import YAML

from fast_agent.config import (
    SHELL_WRITE_TEXT_FILE_MODE_HELP,
    SHELL_WRITE_TEXT_FILE_MODES,
    LoggerSettings,
    ShellSettings,
    load_implicit_settings,
    normalize_shell_write_text_file_mode,
)
from fast_agent.home import (
    PREFERRED_CONFIG_FILENAME,
    discover_config_files,
    resolve_fast_agent_home,
)
from fast_agent.human_input.form_fields import FormSchema, boolean, integer, string
from fast_agent.human_input.simple_form import form_sync
from fast_agent.types.streaming import STREAMING_MODE_HELP, normalize_streaming_mode
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.text import strip_to_none

app = typer.Typer(help="Configure fast-agent settings interactively.", add_completion=False)

DISPLAY_BOOL_DEFAULTS: dict[str, bool] = {
    "render_fences_with_syntax": True,
    "code_word_wrap": True,
    "progress_display": True,
    "show_chat": True,
    "stream_reprint_banner": True,
    "show_tools": True,
    "truncate_tools": True,
    "enable_markup": True,
    "enable_prompt_marks": True,
}

# Use round-trip mode to preserve comments and formatting
_yaml = YAML()
_yaml.preserve_quotes = True

# Common option for specifying config file path
ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config file (default: environment-dir fast-agent.yaml)",
        exists=False,  # Allow non-existent files (will be created)
    ),
]


def _default_config_file() -> Path:
    """Return the discovered implicit config path, or the default env path if none exist."""
    cwd = Path.cwd()
    home = resolve_fast_agent_home(cwd=cwd)
    discovery = discover_config_files(cwd=cwd, home=home)
    if discovery.config_path is not None:
        return discovery.config_path
    if home is not None:
        return home.path / PREFERRED_CONFIG_FILENAME
    return cwd / PREFERRED_CONFIG_FILENAME


def _load_config(config_path: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Load config file, creating if needed. Returns (config, path).

    Args:
        config_path: Optional explicit path to config file. If not provided,
                     edits the discovered config (env, cwd, legacy), creating
                     the environment-dir config if none exist.
    """
    if config_path is not None:
        # Use explicit path
        resolved_path = config_path.resolve()
        if resolved_path.exists():
            with resolved_path.open() as f:
                config = _yaml.load(f) or {}
            return config, resolved_path
        # File doesn't exist yet - will be created
        return {}, resolved_path

    found_path = _default_config_file().resolve()

    if found_path.exists():
        with found_path.open() as f:
            config = _yaml.load(f) or {}
        return config, found_path

    return {}, found_path


def _load_effective_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load the effective settings used to prefill interactive forms."""
    if config_path is not None:
        resolved_path = config_path.resolve()
        if resolved_path.exists():
            with resolved_path.open() as f:
                return _yaml.load(f) or {}
        return {}

    discovered_config, _ = load_implicit_settings(start_path=Path.cwd())
    return discovered_config


def _overlay_section_updates(
    *,
    minimal_write: bool,
    updates: dict[str, Any],
    baseline: ShellSettings | LoggerSettings,
) -> dict[str, Any]:
    """Return the raw section contents to persist for the target file."""
    if not minimal_write:
        return updates

    baseline_values = baseline.model_dump(mode="python")
    return {key: value for key, value in updates.items() if baseline_values.get(key) != value}


def _replace_config_section(
    config_data: dict[str, Any],
    *,
    section_name: str,
    section_updates: dict[str, Any],
) -> None:
    """Replace a config section in-place, removing cleared keys and empty sections."""
    current_section = config_data.get(section_name)
    if current_section is None or not isinstance(current_section, dict):
        current_section = {}
        if section_updates:
            config_data[section_name] = current_section

    if not section_updates:
        config_data.pop(section_name, None)
        return

    for key in list(current_section):
        if key not in section_updates:
            current_section.pop(key, None)

    for key, value in section_updates.items():
        current_section[key] = value


def _save_config(config: dict[str, Any], config_path: Path) -> None:
    """Save config to file, preserving comments."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w") as f:
        _yaml.dump(config, f)


def _settings_field_description(
    settings_type: type[ShellSettings] | type[LoggerSettings],
    field_name: str,
) -> str:
    """Get description from a settings model field."""
    field_info = settings_type.model_fields.get(field_name)
    return field_info.description if field_info and field_info.description else ""


def _field_title(field_name: str) -> str:
    return field_name.replace("_", " ").title()


def _build_shell_form(current: ShellSettings) -> FormSchema:
    """Build form schema for shell settings from ShellSettings model."""
    # Build form dynamically using descriptions from model fields
    fields: dict[str, Any] = {}
    current_values = current.model_dump()

    for name, field_info in ShellSettings.model_fields.items():
        # Skip internal fields
        if name in ("model_config", "interactive_use_pty"):
            continue

        desc = field_info.description or ""
        current_value = current_values[name]
        annotation = field_info.annotation

        if name == "write_text_file_mode":
            fields[name] = string(
                title=_field_title(name),
                description=f"{desc} ({SHELL_WRITE_TEXT_FILE_MODE_HELP})",
                default=str(current_value or "auto"),
                max_length=max(len(mode) for mode in SHELL_WRITE_TEXT_FILE_MODES),
            )
            continue

        # Determine field type and build appropriate form field
        if annotation is bool:
            fields[name] = boolean(
                title=_field_title(name),
                description=desc,
                default=current_value,
            )
        elif annotation is int:
            fields[name] = integer(
                title=_field_title(name),
                description=desc,
                default=current_value,
                minimum=1,
                maximum=3600 if "timeout" in name or "interval" in name else 300,
            )
        elif annotation == int | None:
            # Handle optional integers (like output_display_lines, output_byte_limit)
            max_val = 1000 if "lines" in name else 1048576
            if name == "output_display_lines":
                fields[name] = integer(
                    title=_field_title(name),
                    description=f"{desc} (-1 = show all, 0 = show none)",
                    default=current_value if current_value is not None else -1,
                    minimum=-1,
                    maximum=max_val,
                )
                continue

            if name == "output_byte_limit":
                fields[name] = integer(
                    title=_field_title(name),
                    description=f"{desc} (0 = auto)",
                    default=current_value if current_value is not None else 0,
                    minimum=0,
                    maximum=max_val,
                )
                continue

            fields[name] = integer(
                title=_field_title(name),
                description=desc,
                default=current_value if current_value is not None else 0,
                minimum=0,
                maximum=max_val,
            )

    return FormSchema(**fields)


def _build_display_form(current: LoggerSettings) -> FormSchema:
    """Build form schema for display-related logger settings."""
    return FormSchema(
        theme_file=string(
            title="Theme File",
            description=f"{_settings_field_description(LoggerSettings, 'theme_file')} (blank = default)",
            default=current.theme_file or "",
            max_length=240,
        ),
        code_theme=string(
            title="Code Theme",
            description=(
                f"{_settings_field_description(LoggerSettings, 'code_theme')} "
                "(examples: native, monokai, emacs, ansi_dark)"
            ),
            default=current.code_theme or "native",
            max_length=80,
        ),
        streaming=string(
            title="Streaming Mode",
            description=(
                f"{_settings_field_description(LoggerSettings, 'streaming')} "
                f"({STREAMING_MODE_HELP})"
            ),
            default=current.streaming,
            max_length=16,
        ),
        render_fences_with_syntax=boolean(
            title="Syntax Fences",
            description=_settings_field_description(LoggerSettings, "render_fences_with_syntax"),
            default=current.render_fences_with_syntax,
        ),
        code_word_wrap=boolean(
            title="Wrap Code",
            description=_settings_field_description(LoggerSettings, "code_word_wrap"),
            default=current.code_word_wrap,
        ),
        apply_patch_preview_max_lines=integer(
            title="Apply Patch Preview Lines",
            description=(
                f"{_settings_field_description(LoggerSettings, 'apply_patch_preview_max_lines')} "
                "(0 = show all)"
            ),
            default=current.apply_patch_preview_max_lines
            if current.apply_patch_preview_max_lines is not None
            else 0,
            minimum=0,
            maximum=10000,
        ),
        progress_display=boolean(
            title="Progress Display",
            description=_settings_field_description(LoggerSettings, "progress_display"),
            default=current.progress_display,
        ),
        show_chat=boolean(
            title="Show Chat",
            description=_settings_field_description(LoggerSettings, "show_chat"),
            default=current.show_chat,
        ),
        stream_reprint_banner=boolean(
            title="Stream Reprint Banner",
            description=_settings_field_description(LoggerSettings, "stream_reprint_banner"),
            default=current.stream_reprint_banner,
        ),
        show_tools=boolean(
            title="Show Tools",
            description=_settings_field_description(LoggerSettings, "show_tools"),
            default=current.show_tools,
        ),
        truncate_tools=boolean(
            title="Truncate Tools",
            description=_settings_field_description(LoggerSettings, "truncate_tools"),
            default=current.truncate_tools,
        ),
        enable_markup=boolean(
            title="Enable Markup",
            description=_settings_field_description(LoggerSettings, "enable_markup"),
            default=current.enable_markup,
        ),
        enable_prompt_marks=boolean(
            title="Prompt Marks",
            description=_settings_field_description(LoggerSettings, "enable_prompt_marks"),
            default=current.enable_prompt_marks,
        ),
    )


def _normalize_shell_updates(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize shell form results into config values."""
    shell_updates: dict[str, Any] = {}

    timeout_seconds = positive_int_or_none(result.get("timeout_seconds"))
    if timeout_seconds is not None:
        shell_updates["timeout_seconds"] = timeout_seconds

    warning_interval_seconds = positive_int_or_none(result.get("warning_interval_seconds"))
    if warning_interval_seconds is not None:
        shell_updates["warning_interval_seconds"] = warning_interval_seconds

    # output_display_lines: -1 means show all (None), 0 means show none, >0 means show amount.
    output_lines = result.get("output_display_lines", -1)
    if output_lines == -1:
        shell_updates["output_display_lines"] = None
    else:
        shell_updates["output_display_lines"] = output_lines

    # output_byte_limit: 0 means auto (None), >0 means explicit cap.
    byte_limit = result.get("output_byte_limit", 0)
    if byte_limit == 0:
        shell_updates["output_byte_limit"] = None
    else:
        shell_updates["output_byte_limit"] = byte_limit

    shell_updates["show_bash"] = result.get("show_bash", True)
    shell_updates["enable_read_text_file"] = result.get("enable_read_text_file", True)

    # write_text_file mode: auto|on|off|apply_patch (defaults to auto).
    mode_raw = result.get("write_text_file_mode", "auto")
    shell_updates["write_text_file_mode"] = normalize_shell_write_text_file_mode(mode_raw) or "auto"

    return shell_updates


def _normalize_display_updates(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize display form results into logger config values."""
    logger_updates: dict[str, Any] = {}

    theme_file = result.get("theme_file", "")
    if isinstance(theme_file, str):
        logger_updates["theme_file"] = strip_to_none(theme_file)

    code_theme = result.get("code_theme", "")
    if isinstance(code_theme, str):
        logger_updates["code_theme"] = strip_to_none(code_theme) or "native"

    streaming = result.get("streaming", "markdown")
    if isinstance(streaming, str):
        logger_updates["streaming"] = normalize_streaming_mode(streaming)

    apply_patch_preview_max_lines = result.get("apply_patch_preview_max_lines", 120)
    if isinstance(apply_patch_preview_max_lines, int):
        logger_updates["apply_patch_preview_max_lines"] = (
            None if apply_patch_preview_max_lines == 0 else apply_patch_preview_max_lines
        )

    for key, default in DISPLAY_BOOL_DEFAULTS.items():
        logger_updates[key] = bool(result.get(key, default))

    return logger_updates


def _form_message(action: str, config_path: Path) -> str:
    return f"{action}\n\nEditing: {config_path}"


@app.command("shell")
def config_shell(config: ConfigOption = None) -> None:
    """Configure shell execution settings interactively."""
    from rich import print as rprint

    config_data, config_path = _load_config(config)
    effective_config = _load_effective_config(config)
    minimal_write = config is None and not config_path.exists()

    # Load current settings
    current = ShellSettings(**(effective_config.get("shell_execution", {}) or {}))

    # Build and show form
    schema = _build_shell_form(current)
    result = form_sync(
        schema,
        message=_form_message("Configure shell execution behavior", config_path),
        title="Shell Settings",
    )

    if result is None:
        rprint("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    # Process results - handle special cases
    shell_updates = _normalize_shell_updates(result)
    baseline = ShellSettings()
    persisted_updates = _overlay_section_updates(
        minimal_write=minimal_write,
        updates=shell_updates,
        baseline=baseline,
    )

    _replace_config_section(
        config_data,
        section_name="shell_execution",
        section_updates=persisted_updates,
    )

    # Save
    _save_config(config_data, config_path)
    rprint(f"[green]Shell settings saved to {config_path}[/green]")


@app.command("display")
def config_display(config: ConfigOption = None) -> None:
    """Configure display and markdown rendering settings interactively."""
    from rich import print as rprint

    config_data, config_path = _load_config(config)
    effective_config = _load_effective_config(config)
    minimal_write = config is None and not config_path.exists()

    current = LoggerSettings(**(effective_config.get("logger", {}) or {}))

    schema = _build_display_form(current)
    result = form_sync(
        schema,
        message=_form_message("Configure display and markdown rendering behavior", config_path),
        title="Display Settings",
    )

    if result is None:
        rprint("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit(0)

    logger_updates = _normalize_display_updates(result)
    baseline = LoggerSettings()
    persisted_updates = _overlay_section_updates(
        minimal_write=minimal_write,
        updates=logger_updates,
        baseline=baseline,
    )

    _replace_config_section(
        config_data,
        section_name="logger",
        section_updates=persisted_updates,
    )
    logger_config = config_data.get("logger")
    if isinstance(logger_config, dict):
        if logger_config.get("theme_file") in ("", None):
            logger_config.pop("theme_file", None)
        if logger_config.get("code_theme") in ("", "native"):
            logger_config.pop("code_theme", None)

    _save_config(config_data, config_path)
    rprint(f"[green]Display settings saved to {config_path}[/green]")


@app.callback(invoke_without_command=True)
def config_main(ctx: typer.Context) -> None:
    """Configure fast-agent settings interactively.

    Use subcommands to configure specific areas:
      - shell: Shell execution settings (timeout, output limits, etc.)
      - display: Console display and markdown rendering
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
        table.add_row("display", "Configure display and markdown rendering")

        rprint(table)
        rprint("\nExample: [cyan]fast-agent config shell[/cyan]")
