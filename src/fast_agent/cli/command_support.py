"""Shared helpers for top-level CLI command state."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_agent.utils.text import strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    import typer

    from fast_agent.config import Settings

_MCP_TARGETS_MIGRATION_ERROR = (
    "Value error, `mcp.targets` is no longer supported. Run "
    "`fast-agent config migrate-mcp` to migrate it to `mcp.servers`."
)
_SAFE_VALIDATION_MESSAGES = {
    "extra_forbidden": "Extra inputs are not permitted",
    "missing": "Field required",
    "none_required": "Input should be null",
}


def ensure_context_object(ctx: typer.Context) -> dict[str, Any]:
    """Return a mutable context object dictionary for a Typer command tree."""
    if isinstance(ctx.obj, dict):
        return ctx.obj
    if ctx.obj is None:
        ctx.obj = {}
        return ctx.obj
    return {}


def resolve_context_string_option(
    ctx: typer.Context,
    *,
    key: str,
    command_value: str | None = None,
) -> str | None:
    """Resolve a string option from the current command, then the shared context."""
    if (resolved_value := strip_to_none(command_value)) is not None:
        return resolved_value
    ctx_value = ensure_context_object(ctx).get(key)
    return strip_str_to_none(ctx_value)


def resolve_context_path_option(
    ctx: typer.Context,
    *,
    key: str,
    command_value: Path | None = None,
) -> Path | None:
    """Resolve a path option from the current command, then the shared context."""
    if command_value is not None:
        return command_value
    ctx_value = ensure_context_object(ctx).get(key)
    if isinstance(ctx_value, Path):
        return ctx_value
    if (resolved_value := strip_str_to_none(ctx_value)) is not None:
        return Path(resolved_value)
    return None


def get_settings_or_exit(
    config_path: str | Path | None = None,
    *,
    home: str | Path | None = None,
    no_home: bool = False,
) -> "Settings":
    """Load settings or exit with a concise user-facing error."""
    import typer
    from pydantic import ValidationError

    from fast_agent.config import get_settings
    from fast_agent.core.exceptions import FastAgentError, format_fast_agent_error
    from fast_agent.io.source_resolver import materialize_text_source

    selected_config_path: Path | None = None
    try:
        resolved_config_path = (
            materialize_text_source(config_path, label="config file", suffix=".yaml")
            if config_path is not None
            else None
        )
        selected_config_path = _selected_config_path(
            resolved_config_path,
            home=home,
            no_home=no_home,
        )
        return get_settings(resolved_config_path, home=home, no_home=no_home)
    except ValidationError as exc:
        typer.echo(
            "Error loading fast-agent settings: "
            f"{_format_validation_error(exc, config_path=selected_config_path)}",
            err=True,
        )
        raise typer.Exit(1) from exc
    except FastAgentError as exc:
        typer.echo(f"Error loading fast-agent settings: {format_fast_agent_error(exc)}", err=True)
        raise typer.Exit(1) from exc
    except Exception as exc:
        typer.echo(f"Error loading fast-agent settings: {exc}", err=True)
        raise typer.Exit(1) from exc


def _selected_config_path(
    explicit_config_path: Path | None,
    *,
    home: str | Path | None,
    no_home: bool,
) -> Path | None:
    """Return the config path selected by the settings discovery rules."""
    from fast_agent.home import discover_config_files, resolve_fast_agent_home

    cwd = Path.cwd()
    discovery = discover_config_files(
        cwd=cwd,
        home=resolve_fast_agent_home(
            cwd=cwd,
            cli_override=home,
            no_home=no_home,
        ),
        explicit_config_path=explicit_config_path,
    )
    return discovery.config_path


def _format_validation_error(
    exc: Any,
    *,
    config_path: Path | None,
) -> str:
    """Format Pydantic validation errors without echoing configuration values."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    lines = [f"{len(errors)} validation error{'s' if len(errors) != 1 else ''} for {exc.title}"]
    for error in errors:
        location = next(
            (str(part) for part in error.get("loc", ()) if isinstance(part, str)),
            "settings",
        )
        raw_message = str(error.get("msg", ""))
        error_type = str(error.get("type", ""))
        message = (
            _mcp_targets_migration_error(config_path)
            if raw_message == _MCP_TARGETS_MIGRATION_ERROR
            else _SAFE_VALIDATION_MESSAGES.get(error_type, "Invalid configuration value")
        )
        lines.append(f"{location}: {message}" if location else message)
    return "\n".join(lines)


def _mcp_targets_migration_error(config_path: Path | None) -> str:
    path = str(config_path) if config_path is not None else "<config-path>"
    command = shlex.join(["fast-agent", "config", "migrate-mcp", path, "--write"])
    return (
        "Value error, `mcp.targets` is no longer supported. "
        f"Run `{command}` to migrate it to `mcp.servers`."
    )
