"""CLI command for managing fast-agent command plugins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.table import Table

from fast_agent.cli.command_support import (
    ensure_context_object,
    get_settings_or_exit,
    resolve_context_string_option,
)
from fast_agent.cli.display import (
    DetailDisplayRow,
    UpdateDisplayRow,
    format_display_path,
    print_detail_section,
    print_hint,
    print_update_table,
)
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.paths import resolve_environment_paths
from fast_agent.plugins import operations as plugin_ops
from fast_agent.plugins.configuration import (
    disable_plugin_in_config,
    enable_plugin_in_config,
    get_marketplace_url,
)
from fast_agent.plugins.provenance import format_installed_at_display, format_revision_short
from fast_agent.ui.console import console

if TYPE_CHECKING:
    import click

RegistryOption = Annotated[
    str | None,
    typer.Option("--registry", "-r", help="Override plugin registry URL/path for this invocation."),
]

app = typer.Typer(help="Manage command plugins (list/add/remove/update).", add_completion=False)


def _resolve_registry_input(ctx: typer.Context, command_registry: str | None = None) -> str:
    registry = resolve_context_string_option(ctx, key="registry", command_value=command_registry)
    if registry:
        return registry
    return get_marketplace_url(_settings(ctx))


def _context_env_dir(ctx: typer.Context) -> Path | None:
    current: click.Context | None = ctx
    while current is not None:
        payload = current.obj
        if isinstance(payload, dict):
            env_dir = payload.get("env_dir")
            if isinstance(env_dir, Path):
                return env_dir
            if isinstance(env_dir, str) and env_dir.strip():
                return Path(env_dir)
        current = current.parent
    return None


def _settings(ctx: typer.Context):
    return get_settings_or_exit(env_dir=_context_env_dir(ctx))


def _environment_paths(ctx: typer.Context):
    return resolve_environment_paths(_settings(ctx))


def _print_local_plugins(ctx: typer.Context) -> None:
    env_paths = _environment_paths(ctx)
    plugins = plugin_ops.list_local_plugins(destination_root=env_paths.plugins)
    print_detail_section(
        console,
        "Installed Plugins",
        [DetailDisplayRow(label="plugins directory", value=format_display_path(env_paths.plugins))],
    )
    if not plugins:
        console.print("[yellow]No plugins installed.[/yellow]")
        print_hint(console, "Install with: fast-agent plugins add <number|name>")
        return
    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Commands", style="white", header_style="bold bright_white")
    table.add_column("Keys", style="white", header_style="bold bright_white")
    table.add_column("Provenance", style="dim", header_style="bold bright_white")
    table.add_column("Installed", style="green", header_style="bold bright_white")
    for entry in plugins:
        commands = ", ".join(entry.manifest.commands) if entry.manifest else "invalid manifest"
        keys = _format_plugin_keys(entry)
        if entry.source is None:
            provenance = f"invalid metadata: {entry.metadata_error}" if entry.metadata_error else "unmanaged"
            table.add_row(str(entry.index), entry.name, commands, keys, provenance, "-")
            continue
        source = entry.source
        ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
        provenance = f"{source.repo_url}{ref_label} ({source.repo_path})"
        installed = f"{format_installed_at_display(source.installed_at)} {format_revision_short(source.installed_revision)}"
        table.add_row(str(entry.index), entry.name, commands, keys, provenance, installed)
    console.print(table)


def _format_plugin_keys(entry) -> str:
    if entry.manifest is None:
        return "-"
    key_labels = [
        f"{name}: {spec.key}"
        for name, spec in entry.manifest.commands.items()
        if spec.key is not None and spec.key.strip()
    ]
    return ", ".join(key_labels) if key_labels else "-"


def _print_marketplace_plugins(plugins) -> None:
    if not plugins:
        console.print("[yellow]No plugins found in the marketplace.[/yellow]")
        return
    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Description", style="dim", header_style="bold bright_white")
    for index, entry in enumerate(plugins, 1):
        table.add_row(str(index), entry.name, entry.description or "")
    console.print(table)


def _print_updates(updates, *, title: str) -> None:
    print_update_table(
        console,
        [
            UpdateDisplayRow(
                index=update.index,
                name=update.name,
                source_path=update.plugin_dir,
                current_revision=update.current_revision,
                available_revision=update.available_revision,
                status=update.status,
                detail=update.detail,
            )
            for update in updates
        ],
        format_revision_short=format_revision_short,
    )


@app.callback(invoke_without_command=True)
def plugins_main(ctx: typer.Context, registry: RegistryOption = None) -> None:
    ensure_context_object(ctx)["registry"] = registry
    if ctx.invoked_subcommand is None:
        _print_local_plugins(ctx)


@app.command("list")
def plugins_list(ctx: typer.Context) -> None:
    """List local plugins."""
    _print_local_plugins(ctx)


@app.command("add")
def plugins_add(
    ctx: typer.Context,
    selector: Annotated[str | None, typer.Argument(help="Plugin name or marketplace index.", show_default=False)] = None,
    registry: RegistryOption = None,
    global_install: Annotated[bool, typer.Option("--global", help="Install and enable globally (FAST_AGENT_HOME, or ~/.fast-agent).")] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing plugin.")] = False,
) -> None:
    """Install and enable a command plugin."""
    destination_root, config_path = _target_install_context(ctx, global_install=global_install)
    marketplace_input = _resolve_registry_input(ctx, registry)
    plugins, source = plugin_ops.fetch_marketplace_plugins_with_source_sync(marketplace_input)
    if not selector:
        print_detail_section(console, "Marketplace Plugins", [DetailDisplayRow(label="marketplace", value=source)])
        _print_marketplace_plugins(plugins)
        print_hint(console, "Install with: fast-agent plugins add <number|name>")
        raise typer.Exit(0)

    selected = plugin_ops.select_plugin_by_name_or_index(plugins, selector)
    if selected is None:
        typer.echo(f"Plugin not found: {selector}", err=True)
        raise typer.Exit(1)

    try:
        plugin_dir = plugin_ops.install_marketplace_plugin_sync(
            selected,
            destination_root=destination_root,
            replace_existing=force,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to install plugin: {exc}", err=True)
        raise typer.Exit(1) from exc
    manifest = plugin_ops.load_plugin_manifest(plugin_dir) if hasattr(plugin_ops, "load_plugin_manifest") else None
    plugin_name = manifest.name if manifest else selected.name
    enable_plugin_in_config(config_path, plugin_name)
    print_detail_section(
        console,
        "Plugin Installed",
        [
            DetailDisplayRow(label="name", value=plugin_name),
            DetailDisplayRow(label="location", value=format_display_path(plugin_dir)),
            DetailDisplayRow(label="config", value=format_display_path(config_path)),
        ],
        color="green",
    )


@app.command("remove")
def plugins_remove(
    ctx: typer.Context,
    selector: Annotated[str | None, typer.Argument(help="Installed plugin name or index.", show_default=False)] = None,
    global_remove: Annotated[bool, typer.Option("--global", help="Remove globally (FAST_AGENT_HOME, or ~/.fast-agent).")] = False,
) -> None:
    """Remove an installed plugin."""
    destination_root, config_path = _target_install_context(ctx, global_install=global_remove)
    plugins = plugin_ops.list_local_plugins(destination_root=destination_root)
    if not selector:
        _print_local_plugins(ctx)
        print_hint(console, "Remove with: fast-agent plugins remove <number|name>")
        raise typer.Exit(0)
    selected = plugin_ops.select_local_plugin_by_name_or_index(plugins, selector)
    if selected is None:
        typer.echo(f"Plugin not found: {selector}", err=True)
        raise typer.Exit(1)
    plugin_ops.remove_local_plugin(selected.plugin_dir, destination_root=destination_root)
    disable_plugin_in_config(config_path, selected.name)
    print_detail_section(console, "Plugin Removed", [DetailDisplayRow(label="name", value=selected.name)], color="green")


@app.command("update")
def plugins_update(
    ctx: typer.Context,
    selector: Annotated[str | None, typer.Argument(help="Plugin name, index, or 'all'. Omit to check.", show_default=False)] = None,
    force: Annotated[bool, typer.Option("--force", help="Overwrite local modifications.")] = False,
    yes: Annotated[bool, typer.Option("--yes", help="Confirm multi-plugin apply.")] = False,
) -> None:
    """Check and apply plugin updates."""
    env_paths = _environment_paths(ctx)
    updates = plugin_ops.check_plugin_updates(destination_root=env_paths.plugins)
    if not selector:
        print_detail_section(console, "Plugin Update Check", [DetailDisplayRow(label="plugins directory", value=format_display_path(env_paths.plugins))])
        _print_updates(updates, title="Plugin update check")
        print_hint(console, "Apply with: fast-agent plugins update <number|name|all> [--force] [--yes]")
        raise typer.Exit(0)
    selected = plugin_ops.select_plugin_updates(updates, selector)
    if not selected:
        typer.echo(f"Plugin not found: {selector}", err=True)
        raise typer.Exit(1)
    if len(selected) > 1 and not yes:
        _print_updates(selected, title="Update plan")
        console.print("[yellow]Multiple plugins selected. Re-run with --yes to apply updates.[/yellow]")
        raise typer.Exit(1)
    applied = plugin_ops.apply_plugin_updates(selected, force=force)
    _print_updates(applied, title="Plugin update results")


def _target_install_context(ctx: typer.Context, *, global_install: bool) -> tuple[Path, Path]:
    if global_install:
        root = _global_plugin_root()
        return root / "plugins", root / PREFERRED_CONFIG_FILENAME

    settings = _settings(ctx)
    env_paths = resolve_environment_paths(settings)
    config_path = Path(settings._config_file) if settings._config_file else env_paths.root / PREFERRED_CONFIG_FILENAME
    return env_paths.plugins, config_path


def _global_plugin_root() -> Path:
    configured = os.getenv("FAST_AGENT_HOME")
    if configured:
        root = Path(configured).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        return root.resolve()
    try:
        return (Path.home() / ".fast-agent").resolve()
    except RuntimeError as exc:
        typer.echo(
            "FAST_AGENT_HOME is not set and the user home directory could not be resolved.",
            err=True,
        )
        raise typer.Exit(1) from exc
