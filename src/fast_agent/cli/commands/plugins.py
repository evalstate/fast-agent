"""CLI command for managing fast-agent command plugins."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from fast_agent.cli.command_support import (
    ensure_context_object,
    get_settings_or_exit,
    resolve_context_string_option,
)
from fast_agent.cli.display import (
    DetailDisplayRow,
    UpdateDisplayRow,
    format_display_path,
    indexed_table,
    print_detail_section,
    print_hint,
    print_update_table,
    print_warning,
)
from fast_agent.config import find_config_in_directory, resolve_global_plugin_home_path
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.marketplace.formatting import (
    format_installed_revision_display,
    format_source_provenance,
)
from fast_agent.paths import resolve_home_paths
from fast_agent.plugins import operations as plugin_ops
from fast_agent.plugins.bundles import install_plugin_bundle
from fast_agent.plugins.configuration import (
    disable_plugin_in_config,
    enable_plugin_in_config,
    enabled_plugins_by_scope,
    get_marketplace_url,
)
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.marketplace import plugin_bundle_names
from fast_agent.plugins.provenance import format_revision_short
from fast_agent.ui.console import console
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typer._click.core import Context

    from fast_agent.plugins.models import LocalPlugin, MarketplacePlugin

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


def _context_home(ctx: typer.Context) -> Path | None:
    current: Context | None = ctx
    while current is not None:
        payload = current.obj
        if isinstance(payload, dict):
            home = payload.get("home")
            if isinstance(home, Path):
                return home
            normalized_home = strip_to_none(home) if isinstance(home, str) else None
            if normalized_home is not None:
                return Path(normalized_home)
        current = current.parent
    return None


def _settings(ctx: typer.Context):
    settings = get_settings_or_exit(home=_context_home(ctx))
    if _has_original_fast_agent_home(ctx) and not settings._fast_agent_no_home:
        try:
            global_root = resolve_global_plugin_home_path(
                fast_agent_home=_original_fast_agent_home(ctx),
                home=Path.home(),
                cwd=Path.cwd(),
            )
        except RuntimeError:
            global_root = None
        settings._fast_agent_global_plugin_home = str(global_root) if global_root else None
    return settings


def _home_paths(ctx: typer.Context):
    return resolve_home_paths(_settings(ctx))


def _print_installed_plugins(ctx: typer.Context) -> None:
    settings = _settings(ctx)
    home_paths = resolve_home_paths(settings)
    plugin_roots = _installed_plugin_roots(settings, home_paths.plugins)
    home_enabled, project_enabled = enabled_plugins_by_scope(settings)
    _print_plugin_roots(
        plugin_roots,
        home_enabled=home_enabled,
        project_enabled=project_enabled,
    )


def _print_scoped_plugins(destination_root: Path, *, scope: str) -> None:
    _print_plugin_roots([(scope, destination_root)], home_enabled=[], project_enabled=[])


def _print_plugin_roots(
    plugin_roots: list[tuple[str, Path]],
    *,
    home_enabled: Sequence[str],
    project_enabled: Sequence[str],
) -> None:
    scoped_plugins = [
        (scope, plugin)
        for scope, plugin_root in plugin_roots
        for plugin in plugin_ops.list_local_plugins(destination_root=plugin_root)
    ]
    print_detail_section(
        console,
        "Installed Plugins",
        [
            DetailDisplayRow(label=f"{scope} plugins directory", value=format_display_path(root))
            for scope, root in plugin_roots
        ],
    )
    if not scoped_plugins:
        print_warning(console, "No plugins installed.")
        print_hint(console, "Install with: fast-agent plugins add <number|name>")
        return
    by_name: dict[str, list[tuple[str, LocalPlugin]]] = {}
    for scope, entry in scoped_plugins:
        by_name.setdefault(entry.name, []).append((scope, entry))

    table = indexed_table(
        ("Scope", "white"),
        ("Name", "cyan"),
        ("Version", "white"),
        ("Commands", "white"),
        ("Keys", "white"),
        ("Provenance", "dim"),
        ("Installed", "green"),
    )
    for column in table.columns[1:4]:
        column.no_wrap = True
    for index, (scope, entry) in enumerate(scoped_plugins, start=1):
        status = _plugin_status(
            entry.name,
            scope=scope,
            copies=by_name[entry.name],
            home_enabled=home_enabled,
            project_enabled=project_enabled,
        )
        scope_display = scope if status == "-" else f"{scope} ({status})"
        commands = ", ".join(entry.manifest.commands) if entry.manifest else "invalid manifest"
        version = entry.manifest.version if entry.manifest and entry.manifest.version else "-"
        keys = _format_plugin_keys(entry)
        if entry.source is None:
            provenance = (
                f"invalid metadata: {entry.metadata_error}" if entry.metadata_error else "unmanaged"
            )
            table.add_row(
                str(index), scope_display, entry.name, version, commands, keys, provenance, "-"
            )
            continue
        source = entry.source
        provenance = format_source_provenance(source.repo_url, source.repo_ref, source.repo_path)
        installed = format_installed_revision_display(
            source.installed_at,
            source.installed_revision,
            revision_label="",
        )
        table.add_row(
            str(index), scope_display, entry.name, version, commands, keys, provenance, installed
        )
    console.print(table)


def _plugin_status(
    name: str,
    *,
    scope: str,
    copies: Sequence[tuple[str, LocalPlugin]],
    home_enabled: Sequence[str],
    project_enabled: Sequence[str],
) -> str:
    if len(copies) < 2:
        return "-"

    active_scope = _active_scope_for(
        name, home_enabled=home_enabled, project_enabled=project_enabled
    )
    if active_scope is None:
        return "-"
    return "active" if active_scope == scope else f"shadowed by {active_scope}"


def _active_scope_for(
    name: str,
    *,
    home_enabled: Sequence[str],
    project_enabled: Sequence[str],
) -> str | None:
    if name in project_enabled:
        return "project"
    if name in home_enabled:
        return "global"
    return None


def _installed_plugin_roots(settings, project_plugins: Path) -> list[tuple[str, Path]]:
    from fast_agent.plugins.configuration import installed_plugin_roots

    return installed_plugin_roots(settings, project_plugins=project_plugins)


def _format_plugin_keys(entry: LocalPlugin) -> str:
    if entry.manifest is None:
        return "-"
    key_labels = [
        f"{name}: {normalized_key}"
        for name, spec in entry.manifest.commands.items()
        if (normalized_key := strip_to_none(spec.key)) is not None
    ]
    return ", ".join(key_labels) if key_labels else "-"


def _print_marketplace_plugins(plugins) -> None:
    if not plugins:
        print_warning(console, "No plugins found in the marketplace.")
        return
    table = indexed_table(("Name", "cyan"), ("Description", "dim"))
    for index, entry in enumerate(plugins, 1):
        table.add_row(str(index), entry.name, entry.description or "")
    console.print(table)


def _print_updates(updates) -> None:
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
        _print_installed_plugins(ctx)


@app.command("list")
def plugins_list(ctx: typer.Context) -> None:
    """List installed project and global plugins."""
    _print_installed_plugins(ctx)


@app.command("add")
def plugins_add(
    ctx: typer.Context,
    selector: Annotated[
        str | None, typer.Argument(help="Plugin name or marketplace index.", show_default=False)
    ] = None,
    registry: RegistryOption = None,
    global_install: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Install and enable globally (default; FAST_AGENT_HOME, or ~/.fast-agent).",
        ),
    ] = False,
    project_install: Annotated[
        bool,
        typer.Option("--project", help="Install and enable only in the active project."),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing plugin.")] = False,
    bundle: Annotated[
        str | None,
        typer.Option(
            "--bundle",
            help="Install and enable every plugin in a marketplace bundle (e.g. recommended).",
        ),
    ] = None,
) -> None:
    """Install and enable a command plugin."""
    if global_install and project_install:
        typer.echo("Choose one install scope: --global or --project.", err=True)
        raise typer.Exit(1)
    if bundle and selector:
        typer.echo("Choose a plugin or --bundle, not both.", err=True)
        raise typer.Exit(1)
    use_global = _resolve_default_global_scope(
        ctx,
        global_flag=global_install,
        project_flag=project_install,
    )
    destination_root, config_path = _target_install_context(ctx, global_install=use_global)
    marketplace_input = _resolve_registry_input(ctx, registry)
    plugins, source = plugin_ops.fetch_marketplace_plugins_with_source_sync(marketplace_input)
    if bundle:
        _install_bundle(plugins, bundle, destination_root=destination_root, config_path=config_path)
        return
    if not selector:
        print_detail_section(
            console, "Marketplace Plugins", [DetailDisplayRow(label="marketplace", value=source)]
        )
        _print_marketplace_plugins(plugins)
        print_hint(console, "Install with: fast-agent plugins add <number|name>")
        if bundles := plugin_bundle_names(plugins):
            print_hint(
                console,
                f"Bundles: {', '.join(bundles)} (fast-agent plugins add --bundle <name>)",
            )
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
    except Exception as exc:
        typer.echo(f"Failed to install plugin: {exc}", err=True)
        raise typer.Exit(1) from exc
    manifest = load_plugin_manifest(plugin_dir)
    plugin_name = manifest.name
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


def _install_bundle(
    plugins: Sequence[MarketplacePlugin],
    bundle: str,
    *,
    destination_root: Path,
    config_path: Path,
) -> None:
    try:
        results = install_plugin_bundle(
            plugins, bundle, destination_root=destination_root, config_path=config_path
        )
    except Exception as exc:
        typer.echo(f"Failed to install plugin bundle: {exc}", err=True)
        raise typer.Exit(1) from exc
    print_detail_section(
        console,
        f"Plugin Bundle Installed: {bundle}",
        [
            *(
                DetailDisplayRow(
                    label=result.name,
                    value="installed" if result.installed else "already installed (enabled)",
                )
                for result in results
            ),
            DetailDisplayRow(label="config", value=format_display_path(config_path)),
        ],
        color="green",
    )


@app.command("remove")
def plugins_remove(
    ctx: typer.Context,
    selector: Annotated[
        str | None, typer.Argument(help="Installed plugin name or index.", show_default=False)
    ] = None,
    global_remove: Annotated[
        bool,
        typer.Option(
            "--global",
            help="Remove globally (default; FAST_AGENT_HOME, or ~/.fast-agent).",
        ),
    ] = False,
    project_remove: Annotated[
        bool,
        typer.Option("--project", help="Remove only from the active project."),
    ] = False,
) -> None:
    """Remove an installed plugin."""
    if global_remove and project_remove:
        typer.echo("Choose one remove scope: --global or --project.", err=True)
        raise typer.Exit(1)
    use_global = _resolve_default_global_scope(
        ctx,
        global_flag=global_remove,
        project_flag=project_remove,
    )
    destination_root, config_path = _target_install_context(ctx, global_install=use_global)
    plugins = plugin_ops.list_local_plugins(destination_root=destination_root)
    if not selector:
        _print_scoped_plugins(destination_root, scope="global" if use_global else "project")
        scope_flag = " --project" if not use_global else ""
        print_hint(console, f"Remove with: fast-agent plugins remove <number|name>{scope_flag}")
        raise typer.Exit(0)
    selected = plugin_ops.select_local_plugin_by_name_or_index(plugins, selector)
    if selected is None:
        typer.echo(f"Plugin not found: {selector}", err=True)
        raise typer.Exit(1)
    plugin_ops.remove_local_plugin(selected.plugin_dir, destination_root=destination_root)
    disable_plugin_in_config(config_path, selected.name)
    print_detail_section(
        console,
        "Plugin Removed",
        [DetailDisplayRow(label="name", value=selected.name)],
        color="green",
    )


@app.command("update")
def plugins_update(
    ctx: typer.Context,
    selector: Annotated[
        str | None,
        typer.Argument(
            help="Plugin name, index, or 'all'. Omit to update all.",
            show_default=False,
        ),
    ] = None,
    global_update: Annotated[
        bool,
        typer.Option("--global", help="Check or update only globally installed plugins."),
    ] = False,
    project_update: Annotated[
        bool,
        typer.Option("--project", help="Check or update only project plugins."),
    ] = False,
    check: Annotated[
        bool,
        typer.Option("--check", help="Check for updates without applying."),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Overwrite local modifications.")] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Accepted for compatibility; updates apply immediately."),
    ] = False,
) -> None:
    """Check and apply plugin updates."""
    del yes
    if global_update and project_update:
        typer.echo("Choose one update scope: --global or --project.", err=True)
        raise typer.Exit(1)
    scope = "global" if global_update else "project" if project_update else None
    settings = _settings(ctx)
    home_paths = resolve_home_paths(settings)
    plugin_roots = _installed_plugin_roots(settings, home_paths.plugins)
    selected_roots = [
        (root_scope, root)
        for root_scope, root in plugin_roots
        if scope is None or root_scope == scope
    ]
    if not selected_roots and scope == "global":
        global_home_value = strip_to_none(settings._fast_agent_global_plugin_home)
        if global_home_value is not None:
            selected_roots = [("global", Path(global_home_value).expanduser() / "plugins")]
    if not selected_roots:
        typer.echo(f"Plugin {scope} scope is unavailable.", err=True)
        raise typer.Exit(1)
    all_updates = plugin_ops.check_plugin_updates_in_roots(
        destination_roots=[root for _, root in plugin_roots]
    )
    selected_root_paths = {root.resolve() for _, root in selected_roots}
    updates = [update for update in all_updates if update.plugin_dir.parent in selected_root_paths]
    if selector is not None and selector.strip().lower() != "all":
        selected = plugin_ops.select_plugin_updates(updates, selector)
        if not selected:
            scope_suffix = f" in {scope} scope" if scope is not None else ""
            typer.echo(f"Plugin not found{scope_suffix}: {selector}", err=True)
            raise typer.Exit(1)
    else:
        selected = updates
    if check:
        print_detail_section(
            console,
            "Plugin Update Check",
            [
                DetailDisplayRow(
                    label=f"{root_scope} plugins directory",
                    value=format_display_path(root),
                )
                for root_scope, root in selected_roots
            ],
        )
        _print_updates(selected)
        print_hint(
            console,
            "Apply with: fast-agent plugins update [<number|name|all>] "
            "[--global|--project] [--force]",
        )
        raise typer.Exit(0)
    if not selected:
        _print_updates([])
        raise typer.Exit(0)
    applied = plugin_ops.apply_plugin_updates(selected, force=force)
    _print_updates(applied)


def _has_explicit_home(ctx: typer.Context) -> bool:
    current: Context | None = ctx
    while current is not None:
        payload = current.obj
        if isinstance(payload, dict) and "explicit_home" in payload:
            return bool(payload["explicit_home"])
        current = current.parent
    return _context_home(ctx) is not None


def _has_original_fast_agent_home(ctx: typer.Context) -> bool:
    current: Context | None = ctx
    while current is not None:
        payload = current.obj
        if isinstance(payload, dict) and "original_fast_agent_home" in payload:
            return True
        current = current.parent
    return False


def _original_fast_agent_home(ctx: typer.Context) -> str | None:
    current: Context | None = ctx
    while current is not None:
        payload = current.obj
        if isinstance(payload, dict) and "original_fast_agent_home" in payload:
            value = payload["original_fast_agent_home"]
            return value if isinstance(value, str) else None
        current = current.parent
    return os.getenv("FAST_AGENT_HOME")


def _resolve_default_global_scope(
    ctx: typer.Context,
    *,
    global_flag: bool,
    project_flag: bool,
) -> bool:
    if global_flag:
        return True
    if project_flag:
        return False
    return not _has_explicit_home(ctx)


def _target_install_context(ctx: typer.Context, *, global_install: bool) -> tuple[Path, Path]:
    if global_install:
        root = _global_plugin_root(ctx)
        return (
            root / "plugins",
            find_config_in_directory(root) or root / PREFERRED_CONFIG_FILENAME,
        )

    settings = _settings(ctx)
    home_paths = resolve_home_paths(settings)
    config_path = (
        Path(settings._config_file)
        if settings._config_file
        else home_paths.root / PREFERRED_CONFIG_FILENAME
    )
    return home_paths.plugins, config_path


def _global_plugin_root(ctx: typer.Context) -> Path:
    try:
        root = resolve_global_plugin_home_path(
            fast_agent_home=_original_fast_agent_home(ctx),
            home=Path.home(),
            cwd=Path.cwd(),
        )
    except RuntimeError as exc:
        typer.echo(
            "FAST_AGENT_HOME is not set and the user home directory could not be resolved.",
            err=True,
        )
        raise typer.Exit(1) from exc
    if root is None:
        typer.echo("Global plugin installs are disabled in no_home mode.", err=True)
        raise typer.Exit(1)
    return root
