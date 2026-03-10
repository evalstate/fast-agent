"""CLI command for managing skills."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from fast_agent.cli.command_support import (
    ensure_context_object,
    resolve_context_path_option,
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
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.config import get_settings
from fast_agent.skills.command_support import filter_marketplace_skills
from fast_agent.skills.configuration import (
    format_marketplace_display_url,
    get_marketplace_url,
)
from fast_agent.skills.models import DEFAULT_MARKETPLACE_URL, MarketplaceSkill, SkillUpdateInfo
from fast_agent.skills.operations import (
    apply_skill_updates,
    check_skill_updates,
    fetch_marketplace_skills_with_source,
    select_manifest_by_name_or_index,
    select_skill_updates,
)
from fast_agent.skills.provenance import format_revision_short, format_skill_provenance_details
from fast_agent.skills.registry import SkillManifest, SkillRegistry
from fast_agent.skills.scope import (
    get_manager_directory,
    order_skill_directories_for_display,
    resolve_skills_management_scope,
)
from fast_agent.skills.service import install_skill_sync, remove_skill
from fast_agent.ui.console import console

DEFAULT_CLI_SKILLS_REGISTRY = DEFAULT_MARKETPLACE_URL

RegistryOption = Annotated[
    str | None,
    typer.Option(
        "--registry",
        "-r",
        help=(
            "Override skills registry URL/path for this invocation "
            "(defaults to skills.marketplace_url / skills.marketplace_urls, "
            f"or {DEFAULT_CLI_SKILLS_REGISTRY})."
        ),
    ),
]

SkillsDirOption = Annotated[
    Path | None,
    typer.Option(
        "--skills-dir",
        help="Override the managed skills directory for this invocation.",
    ),
]

EnvOption = Annotated[
    Path | None,
    typer.Option(
        "--env",
        help=(
            "Override the base fast-agent environment directory for this invocation "
            "(same behavior as the top-level --env)."
        ),
    ),
]

app = typer.Typer(
    help="Manage skills (list/available/search/add/remove/update).",
    add_completion=False,
)


def _resolve_registry_input(ctx: typer.Context, command_registry: str | None = None) -> str:
    ctx_registry = resolve_context_string_option(
        ctx,
        key="registry",
        command_value=command_registry,
    )
    if ctx_registry:
        return ctx_registry
    return get_marketplace_url(get_settings())


def _resolve_skills_dir_input(
    ctx: typer.Context,
    command_skills_dir: Path | None = None,
) -> Path | None:
    return resolve_context_path_option(
        ctx,
        key="skills_dir",
        command_value=command_skills_dir,
    )


def _print_skill_table(
    manifests: list[SkillManifest],
    *,
    show_index: bool,
) -> None:
    table = Table(show_header=True, box=None)
    if show_index:
        table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Source", style="dim", header_style="bold bright_white")
    table.add_column("Provenance", style="white", header_style="bold bright_white")
    table.add_column("Installed", style="green", header_style="bold bright_white")

    for index, manifest in enumerate(manifests, 1):
        source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
        provenance_text, installed_text = format_skill_provenance_details(source_path)
        row = [
            manifest.name,
            format_display_path(source_path),
            provenance_text,
            installed_text or "—",
        ]
        if show_index:
            row = [str(index), *row]
        table.add_row(*row)

    console.print(table)


def _print_local_skills(*, managed_directory_override: Path | None = None) -> None:
    settings = get_settings()
    management_scope = resolve_skills_management_scope(
        settings,
        managed_directory_override=managed_directory_override,
    )
    discovered_directories = order_skill_directories_for_display(
        management_scope.discovered_directories,
        settings=settings,
        managed_directory_override=managed_directory_override,
    )

    print_detail_section(
        console,
        "Installed Skills",
        [
            DetailDisplayRow(
                label="managed directory",
                value=format_display_path(management_scope.managed_directory),
            )
        ],
    )

    total_skills = 0
    for directory in discovered_directories:
        manifests = SkillRegistry.load_directory(directory) if directory.exists() else []
        total_skills += len(manifests)

        directory_label = format_display_path(directory)
        if directory == management_scope.managed_directory:
            directory_label = f"{directory_label} (managed)"

        console.print()
        console.print(f"[bold bright_white]{directory_label}[/bold bright_white]")

        if not manifests:
            console.print("[yellow]No skills found in this directory.[/yellow]")
            continue

        _print_skill_table(manifests, show_index=False)

    if total_skills == 0:
        print_hint(console, "Browse marketplace with: fast-agent skills available")


def _print_marketplace_skills(
    marketplace: list[MarketplaceSkill],
    *,
    title: str,
    registry_url: str,
) -> None:
    print_detail_section(
        console,
        title,
        [
            DetailDisplayRow(
                label="registry",
                value=format_marketplace_display_url(registry_url),
            )
        ],
    )

    if not marketplace:
        console.print("[yellow]No skills found in the marketplace.[/yellow]")
        return

    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Bundle", style="white", header_style="bold bright_white")
    table.add_column("Description", style="dim", header_style="bold bright_white")

    for index, entry in enumerate(marketplace, 1):
        table.add_row(
            str(index),
            entry.name,
            entry.bundle_name or "—",
            entry.description or "",
        )

    console.print(table)


def _print_managed_skill_selection(*, managed_directory: Path) -> None:
    print_detail_section(
        console,
        "Managed Skills",
        [
            DetailDisplayRow(
                label="managed directory",
                value=format_display_path(managed_directory),
            )
        ],
    )

    manifests = SkillRegistry.load_directory(managed_directory) if managed_directory.exists() else []
    if not manifests:
        console.print("[yellow]No local skills found in the managed directory.[/yellow]")
        return

    _print_skill_table(manifests, show_index=True)


def _print_updates(
    updates: list[SkillUpdateInfo],
    *,
    title: str,
    managed_directory: Path,
) -> None:
    print_detail_section(
        console,
        title.rstrip(":"),
        [
            DetailDisplayRow(
                label="managed directory",
                value=format_display_path(managed_directory),
            )
        ],
    )

    if not updates:
        console.print("[yellow]No managed skills found.[/yellow]")
        return
    print_update_table(
        console,
        [
            UpdateDisplayRow(
                index=update.index,
                name=update.name,
                source_path=update.skill_dir,
                current_revision=update.current_revision,
                available_revision=update.available_revision,
                status=update.status,
                detail=update.detail,
            )
            for update in updates
        ],
        format_revision_short=format_revision_short,
    )


def _load_marketplace(
    ctx: typer.Context,
    *,
    registry: str | None = None,
) -> tuple[list[MarketplaceSkill], str]:
    marketplace_input = _resolve_registry_input(ctx, registry)
    try:
        scan_result = asyncio.run(fetch_marketplace_skills_with_source(marketplace_input))
        return scan_result
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to load marketplace: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.callback(invoke_without_command=True)
def skills_main(
    ctx: typer.Context,
    env: EnvOption = None,
    registry: RegistryOption = None,
    skills_dir: SkillsDirOption = None,
) -> None:
    """Manage skills."""
    resolve_environment_dir_option(ctx, env)
    ctx_object = ensure_context_object(ctx)
    ctx_object["registry"] = registry
    ctx_object["skills_dir"] = skills_dir
    if ctx.invoked_subcommand is None:
        _print_local_skills(managed_directory_override=skills_dir)


@app.command("list")
def skills_list(
    ctx: typer.Context,
    skills_dir: SkillsDirOption = None,
) -> None:
    """List local skills from discovered directories."""
    _print_local_skills(managed_directory_override=_resolve_skills_dir_input(ctx, skills_dir))


@app.command("available")
def skills_available(
    ctx: typer.Context,
    registry: RegistryOption = None,
) -> None:
    """List available skills from the marketplace."""
    marketplace, marketplace_url = _load_marketplace(ctx, registry=registry)
    _print_marketplace_skills(
        marketplace,
        title="Marketplace Skills",
        registry_url=marketplace_url,
    )
    print_hint(console, "Install with: fast-agent skills add <number|name>")


@app.command("search")
def skills_search(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Search query.")],
    registry: RegistryOption = None,
) -> None:
    """Search marketplace skills."""
    marketplace, marketplace_url = _load_marketplace(ctx, registry=registry)
    selected_marketplace = filter_marketplace_skills(marketplace, query)

    title = f"Marketplace Skills (search: {query})"
    _print_marketplace_skills(
        selected_marketplace,
        title=title,
        registry_url=marketplace_url,
    )
    if selected_marketplace:
        print_hint(console, "Install with: fast-agent skills add <number|name>")
    else:
        print_hint(console, "Browse all skills with: fast-agent skills available")


@app.command("add")
def skills_add(
    ctx: typer.Context,
    selector: Annotated[
        str | None,
        typer.Argument(help="Skill name or marketplace index.", show_default=False),
    ] = None,
    registry: RegistryOption = None,
    skills_dir: SkillsDirOption = None,
) -> None:
    """Install a skill from the selected marketplace."""
    managed_directory = get_manager_directory(
        get_settings(),
        managed_directory_override=_resolve_skills_dir_input(ctx, skills_dir),
    )
    marketplace, marketplace_url = _load_marketplace(ctx, registry=registry)

    if not selector:
        _print_marketplace_skills(
            marketplace,
            title="Marketplace Skills",
            registry_url=marketplace_url,
        )
        print_hint(console, "Install with: fast-agent skills add <number|name>")
        raise typer.Exit(0)

    try:
        installed = install_skill_sync(
            marketplace_url,
            selector,
            destination_root=managed_directory,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to install skill: {exc}", err=True)
        raise typer.Exit(1) from exc

    print_detail_section(
        console,
        "Skill Installed",
        [
            DetailDisplayRow(label="name", value=installed.name),
            DetailDisplayRow(label="location", value=format_display_path(installed.skill_dir)),
            DetailDisplayRow(
                label="managed directory",
                value=format_display_path(managed_directory),
            ),
        ],
        color="green",
    )


@app.command("remove")
def skills_remove(
    ctx: typer.Context,
    selector: Annotated[
        str | None,
        typer.Argument(help="Installed skill name or index.", show_default=False),
    ] = None,
    skills_dir: SkillsDirOption = None,
) -> None:
    """Remove an installed skill from the managed directory."""
    managed_directory = get_manager_directory(
        get_settings(),
        managed_directory_override=_resolve_skills_dir_input(ctx, skills_dir),
    )
    manifests = SkillRegistry.load_directory(managed_directory) if managed_directory.exists() else []
    if not manifests:
        console.print("[yellow]No local skills to remove.[/yellow]")
        raise typer.Exit(0)

    if not selector:
        _print_managed_skill_selection(managed_directory=managed_directory)
        print_hint(console, "Remove with: fast-agent skills remove <number|name>")
        raise typer.Exit(0)

    manifest = select_manifest_by_name_or_index(manifests, selector)
    if manifest is None:
        typer.echo(f"Skill not found: {selector}", err=True)
        raise typer.Exit(1)

    try:
        removed = remove_skill(managed_directory, selector)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to remove skill: {exc}", err=True)
        raise typer.Exit(1) from exc

    print_detail_section(
        console,
        "Skill Removed",
        [DetailDisplayRow(label="name", value=removed.name)],
        color="green",
    )


@app.command("update")
def skills_update(
    ctx: typer.Context,
    selector: Annotated[
        str | None,
        typer.Argument(
            help="Skill name, index, or 'all'. Omit to run update check.",
            show_default=False,
        ),
    ] = None,
    skills_dir: SkillsDirOption = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite local modifications."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm multi-skill apply."),
    ] = False,
) -> None:
    """Check and apply skill updates."""
    managed_directory = get_manager_directory(
        get_settings(),
        managed_directory_override=_resolve_skills_dir_input(ctx, skills_dir),
    )
    updates = check_skill_updates(destination_root=managed_directory)

    if not selector:
        _print_updates(
            updates,
            title="Skill update check:",
            managed_directory=managed_directory,
        )
        print_hint(
            console,
            "Apply with: fast-agent skills update <number|name|all> [--force] [--yes]",
        )
        raise typer.Exit(0)

    selected = select_skill_updates(updates, selector)
    if not selected:
        typer.echo(f"Skill not found: {selector}", err=True)
        raise typer.Exit(1)

    if len(selected) > 1 and not yes:
        _print_updates(
            selected,
            title="Update plan:",
            managed_directory=managed_directory,
        )
        console.print("[yellow]Multiple skills selected. Re-run with --yes to apply updates.[/yellow]")
        raise typer.Exit(1)

    applied = apply_skill_updates(selected, force=force)
    _print_updates(
        applied,
        title="Skill update results:",
        managed_directory=managed_directory,
    )
