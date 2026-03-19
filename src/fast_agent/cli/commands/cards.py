"""CLI command for managing card packs."""

from __future__ import annotations

import asyncio
from pathlib import Path  # noqa: TC003
from typing import Annotated

import typer
from rich.markdown import Markdown
from rich.table import Table

from fast_agent.cards import manager as card_manager
from fast_agent.cli.command_support import ensure_context_object, resolve_context_string_option
from fast_agent.cli.display import (
    DetailDisplayRow,
    UpdateDisplayRow,
    format_display_path,
    print_detail_section,
    print_hint,
    print_section_header,
    print_update_table,
)
from fast_agent.config import get_settings
from fast_agent.paths import resolve_environment_paths
from fast_agent.ui.console import console

DEFAULT_CLI_CARD_REGISTRY = "https://github.com/fast-agent-ai/card-packs"

RegistryOption = Annotated[
    str | None,
    typer.Option(
        "--registry",
        "-r",
        help=(
            "Override card registry URL/path for this invocation "
            "(defaults to cards.marketplace_url / cards.marketplace_urls, "
            f"or {DEFAULT_CLI_CARD_REGISTRY})."
        ),
    ),
]

app = typer.Typer(help="Manage card packs (list/add/remove/update/publish).")


def _resolve_registry_input(ctx: typer.Context, command_registry: str | None = None) -> str:
    registry = resolve_context_string_option(
        ctx,
        key="registry",
        command_value=command_registry,
    )
    if registry:
        return registry
    return card_manager.get_marketplace_url(get_settings())


def _environment_paths():
    settings = get_settings()
    return resolve_environment_paths(settings)


def _print_local_packs() -> None:
    env_paths = _environment_paths()
    packs = card_manager.list_local_card_packs(environment_paths=env_paths)
    print_detail_section(
        console,
        "Installed Card Packs",
        [
            DetailDisplayRow(
                label="card packs directory",
                value=format_display_path(env_paths.card_packs),
            )
        ],
    )

    if not packs:
        console.print("[yellow]No card packs installed.[/yellow]")
        print_hint(console, "Install with: fast-agent cards add <number|name>")
        return

    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Source", style="dim", header_style="bold bright_white")
    table.add_column("Provenance", style="white", header_style="bold bright_white")
    table.add_column("Installed", style="green", header_style="bold bright_white")

    for entry in packs:
        source_path = format_display_path(entry.pack_dir)
        if entry.source is None:
            provenance = "unmanaged"
            if entry.metadata_error:
                provenance = f"invalid metadata: {entry.metadata_error}"
            table.add_row(str(entry.index), entry.name, source_path, provenance, "—")
            continue

        source = entry.source
        ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
        provenance = f"{source.repo_url}{ref_label} ({source.repo_path})"
        installed = (
            f"{card_manager.format_installed_at_display(source.installed_at)} "
            f"· {card_manager.format_revision_short(source.installed_revision)}"
        )
        table.add_row(str(entry.index), entry.name, source_path, provenance, installed)

    console.print(table)


def _print_marketplace_packs(packs: list[card_manager.MarketplaceCardPack]) -> None:
    if not packs:
        console.print("[yellow]No card packs found in the marketplace.[/yellow]")
        return

    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Kind", style="white", header_style="bold bright_white")
    table.add_column("Description", style="dim", header_style="bold bright_white")

    for index, entry in enumerate(packs, 1):
        table.add_row(str(index), entry.name, entry.kind, entry.description or "")

    console.print(table)


def _print_updates(updates: list[card_manager.CardPackUpdateInfo], *, title: str) -> None:
    print_section_header(console, title.rstrip(":"), color="blue")
    if not updates:
        console.print("[yellow]No managed card packs found.[/yellow]")
        return
    print_update_table(
        console,
        [
            UpdateDisplayRow(
                index=update.index,
                name=update.name,
                source_path=update.pack_dir,
                current_revision=update.current_revision,
                available_revision=update.available_revision,
                status=update.status,
                detail=update.detail,
            )
            for update in updates
        ],
        format_revision_short=card_manager.format_revision_short,
    )


def _print_publish_result(result: card_manager.CardPackPublishResult) -> None:
    rows = [
        DetailDisplayRow(label="pack", value=result.pack_name),
        DetailDisplayRow(label="source", value=format_display_path(result.pack_dir)),
    ]
    if result.repo_root is not None:
        repo_label = format_display_path(result.repo_root)
        if result.repo_path:
            repo_label = f"{repo_label} ({result.repo_path})"
        rows.append(DetailDisplayRow(label="repo", value=repo_label))
    if result.commit:
        rows.append(
            DetailDisplayRow(
                label="commit",
                value=card_manager.format_revision_short(result.commit),
            )
        )
    if result.patch_path is not None:
        rows.append(
            DetailDisplayRow(label="patch", value=format_display_path(result.patch_path))
        )
    if result.retained_temp_dir is not None:
        rows.append(
            DetailDisplayRow(
                label="temp clone",
                value=format_display_path(result.retained_temp_dir),
            )
        )
    print_detail_section(console, "Card Pack Publish", rows)

    status = result.status.replace("_", " ")
    if result.detail:
        status = f"{status}: {result.detail}"

    style = "yellow"
    if result.status in {"published", "committed", "no_changes"}:
        style = "green"
    elif result.status == "unmanaged":
        style = "white"

    console.print(f"[{style}]Status: {status}[/{style}]")


def _print_card_pack_readme(pack_name: str, pack_dir: Path) -> None:
    readme = card_manager.load_card_pack_readme(pack_dir)
    if not readme:
        console.print(f"[yellow]Card pack '{pack_name}' does not include a README.md.[/yellow]")
        raise typer.Exit(0)

    print_section_header(console, f"{pack_name} README", color="cyan")
    console.print(Markdown(readme))


@app.callback(invoke_without_command=True)
def cards_main(ctx: typer.Context, registry: RegistryOption = None) -> None:
    """Manage card packs."""
    ensure_context_object(ctx)["registry"] = registry
    if ctx.invoked_subcommand is None:
        _print_local_packs()


@app.command("list")
def cards_list() -> None:
    """List local card packs."""
    _print_local_packs()


@app.command("add")
def cards_add(
    ctx: typer.Context,
    selector: Annotated[
        str | None,
        typer.Argument(help="Card pack name or marketplace index.", show_default=False),
    ] = None,
    registry: RegistryOption = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite files owned by other packs."),
    ] = False,
) -> None:
    """Install a card pack from the selected marketplace."""
    settings = get_settings()
    marketplace_input = _resolve_registry_input(ctx, registry)
    env_paths = resolve_environment_paths(settings)

    try:
        marketplace, marketplace_url = asyncio.run(
            card_manager.fetch_marketplace_card_packs_with_source(marketplace_input)
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to load marketplace: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not selector:
        print_detail_section(
            console,
            "Marketplace Card Packs",
            [
                DetailDisplayRow(
                    label="marketplace",
                    value=card_manager.format_marketplace_display_url(marketplace_url),
                )
            ],
        )
        _print_marketplace_packs(marketplace)
        print_hint(console, "Install with: fast-agent cards add <number|name>")
        raise typer.Exit(0)

    pack = card_manager.select_card_pack_by_name_or_index(marketplace, selector)
    if pack is None:
        typer.echo(f"Card pack not found: {selector}", err=True)
        raise typer.Exit(1)

    try:
        result = asyncio.run(
            card_manager.install_marketplace_card_pack(
                pack,
                environment_paths=env_paths,
                force=force,
            )
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to install card pack: {exc}", err=True)
        raise typer.Exit(1) from exc

    print_detail_section(
        console,
        "Card Pack Installed",
        [
            DetailDisplayRow(label="name", value=pack.name),
            DetailDisplayRow(label="location", value=format_display_path(result.pack_dir)),
            DetailDisplayRow(
                label="managed files",
                value=len(result.installed_files),
                value_style="green",
            ),
        ],
        color="green",
    )
    if card_manager.load_card_pack_readme(result.pack_dir):
        console.print()
        _print_card_pack_readme(pack.name, result.pack_dir)


@app.command("remove")
def cards_remove(
    selector: Annotated[
        str | None,
        typer.Argument(help="Installed card pack name or index.", show_default=False),
    ] = None,
) -> None:
    """Remove an installed card pack."""
    env_paths = _environment_paths()
    packs = card_manager.list_local_card_packs(environment_paths=env_paths)
    if not packs:
        console.print("[yellow]No local card packs to remove.[/yellow]")
        raise typer.Exit(0)

    if not selector:
        _print_local_packs()
        print_hint(console, "Remove with: fast-agent cards remove <number|name>")
        raise typer.Exit(0)

    selected = card_manager.select_installed_card_pack_by_name_or_index(packs, selector)
    if selected is None:
        typer.echo(f"Card pack not found: {selector}", err=True)
        raise typer.Exit(1)

    try:
        removal = card_manager.remove_local_card_pack(
            selected.pack_dir.name,
            environment_paths=env_paths,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to remove card pack: {exc}", err=True)
        raise typer.Exit(1) from exc

    rows = [
        DetailDisplayRow(label="name", value=removal.pack_name),
        DetailDisplayRow(
            label="removed files",
            value=len(removal.removed_paths),
            value_style="green",
        ),
    ]
    if removal.skipped_paths:
        rows.append(
            DetailDisplayRow(
                label="skipped files",
                value=len(removal.skipped_paths),
                value_style="yellow",
            )
        )
    print_detail_section(console, "Card Pack Removed", rows, color="green")


@app.command("readme")
def cards_readme(
    selector: Annotated[
        str | None,
        typer.Argument(help="Installed card pack name or index.", show_default=False),
    ] = None,
) -> None:
    """Show an installed card pack README."""
    env_paths = _environment_paths()
    packs = card_manager.list_local_card_packs(environment_paths=env_paths)
    if not packs:
        console.print("[yellow]No local card packs installed.[/yellow]")
        raise typer.Exit(0)

    if not selector:
        if len(packs) == 1:
            selected = packs[0]
            _print_card_pack_readme(selected.name, selected.pack_dir)
            return
        _print_local_packs()
        print_hint(console, "Show with: fast-agent cards readme <number|name>")
        raise typer.Exit(0)

    selected = card_manager.select_installed_card_pack_by_name_or_index(packs, selector)
    if selected is None:
        typer.echo(f"Card pack not found: {selector}", err=True)
        raise typer.Exit(1)

    _print_card_pack_readme(selected.name, selected.pack_dir)


@app.command("update")
def cards_update(
    selector: Annotated[
        str | None,
        typer.Argument(
            help="Card pack name, index, or 'all'. Omit to run update check.",
            show_default=False,
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite local modifications."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", help="Confirm multi-pack apply."),
    ] = False,
) -> None:
    """Check and apply card pack updates."""
    env_paths = _environment_paths()
    updates = card_manager.check_card_pack_updates(environment_paths=env_paths)

    if not selector:
        _print_updates(updates, title="Card pack update check:")
        print_hint(
            console,
            "Apply with: fast-agent cards update <number|name|all> [--force] [--yes]",
        )
        raise typer.Exit(0)

    selected = card_manager.select_card_pack_updates(updates, selector)
    if not selected:
        typer.echo(f"Card pack not found: {selector}", err=True)
        raise typer.Exit(1)

    if len(selected) > 1 and not yes:
        _print_updates(selected, title="Update plan:")
        console.print("[yellow]Multiple card packs selected. Re-run with --yes to apply updates.[/yellow]")
        raise typer.Exit(1)

    applied = card_manager.apply_card_pack_updates(
        selected,
        environment_paths=env_paths,
        force=force,
    )
    _print_updates(applied, title="Card pack update results:")


@app.command("publish")
def cards_publish(
    selector: Annotated[
        str | None,
        typer.Argument(help="Installed card pack name or index.", show_default=False),
    ] = None,
    no_push: Annotated[
        bool,
        typer.Option("--no-push", help="Commit locally but skip git push."),
    ] = False,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Commit message for published changes."),
    ] = None,
    temp_dir: Annotated[
        Path | None,
        typer.Option(
            "--temp-dir",
            help="Directory for temporary clone checkout when source repo is remote.",
        ),
    ] = None,
    keep_temp: Annotated[
        bool,
        typer.Option(
            "--keep-temp",
            help="Retain temporary clone checkout on disk for inspection.",
        ),
    ] = False,
) -> None:
    """Publish local card pack changes back to the source repository."""
    env_paths = _environment_paths()
    packs = card_manager.list_local_card_packs(environment_paths=env_paths)
    if not packs:
        console.print("[yellow]No local card packs to publish.[/yellow]")
        raise typer.Exit(0)

    if not selector:
        _print_local_packs()
        print_hint(
            console,
            "Publish with: fast-agent cards publish <number|name> "
            "[--no-push] [--message ...] [--temp-dir <path>] [--keep-temp]"
        )
        raise typer.Exit(0)

    selected = card_manager.select_installed_card_pack_by_name_or_index(packs, selector)
    if selected is None:
        typer.echo(f"Card pack not found: {selector}", err=True)
        raise typer.Exit(1)

    result = card_manager.publish_local_card_pack(
        selected.pack_dir,
        environment_paths=env_paths,
        push=not no_push,
        commit_message=message,
        temp_dir=temp_dir,
        keep_temp=keep_temp,
    )
    _print_publish_result(result)

    if result.status in {"published", "committed", "no_changes"}:
        raise typer.Exit(0)

    if result.status == "publish_failed" and result.patch_path is not None:
        console.print(
            "[yellow]Push was rejected. Share the generated patch with a maintainer or open a PR.[/yellow]"
        )
    raise typer.Exit(1)
