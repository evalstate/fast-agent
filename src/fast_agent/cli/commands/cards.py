"""CLI command for managing card packs."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table
from rich.text import Text

from fast_agent.cards import manager as card_manager
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


def _print_hint(message: str) -> None:
    console.print(f"[dim]▎• {message}[/dim]")


def _ctx_object(ctx: typer.Context) -> dict[str, Any]:
    if isinstance(ctx.obj, dict):
        return ctx.obj
    if ctx.obj is None:
        ctx.obj = {}
        return ctx.obj
    return {}


def _resolve_registry_input(ctx: typer.Context, command_registry: str | None = None) -> str:
    if command_registry:
        return command_registry
    ctx_registry = _ctx_object(ctx).get("registry")
    if isinstance(ctx_registry, str) and ctx_registry.strip():
        return ctx_registry
    return card_manager.get_marketplace_url(get_settings())


def _environment_paths():
    settings = get_settings()
    return resolve_environment_paths(settings)


def _format_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _print_local_packs() -> None:
    env_paths = _environment_paths()
    packs = card_manager.list_local_card_packs(environment_paths=env_paths)
    _print_section_header("Installed Card Packs", color="blue")
    console.print(f"[dim]▎• Card packs directory:[/dim] [cyan]{_format_path(env_paths.card_packs)}[/cyan]")

    if not packs:
        console.print("[yellow]No card packs installed.[/yellow]")
        _print_hint("Install with: fast-agent cards add <number|name>")
        return

    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Source", style="dim", header_style="bold bright_white")
    table.add_column("Provenance", style="white", header_style="bold bright_white")
    table.add_column("Installed", style="green", header_style="bold bright_white")

    for entry in packs:
        source_path = _format_path(entry.pack_dir)
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
    _print_section_header(title.rstrip(":"), color="blue")
    if not updates:
        console.print("[yellow]No managed card packs found.[/yellow]")
        return

    table = Table(show_header=True, box=None)
    table.add_column("#", justify="right", style="dim", header_style="bold bright_white")
    table.add_column("Name", style="cyan", header_style="bold bright_white")
    table.add_column("Source", style="dim", header_style="bold bright_white")
    table.add_column("Revision", style="white", header_style="bold bright_white")
    table.add_column("Status", style="green", header_style="bold bright_white")

    for update in updates:
        revision_display = ""
        if update.current_revision or update.available_revision:
            current = card_manager.format_revision_short(update.current_revision)
            available = card_manager.format_revision_short(update.available_revision)
            revision_display = f"{current} -> {available}"

        status = update.status.replace("_", " ")
        if update.detail:
            status = f"{status}: {update.detail}"

        table.add_row(
            str(update.index),
            update.name,
            _format_path(update.pack_dir),
            revision_display,
            status,
        )

    console.print(table)


def _print_publish_result(result: card_manager.CardPackPublishResult) -> None:
    _print_section_header("Card Pack Publish", color="blue")
    console.print(f"[dim]▎• pack:[/dim] [cyan]{result.pack_name}[/cyan]")
    console.print(f"[dim]▎• source:[/dim] [cyan]{_format_path(result.pack_dir)}[/cyan]")
    if result.repo_root is not None:
        repo_label = _format_path(result.repo_root)
        if result.repo_path:
            repo_label = f"{repo_label} ({result.repo_path})"
        console.print(f"[dim]▎• repo:[/dim] [cyan]{repo_label}[/cyan]")
    if result.commit:
        console.print(
            "[dim]▎• commit:[/dim] "
            f"[cyan]{card_manager.format_revision_short(result.commit)}[/cyan]"
        )
    if result.patch_path is not None:
        console.print(f"[dim]▎• patch:[/dim] [cyan]{_format_path(result.patch_path)}[/cyan]")
    if result.retained_temp_dir is not None:
        console.print(
            f"[dim]▎• temp clone:[/dim] [cyan]{_format_path(result.retained_temp_dir)}[/cyan]"
        )

    status = result.status.replace("_", " ")
    if result.detail:
        status = f"{status}: {result.detail}"

    style = "yellow"
    if result.status in {"published", "committed", "no_changes"}:
        style = "green"
    elif result.status == "unmanaged":
        style = "white"

    console.print(f"[{style}]Status: {status}[/{style}]")


@app.callback(invoke_without_command=True)
def cards_main(ctx: typer.Context, registry: RegistryOption = None) -> None:
    """Manage card packs."""
    _ctx_object(ctx)["registry"] = registry
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
        _print_section_header("Marketplace Card Packs", color="blue")
        console.print(
            "[dim]▎• Marketplace:[/dim] "
            f"[cyan]{card_manager.format_marketplace_display_url(marketplace_url)}[/cyan]"
        )
        _print_marketplace_packs(marketplace)
        _print_hint("Install with: fast-agent cards add <number|name>")
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

    console.print(f"[green]Installed card pack: {pack.name}[/green]")
    console.print(f"[dim]▎• location:[/dim] [cyan]{_format_path(result.pack_dir)}[/cyan]")
    console.print(f"[dim]▎• managed files:[/dim] {len(result.installed_files)}")


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
        _print_hint("Remove with: fast-agent cards remove <number|name>")
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

    console.print(f"[green]Removed card pack: {removal.pack_name}[/green]")
    console.print(f"[dim]▎• removed files:[/dim] {len(removal.removed_paths)}")
    if removal.skipped_paths:
        console.print(f"[yellow]▎• skipped files:[/yellow] {len(removal.skipped_paths)}")


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
        _print_hint("Apply with: fast-agent cards update <number|name|all> [--force] [--yes]")
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
        _print_hint(
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
