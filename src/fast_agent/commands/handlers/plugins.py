"""Shared /plugins command handlers for the interactive prompt."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

from rich.text import Text

from fast_agent.commands.handlers._marketplace_argument_parsing import parse_update_argument
from fast_agent.commands.handlers._text_formatting import append_heading, append_wrapped_text
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.config import get_settings
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.paths import resolve_environment_paths
from fast_agent.plugins.configuration import (
    disable_plugin_in_config,
    enable_plugin_in_config,
    get_marketplace_url,
    resolve_registries,
)
from fast_agent.plugins.manifest import load_plugin_manifest
from fast_agent.plugins.operations import (
    apply_plugin_updates,
    check_plugin_updates,
    fetch_marketplace_plugins_with_source,
    install_marketplace_plugin_sync,
    list_local_plugins,
    remove_local_plugin,
    select_local_plugin_by_name_or_index,
    select_plugin_by_name_or_index,
    select_plugin_updates,
)
from fast_agent.plugins.provenance import format_installed_at_display, format_revision_short

if TYPE_CHECKING:
    from fast_agent.command_actions.models import PluginCommandActionSpec
    from fast_agent.commands.context import CommandContext
    from fast_agent.plugins.models import (
        LocalPlugin,
        MarketplacePlugin,
        PluginUpdateInfo,
    )


@runtime_checkable
class _PluginCommandProvider(Protocol):
    def set_plugin_commands(
        self,
        commands: dict[str, "PluginCommandActionSpec"] | None,
        *,
        base_path: Path | None,
    ) -> None: ...


def _plugins_usage_lines() -> list[str]:
    return [
        "Usage: /plugins [list|available|add|remove|update|registry|help] [args]",
        "",
        "Examples:",
        "- /plugins available",
        "- /plugins add <number|name>",
        "- /plugins remove <number|name>",
        "- /plugins update all --yes",
        "- /plugins registry",
    ]


def _is_help_flag(value: str | None) -> bool:
    token = (value or "").strip().lower()
    return token in {"help", "--help", "-h"}


def _config_path_for_settings(ctx: CommandContext) -> Path:
    settings = ctx.resolve_settings()
    if settings._config_file:
        return Path(settings._config_file)
    return resolve_environment_paths(settings).root / PREFERRED_CONFIG_FILENAME


def _format_plugin_keys(entry: LocalPlugin) -> str:
    if entry.manifest is None:
        return "-"
    labels = [
        f"{name}: {spec.key}"
        for name, spec in entry.manifest.commands.items()
        if spec.key is not None and spec.key.strip()
    ]
    return ", ".join(labels) if labels else "-"


def _format_local_plugins(*, plugins_dir: Path, plugins: Sequence[LocalPlugin]) -> Text:
    content = Text()
    try:
        display_dir = plugins_dir.relative_to(Path.cwd())
    except ValueError:
        display_dir = plugins_dir

    append_heading(content, f"Plugins in {display_dir}:")
    if not plugins:
        content.append_text(Text("No plugins installed.", style="yellow"))
        content.append("\n")
        content.append_text(Text("Install with /plugins add <number|name>", style="dim"))
        return content

    for entry in plugins:
        row = Text()
        row.append(f"[{entry.index:2}] ", style="dim cyan")
        row.append(entry.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        try:
            source_display = entry.plugin_dir.relative_to(Path.cwd())
        except ValueError:
            source_display = entry.plugin_dir
        content.append("     ", style="dim")
        content.append(f"source: {source_display}", style="dim green")
        content.append("\n")

        if entry.manifest is None:
            content.append("     ", style="dim")
            content.append(f"manifest: invalid: {entry.manifest_error}", style="yellow")
            content.append("\n")
        else:
            commands = ", ".join(entry.manifest.commands) or "-"
            content.append("     ", style="dim")
            content.append(f"commands: {commands}", style="dim")
            content.append("\n")
            keys = _format_plugin_keys(entry)
            if keys != "-":
                content.append("     ", style="dim")
                content.append(f"keys: {keys}", style="dim")
                content.append("\n")

        if entry.source is None:
            provenance = (
                f"invalid metadata: {entry.metadata_error}"
                if entry.metadata_error
                else "unmanaged"
            )
            content.append("     ", style="dim")
            content.append(f"provenance: {provenance}", style="dim")
            content.append("\n\n")
            continue

        source = entry.source
        ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
        provenance = f"{source.repo_url}{ref_label} ({source.repo_path})"
        content.append("     ", style="dim")
        content.append(f"provenance: {provenance}", style="dim")
        content.append("\n")
        content.append("     ", style="dim")
        content.append(
            f"installed: {format_installed_at_display(source.installed_at)} "
            f"revision: {format_revision_short(source.installed_revision)}",
            style="dim",
        )
        content.append("\n\n")

    content.append_text(Text("Browse marketplace plugins with /plugins available", style="dim"))
    content.append("\n")
    content.append_text(Text("Remove with /plugins remove <number|name>", style="dim"))
    return content


def _format_marketplace_plugins(plugins: Sequence[MarketplacePlugin], *, source: str) -> Text:
    content = Text()
    append_heading(content, "Marketplace plugins:")
    content.append_text(Text(f"Registry: {source}", style="dim"))
    content.append("\n\n")

    current_bundle = None
    for index, entry in enumerate(plugins, 1):
        if entry.bundle_name and entry.bundle_name != current_bundle:
            current_bundle = entry.bundle_name
            append_heading(content, entry.bundle_name)

        row = Text()
        row.append(f"[{index:2}] ", style="dim cyan")
        row.append(entry.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        if entry.description:
            append_wrapped_text(content, entry.description, indent="     ")
        if entry.source_url:
            content.append("     ", style="dim")
            content.append(f"source: {entry.source_url}", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _format_install_result(plugin_name: str, install_path: Path, config_path: Path) -> Text:
    try:
        display_path = install_path.relative_to(Path.cwd())
    except ValueError:
        display_path = install_path
    try:
        display_config = config_path.relative_to(Path.cwd())
    except ValueError:
        display_config = config_path

    content = Text()
    content.append(f"Installed plugin: {plugin_name}", style="green")
    content.append("\n")
    content.append(f"location: {display_path}", style="dim green")
    content.append("\n")
    content.append(f"enabled in: {display_config}", style="dim green")
    return content


def _format_update_results(updates: Sequence[PluginUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        content.append_text(Text("No managed plugins found.", style="yellow"))
        return content

    status_labels: dict[str, str] = {
        "up_to_date": "already up to date",
        "update_available": "update available",
        "updated": "updated",
        "unmanaged": "unmanaged",
        "invalid_metadata": "invalid metadata",
        "invalid_local_plugin": "invalid local plugin",
        "unknown_revision": "unknown revision",
        "source_unreachable": "source unreachable",
        "source_ref_missing": "source ref missing",
        "source_path_missing": "source path missing",
        "skipped_dirty": "skipped (local modifications)",
    }
    detail_statuses = {
        "invalid_metadata",
        "invalid_local_plugin",
        "unknown_revision",
        "source_unreachable",
        "source_ref_missing",
        "source_path_missing",
        "skipped_dirty",
    }

    for update in updates:
        row = Text()
        row.append(f"[{update.index:2}] ", style="dim cyan")
        row.append(update.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        try:
            source_display = update.plugin_dir.relative_to(Path.cwd())
        except ValueError:
            source_display = update.plugin_dir
        content.append("  - ", style="dim")
        content.append(f"source: {source_display}", style="dim green")
        content.append("\n")

        if update.current_revision or update.available_revision:
            content.append("  - ", style="dim")
            content.append(
                "revision: "
                f"{format_revision_short(update.current_revision)} -> "
                f"{format_revision_short(update.available_revision)}",
                style="dim",
            )
            content.append("\n")

        status = status_labels.get(update.status, update.status.replace("_", " "))
        if update.status in detail_statuses and update.detail:
            status = f"{status}: {update.detail}"
        style = None
        if update.status in {"up_to_date", "updated"}:
            style = "green"
        elif update.status == "update_available":
            style = "bold bright_yellow"
        elif update.status != "unmanaged":
            style = "yellow"

        content.append("  - ", style="dim")
        content.append("status: ", style="dim")
        content.append(status, style=style)
        content.append("\n\n")

    return content


def _refresh_provider_plugins(ctx: CommandContext, config_path: Path) -> None:
    settings = get_settings(config_path=str(config_path))
    provider = ctx.agent_provider
    if isinstance(provider, _PluginCommandProvider):
        provider.set_plugin_commands(settings.commands, base_path=config_path.parent)


async def handle_list_plugins(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    env_paths = resolve_environment_paths(ctx.resolve_settings())
    plugins = list_local_plugins(destination_root=env_paths.plugins)
    outcome.add_message(
        _format_local_plugins(plugins_dir=env_paths.plugins, plugins=plugins),
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome


def handle_plugins_help(*, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    outcome.add_message(
        "\n".join(_plugins_usage_lines()),
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome


async def handle_list_marketplace_plugins(
    ctx: CommandContext,
    *,
    agent_name: str,
) -> CommandOutcome:
    outcome = CommandOutcome()
    marketplace_url = get_marketplace_url(ctx.resolve_settings())
    try:
        plugins, source = await fetch_marketplace_plugins_with_source(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not plugins:
        outcome.add_message("No plugins found in the marketplace.", channel="warning")
        return outcome

    outcome.add_message(
        _format_marketplace_plugins(plugins, source=source),
        right_info="plugins",
        agent_name=agent_name,
    )
    outcome.add_message(
        "Install with `/plugins add <number|name>`.",
        channel="info",
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome


async def handle_add_plugin(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    env_paths = resolve_environment_paths(settings)
    config_path = _config_path_for_settings(ctx)
    marketplace_url = get_marketplace_url(settings)

    try:
        plugins, source = await fetch_marketplace_plugins_with_source(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not plugins:
        outcome.add_message("No plugins found in the marketplace.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = _format_marketplace_plugins(plugins, source=source)
        if not interactive:
            outcome.add_message(content, right_info="plugins", agent_name=agent_name)
            outcome.add_message(
                "Install with `/plugins add <number|name>`.",
                channel="info",
                right_info="plugins",
                agent_name=agent_name,
            )
            return outcome

        await ctx.io.emit(
            CommandMessage(text=content, right_info="plugins", agent_name=agent_name)
        )
        selection = await ctx.io.prompt_selection(
            "Install plugin by number or name (empty to cancel): ",
            options=[entry.name for entry in plugins],
            allow_cancel=True,
        )
        if selection is None:
            return outcome

    selected = select_plugin_by_name_or_index(plugins, selection)
    if selected is None:
        outcome.add_message(f"Plugin not found: {selection}", channel="error")
        outcome.add_message(
            "Run `/plugins available` to browse plugins.",
            channel="info",
            right_info="plugins",
            agent_name=agent_name,
        )
        return outcome

    try:
        install_path = await asyncio.to_thread(
            install_marketplace_plugin_sync,
            selected,
            destination_root=env_paths.plugins,
        )
        manifest = load_plugin_manifest(install_path)
        enable_plugin_in_config(config_path, manifest.name)
        _refresh_provider_plugins(ctx, config_path)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to install plugin: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(manifest.name, install_path, config_path),
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome


async def handle_remove_plugin(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    env_paths = resolve_environment_paths(settings)
    config_path = _config_path_for_settings(ctx)
    plugins = list_local_plugins(destination_root=env_paths.plugins)

    if not plugins:
        outcome.add_message("No local plugins to remove.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = _format_local_plugins(plugins_dir=env_paths.plugins, plugins=plugins)
        if not interactive:
            outcome.add_message(content, right_info="plugins", agent_name=agent_name)
            outcome.add_message(
                "Remove with `/plugins remove <number|name>`.",
                channel="info",
                right_info="plugins",
                agent_name=agent_name,
            )
            return outcome

        await ctx.io.emit(
            CommandMessage(text=content, right_info="plugins", agent_name=agent_name)
        )
        selection = await ctx.io.prompt_selection(
            "Remove plugin by number or name (empty to cancel): ",
            options=[entry.name for entry in plugins],
            allow_cancel=True,
        )
        if selection is None:
            return outcome

    selected = select_local_plugin_by_name_or_index(plugins, selection)
    if selected is None:
        outcome.add_message(f"Plugin not found: {selection}", channel="error")
        return outcome

    try:
        remove_local_plugin(selected.plugin_dir, destination_root=env_paths.plugins)
        disable_plugin_in_config(config_path, selected.name)
        _refresh_provider_plugins(ctx, config_path)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to remove plugin: {exc}", channel="error")
        return outcome

    outcome.add_message(
        f"Removed plugin: {selected.name}",
        channel="info",
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome


async def handle_update_plugin(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    selector, force, yes, parse_error = parse_update_argument(argument)
    if parse_error:
        outcome.add_message(parse_error, channel="error")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    updates = await asyncio.to_thread(check_plugin_updates, destination_root=env_paths.plugins)

    if selector is None:
        outcome.add_message(
            _format_update_results(updates, title="Plugin update check:"),
            right_info="plugins",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Apply with `/plugins update <number|name|all> [--force] [--yes]`.",
            channel="info",
            right_info="plugins",
            agent_name=agent_name,
        )
        return outcome

    selected = select_plugin_updates(updates, selector)
    if not selected:
        outcome.add_message(f"Plugin not found: {selector}", channel="error")
        return outcome

    if len(selected) > 1 and not yes:
        outcome.add_message(
            _format_update_results(selected, title="Update plan:"),
            right_info="plugins",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Multiple plugins selected. Re-run with `--yes` to apply updates.",
            channel="warning",
            right_info="plugins",
            agent_name=agent_name,
        )
        return outcome

    applied = await asyncio.to_thread(apply_plugin_updates, selected, force=force)
    outcome.add_message(
        _format_update_results(applied, title="Plugin update results:"),
        right_info="plugins",
        agent_name=agent_name,
    )
    if any(update.status == "updated" for update in applied):
        _refresh_provider_plugins(ctx, _config_path_for_settings(ctx))
    return outcome


async def handle_set_plugins_registry(
    ctx: CommandContext,
    *,
    argument: str | None,
    agent_name: str,
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    configured_urls = resolve_registries(settings)

    if not argument:
        current = get_marketplace_url(settings)
        content = Text()
        for index, url in enumerate(configured_urls, 1):
            row = Text()
            row.append(f"[{index:2}] ", style="dim cyan")
            row.append(url, style="bright_blue bold")
            if url == current:
                row.append(" • ", style="dim")
                row.append("current", style="dim green")
            content.append_text(row)
            content.append("\n")
        content.append("\n")
        content.append_text(Text("Usage: /plugins registry <number|url|path>", style="dim"))
        outcome.add_message(content, right_info="plugins", agent_name=agent_name)
        return outcome

    arg = argument.strip()
    if arg.isdigit():
        index = int(arg)
        if 1 <= index <= len(configured_urls):
            url = configured_urls[index - 1]
        else:
            outcome.add_message(
                f"Invalid registry number. Use 1-{len(configured_urls)}.",
                channel="warning",
            )
            return outcome
    else:
        url = arg

    try:
        plugins, resolved_url = await fetch_marketplace_plugins_with_source(url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    plugins_settings = settings.plugins
    plugins_settings.marketplace_url = resolved_url

    content = Text()
    if resolved_url != url:
        content.append_text(Text(f"Resolved from: {url}", style="dim"))
        content.append("\n")
    content.append_text(Text(f"Registry set to: {resolved_url}", style="green"))
    content.append("\n")
    content.append_text(Text(f"Plugins discovered: {len(plugins)}", style="dim"))
    outcome.add_message(content, right_info="plugins", agent_name=agent_name)
    return outcome


async def handle_plugins_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = str(action or "list").lower()

    if _is_help_flag(action) or _is_help_flag(argument):
        return handle_plugins_help(agent_name=agent_name)
    if normalized == "help":
        return handle_plugins_help(agent_name=agent_name)
    if normalized in {"list", ""}:
        return await handle_list_plugins(ctx, agent_name=agent_name)
    if normalized in {"available", "marketplace", "browse"}:
        return await handle_list_marketplace_plugins(ctx, agent_name=agent_name)
    if normalized in {"add", "install"}:
        return await handle_add_plugin(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"remove", "rm", "delete", "uninstall"}:
        return await handle_remove_plugin(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"update", "refresh", "upgrade"}:
        return await handle_update_plugin(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"registry", "source"}:
        return await handle_set_plugins_registry(
            ctx,
            argument=argument,
            agent_name=agent_name,
        )

    outcome = CommandOutcome()
    outcome.add_message(
        (
            f"Unknown /plugins action: {normalized}. "
            "Use list/available/add/remove/update/registry/help."
        ),
        channel="warning",
        right_info="plugins",
        agent_name=agent_name,
    )
    outcome.add_message(
        "\n".join(_plugins_usage_lines()),
        channel="info",
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome
