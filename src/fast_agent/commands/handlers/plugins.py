"""Shared /plugins command handlers for the interactive prompt."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rich.text import Text

from fast_agent.commands.command_catalog import (
    command_usage_lines,
    format_unknown_command_action,
    normalize_command_action,
)
from fast_agent.commands.handlers._marketplace_argument_parsing import (
    optional_selector,
    parse_add_argument,
    parse_update_argument,
    resolve_registry_argument,
)
from fast_agent.commands.handlers._text_formatting import (
    append_detail_line,
    append_heading,
    append_indexed_current_line,
    append_indexed_name_line,
    append_revision_line,
    append_status_line,
    append_warning_line,
    append_wrapped_text,
    format_display_path,
    update_status_text,
)
from fast_agent.commands.handlers.shared import (
    add_info_messages,
    prompt_selection_after_message,
    unique_selection_options,
)
from fast_agent.commands.results import CommandOutcome
from fast_agent.config import get_settings
from fast_agent.home import PREFERRED_CONFIG_FILENAME
from fast_agent.marketplace.formatting import (
    format_installed_revision_display,
    format_source_provenance,
)
from fast_agent.marketplace.update_status import is_update_applied
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
from fast_agent.plugins.provenance import format_revision_short
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _config_path_for_settings(ctx: CommandContext) -> Path:
    settings = ctx.resolve_settings()
    if settings._config_file:
        return Path(settings._config_file)
    return resolve_environment_paths(settings).root / PREFERRED_CONFIG_FILENAME


def _format_plugin_keys(entry: LocalPlugin) -> str:
    if entry.manifest is None:
        return "-"
    labels = [
        f"{name}: {key}"
        for name, spec in entry.manifest.commands.items()
        if (key := strip_to_none(spec.key)) is not None
    ]
    return ", ".join(labels) if labels else "-"


def _format_local_plugins(*, plugins_dir: Path, plugins: Sequence[LocalPlugin]) -> Text:
    content = Text()
    append_heading(content, f"Plugins in {format_display_path(plugins_dir)}:")
    if not plugins:
        content.append_text(Text("No plugins installed.", style="yellow"))
        content.append("\n")
        content.append_text(Text("Install with /plugins add <number|name>", style="dim"))
        return content

    for entry in plugins:
        append_indexed_name_line(content, entry.index, entry.name)

        content.append("     ", style="dim")
        content.append(f"source: {format_display_path(entry.plugin_dir)}", style="dim green")
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
                f"invalid metadata: {entry.metadata_error}" if entry.metadata_error else "unmanaged"
            )
            content.append("     ", style="dim")
            content.append(f"provenance: {provenance}", style="dim")
            content.append("\n\n")
            continue

        source = entry.source
        provenance = format_source_provenance(source.repo_url, source.repo_ref, source.repo_path)
        content.append("     ", style="dim")
        content.append(f"provenance: {provenance}", style="dim")
        content.append("\n")
        content.append("     ", style="dim")
        content.append(
            f"installed: {format_installed_revision_display(source.installed_at, source.installed_revision)}",
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

        append_indexed_name_line(content, index, entry.name)

        if entry.description:
            append_wrapped_text(content, entry.description, indent="     ")
        if entry.source_url:
            content.append("     ", style="dim")
            content.append(f"source: {entry.source_url}", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _add_empty_plugins_registry_warning(
    outcome: CommandOutcome, url: str, *, agent_name: str
) -> None:
    content = Text()
    content.append_text(
        Text("No plugins found in the registry; registry unchanged.", style="yellow")
    )
    content.append("\n")
    content.append_text(Text(f"Registry: {url}", style="dim"))
    outcome.add_message(
        content,
        channel="warning",
        right_info="plugins",
        agent_name=agent_name,
    )


def _plugin_selection_options(plugins: Sequence[MarketplacePlugin]) -> list[str]:
    return unique_selection_options(
        option for entry in plugins for option in (entry.name, entry.install_dir_name)
    )


def _local_plugin_selection_options(plugins: Sequence[LocalPlugin]) -> list[str]:
    return unique_selection_options(
        option for entry in plugins for option in (entry.name, entry.plugin_dir.name)
    )


def _format_install_result(plugin_name: str, install_path: Path, config_path: Path) -> Text:
    content = Text()
    content.append(f"Installed plugin: {plugin_name}", style="green")
    content.append("\n")
    content.append(f"location: {format_display_path(install_path)}", style="dim green")
    content.append("\n")
    content.append(f"enabled in: {format_display_path(config_path)}", style="dim green")
    return content


def _add_plugin_install_guidance(
    outcome: CommandOutcome,
    *,
    agent_name: str,
) -> None:
    add_info_messages(
        outcome,
        ("Install with `/plugins add <number|name>`.",),
        right_info="plugins",
        agent_name=agent_name,
    )


def _format_update_results(updates: Sequence[PluginUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        append_warning_line(content, "No managed plugins found.")
        return content

    for update in updates:
        append_indexed_name_line(content, update.index, update.name)

        append_detail_line(
            content,
            "source",
            format_display_path(update.plugin_dir),
            value_style="dim green",
        )

        append_revision_line(
            content,
            update.current_revision,
            update.available_revision,
            format_revision=format_revision_short,
        )

        status = update_status_text(
            update.status,
            detail=update.detail,
        )

        append_status_line(content, status)
        content.append("\n")

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
        "\n".join(command_usage_lines("plugins")),
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
    except Exception as exc:
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
    _add_plugin_install_guidance(outcome, agent_name=agent_name)
    return outcome


async def _select_plugin_for_add(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    plugins: Sequence[MarketplacePlugin],
    source: str,
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> str | None:
    selection = optional_selector(argument)
    if selection:
        return selection

    content = _format_marketplace_plugins(plugins, source=source)
    if not interactive:
        outcome.add_message(content, right_info="plugins", agent_name=agent_name)
        _add_plugin_install_guidance(outcome, agent_name=agent_name)
        return None

    return await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="plugins",
        agent_name=agent_name,
        prompt="Install plugin by number or name (empty to cancel): ",
        options=_plugin_selection_options(plugins),
        allow_cancel=True,
    )


async def handle_add_plugin(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = parse_add_argument(argument, allow_skills_dir=False, allow_force=False)
    if parsed.error is not None:
        outcome.add_message(parsed.error, channel="warning")
        return outcome
    settings = ctx.resolve_settings()
    env_paths = resolve_environment_paths(settings)
    config_path = _config_path_for_settings(ctx)
    marketplace_url = parsed.registry or get_marketplace_url(settings)

    try:
        plugins, source = await fetch_marketplace_plugins_with_source(marketplace_url)
    except Exception as exc:
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not plugins:
        outcome.add_message("No plugins found in the marketplace.", channel="warning")
        return outcome

    selection = await _select_plugin_for_add(
        ctx,
        outcome,
        plugins=plugins,
        source=source,
        agent_name=agent_name,
        argument=parsed.selector,
        interactive=interactive,
    )
    if selection is None:
        return outcome

    selected = select_plugin_by_name_or_index(plugins, selection)
    if selected is None:
        outcome.add_message(f"Plugin not found: {selection}", channel="error")
        add_info_messages(
            outcome,
            ("Run `/plugins available` to browse plugins.",),
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
    except Exception as exc:
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

    selection = optional_selector(argument)
    if not selection:
        content = _format_local_plugins(plugins_dir=env_paths.plugins, plugins=plugins)
        if not interactive:
            outcome.add_message(content, right_info="plugins", agent_name=agent_name)
            add_info_messages(
                outcome,
                ("Remove with `/plugins remove <number|name>`.",),
                right_info="plugins",
                agent_name=agent_name,
            )
            return outcome

        selection = await prompt_selection_after_message(
            ctx,
            content=content,
            right_info="plugins",
            agent_name=agent_name,
            prompt="Remove plugin by number or name (empty to cancel): ",
            options=_local_plugin_selection_options(plugins),
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
    except Exception as exc:
        outcome.add_message(f"Failed to remove plugin: {exc}", channel="error")
        return outcome

    add_info_messages(
        outcome,
        (f"Removed plugin: {selected.name}",),
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

    parsed = parse_update_argument(argument)
    if parsed.error:
        outcome.add_message(parsed.error, channel="error")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    updates = await asyncio.to_thread(check_plugin_updates, destination_root=env_paths.plugins)

    if parsed.selector is None:
        outcome.add_message(
            _format_update_results(updates, title="Plugin update check:"),
            right_info="plugins",
            agent_name=agent_name,
        )
        add_info_messages(
            outcome,
            ("Apply with `/plugins update <number|name|all> [--force] [--yes]`.",),
            right_info="plugins",
            agent_name=agent_name,
        )
        return outcome

    selected = select_plugin_updates(updates, parsed.selector)
    if not selected:
        outcome.add_message(f"Plugin not found: {parsed.selector}", channel="error")
        return outcome

    if len(selected) > 1 and not parsed.yes:
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

    applied = await asyncio.to_thread(apply_plugin_updates, selected, force=parsed.force)
    outcome.add_message(
        _format_update_results(applied, title="Plugin update results:"),
        right_info="plugins",
        agent_name=agent_name,
    )
    if any(is_update_applied(update.status) for update in applied):
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

    registry_arg = optional_selector(argument)
    if registry_arg is None:
        current = get_marketplace_url(settings)
        content = Text()
        for index, url in enumerate(configured_urls, 1):
            append_indexed_current_line(content, index, url, is_current=url == current)
        content.append("\n")
        content.append_text(Text("Usage: /plugins registry <number|url|path>", style="dim"))
        outcome.add_message(content, right_info="plugins", agent_name=agent_name)
        return outcome

    resolved = resolve_registry_argument(registry_arg, configured_urls)
    if resolved.warning is not None:
        outcome.add_message(resolved.warning, channel="warning")
        return outcome
    if resolved.url is None:
        return outcome
    url = resolved.url

    try:
        plugins, resolved_url = await fetch_marketplace_plugins_with_source(url)
    except Exception as exc:
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    if not plugins:
        _add_empty_plugins_registry_warning(outcome, url, agent_name=agent_name)
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


type _PluginsActionHandler = Callable[
    ["CommandContext", str, str | None],
    Awaitable[CommandOutcome],
]


async def _handle_plugins_list_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
) -> CommandOutcome:
    return await handle_list_plugins(ctx, agent_name=agent_name)


async def _handle_plugins_available_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
) -> CommandOutcome:
    return await handle_list_marketplace_plugins(ctx, agent_name=agent_name)


async def _handle_plugins_add_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_add_plugin(ctx, agent_name=agent_name, argument=argument)


async def _handle_plugins_remove_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_remove_plugin(ctx, agent_name=agent_name, argument=argument)


async def _handle_plugins_update_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_update_plugin(ctx, agent_name=agent_name, argument=argument)


async def _handle_plugins_registry_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_set_plugins_registry(
        ctx,
        argument=argument,
        agent_name=agent_name,
    )


_PLUGINS_ACTION_HANDLERS: dict[str, _PluginsActionHandler] = {
    "list": _handle_plugins_list_action,
    "available": _handle_plugins_available_action,
    "add": _handle_plugins_add_action,
    "remove": _handle_plugins_remove_action,
    "update": _handle_plugins_update_action,
    "registry": _handle_plugins_registry_action,
}


async def handle_plugins_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = normalize_command_action("plugins", action)

    if is_help_flag(action) or is_help_flag(argument):
        return handle_plugins_help(agent_name=agent_name)

    handler = _PLUGINS_ACTION_HANDLERS.get(normalized)
    if handler is not None:
        return await handler(ctx, agent_name, argument)

    outcome = CommandOutcome()
    outcome.add_message(
        format_unknown_command_action("plugins", normalized),
        channel="warning",
        right_info="plugins",
        agent_name=agent_name,
    )
    add_info_messages(
        outcome,
        ("\n".join(command_usage_lines("plugins")),),
        right_info="plugins",
        agent_name=agent_name,
    )
    return outcome
