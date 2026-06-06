"""Shared /cards command handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.cards import service as card_service
from fast_agent.cards.manager import (
    CardPackPublishResult,
    CardPackUpdateInfo,
    format_card_pack_publish_status,
    format_marketplace_display_url,
    format_revision_short,
    get_marketplace_url,
    resolve_card_registries,
)
from fast_agent.commands.command_catalog import (
    command_usage_lines,
    format_unknown_command_action,
    normalize_command_action,
)
from fast_agent.commands.handlers._marketplace_argument_parsing import (
    optional_selector,
    parse_add_argument,
    parse_publish_argument,
    parse_update_argument,
    resolve_registry_argument,
)
from fast_agent.commands.handlers._text_formatting import (
    StatusText,
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
from fast_agent.commands.handlers.plugins import (
    _config_path_for_settings,
    _refresh_provider_plugins,
)
from fast_agent.commands.handlers.shared import add_info_messages, prompt_selection_after_message
from fast_agent.commands.results import CommandOutcome
from fast_agent.marketplace.formatting import (
    format_installed_revision_display,
    format_source_provenance,
)
from fast_agent.paths import resolve_environment_paths
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.count_display import format_count

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from fast_agent.cards.manager import LocalCardPack, MarketplaceCardPack
    from fast_agent.commands.context import CommandContext


def _format_local_card_packs(*, environment_paths, packs) -> Text:
    content = Text()
    manager_dir = environment_paths.card_packs
    append_heading(content, f"Card packs in {format_display_path(manager_dir)}:")
    if not packs:
        content.append_text(Text("No card packs installed.", style="yellow"))
        content.append("\n")
        content.append_text(Text("Use /cards add to install a card pack", style="dim"))
        return content

    for entry in packs:
        append_indexed_name_line(content, entry.index, entry.name)

        content.append("     ", style="dim")
        content.append(f"source: {format_display_path(entry.pack_dir)}", style="dim green")
        content.append("\n")

        if entry.source is None:
            summary = "unmanaged"
            if entry.metadata_error:
                summary = f"invalid metadata: {entry.metadata_error}"
            content.append("     ", style="dim")
            content.append(f"provenance: {summary}", style="dim")
            content.append("\n")
            content.append("\n")
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
        content.append("\n")
        content.append("\n")

    content.append_text(Text("Install with /cards add <number|name>", style="dim"))
    content.append("\n")
    content.append_text(Text("Remove with /cards remove <number|name>", style="dim"))
    return content


def _format_marketplace_packs(marketplace: Sequence[MarketplaceCardPack]) -> Text:
    content = Text()
    current_bundle = None

    for index, entry in enumerate(marketplace, 1):
        bundle_name = entry.bundle_name
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            append_heading(content, bundle_name)

        append_indexed_name_line(content, index, entry.name)

        if entry.description:
            append_wrapped_text(content, entry.description, indent="     ")
        content.append("     ", style="dim")
        content.append(f"kind: {entry.kind}", style="dim")
        content.append("\n")
        if entry.source_url:
            content.append("     ", style="dim")
            content.append(f"source: {entry.source_url}", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _add_empty_cards_registry_warning(outcome: CommandOutcome, url: str) -> None:
    content = Text()
    content.append_text(
        Text("No card packs found in the registry; registry unchanged.", style="yellow")
    )
    content.append("\n")
    content.append_text(Text(f"Registry: {format_marketplace_display_url(url)}", style="dim"))
    outcome.add_message(content, channel="warning", right_info="cards")


def _format_install_result(
    *, pack_name: str, install_path: Path, installed_files: Sequence[str]
) -> Text:
    content = Text()
    content.append(f"Installed card pack: {pack_name}", style="green")
    content.append("\n")
    content.append(f"location: {format_display_path(install_path)}", style="dim green")
    content.append("\n")
    content.append(f"managed files: {format_count(len(installed_files), 'file')}", style="dim")
    return content


def _format_remove_result(*, pack_name: str, skipped_paths: Sequence[str]) -> Text:
    content = Text()
    content.append(f"Removed card pack: {pack_name}", style="green")
    if skipped_paths:
        content.append("\n")
        content.append(
            f"Skipped {format_count(len(skipped_paths), 'path')} with shared ownership.",
            style="yellow",
        )
    return content


def _format_update_results(updates: Sequence[CardPackUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        append_warning_line(content, "No managed card packs found.")
        return content

    for update in updates:
        append_indexed_name_line(content, update.index, update.name)

        append_detail_line(
            content,
            "source",
            format_display_path(update.pack_dir),
            value_style="dim green",
        )

        if update.managed_source is not None:
            source = update.managed_source
            provenance = format_source_provenance(
                source.repo_url,
                source.repo_ref,
                source.repo_path,
            )
            append_detail_line(content, "provenance", provenance, value_style="dim")
            append_detail_line(
                content,
                "installed",
                format_installed_revision_display(source.installed_at, source.installed_revision),
                value_style="dim",
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


def _publish_status_text(result: CardPackPublishResult) -> StatusText:
    text, style = format_card_pack_publish_status(result)
    return StatusText(text, style)


def _publish_failure_warning(result: CardPackPublishResult) -> str | None:
    if not card_service.is_card_pack_publish_failure(result.status):
        return None
    if result.patch_path is not None:
        return "Push was rejected. Share the generated patch with a maintainer or open a PR from your branch."
    return "Publish failed after committing locally. Push manually or ask a maintainer with write access."


def _format_publish_result(result: CardPackPublishResult, *, title: str) -> Text:
    content = Text()
    append_heading(content, title)

    row = Text()
    row.append(result.pack_name, style="bright_blue bold")
    content.append_text(row)
    content.append("\n")

    append_detail_line(
        content,
        "source",
        format_display_path(result.pack_dir),
        value_style="dim green",
    )

    if result.repo_root is not None:
        repo_label = format_display_path(result.repo_root)
        if result.repo_path:
            repo_label = f"{repo_label} ({result.repo_path})"
        append_detail_line(content, "repo", repo_label, value_style="dim")

    if result.commit:
        append_detail_line(
            content,
            "commit",
            format_revision_short(result.commit),
            value_style="dim",
        )

    if result.patch_path is not None:
        append_detail_line(
            content,
            "patch",
            format_display_path(result.patch_path),
            value_style="dim",
        )

    if result.retained_temp_dir is not None:
        append_detail_line(
            content,
            "temp clone",
            format_display_path(result.retained_temp_dir),
            value_style="dim",
        )

    append_status_line(content, _publish_status_text(result))

    return content


async def handle_list_cards(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    env_paths = resolve_environment_paths(ctx.resolve_settings())
    packs = card_service.list_installed_packs(environment_paths=env_paths)
    outcome.add_message(
        _format_local_card_packs(environment_paths=env_paths, packs=packs),
        right_info="cards",
        agent_name=agent_name,
    )
    return outcome


async def handle_set_cards_registry(
    ctx: CommandContext,
    *,
    argument: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    configured_urls = resolve_card_registries(settings)

    registry_arg = optional_selector(argument)
    if registry_arg is None:
        current = get_marketplace_url(settings)
        current_display = format_marketplace_display_url(current)
        content = Text()
        for index, url in enumerate(configured_urls, 1):
            display = format_marketplace_display_url(url)
            append_indexed_current_line(
                content,
                index,
                display,
                is_current=display == current_display,
            )

        content.append("\n")
        content.append_text(Text("Usage: /cards registry <number|url|path>", style="dim"))
        outcome.add_message(content, right_info="cards")
        return outcome

    resolved = resolve_registry_argument(registry_arg, configured_urls)
    if resolved.warning is not None:
        outcome.add_message(resolved.warning, channel="warning")
        return outcome
    if resolved.url is None:
        return outcome
    selected_url = resolved.url

    try:
        marketplace = await card_service.scan_marketplace(selected_url)
    except Exception as exc:
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    if not marketplace.packs:
        _add_empty_cards_registry_warning(outcome, selected_url)
        return outcome

    settings.cards.marketplace_url = marketplace.source

    content = Text()
    if marketplace.source != selected_url:
        content.append_text(Text(f"Resolved from: {selected_url}", style="dim"))
        content.append("\n")
    content.append_text(
        Text(
            f"Registry set to: {format_marketplace_display_url(marketplace.source)}",
            style="green",
        )
    )
    content.append("\n")
    content.append_text(Text(f"Card packs discovered: {len(marketplace.packs)}", style="dim"))
    outcome.add_message(content, right_info="cards")
    return outcome


async def _select_card_pack_for_add(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    packs: Sequence[MarketplaceCardPack],
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> str | None:
    selection = optional_selector(argument)
    if selection:
        return selection

    content = Text()
    append_heading(content, "Marketplace card packs:")
    content.append_text(_format_marketplace_packs(packs))

    if not interactive:
        outcome.add_message(content, right_info="cards", agent_name=agent_name)
        add_info_messages(
            outcome,
            ("Install with `/cards add <number|name>`.",),
            right_info="cards",
            agent_name=agent_name,
        )
        return None

    return await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="cards",
        agent_name=agent_name,
        prompt="Install card pack by number or name (empty to cancel): ",
        options=[entry.name for entry in packs],
        allow_cancel=True,
    )


async def _select_local_card_pack(
    ctx: CommandContext,
    outcome: CommandOutcome,
    *,
    environment_paths,
    packs: Sequence[LocalCardPack],
    agent_name: str,
    argument: str | None,
    interactive: bool,
    prompt: str,
    usage_hint: str,
) -> str | None:
    selection = optional_selector(argument)
    if selection:
        return selection
    if len(packs) == 1:
        return packs[0].name

    content = _format_local_card_packs(environment_paths=environment_paths, packs=packs)
    if not interactive:
        outcome.add_message(content, right_info="cards", agent_name=agent_name)
        add_info_messages(
            outcome,
            (usage_hint,),
            right_info="cards",
            agent_name=agent_name,
        )
        return None

    return await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="cards",
        agent_name=agent_name,
        prompt=prompt,
        options=[entry.name for entry in packs],
        allow_cancel=True,
    )


async def handle_add_card_pack(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = parse_add_argument(argument, allow_skills_dir=False)
    if parsed.error is not None:
        outcome.add_message(parsed.error, channel="warning")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    marketplace_url = parsed.registry or get_marketplace_url(ctx.resolve_settings())
    try:
        marketplace = await card_service.scan_marketplace(marketplace_url)
    except Exception as exc:
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace.packs:
        outcome.add_message("No card packs found in the marketplace.", channel="warning")
        return outcome

    selection = await _select_card_pack_for_add(
        ctx,
        outcome,
        packs=marketplace.packs,
        agent_name=agent_name,
        argument=parsed.selector,
        interactive=interactive,
    )
    if selection is None:
        return outcome

    try:
        install_result = await card_service.install_selected_pack(
            card_service.select_marketplace_pack(marketplace.packs, selection),
            environment_paths=env_paths,
            force=parsed.force,
            marketplace_source=marketplace.source,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome
    except Exception as exc:
        outcome.add_message(f"Failed to install card pack: {exc}", channel="error")
        return outcome
    _refresh_provider_plugins(ctx, _config_path_for_settings(ctx))

    outcome.add_message(
        _format_install_result(
            pack_name=install_result.pack.name,
            install_path=install_result.install_result.pack_dir,
            installed_files=install_result.install_result.installed_files,
        ),
        right_info="cards",
        agent_name=agent_name,
    )
    if install_result.readme:
        outcome.add_message(
            install_result.readme,
            title=f"{install_result.pack.name} README",
            right_info="cards",
            agent_name=agent_name,
            render_markdown=True,
        )
    return outcome


async def handle_remove_card_pack(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    env_paths = resolve_environment_paths(ctx.resolve_settings())
    packs = card_service.list_installed_packs(environment_paths=env_paths)
    if not packs:
        outcome.add_message("No local card packs to remove.", channel="warning")
        return outcome

    selector = await _select_local_card_pack(
        ctx,
        outcome,
        environment_paths=env_paths,
        packs=packs,
        agent_name=agent_name,
        argument=argument,
        interactive=interactive,
        prompt="Remove card pack by number or name (empty to cancel): ",
        usage_hint="Remove with `/cards remove <number|name>`.",
    )
    if selector is None:
        return outcome

    try:
        removal = card_service.remove_pack(
            environment_paths=env_paths,
            selector=selector,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome
    except Exception as exc:
        outcome.add_message(f"Failed to remove card pack: {exc}", channel="error")
        return outcome

    message = _format_remove_result(
        pack_name=removal.pack_name,
        skipped_paths=removal.skipped_paths,
    )
    outcome.add_message(message, right_info="cards", agent_name=agent_name)
    return outcome


async def handle_card_pack_readme(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()
    env_paths = resolve_environment_paths(ctx.resolve_settings())
    packs = card_service.list_installed_packs(environment_paths=env_paths)
    if not packs:
        outcome.add_message("No local card packs installed.", channel="warning")
        return outcome

    selected_name = await _select_local_card_pack(
        ctx,
        outcome,
        environment_paths=env_paths,
        packs=packs,
        agent_name=agent_name,
        argument=argument,
        interactive=interactive,
        prompt="Show README for card pack by number or name (empty to cancel): ",
        usage_hint="Show with `/cards readme <number|name>`.",
    )
    if selected_name is None:
        return outcome

    try:
        readme_record = card_service.read_installed_pack_readme(
            environment_paths=env_paths,
            selector=selected_name,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome

    if not readme_record.readme:
        outcome.add_message(
            f"Card pack '{readme_record.pack_name}' does not include a README.md.",
            channel="warning",
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        readme_record.readme,
        title=f"{readme_record.pack_name} README",
        right_info="cards",
        agent_name=agent_name,
        render_markdown=True,
    )
    return outcome


async def handle_update_card_pack(
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
    updates = card_service.check_updates(environment_paths=env_paths)

    if parsed.selector is None:
        outcome.add_message(
            _format_update_results(updates, title="Card pack update check:"),
            right_info="cards",
            agent_name=agent_name,
        )
        add_info_messages(
            outcome,
            ("Apply with `/cards update <number|name|all> [--force] [--yes]`.",),
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    try:
        plan = card_service.plan_updates(environment_paths=env_paths, selector=parsed.selector)
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome

    if len(plan.selected) > 1 and not parsed.yes:
        outcome.add_message(
            _format_update_results(plan.selected, title="Update plan:"),
            right_info="cards",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Multiple card packs selected. Re-run with `--yes` to apply updates.",
            channel="warning",
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    applied = await asyncio.to_thread(
        card_service.apply_update_plan,
        plan.selected,
        environment_paths=env_paths,
        force=parsed.force,
    )
    _refresh_provider_plugins(ctx, _config_path_for_settings(ctx))
    outcome.add_message(
        _format_update_results(applied.applied, title="Card pack update results:"),
        right_info="cards",
        agent_name=agent_name,
    )

    # Show READMEs for successfully updated packs
    for readme_record in applied.readmes:
        if readme_record.readme:
            outcome.add_message(
                readme_record.readme,
                title=f"{readme_record.pack_name} README (updated)",
                right_info="cards",
                agent_name=agent_name,
                render_markdown=True,
            )

    return outcome


async def handle_publish_card_pack(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = parse_publish_argument(argument)
    if parsed.error:
        outcome.add_message(parsed.error, channel="error")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    packs = card_service.list_installed_packs(environment_paths=env_paths)
    if not packs:
        outcome.add_message("No local card packs to publish.", channel="warning")
        return outcome

    if parsed.selector is None:
        outcome.add_message(
            _format_local_card_packs(environment_paths=env_paths, packs=packs),
            right_info="cards",
            agent_name=agent_name,
        )
        add_info_messages(
            outcome,
            [
                "Publish with `/cards publish <number|name> [--no-push] [--message ...] "
                "[--temp-dir <path>] [--keep-temp]`."
            ],
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    try:
        result = await asyncio.to_thread(
            card_service.publish_pack,
            environment_paths=env_paths,
            selector=parsed.selector,
            push=parsed.push,
            commit_message=parsed.message,
            temp_dir=parsed.temp_dir,
            keep_temp=parsed.keep_temp,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome

    outcome.add_message(
        _format_publish_result(result, title="Card pack publish:"),
        right_info="cards",
        agent_name=agent_name,
    )

    warning = _publish_failure_warning(result)
    if warning is not None:
        outcome.add_message(
            warning,
            channel="warning",
            right_info="cards",
            agent_name=agent_name,
        )

    if result.retained_temp_dir is not None:
        add_info_messages(
            outcome,
            (f"Retained temporary clone at: {result.retained_temp_dir}",),
            right_info="cards",
            agent_name=agent_name,
        )

    return outcome


type _CardsActionHandler = Callable[
    ["CommandContext", str, str | None],
    Awaitable[CommandOutcome],
]


async def _handle_cards_list_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
) -> CommandOutcome:
    return await handle_list_cards(ctx, agent_name=agent_name)


async def _handle_cards_add_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_add_card_pack(ctx, agent_name=agent_name, argument=argument)


async def _handle_cards_registry_action(
    ctx: "CommandContext",
    _agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_set_cards_registry(ctx, argument=argument)


async def _handle_cards_remove_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_remove_card_pack(ctx, agent_name=agent_name, argument=argument)


async def _handle_cards_readme_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_card_pack_readme(ctx, agent_name=agent_name, argument=argument)


async def _handle_cards_update_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_update_card_pack(ctx, agent_name=agent_name, argument=argument)


async def _handle_cards_publish_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_publish_card_pack(ctx, agent_name=agent_name, argument=argument)


_CARDS_ACTION_HANDLERS: dict[str, _CardsActionHandler] = {
    "list": _handle_cards_list_action,
    "add": _handle_cards_add_action,
    "registry": _handle_cards_registry_action,
    "remove": _handle_cards_remove_action,
    "readme": _handle_cards_readme_action,
    "update": _handle_cards_update_action,
    "publish": _handle_cards_publish_action,
}


async def handle_cards_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = normalize_command_action("cards", action)

    if is_help_flag(action) or is_help_flag(argument):
        outcome = CommandOutcome()
        add_info_messages(outcome, ("\n".join(command_usage_lines("cards")),), right_info="cards")
        return outcome

    handler = _CARDS_ACTION_HANDLERS.get(normalized)
    if handler is not None:
        return await handler(ctx, agent_name, argument)

    outcome = CommandOutcome()
    outcome.add_message(
        format_unknown_command_action("cards", normalized),
        channel="warning",
        right_info="cards",
    )
    add_info_messages(outcome, ("\n".join(command_usage_lines("cards")),), right_info="cards")
    return outcome
