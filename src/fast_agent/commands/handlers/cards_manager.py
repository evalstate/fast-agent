"""Shared /cards command handlers."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.cards import service as card_service
from fast_agent.cards.manager import (
    CardPackPublishResult,
    CardPackUpdateInfo,
    format_installed_at_display,
    format_marketplace_display_url,
    format_revision_short,
    get_marketplace_url,
    resolve_card_registries,
)
from fast_agent.commands.command_catalog import suggest_command_action
from fast_agent.commands.handlers._marketplace_argument_parsing import parse_update_argument
from fast_agent.commands.handlers._text_formatting import append_heading, append_wrapped_text
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.paths import resolve_environment_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext


def _cards_usage_lines() -> list[str]:
    return [
        "Usage: /cards [list|add|remove|readme|update|publish|registry|help] [args]",
        "",
        "Examples:",
        "- /cards add <number|name>",
        "- /cards readme <number|name>",
        "- /cards update all --yes",
        "- /cards registry",
    ]


def _is_help_flag(value: str | None) -> bool:
    token = (value or "").strip().lower()
    return token in {"help", "--help", "-h"}


def _format_local_card_packs(*, environment_paths, packs) -> Text:
    content = Text()
    manager_dir = environment_paths.card_packs
    try:
        display_dir = manager_dir.relative_to(Path.cwd())
    except ValueError:
        display_dir = manager_dir

    append_heading(content, f"Card packs in {display_dir}:")
    if not packs:
        content.append_text(Text("No card packs installed.", style="yellow"))
        content.append("\n")
        content.append_text(Text("Use /cards add to install a card pack", style="dim"))
        return content

    for entry in packs:
        row = Text()
        row.append(f"[{entry.index:2}] ", style="dim cyan")
        row.append(entry.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        try:
            source_display = entry.pack_dir.relative_to(Path.cwd())
        except ValueError:
            source_display = entry.pack_dir
        content.append("     ", style="dim")
        content.append(f"source: {source_display}", style="dim green")
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
        content.append("\n")
        content.append("\n")

    content.append_text(Text("Install with /cards add <number|name>", style="dim"))
    content.append("\n")
    content.append_text(Text("Remove with /cards remove <number|name>", style="dim"))
    return content


def _format_marketplace_packs(marketplace) -> Text:
    content = Text()
    current_bundle = None

    for index, entry in enumerate(marketplace, 1):
        bundle_name = getattr(entry, "bundle_name", None)
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            append_heading(content, bundle_name)

        line = Text()
        line.append(f"[{index:2}] ", style="dim cyan")
        line.append(entry.name, style="bright_blue bold")
        content.append_text(line)
        content.append("\n")

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


def _format_install_result(*, pack_name: str, install_path: Path, installed_files: Sequence[str]) -> Text:
    try:
        display_path = install_path.relative_to(Path.cwd())
    except ValueError:
        display_path = install_path

    content = Text()
    content.append(f"Installed card pack: {pack_name}", style="green")
    content.append("\n")
    content.append(f"location: {display_path}", style="dim green")
    content.append("\n")
    content.append(f"managed files: {len(installed_files)}", style="dim")
    return content
def _parse_publish_argument(
    argument: str | None,
) -> tuple[str | None, bool, str | None, Path | None, bool, str | None]:
    if argument is None:
        return None, True, None, None, False, None

    try:
        tokens = shlex.split(argument)
    except ValueError as exc:
        return None, True, None, None, False, f"Invalid publish arguments: {exc}"

    selector: str | None = None
    push = True
    message: str | None = None
    temp_dir: Path | None = None
    keep_temp = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--no-push":
            push = False
            index += 1
            continue
        if token == "--push":
            push = True
            index += 1
            continue
        if token in {"--message", "-m"}:
            if index + 1 >= len(tokens):
                return None, True, None, None, False, "Missing value for --message"
            message = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--message="):
            message = token.split("=", maxsplit=1)[1]
            index += 1
            continue
        if token == "--keep-temp":
            keep_temp = True
            index += 1
            continue
        if token == "--temp-dir":
            if index + 1 >= len(tokens):
                return None, True, None, None, False, "Missing value for --temp-dir"
            temp_dir = Path(tokens[index + 1]).expanduser()
            index += 2
            continue
        if token.startswith("--temp-dir="):
            temp_dir = Path(token.split("=", maxsplit=1)[1]).expanduser()
            index += 1
            continue
        if token.startswith("--"):
            return None, True, None, None, False, f"Unknown option: {token}"
        if selector is not None:
            return None, True, None, None, False, "Only one selector is allowed."
        selector = token
        index += 1

    return selector, push, message, temp_dir, keep_temp, None


def _format_update_results(updates: Sequence[CardPackUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        content.append_text(Text("No managed card packs found.", style="yellow"))
        return content

    status_labels: dict[str, str] = {
        "up_to_date": "already up to date",
        "update_available": "update available",
        "updated": "updated",
        "unmanaged": "unmanaged",
        "invalid_metadata": "invalid metadata",
        "invalid_local_pack": "invalid local pack",
        "unknown_revision": "unknown revision",
        "source_unreachable": "source unreachable",
        "source_ref_missing": "source ref missing",
        "source_path_missing": "source path missing",
        "skipped_dirty": "skipped (local modifications)",
        "ownership_conflict": "ownership conflict",
    }
    detail_statuses = {
        "invalid_metadata",
        "invalid_local_pack",
        "unknown_revision",
        "source_unreachable",
        "source_ref_missing",
        "source_path_missing",
        "skipped_dirty",
        "ownership_conflict",
    }

    for update in updates:
        row = Text()
        row.append(f"[{update.index:2}] ", style="dim cyan")
        row.append(update.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        try:
            source_display = update.pack_dir.relative_to(Path.cwd())
        except ValueError:
            source_display = update.pack_dir
        content.append("  - ", style="dim")
        content.append(f"source: {source_display}", style="dim green")
        content.append("\n")

        if update.managed_source is not None:
            source = update.managed_source
            ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
            provenance = f"{source.repo_url}{ref_label} ({source.repo_path})"
            content.append("  - ", style="dim")
            content.append(f"provenance: {provenance}", style="dim")
            content.append("\n")
            content.append("  - ", style="dim")
            content.append(
                f"installed: {format_installed_at_display(source.installed_at)} "
                f"revision: {format_revision_short(source.installed_revision)}",
                style="dim",
            )
            content.append("\n")

        if update.current_revision or update.available_revision:
            current = format_revision_short(update.current_revision)
            available = format_revision_short(update.available_revision)
            content.append("  - ", style="dim")
            content.append(f"revision: {current} -> {available}", style="dim")
            content.append("\n")

        status_text = status_labels.get(update.status, update.status.replace("_", " "))
        if update.status in detail_statuses and update.detail:
            status_text = f"{status_text}: {update.detail}"

        status_style: str | None = None
        if update.status in {"up_to_date", "updated"}:
            status_style = "green"
        elif update.status == "update_available":
            status_style = "bold bright_yellow"
        elif update.status not in {"unmanaged"}:
            status_style = "yellow"

        content.append("  - ", style="dim")
        content.append("status: ", style="dim")
        if status_style:
            content.append(status_text, style=status_style)
        else:
            content.append(status_text)
        content.append("\n\n")

    return content


def _format_publish_result(result: CardPackPublishResult, *, title: str) -> Text:
    content = Text()
    append_heading(content, title)

    row = Text()
    row.append(result.pack_name, style="bright_blue bold")
    content.append_text(row)
    content.append("\n")

    try:
        source_display = result.pack_dir.relative_to(Path.cwd())
    except ValueError:
        source_display = result.pack_dir
    content.append("  - ", style="dim")
    content.append(f"source: {source_display}", style="dim green")
    content.append("\n")

    if result.repo_root is not None:
        try:
            repo_display = result.repo_root.relative_to(Path.cwd())
        except ValueError:
            repo_display = result.repo_root
        repo_label = str(repo_display)
        if result.repo_path:
            repo_label = f"{repo_label} ({result.repo_path})"
        content.append("  - ", style="dim")
        content.append(f"repo: {repo_label}", style="dim")
        content.append("\n")

    if result.commit:
        content.append("  - ", style="dim")
        content.append(f"commit: {format_revision_short(result.commit)}", style="dim")
        content.append("\n")

    if result.patch_path is not None:
        try:
            patch_display = result.patch_path.relative_to(Path.cwd())
        except ValueError:
            patch_display = result.patch_path
        content.append("  - ", style="dim")
        content.append(f"patch: {patch_display}", style="dim")
        content.append("\n")

    if result.retained_temp_dir is not None:
        try:
            temp_display = result.retained_temp_dir.relative_to(Path.cwd())
        except ValueError:
            temp_display = result.retained_temp_dir
        content.append("  - ", style="dim")
        content.append(f"temp clone: {temp_display}", style="dim")
        content.append("\n")

    status_labels = {
        "published": "published",
        "committed": "committed locally",
        "no_changes": "no changes",
        "unmanaged": "unmanaged",
        "invalid_metadata": "invalid metadata",
        "source_unreachable": "source unavailable",
        "source_path_missing": "source path missing",
        "missing_managed_files": "missing managed files",
        "publish_failed": "publish failed",
    }
    status_text = status_labels.get(result.status, result.status.replace("_", " "))
    if result.detail:
        status_text = f"{status_text}: {result.detail}"

    status_style: str | None = None
    if result.status in {"published", "committed", "no_changes"}:
        status_style = "green"
    elif result.status != "unmanaged":
        status_style = "yellow"

    content.append("  - ", style="dim")
    content.append("status: ", style="dim")
    if status_style:
        content.append(status_text, style=status_style)
    else:
        content.append(status_text)
    content.append("\n")

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

    if not argument:
        current = get_marketplace_url(settings)
        current_display = format_marketplace_display_url(current)
        content = Text()
        for index, url in enumerate(configured_urls, 1):
            display = format_marketplace_display_url(url)
            row = Text()
            row.append(f"[{index:2}] ", style="dim cyan")
            row.append(display, style="bright_blue bold")
            if display == current_display:
                row.append(" • ", style="dim")
                row.append("current", style="dim green")
            content.append_text(row)
            content.append("\n")

        content.append("\n")
        content.append_text(Text("Usage: /cards registry <number|url|path>", style="dim"))
        outcome.add_message(content, right_info="cards")
        return outcome

    argument_clean = argument.strip()
    if argument_clean.isdigit():
        index = int(argument_clean)
        if not configured_urls:
            outcome.add_message("No registries configured.", channel="warning")
            return outcome
        if not (1 <= index <= len(configured_urls)):
            outcome.add_message(
                f"Invalid registry number. Use 1-{len(configured_urls)}.",
                channel="warning",
            )
            return outcome
        selected_url = configured_urls[index - 1]
    else:
        selected_url = argument_clean

    try:
        marketplace = await card_service.scan_marketplace(selected_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    cards_settings = getattr(settings, "cards", None)
    if cards_settings is not None:
        cards_settings.marketplace_url = marketplace.source

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


async def handle_add_card_pack(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    marketplace_url = get_marketplace_url(ctx.resolve_settings())
    try:
        marketplace = await card_service.scan_marketplace(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace.packs:
        outcome.add_message("No card packs found in the marketplace.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = Text()
        append_heading(content, "Marketplace card packs:")
        content.append_text(_format_marketplace_packs(marketplace.packs))

        if not interactive:
            outcome.add_message(content, right_info="cards", agent_name=agent_name)
            outcome.add_message(
                "Install with `/cards add <number|name>`.",
                channel="info",
                right_info="cards",
                agent_name=agent_name,
            )
            return outcome

        await ctx.io.emit(CommandMessage(text=content, right_info="cards", agent_name=agent_name))
        selection = await ctx.io.prompt_selection(
            "Install card pack by number or name (empty to cancel): ",
            options=[entry.name for entry in marketplace.packs],
            allow_cancel=True,
        )
        if selection is None:
            return outcome

    try:
        install_result = await card_service.install_selected_pack(
            card_service.select_marketplace_pack(marketplace.packs, selection),
            environment_paths=env_paths,
            force=False,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to install card pack: {exc}", channel="error")
        return outcome

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

    selected = None
    selection = argument
    if not selection:
        if len(packs) == 1:
            selected = packs[0]
        else:
            content = _format_local_card_packs(environment_paths=env_paths, packs=packs)
            if not interactive:
                outcome.add_message(content, right_info="cards", agent_name=agent_name)
                outcome.add_message(
                    "Remove with `/cards remove <number|name>`.",
                    channel="info",
                    right_info="cards",
                    agent_name=agent_name,
                )
                return outcome

            await ctx.io.emit(CommandMessage(text=content, right_info="cards", agent_name=agent_name))
            selection = await ctx.io.prompt_selection(
                "Remove card pack by number or name (empty to cancel): ",
                options=[entry.name for entry in packs],
                allow_cancel=True,
            )
            if selection is None:
                return outcome

    try:
        if selected is None:
            assert selection is not None
            removal = card_service.remove_pack(
                environment_paths=env_paths,
                selector=selection,
            )
        else:
            removal = card_service.remove_pack(
                environment_paths=env_paths,
                selector=selected.name,
            )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to remove card pack: {exc}", channel="error")
        return outcome

    message = Text()
    message.append(f"Removed card pack: {removal.pack_name}", style="green")
    if removal.skipped_paths:
        message.append("\n")
        message.append(
            f"Skipped {len(removal.skipped_paths)} path(s) with shared ownership.",
            style="yellow",
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

    selected_name: str | None = None
    selection = argument
    if not selection:
        if len(packs) == 1:
            selected_name = packs[0].name
        else:
            content = _format_local_card_packs(environment_paths=env_paths, packs=packs)
            if not interactive:
                outcome.add_message(content, right_info="cards", agent_name=agent_name)
                outcome.add_message(
                    "Show with `/cards readme <number|name>`.",
                    channel="info",
                    right_info="cards",
                    agent_name=agent_name,
                )
                return outcome

            await ctx.io.emit(CommandMessage(text=content, right_info="cards", agent_name=agent_name))
            selection = await ctx.io.prompt_selection(
                "Show README for card pack by number or name (empty to cancel): ",
                options=[entry.name for entry in packs],
                allow_cancel=True,
            )
            if selection is None:
                return outcome

    if selected_name is None:
        assert selection is not None
        selected_name = selection

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
    selector, force, yes, parse_error = parse_update_argument(argument)
    if parse_error:
        outcome.add_message(parse_error, channel="error")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    updates = card_service.check_updates(environment_paths=env_paths)

    if selector is None:
        outcome.add_message(
            _format_update_results(updates, title="Card pack update check:"),
            right_info="cards",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Apply with `/cards update <number|name|all> [--force] [--yes]`.",
            channel="info",
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    try:
        plan = card_service.plan_updates(environment_paths=env_paths, selector=selector)
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome

    if len(plan.selected) > 1 and not yes:
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
        force=force,
    )
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
    selector, push, message, temp_dir, keep_temp, parse_error = _parse_publish_argument(argument)
    if parse_error:
        outcome.add_message(parse_error, channel="error")
        return outcome

    env_paths = resolve_environment_paths(ctx.resolve_settings())
    packs = card_service.list_installed_packs(environment_paths=env_paths)
    if not packs:
        outcome.add_message("No local card packs to publish.", channel="warning")
        return outcome

    if selector is None:
        outcome.add_message(
            _format_local_card_packs(environment_paths=env_paths, packs=packs),
            right_info="cards",
            agent_name=agent_name,
        )
        outcome.add_message(
            (
                "Publish with `/cards publish <number|name> [--no-push] [--message ...] "
                "[--temp-dir <path>] [--keep-temp]`."
            ),
            channel="info",
            right_info="cards",
            agent_name=agent_name,
        )
        return outcome

    try:
        result = await asyncio.to_thread(
            card_service.publish_pack,
            environment_paths=env_paths,
            selector=selector,
            push=push,
            commit_message=message,
            temp_dir=temp_dir,
            keep_temp=keep_temp,
        )
    except card_service.CardPackLookupError as exc:
        outcome.add_message(str(exc), channel="error")
        return outcome

    outcome.add_message(
        _format_publish_result(result, title="Card pack publish:"),
        right_info="cards",
        agent_name=agent_name,
    )

    if result.status == "publish_failed" and result.patch_path is not None:
        outcome.add_message(
            "Push was rejected. Share the generated patch with a maintainer or open a PR from your branch.",
            channel="warning",
            right_info="cards",
            agent_name=agent_name,
        )
    elif result.status == "publish_failed":
        outcome.add_message(
            "Publish failed after committing locally. Push manually or ask a maintainer with write access.",
            channel="warning",
            right_info="cards",
            agent_name=agent_name,
        )

    if result.retained_temp_dir is not None:
        outcome.add_message(
            f"Retained temporary clone at: {result.retained_temp_dir}",
            channel="info",
            right_info="cards",
            agent_name=agent_name,
        )

    return outcome


async def handle_cards_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = str(action or "list").lower()

    if _is_help_flag(action) or _is_help_flag(argument):
        outcome = CommandOutcome()
        outcome.add_message("\n".join(_cards_usage_lines()), channel="info", right_info="cards")
        return outcome

    if normalized in {"list", ""}:
        return await handle_list_cards(ctx, agent_name=agent_name)
    if normalized in {"add", "install"}:
        return await handle_add_card_pack(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"registry", "marketplace", "source"}:
        return await handle_set_cards_registry(ctx, argument=argument)
    if normalized in {"remove", "rm", "delete", "uninstall"}:
        return await handle_remove_card_pack(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"readme", "show", "cat"}:
        return await handle_card_pack_readme(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"update", "refresh", "upgrade"}:
        return await handle_update_card_pack(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"publish"}:
        return await handle_publish_card_pack(ctx, agent_name=agent_name, argument=argument)

    outcome = CommandOutcome()
    suggestions = suggest_command_action("cards", normalized)
    suggestion_text = ""
    if suggestions:
        suggestion_text = " Did you mean: " + ", ".join(f"`{name}`" for name in suggestions)
    outcome.add_message(
        (
            f"Unknown /cards action: {normalized}. "
            f"Use list/add/remove/update/publish/registry/help.{suggestion_text}"
        ),
        channel="warning",
        right_info="cards",
    )
    outcome.add_message(
        "\n".join(_cards_usage_lines()),
        channel="info",
        right_info="cards",
    )
    return outcome
