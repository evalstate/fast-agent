"""Shared skills command handlers."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.skills.manager import (
    DEFAULT_SKILL_REGISTRIES,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    format_marketplace_display_url,
    get_manager_directory,
    get_marketplace_url,
    install_marketplace_skill,
    list_local_skills,
    reload_skill_manifests,
    remove_local_skill,
    resolve_skill_directories,
    select_manifest_by_name_or_index,
    select_skill_by_name_or_index,
)
from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext


def _append_heading(content: Text, heading: str) -> None:
    if content.plain:
        content.append("\n")
    content.append_text(Text.from_markup(f"[bold]{heading}[/bold]\n\n"))


def _append_wrapped_text(content: Text, value: str, *, indent: str = "") -> None:
    wrapped_lines = textwrap.wrap(value.strip(), width=72)
    for line in wrapped_lines:
        content.append(indent)
        content.append_text(Text(line))
        content.append("\n")


def _append_manifest_entry(content: Text, manifest: SkillManifest, index: int) -> None:
    entry = Text()
    entry.append(f"[{index:2}] ", style="dim cyan")
    entry.append(manifest.name, style="bright_blue bold")
    content.append_text(entry)
    content.append("\n")

    if manifest.description:
        _append_wrapped_text(content, manifest.description, indent="     ")

    source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
    try:
        source_display = source_path.relative_to(Path.cwd())
    except ValueError:
        source_display = source_path
    content.append("     ", style="dim")
    content.append(f"source: {source_display}", style="dim green")
    content.append("\n\n")


def _format_local_skills_by_directory(manifests_by_dir: dict[Path, list[SkillManifest]]) -> Text:
    content = Text()
    skill_index = 0
    total_skills = sum(len(manifests) for manifests in manifests_by_dir.values())

    for directory, manifests in manifests_by_dir.items():
        try:
            display_dir = directory.relative_to(Path.cwd())
        except ValueError:
            display_dir = directory

        _append_heading(content, f"Skills in {display_dir}:")

        if not manifests:
            content.append_text(Text("No skills in this directory", style="yellow"))
            content.append("\n")
            continue

        for manifest in manifests:
            skill_index += 1
            _append_manifest_entry(content, manifest, skill_index)

    if total_skills == 0:
        content.append_text(Text("Use /skills add to install a skill", style="dim"))
    else:
        content.append_text(Text("Use /skills add to install a skill", style="dim"))
        content.append("\n")
        content.append_text(Text("Remove a skill with /skills remove <number|name>", style="dim"))

    return content


def _format_agent_skills_override(
    manifests: list[SkillManifest],
    *,
    source_paths: list[str],
) -> Text:
    content = Text()
    _append_heading(content, "Active agent skills (override):")
    content.append_text(
        Text(
            "Note: this agent has an explicit skills configuration. /skills lists global skills directories from settings, not per-agent overrides. Update settings.skills.directories or the --skills flag to change this list.",
            style="dim",
        )
    )
    content.append("\n")
    if source_paths:
        sources_display = ", ".join(source_paths)
        content.append_text(Text(f"Sources: {sources_display}", style="dim"))
        content.append("\n")
    if not manifests:
        content.append_text(Text("No skills configured for this agent.", style="yellow"))
        return content

    for index, manifest in enumerate(manifests, 1):
        _append_manifest_entry(content, manifest, index)

    return content


def _format_marketplace_skills(marketplace: Sequence[object]) -> Text:
    content = Text()
    current_bundle = None

    for index, entry in enumerate(marketplace, 1):
        bundle_name = getattr(entry, "bundle_name", None)
        bundle_description = getattr(entry, "bundle_description", None)
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            _append_heading(content, bundle_name)
            if bundle_description:
                _append_wrapped_text(content, bundle_description)
            content.append("\n")

        name = getattr(entry, "name", "")
        description = getattr(entry, "description", "")
        source_url = getattr(entry, "source_url", None)

        entry_line = Text()
        entry_line.append(f"[{index:2}] ", style="dim cyan")
        entry_line.append(str(name), style="bright_blue bold")
        content.append_text(entry_line)
        content.append("\n")

        if description:
            _append_wrapped_text(content, str(description), indent="     ")
        if source_url:
            content.append("     ", style="dim")
            content.append(f"source: {source_url}", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _format_install_result(skill_name: str, install_path: Path) -> Text:
    try:
        display_path = install_path.relative_to(Path.cwd())
    except ValueError:
        display_path = install_path
    content = Text()
    content.append(f"Installed skill: {skill_name}", style="green")
    content.append("\n")
    content.append(f"location: {display_path}", style="dim green")
    return content


def _get_agent_skill_override_sources(manifests: list[SkillManifest]) -> list[str]:
    sources: list[str] = []
    for manifest in manifests:
        path = Path(getattr(manifest, "path", Path(".")))
        source_path = path.parent if path.is_file() else path
        try:
            display_path = source_path.relative_to(Path.cwd())
        except ValueError:
            display_path = source_path
        sources.append(str(display_path))
    return sorted(set(sources))


async def _refresh_agent_skills(ctx: CommandContext, agent_name: str) -> None:
    agent = ctx.agent_provider._agent(agent_name)
    override_dirs = resolve_skill_directories(ctx.resolve_settings())
    registry, manifests = reload_skill_manifests(
        base_dir=Path.cwd(), override_directories=override_dirs
    )
    instruction_context = None
    try:
        skills_text = format_skills_for_prompt(manifests, read_tool_name="read_skill")
        instruction_context = {"agentSkills": skills_text}
    except Exception:
        instruction_context = None

    await rebuild_agent_instruction(
        agent,
        skill_manifests=manifests,
        context=instruction_context,
        skill_registry=registry,
    )


async def handle_list_skills(ctx: CommandContext, *, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()

    directories = resolve_skill_directories(ctx.resolve_settings())
    manifests_by_dir: dict[Path, list[SkillManifest]] = {}
    for directory in directories:
        manifests_by_dir[directory] = list_local_skills(directory) if directory.exists() else []

    outcome.add_message(
        _format_local_skills_by_directory(manifests_by_dir),
        right_info="skills",
        agent_name=agent_name,
    )

    agent_obj = ctx.agent_provider._agent(agent_name)
    config = getattr(agent_obj, "config", None)
    if not config or getattr(config, "skills", SKILLS_DEFAULT) is SKILLS_DEFAULT:
        return outcome

    manifests = list(getattr(config, "skill_manifests", []) or [])
    sources = _get_agent_skill_override_sources(manifests)
    outcome.add_message(
        _format_agent_skills_override(manifests, source_paths=sources),
        right_info="skills",
        agent_name=agent_name,
    )

    return outcome


async def handle_set_skills_registry(
    ctx: CommandContext, *, argument: str | None
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    configured_urls = (settings.skills.marketplace_urls if settings.skills else None) or list(
        DEFAULT_SKILL_REGISTRIES
    )

    if not argument:
        current = get_marketplace_url(settings)
        content = Text()
        content.append_text(
            Text(f"Current registry: {format_marketplace_display_url(current)}", style="dim")
        )
        if configured_urls:
            content.append("\n\n")
            content.append_text(Text("Configured registries:", style="dim"))
            content.append("\n")
            for index, reg_url in enumerate(configured_urls, 1):
                display = format_marketplace_display_url(reg_url)
                content.append_text(Text(f"  [{index}] {display}", style="dim"))
                content.append("\n")
        content.append("\n")
        content.append_text(Text("Usage: /skills registry <number|url|path>", style="dim"))
        outcome.add_message(content, right_info="skills")
        return outcome

    arg = str(argument).strip()
    if arg.isdigit():
        index = int(arg)
        if not configured_urls:
            outcome.add_message("No registries configured.", channel="warning")
            return outcome
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
        marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    skills_settings = getattr(settings, "skills", None)
    if skills_settings is not None:
        skills_settings.marketplace_url = resolved_url

    content = Text()
    if resolved_url != url:
        content.append_text(Text(f"Resolved from: {url}", style="dim"))
        content.append("\n")
    content.append_text(
        Text(
            f"Registry set to: {format_marketplace_display_url(resolved_url)}",
            style="green",
        )
    )
    content.append("\n")
    content.append_text(Text(f"Skills discovered: {len(marketplace)}", style="dim"))
    outcome.add_message(content, right_info="skills")
    return outcome


async def handle_add_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()

    manager_dir = get_manager_directory()
    marketplace_url = get_marketplace_url(ctx.resolve_settings())
    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace:
        outcome.add_message("No skills found in the marketplace.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = Text()
        _append_heading(content, "Marketplace skills:")
        repo_hint = None
        if marketplace:
            repo_url = getattr(marketplace[0], "repo_url", None)
            if repo_url:
                repo_ref = getattr(marketplace[0], "repo_ref", None)
                repo_hint = f"{repo_url}@{repo_ref}" if repo_ref else repo_url
        if repo_hint:
            content.append_text(
                Text(
                    f"Repository: {format_marketplace_display_url(repo_hint)}",
                    style="dim",
                )
            )
            content.append("\n\n")
        content.append_text(_format_marketplace_skills(marketplace))

        if not interactive:
            outcome.add_message(content, right_info="skills", agent_name=agent_name)
            outcome.add_message(
                "Install with `/skills add <number|name>`.",
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            outcome.add_message(
                "Change registry with `/skills registry`.",
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome

        await ctx.io.emit(CommandMessage(text=content, right_info="skills", agent_name=agent_name))

        selection = await ctx.io.prompt_selection(
            "Install skill by number or name (empty to cancel): ",
            options=[getattr(entry, "name", "") for entry in marketplace],
            allow_cancel=True,
        )
        if selection is None:
            return outcome

    skill = select_skill_by_name_or_index(marketplace, selection)
    if not skill:
        outcome.add_message(f"Skill not found: {selection}", channel="error")
        return outcome

    try:
        install_path = await install_marketplace_skill(skill, destination_root=manager_dir)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to install skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(getattr(skill, "name", ""), install_path),
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(ctx, agent_name)
    return outcome


async def handle_remove_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    outcome = CommandOutcome()

    manager_dir = get_manager_directory()
    manifests = list_local_skills(manager_dir)
    if not manifests:
        outcome.add_message("No local skills to remove.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = Text()
        _append_heading(content, f"Skills in {manager_dir}:")
        for index, manifest in enumerate(manifests, 1):
            _append_manifest_entry(content, manifest, index)

        if not interactive:
            outcome.add_message(content, right_info="skills", agent_name=agent_name)
            outcome.add_message(
                "Remove with `/skills remove <number|name>`.",
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome

        await ctx.io.emit(CommandMessage(text=content, right_info="skills", agent_name=agent_name))

        selection = await ctx.io.prompt_selection(
            "Remove skill by number or name (empty to cancel): ",
            options=[manifest.name for manifest in manifests],
            allow_cancel=True,
        )
        if selection is None:
            return outcome

    manifest = select_manifest_by_name_or_index(manifests, selection)
    if not manifest:
        outcome.add_message(f"Skill not found: {selection}", channel="error")
        return outcome

    try:
        skill_dir = Path(manifest.path).parent
        remove_local_skill(skill_dir, destination_root=manager_dir)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to remove skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        f"Removed skill: {manifest.name}",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(ctx, agent_name)
    return outcome


async def handle_skills_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = str(action or "list").lower()

    if normalized in {"list", ""}:
        return await handle_list_skills(ctx, agent_name=agent_name)
    if normalized in {"add", "install"}:
        return await handle_add_skill(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"registry", "marketplace", "source"}:
        return await handle_set_skills_registry(ctx, argument=argument)
    if normalized in {"remove", "rm", "delete", "uninstall"}:
        return await handle_remove_skill(ctx, agent_name=agent_name, argument=argument)

    outcome = CommandOutcome()
    outcome.add_message(f"Unknown /skills action: {normalized}", channel="warning")
    return outcome
