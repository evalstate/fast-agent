"""Shared skills command handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from rich.text import Text

from fast_agent.commands.command_catalog import (
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
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.marketplace.formatting import (
    format_installed_revision_display,
    format_source_provenance,
)
from fast_agent.marketplace.update_status import is_update_applied
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.skills.command_support import (
    SKILLS_ADD_HINT_SLASH,
    filter_marketplace_skills,
    marketplace_repository_hint,
    skills_usage_lines,
)
from fast_agent.skills.configuration import (
    format_marketplace_display_url,
    get_marketplace_url,
    resolve_skill_registries,
)
from fast_agent.skills.mcp_registry import (
    McpRegistrySkill,
    McpSkillRegistry,
    install_mcp_registry_skill,
    select_mcp_registry_skill,
    update_mcp_registry_skill,
)
from fast_agent.skills.models import SkillUpdateInfo
from fast_agent.skills.operations import (
    apply_skill_updates,
    check_skill_updates,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    install_marketplace_skill,
    reload_skill_manifests,
    remove_local_skill,
    select_manifest_by_name_or_index,
    select_skill_by_name_or_index,
    select_skill_updates,
)
from fast_agent.skills.provenance import (
    compute_skill_content_fingerprint,
    format_revision_short,
    format_skill_provenance_details,
    read_installed_skill_source,
)
from fast_agent.skills.registry import SkillManifest, SkillRegistry
from fast_agent.skills.scope import (
    order_skill_directories_for_display,
    resolve_skill_directories,
    resolve_skills_management_scope,
)
from fast_agent.skills.service import install_skill_from_selector
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.skills.models import MarketplaceSkill


MCP_REGISTRY_PREFIX = "mcp://"


@runtime_checkable
class _McpSkillRegistryAggregator(Protocol):
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]: ...


@runtime_checkable
class _McpSkillRegistryAgent(Protocol):
    @property
    def aggregator(self) -> _McpSkillRegistryAggregator: ...


def _mcp_registry_source(server_name: str) -> str:
    return f"{MCP_REGISTRY_PREFIX}{server_name}"


def _mcp_registry_server_name(source: str) -> str | None:
    if not source.startswith(MCP_REGISTRY_PREFIX):
        return None
    server_name = source[len(MCP_REGISTRY_PREFIX) :].strip()
    return server_name or None


async def _list_mcp_skill_registries(
    ctx: "CommandContext", *, agent_name: str
) -> list[McpSkillRegistry]:
    try:
        agent = ctx.agent_provider._agent(agent_name)
    except KeyError:
        return []
    if not isinstance(agent, _McpSkillRegistryAgent):
        return []
    aggregator = agent.aggregator
    if not isinstance(aggregator, _McpSkillRegistryAggregator):
        return []
    return await aggregator.list_mcp_skill_registries()


def _find_mcp_registry(
    registries: Sequence[McpSkillRegistry], server_name: str
) -> McpSkillRegistry | None:
    for registry in registries:
        if registry.server_name == server_name:
            return registry
    return None


def _append_manifest_entry(content: Text, manifest: SkillManifest, index: int) -> None:
    append_indexed_name_line(content, index, manifest.name)

    if manifest.description:
        append_wrapped_text(content, manifest.description, indent="     ")

    source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
    content.append("     ", style="dim")
    content.append(f"source: {format_display_path(source_path)}", style="dim green")
    content.append("\n")
    provenance_text, installed_text = format_skill_provenance_details(source_path)
    content.append("     ", style="dim")
    content.append(f"provenance: {provenance_text}", style="dim")
    content.append("\n")
    if installed_text:
        content.append("     ", style="dim")
        content.append(f"installed: {installed_text}", style="dim")
        content.append("\n")
    content.append("\n")


def _append_registry_entry(
    content: Text,
    *,
    display_url: str,
    index: int,
    is_current: bool,
) -> None:
    append_indexed_current_line(content, index, display_url, is_current=is_current)


def _format_local_skills_by_directory(manifests_by_dir: dict[Path, list[SkillManifest]]) -> Text:
    content = Text()
    skill_index = 0
    total_skills = sum(len(manifests) for manifests in manifests_by_dir.values())

    for directory, manifests in manifests_by_dir.items():
        append_heading(content, f"Skills in {format_display_path(directory)}:")

        if not manifests:
            content.append_text(Text("No skills in this directory", style="yellow"))
            content.append("\n")
            continue

        for manifest in manifests:
            skill_index += 1
            _append_manifest_entry(content, manifest, skill_index)

    content.append_text(Text("Browse marketplace skills with /skills available", style="dim"))
    if total_skills > 0:
        content.append("\n")
        content.append_text(Text("Search marketplace skills with /skills search <query>", style="dim"))
        content.append("\n")
        content.append_text(Text("Remove a skill with /skills remove <number|name>", style="dim"))

    return content


def _format_agent_skills_override(
    manifests: list[SkillManifest],
    *,
    source_paths: list[str],
) -> Text:
    content = Text()
    append_heading(content, "Active agent skills (override):")
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


def _format_marketplace_skills(
    marketplace: Sequence[MarketplaceSkill] | Sequence[McpRegistrySkill],
) -> Text:
    content = Text()
    current_bundle = None

    for index, entry in enumerate(marketplace, 1):
        bundle_name = getattr(entry, "bundle_name", None)
        bundle_description = getattr(entry, "bundle_description", None)
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            append_heading(content, bundle_name)
            if bundle_description:
                append_wrapped_text(content, bundle_description)
            content.append("\n")

        append_indexed_name_line(content, index, entry.name)

        if entry.description:
            append_wrapped_text(content, entry.description, indent="     ")
        if entry.source_url:
            content.append("     ", style="dim")
            content.append(f"source: {entry.source_url}", style="dim green")
            content.append("\n")
        if getattr(entry, "digest", None):
            content.append("     ", style="dim")
            content.append("integrity: SHA256 checked", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _format_install_result(skill_name: str, install_path: Path) -> Text:
    content = Text()
    content.append(f"Installed skill: {skill_name}", style="green")
    content.append("\n")
    content.append(f"location: {format_display_path(install_path)}", style="dim green")
    return content


def _format_marketplace_skill_list(marketplace: Sequence[MarketplaceSkill]) -> Text:
    content = Text()
    append_heading(content, "Marketplace skills:")
    repo_hint = marketplace_repository_hint(marketplace)
    if repo_hint:
        content.append_text(
            Text(
                f"Repository: {format_marketplace_display_url(repo_hint)}",
                style="dim",
            )
        )
        content.append("\n\n")
    content.append_text(_format_marketplace_skills(marketplace))
    return content


def _marketplace_skill_selection_options(marketplace: Sequence[MarketplaceSkill]) -> list[str]:
    return unique_selection_options(
        option
        for entry in marketplace
        for option in (entry.name, entry.install_dir_name)
    )


def _mcp_skill_selection_options(skills: Sequence[McpRegistrySkill]) -> list[str]:
    return unique_selection_options(skill.name for skill in skills)


def _format_skill_selection_list(
    manifests: Sequence[SkillManifest],
    *,
    skills_dir: Path,
) -> Text:
    content = Text()
    append_heading(content, f"Skills in {skills_dir}:")
    for index, manifest in enumerate(manifests, 1):
        _append_manifest_entry(content, manifest, index)
    return content


def _add_noninteractive_add_skill_hints(
    outcome: CommandOutcome,
    *,
    agent_name: str,
) -> None:
    add_info_messages(
        outcome,
        (
            SKILLS_ADD_HINT_SLASH,
            "Browse marketplace with `/skills available`.",
            "Search marketplace with `/skills search <query>`.",
            "Change registry with `/skills registry`.",
        ),
        right_info="skills",
        agent_name=agent_name,
    )


async def _install_skill_from_add_selector(
    ctx: CommandContext,
    *,
    agent_name: str,
    marketplace_url: str,
    managed_skills_dir: Path,
    selection: str,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    try:
        installed = await install_skill_from_selector(
            marketplace_url,
            selection,
            destination_root=managed_skills_dir,
        )
    except Exception as exc:
        outcome.add_message(f"Failed to install skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(installed.name, installed.skill_dir),
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(
        ctx,
        agent_name,
        managed_directory_override=managed_directory_override,
    )
    return outcome


async def _install_mcp_skill_from_add_selector(
    ctx: CommandContext,
    *,
    agent_name: str,
    registry: McpSkillRegistry,
    managed_skills_dir: Path,
    selection: str,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    skill = select_mcp_registry_skill(registry.skills, selection)
    if skill is None:
        _add_skill_not_found_message(outcome, selection=selection, agent_name=agent_name)
        return outcome

    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, _McpSkillRegistryAgent):
        outcome.add_message("This agent does not expose MCP skill registries.", channel="error")
        return outcome

    try:
        install_path = await install_mcp_registry_skill(
            agent.aggregator,
            skill,
            destination_root=managed_skills_dir,
        )
    except Exception as exc:
        outcome.add_message(f"Failed to install skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(skill.name, install_path),
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(
        ctx,
        agent_name,
        managed_directory_override=managed_directory_override,
    )
    return outcome


async def _select_skill_for_add(
    ctx: CommandContext,
    *,
    outcome: CommandOutcome,
    marketplace: Sequence[MarketplaceSkill],
    agent_name: str,
    interactive: bool,
) -> str | None:
    content = _format_marketplace_skill_list(marketplace)
    if not interactive:
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        _add_noninteractive_add_skill_hints(outcome, agent_name=agent_name)
        return None

    return await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="skills",
        agent_name=agent_name,
        prompt="Install skill by number or name (empty to cancel): ",
        options=_marketplace_skill_selection_options(marketplace),
        allow_cancel=True,
    )


async def _select_mcp_skill_for_add(
    ctx: CommandContext,
    *,
    outcome: CommandOutcome,
    registry: McpSkillRegistry,
    agent_name: str,
    interactive: bool,
) -> str | None:
    content = Text()
    append_heading(content, f"MCP skills from {registry.display_name}:")
    content.append_text(_format_marketplace_skills(registry.skills))

    if not interactive:
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        _add_noninteractive_add_skill_hints(outcome, agent_name=agent_name)
        return None

    return await prompt_selection_after_message(
        ctx,
        content=content,
        right_info="skills",
        agent_name=agent_name,
        prompt="Install skill by number or name (empty to cancel): ",
        options=_mcp_skill_selection_options(registry.skills),
        allow_cancel=True,
    )


def _add_skill_not_found_message(
    outcome: CommandOutcome,
    *,
    selection: str,
    agent_name: str,
) -> None:
    outcome.add_message(f"Skill not found: {selection}", channel="error")
    outcome.add_message(
        "Run `/skills available` to browse skills or `/skills search <query>` to filter.",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )


async def _install_selected_marketplace_skill(
    ctx: CommandContext,
    *,
    outcome: CommandOutcome,
    skill: MarketplaceSkill,
    managed_skills_dir: Path,
    agent_name: str,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    try:
        install_path = await install_marketplace_skill(
            skill,
            destination_root=managed_skills_dir,
        )
    except Exception as exc:
        outcome.add_message(f"Failed to install skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(skill.name, install_path),
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(
        ctx,
        agent_name,
        managed_directory_override=managed_directory_override,
    )
    return outcome


def _format_update_results(updates: Sequence[SkillUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        append_warning_line(content, "No managed skills found.")
        return content

    for update in updates:
        append_indexed_name_line(content, update.index, update.name)

        append_detail_line(
            content,
            "source",
            format_display_path(update.skill_dir),
            value_style="dim green",
        )

        if update.managed_source is not None:
            source = update.managed_source
            provenance_text = format_source_provenance(
                source.repo_url,
                source.repo_ref,
                source.repo_path,
            )
            installed_text = format_installed_revision_display(
                source.installed_at,
                source.installed_revision,
            )
        else:
            provenance_text, installed_text = format_skill_provenance_details(update.skill_dir)

        append_detail_line(content, "provenance", provenance_text, value_style="dim")
        if installed_text:
            append_detail_line(content, "installed", installed_text, value_style="dim")

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


def _get_agent_skill_override_sources(manifests: list[SkillManifest]) -> list[str]:
    sources: list[str] = []
    for manifest in manifests:
        path = Path(manifest.path)
        source_path = path.parent if path.is_file() else path
        sources.append(format_display_path(source_path))
    return unique_preserve_order(sources)


def _format_skills_registry_overview(
    *,
    current_url: str,
    configured_urls: list[str],
    mcp_registries: Sequence[McpSkillRegistry],
) -> Text:
    current_display = format_marketplace_display_url(current_url)
    configured_displays = [
        format_marketplace_display_url(reg_url) for reg_url in configured_urls
    ]
    current_mcp_server = _mcp_registry_server_name(current_url)
    current_in_configured = current_display in configured_displays or (
        current_mcp_server is not None
        and _find_mcp_registry(mcp_registries, current_mcp_server) is not None
    )
    content = Text()
    if not current_in_configured:
        current_line = Text()
        current_line.append("current", style="dim green")
        current_line.append(" • ", style="dim")
        current_line.append(current_display, style="bright_blue bold")
        content.append_text(current_line)
        content.append("\n\n")
        content.append_text(Text("Configured registries:", style="dim"))
        content.append("\n")

    for index, display in enumerate(configured_displays, 1):
        _append_registry_entry(
            content,
            display_url=display,
            index=index,
            is_current=display == current_display,
        )

    if mcp_registries:
        if configured_displays:
            content.append("\n")
        content.append_text(Text("MCP registries:", style="dim"))
        content.append("\n")
        offset = len(configured_displays)
        for index, registry in enumerate(mcp_registries, offset + 1):
            _append_registry_entry(
                content,
                display_url=registry.display_name,
                index=index,
                is_current=current_mcp_server == registry.server_name,
            )

    content.append("\n")
    content.append_text(
        Text("Usage: /skills registry <number|url|path|mcp-server>", style="dim")
    )
    return content


def _resolve_skills_registry_argument(
    registry_arg: str,
    configured_urls: list[str],
    outcome: CommandOutcome,
) -> str | None:
    resolved = resolve_registry_argument(registry_arg, configured_urls)
    if resolved.warning is not None:
        outcome.add_message(resolved.warning, channel="warning")
    return resolved.url


def _add_empty_skills_registry_warning(outcome: CommandOutcome, url: str) -> None:
    content = Text()
    content.append_text(
        Text("No skills found in the registry; registry unchanged.", style="yellow")
    )
    content.append("\n")
    content.append_text(Text(f"Registry: {format_marketplace_display_url(url)}", style="dim"))
    outcome.add_message(content, channel="warning", right_info="skills")


def _format_skills_registry_success(
    *,
    url: str,
    resolved_url: str,
    skill_count: int,
) -> Text:
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
    content.append_text(Text(f"Skills discovered: {skill_count}", style="dim"))
    return content


async def _refresh_agent_skills(
    ctx: CommandContext,
    agent_name: str,
    *,
    managed_directory_override: str | Path | None = None,
) -> None:
    agent = ctx.agent_provider._agent(agent_name)
    override_dirs = resolve_skill_directories(
        ctx.resolve_settings(),
        managed_directory_override=managed_directory_override,
    )
    registry, manifests = reload_skill_manifests(
        base_dir=Path.cwd(), override_directories=override_dirs
    )

    await rebuild_agent_instruction(
        agent,
        skill_manifests=manifests,
        skill_registry=registry,
    )


async def handle_list_skills(
    ctx: CommandContext,
    *,
    agent_name: str,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    settings = ctx.resolve_settings()
    management_scope = resolve_skills_management_scope(
        settings,
        managed_directory_override=managed_directory_override,
    )
    discovered_directories = order_skill_directories_for_display(
        management_scope.discovered_directories,
        settings=settings,
        managed_directory_override=managed_directory_override,
    )
    manifests_by_dir: dict[Path, list[SkillManifest]] = {}
    for directory in discovered_directories:
        manifests_by_dir[directory] = (
            SkillRegistry.load_directory(directory) if directory.exists() else []
        )

    outcome.add_message(
        _format_local_skills_by_directory(manifests_by_dir),
        right_info="skills",
        agent_name=agent_name,
    )

    agent_obj = ctx.agent_provider._agent(agent_name)
    agent_obj = cast("AgentProtocol", agent_obj)
    config = agent_obj.config
    if config.skills is SKILLS_DEFAULT:
        return outcome

    manifests = list(config.skill_manifests or [])
    sources = _get_agent_skill_override_sources(manifests)
    outcome.add_message(
        _format_agent_skills_override(manifests, source_paths=sources),
        right_info="skills",
        agent_name=agent_name,
    )

    return outcome


async def handle_set_skills_registry(
    ctx: CommandContext, *, argument: str | None, agent_name: str | None = None
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    configured_urls = [
        url for url in resolve_skill_registries(settings) if _mcp_registry_server_name(url) is None
    ]
    mcp_registries = (
        await _list_mcp_skill_registries(ctx, agent_name=agent_name)
        if agent_name is not None
        else []
    )

    registry_arg = optional_selector(argument)
    if registry_arg is None:
        outcome.add_message(
            _format_skills_registry_overview(
                current_url=get_marketplace_url(settings),
                configured_urls=configured_urls,
                mcp_registries=mcp_registries,
            ),
            right_info="skills",
        )
        return outcome

    selected_mcp: McpSkillRegistry | None = None
    if registry_arg.isdigit():
        index = int(registry_arg)
        if len(configured_urls) < index <= len(configured_urls) + len(mcp_registries):
            selected_mcp = mcp_registries[index - len(configured_urls) - 1]
            url = _mcp_registry_source(selected_mcp.server_name)
        else:
            url = _resolve_skills_registry_argument(registry_arg, configured_urls, outcome)
    else:
        explicit_mcp_server = _mcp_registry_server_name(registry_arg) or registry_arg
        selected_mcp = _find_mcp_registry(mcp_registries, explicit_mcp_server)
        url = _mcp_registry_source(selected_mcp.server_name) if selected_mcp else None
        if url is None:
            url = _resolve_skills_registry_argument(registry_arg, configured_urls, outcome)

    if url is None:
        return outcome

    if selected_mcp is not None:
        settings.skills.marketplace_url = url
        content = Text()
        content.append_text(
            Text(f"Registry set to: {selected_mcp.display_name}", style="green")
        )
        content.append("\n")
        content.append_text(Text(f"Skills discovered: {len(selected_mcp.skills)}", style="dim"))
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        return outcome

    try:
        marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
    except Exception as exc:
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    if not marketplace:
        _add_empty_skills_registry_warning(outcome, url)
        return outcome

    settings.skills.marketplace_url = resolved_url

    outcome.add_message(
        _format_skills_registry_success(
            url=url,
            resolved_url=resolved_url,
            skill_count=len(marketplace),
        ),
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


def handle_skills_help(*, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    outcome.add_message(
        "\n".join(skills_usage_lines()),
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


async def handle_list_marketplace_skills(
    ctx: CommandContext,
    *,
    agent_name: str,
    query: str | None = None,
    marketplace_url_override: str | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    marketplace_url = marketplace_url_override or get_marketplace_url(ctx.resolve_settings())
    mcp_server_name = _mcp_registry_server_name(marketplace_url)
    if mcp_server_name is not None:
        mcp_registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)
        mcp_registry = _find_mcp_registry(mcp_registries, mcp_server_name)
        if mcp_registry is None:
            outcome.add_message(
                f"MCP skill registry is not available: {mcp_server_name}",
                channel="error",
            )
            return outcome

        normalized_query = strip_to_none(query)
        selected_marketplace: Sequence[McpRegistrySkill] = mcp_registry.skills
        if normalized_query is not None:
            query_lower = normalized_query.lower()
            selected_marketplace = [
                skill
                for skill in selected_marketplace
                if query_lower in skill.name.lower()
                or query_lower in (skill.description or "").lower()
            ]

        if not selected_marketplace:
            outcome.add_message("No skills found in the MCP registry.", channel="warning")
            return outcome

        content = Text()
        heading = f"MCP skills from {mcp_registry.display_name}:"
        if normalized_query is not None:
            heading = f"MCP skills from {mcp_registry.display_name} (search: {normalized_query}):"
        append_heading(content, heading)
        content.append_text(_format_marketplace_skills(selected_marketplace))
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        add_info_messages(
            outcome,
            (
                SKILLS_ADD_HINT_SLASH,
                "Search with `/skills search <query>`.",
            ),
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace:
        outcome.add_message("No skills found in the marketplace.", channel="warning")
        return outcome

    normalized_query = strip_to_none(query)
    selected_marketplace: Sequence[MarketplaceSkill] = marketplace
    if normalized_query is not None:
        selected_marketplace = filter_marketplace_skills(marketplace, normalized_query)

    content = Text()
    heading = "Marketplace skills:"
    if normalized_query is not None:
        heading = f"Marketplace skills (search: {normalized_query}):"
    append_heading(content, heading)

    repo_hint = marketplace_repository_hint(marketplace)
    if repo_hint:
        content.append_text(
            Text(
                f"Repository: {format_marketplace_display_url(repo_hint)}",
                style="dim",
            )
        )
        content.append("\n\n")

    if not selected_marketplace:
        content.append_text(Text("No matching skills found.", style="yellow"))
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        add_info_messages(
            outcome,
            ("Try `/skills available` to browse all skills.",),
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    content.append_text(_format_marketplace_skills(selected_marketplace))
    outcome.add_message(content, right_info="skills", agent_name=agent_name)
    add_info_messages(
        outcome,
        (
            SKILLS_ADD_HINT_SLASH,
            "Search with `/skills search <query>`.",
        ),
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


async def handle_add_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
    marketplace_url_override: str | None = None,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = parse_add_argument(argument, allow_force=False)
    if parsed.error is not None:
        outcome.add_message(parsed.error, channel="warning")
        return outcome

    management_scope = resolve_skills_management_scope(
        ctx.resolve_settings(),
        managed_directory_override=parsed.skills_dir or managed_directory_override,
    )
    managed_skills_dir = management_scope.managed_directory
    selection = parsed.selector
    marketplace_url = parsed.registry or marketplace_url_override or get_marketplace_url(
        ctx.resolve_settings()
    )
    mcp_server_name = _mcp_registry_server_name(marketplace_url)

    if mcp_server_name is not None:
        mcp_registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)
        mcp_registry = _find_mcp_registry(mcp_registries, mcp_server_name)
        if mcp_registry is None:
            outcome.add_message(
                f"MCP skill registry is not available: {mcp_server_name}",
                channel="error",
            )
            return outcome

        if selection:
            return await _install_mcp_skill_from_add_selector(
                ctx,
                agent_name=agent_name,
                registry=mcp_registry,
                managed_skills_dir=managed_skills_dir,
                selection=selection,
                managed_directory_override=parsed.skills_dir or managed_directory_override,
            )

        selection = await _select_mcp_skill_for_add(
            ctx,
            outcome=outcome,
            registry=mcp_registry,
            agent_name=agent_name,
            interactive=interactive,
        )
        if selection is None:
            return outcome

        return await _install_mcp_skill_from_add_selector(
            ctx,
            agent_name=agent_name,
            registry=mcp_registry,
            managed_skills_dir=managed_skills_dir,
            selection=selection,
            managed_directory_override=parsed.skills_dir or managed_directory_override,
        )

    if selection:
        return await _install_skill_from_add_selector(
            ctx,
            agent_name=agent_name,
            marketplace_url=marketplace_url,
            managed_skills_dir=managed_skills_dir,
            selection=selection,
            managed_directory_override=parsed.skills_dir or managed_directory_override,
        )

    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace:
        outcome.add_message("No skills found in the marketplace.", channel="warning")
        return outcome

    selection = await _select_skill_for_add(
        ctx,
        outcome=outcome,
        marketplace=marketplace,
        agent_name=agent_name,
        interactive=interactive,
    )
    if selection is None:
        return outcome

    skill = select_skill_by_name_or_index(marketplace, selection)
    if not skill:
        _add_skill_not_found_message(outcome, selection=selection, agent_name=agent_name)
        return outcome

    return await _install_selected_marketplace_skill(
        ctx,
        outcome=outcome,
        skill=skill,
        managed_skills_dir=managed_skills_dir,
        agent_name=agent_name,
        managed_directory_override=managed_directory_override,
    )


async def handle_remove_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    parsed = parse_add_argument(argument, allow_registry=False, allow_force=False)
    if parsed.error is not None:
        outcome.add_message(parsed.error, channel="warning")
        return outcome
    directory_override = parsed.skills_dir or managed_directory_override

    management_scope = resolve_skills_management_scope(
        ctx.resolve_settings(),
        managed_directory_override=directory_override,
    )
    managed_skills_dir = management_scope.managed_directory
    manifests = SkillRegistry.load_directory(managed_skills_dir)
    if not manifests:
        outcome.add_message("No local skills to remove.", channel="warning")
        return outcome

    selection = parsed.selector
    if not selection:
        content = _format_skill_selection_list(manifests, skills_dir=managed_skills_dir)

        if not interactive:
            outcome.add_message(content, right_info="skills", agent_name=agent_name)
            outcome.add_message(
                "Remove with `/skills remove <number|name>`.",
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome

        selection = await prompt_selection_after_message(
            ctx,
            content=content,
            right_info="skills",
            agent_name=agent_name,
            prompt="Remove skill by number or name (empty to cancel): ",
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
        remove_local_skill(skill_dir, destination_root=managed_skills_dir)
    except Exception as exc:
        outcome.add_message(f"Failed to remove skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        f"Removed skill: {manifest.name}",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    await _refresh_agent_skills(
        ctx,
        agent_name,
        managed_directory_override=directory_override,
    )
    return outcome


def _is_mcp_update(update: SkillUpdateInfo) -> bool:
    source = update.managed_source
    return source is not None and source.source_origin == "mcp"


def _find_mcp_skill_for_update(
    registries: Sequence[McpSkillRegistry],
    update: SkillUpdateInfo,
) -> McpRegistrySkill | None:
    source = update.managed_source
    if source is None or source.mcp_server_name is None:
        return None
    for registry in registries:
        if registry.server_name != source.mcp_server_name:
            continue
        for skill in registry.skills:
            if skill.source_url == source.source_url or skill.name == update.name:
                return skill
    return None


async def _enrich_mcp_update_infos(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: Sequence[SkillUpdateInfo],
) -> list[SkillUpdateInfo]:
    if not any(_is_mcp_update(update) for update in updates):
        return list(updates)

    registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)
    enriched: list[SkillUpdateInfo] = []
    for update in updates:
        if not _is_mcp_update(update):
            enriched.append(update)
            continue

        source = update.managed_source
        if source is None:
            enriched.append(update)
            continue

        skill = _find_mcp_skill_for_update(registries, update)
        if skill is None:
            enriched.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_path_missing",
                    detail="MCP registry entry not found",
                    current_revision=source.installed_revision,
                    available_revision=source.installed_revision,
                    managed_source=source,
                )
            )
            continue

        status = "up_to_date"
        detail = "already up to date"
        if skill.digest != source.artifact_digest:
            status = "update_available"
            detail = "MCP skill artifact changed"
        enriched.append(
            SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status=status,
                detail=detail,
                current_revision=source.artifact_digest or source.installed_revision,
                available_revision=skill.digest,
                managed_source=source,
            )
        )
    return enriched


async def _apply_mcp_skill_updates(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: Sequence[SkillUpdateInfo],
    force: bool,
) -> list[SkillUpdateInfo]:
    if not updates:
        return []

    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, _McpSkillRegistryAgent):
        return [
            SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status="source_unreachable",
                detail="This agent does not expose MCP skill registries.",
                current_revision=update.current_revision,
                available_revision=update.available_revision,
                managed_source=update.managed_source,
            )
            for update in updates
        ]

    registries = await agent.aggregator.list_mcp_skill_registries()
    results: list[SkillUpdateInfo] = []
    for update in updates:
        source = update.managed_source
        if source is None:
            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="invalid_metadata",
                    detail="missing source metadata",
                )
            )
            continue

        skill = _find_mcp_skill_for_update(registries, update)
        if skill is None:
            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_path_missing",
                    detail="MCP registry entry not found",
                    current_revision=source.installed_revision,
                    available_revision=source.installed_revision,
                    managed_source=source,
                )
            )
            continue

        fingerprint = compute_skill_content_fingerprint(update.skill_dir)
        if fingerprint != source.content_fingerprint and not force:
            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=source.installed_revision,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue

        try:
            await update_mcp_registry_skill(
                agent.aggregator,
                skill,
                skill_dir=update.skill_dir,
            )
            installed_source = read_installed_skill_source(update.skill_dir).source
        except Exception as exc:
            results.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_unreachable",
                    detail=str(exc),
                    current_revision=source.installed_revision,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue

        results.append(
            SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status="updated",
                detail="updated",
                current_revision=source.installed_revision,
                available_revision=skill.digest,
                managed_source=installed_source or source,
            )
        )
    return results


async def handle_update_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    parsed = parse_update_argument(argument, allow_skills_dir=True)
    if parsed.error:
        outcome.add_message(parsed.error, channel="error")
        return outcome
    directory_override = parsed.skills_dir or managed_directory_override

    management_scope = resolve_skills_management_scope(
        ctx.resolve_settings(),
        managed_directory_override=directory_override,
    )
    managed_skills_dir = management_scope.managed_directory
    updates = check_skill_updates(destination_root=managed_skills_dir)
    updates = await _enrich_mcp_update_infos(ctx, agent_name=agent_name, updates=updates)

    if parsed.selector is None:
        outcome.add_message(
            _format_update_results(updates, title="Skill update check:"),
            right_info="skills",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Apply with `/skills update <number|name|all> [--force] [--yes]`.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    selected = select_skill_updates(updates, parsed.selector)
    if not selected:
        outcome.add_message(f"Skill not found: {parsed.selector}", channel="error")
        return outcome

    if len(selected) > 1 and not parsed.yes:
        outcome.add_message(
            _format_update_results(selected, title="Update plan:"),
            right_info="skills",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Multiple skills selected. Re-run with `--yes` to apply updates.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    mcp_selected = [update for update in selected if _is_mcp_update(update)]
    regular_selected = [update for update in selected if not _is_mcp_update(update)]
    applied = [
        *await asyncio.to_thread(apply_skill_updates, regular_selected, force=parsed.force),
        *await _apply_mcp_skill_updates(
            ctx,
            agent_name=agent_name,
            updates=mcp_selected,
            force=parsed.force,
        ),
    ]
    outcome.add_message(
        _format_update_results(applied, title="Skill update results:"),
        right_info="skills",
        agent_name=agent_name,
    )

    if any(is_update_applied(result.status) for result in applied):
        await _refresh_agent_skills(
            ctx,
            agent_name,
            managed_directory_override=directory_override,
        )

    return outcome


type _SkillsActionHandler = Callable[
    ["CommandContext", str, str | None],
    Awaitable[CommandOutcome],
]


async def _handle_skills_list_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
) -> CommandOutcome:
    return await handle_list_skills(ctx, agent_name=agent_name)


async def _handle_skills_available_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
) -> CommandOutcome:
    return await handle_list_marketplace_skills(ctx, agent_name=agent_name, query=None)


async def _handle_skills_add_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_add_skill(ctx, agent_name=agent_name, argument=argument)


async def _handle_skills_registry_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_set_skills_registry(ctx, agent_name=agent_name, argument=argument)


async def _handle_skills_remove_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_remove_skill(ctx, agent_name=agent_name, argument=argument)


async def _handle_skills_update_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    return await handle_update_skill(ctx, agent_name=agent_name, argument=argument)


_SKILLS_ACTION_HANDLERS: dict[str, _SkillsActionHandler] = {
    "list": _handle_skills_list_action,
    "available": _handle_skills_available_action,
    "add": _handle_skills_add_action,
    "registry": _handle_skills_registry_action,
    "remove": _handle_skills_remove_action,
    "update": _handle_skills_update_action,
}


async def handle_skills_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = normalize_command_action("skills", action)

    if is_help_flag(action) or is_help_flag(argument):
        return handle_skills_help(agent_name=agent_name)

    if normalized == "search":
        query = strip_to_none(argument) or ""
        if not query:
            outcome = CommandOutcome()
            outcome.add_message(
                "Usage: /skills search <query>",
                channel="warning",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome
        return await handle_list_marketplace_skills(ctx, agent_name=agent_name, query=query)

    handler = _SKILLS_ACTION_HANDLERS.get(normalized)
    if handler is not None:
        return await handler(ctx, agent_name, argument)

    outcome = CommandOutcome()
    outcome.add_message(
        format_unknown_command_action("skills", normalized),
        channel="warning",
        right_info="skills",
        agent_name=agent_name,
    )
    outcome.add_message(
        "\n".join(skills_usage_lines()),
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome
