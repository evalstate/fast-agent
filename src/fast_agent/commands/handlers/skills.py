"""Shared skills command handlers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from rich.text import Text

from fast_agent.commands.handlers._marketplace_argument_parsing import parse_update_argument
from fast_agent.commands.handlers._text_formatting import append_heading, append_wrapped_text
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
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
    format_installed_at_display,
    format_revision_short,
    format_skill_provenance_details,
    read_installed_skill_source,
)
from fast_agent.skills.registry import SkillManifest, SkillRegistry, format_skills_for_prompt
from fast_agent.skills.scope import (
    order_skill_directories_for_display,
    resolve_skill_directories,
    resolve_skills_management_scope,
)
from fast_agent.skills.service import install_skill_from_selector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import AgentProtocol


MCP_REGISTRY_PREFIX = "mcp://"


@runtime_checkable
class _McpSkillRegistryAggregator(Protocol):
    async def list_mcp_skill_registries(self) -> list[McpSkillRegistry]: ...


@runtime_checkable
class _McpSkillRegistryAgent(Protocol):
    @property
    def aggregator(self) -> _McpSkillRegistryAggregator: ...


_parse_update_argument = parse_update_argument


def _mcp_registry_source(server_name: str) -> str:
    return f"{MCP_REGISTRY_PREFIX}{server_name}"


def _mcp_registry_server_name(source: str) -> str | None:
    if not source.startswith(MCP_REGISTRY_PREFIX):
        return None
    server_name = source[len(MCP_REGISTRY_PREFIX) :].strip()
    return server_name or None


async def _list_mcp_skill_registries(
    ctx: CommandContext, *, agent_name: str
) -> list[McpSkillRegistry]:
    agent = ctx.agent_provider._agent(agent_name)
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


def _append_manifest_entry(
    content: Text,
    manifest: SkillManifest,
    index: int,
    *,
    disabled: bool = False,
) -> None:
    entry = Text()
    entry.append(f"[{index:2}] ", style="dim cyan")
    entry.append(manifest.name, style="bright_blue bold")
    if disabled:
        entry.append("  (disabled)", style="dim yellow")
    content.append_text(entry)
    content.append("\n")

    if manifest.description:
        append_wrapped_text(content, manifest.description, indent="     ")

    # URI-backed (Skills-over-MCP) manifests have no filesystem path —
    # the provenance is the publishing MCP server. Don't try to derive
    # a `source_path` for them; the SEP-required provenance is the
    # server identity, not a directory.
    if manifest.path is None:
        content.append("     ", style="dim")
        server = manifest.server_name or "unknown"
        content.append(f"source: mcp-server {server}", style="dim green")
        content.append("\n")
        if manifest.uri:
            content.append("     ", style="dim")
            content.append(f"uri: {manifest.uri}", style="dim")
            content.append("\n")
        content.append("\n")
        return

    source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
    try:
        source_display = source_path.relative_to(Path.cwd())
    except ValueError:
        source_display = source_path
    content.append("     ", style="dim")
    content.append(f"source: {source_display}", style="dim green")
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
    entry = Text()
    entry.append(f"[{index}] ", style="dim cyan")
    entry.append(display_url, style="bright_blue bold")
    if is_current:
        entry.append(" • ", style="dim")
        entry.append("current", style="dim green")
    content.append_text(entry)
    content.append("\n")


def _format_local_skills_by_directory(manifests_by_dir: dict[Path, list[SkillManifest]]) -> Text:
    content = Text()
    skill_index = 0
    total_skills = sum(len(manifests) for manifests in manifests_by_dir.values())

    for directory, manifests in manifests_by_dir.items():
        try:
            display_dir = directory.relative_to(Path.cwd())
        except ValueError:
            display_dir = directory

        append_heading(content, f"Skills in {display_dir}:")

        if not manifests:
            content.append_text(Text("No skills in this directory", style="yellow"))
            content.append("\n")
            continue

        for manifest in manifests:
            skill_index += 1
            _append_manifest_entry(content, manifest, skill_index)

    if total_skills == 0:
        content.append_text(Text("Browse marketplace skills with /skills available", style="dim"))
    else:
        content.append_text(Text("Browse marketplace skills with /skills available", style="dim"))
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


def _format_marketplace_skills(marketplace: Sequence[object]) -> Text:
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

        name = getattr(entry, "name", "")
        description = getattr(entry, "description", "")
        source_url = getattr(entry, "source_url", None)
        digest = getattr(entry, "digest", None)

        entry_line = Text()
        entry_line.append(f"[{index:2}] ", style="dim cyan")
        entry_line.append(str(name), style="bright_blue bold")
        content.append_text(entry_line)
        content.append("\n")

        if description:
            append_wrapped_text(content, str(description), indent="     ")
        if source_url:
            content.append("     ", style="dim")
            content.append(f"source: {source_url}", style="dim green")
            content.append("\n")
        if digest:
            content.append("     ", style="dim")
            content.append("integrity: SHA256 checked", style="dim green")
            content.append("\n")
        content.append("\n")

    return content


def _is_help_flag(value: str | None) -> bool:
    token = (value or "").strip().lower()
    return token in {"help", "--help", "-h"}


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


def _format_update_results(updates: Sequence[SkillUpdateInfo], *, title: str) -> Text:
    content = Text()
    append_heading(content, title)
    if not updates:
        content.append_text(Text("No managed skills found.", style="yellow"))
        return content

    status_labels: dict[str, str] = {
        "up_to_date": "already up to date",
        "update_available": "update available",
        "updated": "updated",
        "unmanaged": "unmanaged",
        "invalid_metadata": "invalid metadata",
        "invalid_local_skill": "invalid local skill",
        "unknown_revision": "unknown revision",
        "source_unreachable": "source unreachable",
        "source_ref_missing": "source ref missing",
        "source_path_missing": "source path missing",
        "skipped_dirty": "skipped (local modifications)",
        "integrity_error": "integrity error",
    }
    status_detail_channels = {
        "invalid_metadata",
        "invalid_local_skill",
        "unknown_revision",
        "source_unreachable",
        "source_ref_missing",
        "source_path_missing",
        "skipped_dirty",
        "integrity_error",
    }
    detail_prefix = "  - "

    for update in updates:
        row = Text()
        row.append(f"[{update.index:2}] ", style="dim cyan")
        row.append(update.name, style="bright_blue bold")
        content.append_text(row)
        content.append("\n")

        source_path = update.skill_dir
        try:
            source_display = source_path.relative_to(Path.cwd())
        except ValueError:
            source_display = source_path
        content.append(detail_prefix, style="dim")
        content.append(f"source: {source_display}", style="dim green")
        content.append("\n")

        if update.managed_source is not None:
            source = update.managed_source
            ref_label = f"@{source.repo_ref}" if source.repo_ref else ""
            provenance_text = f"{source.repo_url}{ref_label} ({source.repo_path})"
            installed_text = (
                f"{format_installed_at_display(source.installed_at)} "
                f"revision: {format_revision_short(source.installed_revision)}"
            )
        else:
            provenance_text, installed_text = format_skill_provenance_details(update.skill_dir)

        content.append(detail_prefix, style="dim")
        content.append(f"provenance: {provenance_text}", style="dim")
        content.append("\n")
        if installed_text:
            content.append(detail_prefix, style="dim")
            content.append(f"installed: {installed_text}", style="dim")
            content.append("\n")

        if update.current_revision or update.available_revision:
            installed_revision = format_revision_short(update.current_revision)
            current_revision = format_revision_short(update.available_revision)
            content.append(detail_prefix, style="dim")
            content.append(f"revision: {installed_revision} -> {current_revision}", style="dim")
            content.append("\n")

        if update.status != "unmanaged":
            status_text = status_labels.get(update.status, update.status.replace("_", " "))
            if update.status in status_detail_channels and update.detail:
                status_text = f"{status_text}: {update.detail}"

            status_style: str | None = None
            if update.status in {"up_to_date", "updated"}:
                status_style = "green"
            elif update.status == "update_available":
                status_style = "bold bright_yellow"
            elif update.status not in {"unmanaged"}:
                status_style = "yellow"

            content.append(detail_prefix, style="dim")
            content.append("status: ", style="dim")
            if status_style is None:
                content.append(status_text)
            else:
                content.append(status_text, style=status_style)
            content.append("\n")

        content.append("\n")

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

    settings = ctx.resolve_settings()
    management_scope = resolve_skills_management_scope(settings)
    discovered_directories = order_skill_directories_for_display(
        management_scope.discovered_directories,
        settings=settings,
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
    ctx: CommandContext, *, agent_name: str, argument: str | None
) -> CommandOutcome:
    outcome = CommandOutcome()
    settings = ctx.resolve_settings()
    configured_urls = [
        url for url in resolve_skill_registries(settings) if _mcp_registry_server_name(url) is None
    ]
    mcp_registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)

    if not argument:
        current = get_marketplace_url(settings)
        current_display = format_marketplace_display_url(current)
        configured_displays = [
            format_marketplace_display_url(reg_url) for reg_url in configured_urls
        ]
        mcp_displays = [registry.display_name for registry in mcp_registries]
        current_mcp_server = _mcp_registry_server_name(current)
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

        if configured_displays:
            content.append_text(Text("Configured registries:", style="dim"))
            content.append("\n")

        for index, display in enumerate(configured_displays, 1):
            _append_registry_entry(
                content,
                display_url=display,
                index=index,
                is_current=display == current_display,
            )
        if mcp_displays:
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
        outcome.add_message(content, right_info="skills")
        return outcome

    arg = str(argument).strip()
    selected_mcp: McpSkillRegistry | None = None
    if arg.isdigit():
        index = int(arg)
        combined_count = len(configured_urls) + len(mcp_registries)
        if combined_count == 0:
            outcome.add_message("No registries configured.", channel="warning")
            return outcome
        if 1 <= index <= len(configured_urls):
            url = configured_urls[index - 1]
        elif len(configured_urls) < index <= combined_count:
            selected_mcp = mcp_registries[index - len(configured_urls) - 1]
            url = _mcp_registry_source(selected_mcp.server_name)
        else:
            outcome.add_message(
                f"Invalid registry number. Use 1-{combined_count}.",
                channel="warning",
            )
            return outcome
    else:
        explicit_mcp_server = _mcp_registry_server_name(arg) or arg
        selected_mcp = _find_mcp_registry(mcp_registries, explicit_mcp_server)
        url = _mcp_registry_source(selected_mcp.server_name) if selected_mcp is not None else arg

    if selected_mcp is not None:
        skills_settings = getattr(settings, "skills", None)
        if skills_settings is not None:
            skills_settings.marketplace_url = url

        content = Text()
        content.append_text(
            Text(
                f"Registry set to: {selected_mcp.display_name}",
                style="green",
            )
        )
        content.append("\n")
        content.append_text(Text(f"Skills discovered: {len(selected_mcp.skills)}", style="dim"))
        outcome.add_message(content, right_info="skills")
        return outcome

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
) -> CommandOutcome:
    outcome = CommandOutcome()

    marketplace_url = get_marketplace_url(ctx.resolve_settings())
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
        marketplace: Sequence[McpRegistrySkill] = mcp_registry.skills
        if query and query.strip():
            query_lower = query.strip().lower()
            marketplace = [
                skill
                for skill in marketplace
                if query_lower in skill.name.lower()
                or query_lower in (skill.description or "").lower()
            ]
        if not marketplace:
            outcome.add_message("No skills found in the MCP registry.", channel="warning")
            return outcome
        content = Text()
        heading = f"MCP skills from {mcp_registry.display_name}:"
        if query and query.strip():
            heading = f"MCP skills from {mcp_registry.display_name} (search: {query.strip()}):"
        append_heading(content, heading)
        content.append_text(_format_marketplace_skills(marketplace))
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        outcome.add_message(
            SKILLS_ADD_HINT_SLASH,
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        outcome.add_message(
            "Search with `/skills search <query>`.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace:
        outcome.add_message("No skills found in the marketplace.", channel="warning")
        return outcome

    selected_marketplace: Sequence[object] = marketplace
    if query and query.strip():
        selected_marketplace = filter_marketplace_skills(marketplace, query)

    content = Text()
    heading = "Marketplace skills:"
    if query and query.strip():
        heading = f"Marketplace skills (search: {query.strip()}):"
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
        outcome.add_message(
            "Try `/skills available` to browse all skills.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    content.append_text(_format_marketplace_skills(selected_marketplace))
    outcome.add_message(content, right_info="skills", agent_name=agent_name)
    outcome.add_message(
        SKILLS_ADD_HINT_SLASH,
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    outcome.add_message(
        "Search with `/skills search <query>`.",
        channel="info",
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
) -> CommandOutcome:
    outcome = CommandOutcome()

    management_scope = resolve_skills_management_scope(ctx.resolve_settings())
    managed_skills_dir = management_scope.managed_directory
    selection = argument
    marketplace_url = get_marketplace_url(ctx.resolve_settings())
    mcp_server_name = _mcp_registry_server_name(marketplace_url)

    if mcp_server_name is not None:
        agent = ctx.agent_provider._agent(agent_name)
        if not isinstance(agent, _McpSkillRegistryAgent):
            outcome.add_message("This agent does not expose MCP skill registries.", channel="error")
            return outcome
        mcp_registries = await agent.aggregator.list_mcp_skill_registries()
        mcp_registry = _find_mcp_registry(mcp_registries, mcp_server_name)
        if mcp_registry is None:
            outcome.add_message(
                f"MCP skill registry is not available: {mcp_server_name}",
                channel="error",
            )
            return outcome

        if not selection:
            content = Text()
            append_heading(content, f"MCP skills from {mcp_registry.display_name}:")
            content.append_text(_format_marketplace_skills(mcp_registry.skills))
            if not interactive:
                outcome.add_message(content, right_info="skills", agent_name=agent_name)
                outcome.add_message(
                    SKILLS_ADD_HINT_SLASH,
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

            await ctx.io.emit(
                CommandMessage(text=content, right_info="skills", agent_name=agent_name)
            )
            selection = await ctx.io.prompt_selection(
                "Install skill by number or name (empty to cancel): ",
                options=[skill.name for skill in mcp_registry.skills],
                allow_cancel=True,
            )
            if selection is None:
                return outcome

        mcp_skill = select_mcp_registry_skill(mcp_registry.skills, selection)
        if mcp_skill is None:
            outcome.add_message(f"Skill not found: {selection}", channel="error")
            return outcome

        try:
            install_path = await install_mcp_registry_skill(
                cast("Any", agent.aggregator),
                mcp_skill,
                destination_root=managed_skills_dir,
            )
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"Failed to install skill: {exc}", channel="error")
            return outcome

        outcome.add_message(
            _format_install_result(mcp_skill.name, install_path),
            right_info="skills",
            agent_name=agent_name,
        )
        await _refresh_agent_skills(ctx, agent_name)
        return outcome

    if selection:
        try:
            installed = await install_skill_from_selector(
                marketplace_url,
                selection,
                destination_root=managed_skills_dir,
            )
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(f"Failed to install skill: {exc}", channel="error")
            return outcome

        outcome.add_message(
            _format_install_result(installed.name, installed.skill_dir),
            right_info="skills",
            agent_name=agent_name,
        )
        await _refresh_agent_skills(ctx, agent_name)
        return outcome

    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(f"Failed to load marketplace: {exc}", channel="error")
        return outcome

    if not marketplace:
        outcome.add_message("No skills found in the marketplace.", channel="warning")
        return outcome

    if not selection:
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

        if not interactive:
            outcome.add_message(content, right_info="skills", agent_name=agent_name)
            outcome.add_message(
                SKILLS_ADD_HINT_SLASH,
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            outcome.add_message(
                "Browse marketplace with `/skills available`.",
                channel="info",
                right_info="skills",
                agent_name=agent_name,
            )
            outcome.add_message(
                "Search marketplace with `/skills search <query>`.",
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
        outcome.add_message(
            "Run `/skills available` to browse skills or `/skills search <query>` to filter.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    try:
        install_path = await install_marketplace_skill(
            skill,
            destination_root=managed_skills_dir,
        )
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

    management_scope = resolve_skills_management_scope(ctx.resolve_settings())
    managed_skills_dir = management_scope.managed_directory
    manifests = SkillRegistry.load_directory(managed_skills_dir)
    if not manifests:
        outcome.add_message("No local skills to remove.", channel="warning")
        return outcome

    selection = argument
    if not selection:
        content = Text()
        append_heading(content, f"Skills in {managed_skills_dir}:")
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
        remove_local_skill(skill_dir, destination_root=managed_skills_dir)
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


async def _enrich_mcp_update_infos(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: list[SkillUpdateInfo],
) -> list[SkillUpdateInfo]:
    mcp_updates = [update for update in updates if _is_mcp_update(update)]
    if not mcp_updates:
        return updates

    registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)
    enriched: list[SkillUpdateInfo] = []
    for update in updates:
        if not _is_mcp_update(update):
            enriched.append(update)
            continue
        skill = _find_mcp_skill_for_update(registries, update)
        source = update.managed_source
        if source is None:
            enriched.append(update)
            continue
        if skill is None:
            enriched.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_path_missing",
                    detail="MCP registry entry not found",
                    current_revision=source.artifact_digest or source.installed_revision,
                    managed_source=source,
                )
            )
            continue
        current_digest = source.artifact_digest or source.installed_revision
        status = "up_to_date" if current_digest == skill.digest else "update_available"
        detail = "already up to date" if status == "up_to_date" else "skill artifact changed"
        enriched.append(
            SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status=status,
                detail=detail,
                current_revision=current_digest,
                available_revision=skill.digest,
                managed_source=source,
            )
        )
    return enriched


async def _apply_mcp_skill_updates(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: list[SkillUpdateInfo],
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
    applied: list[SkillUpdateInfo] = []
    for update in updates:
        source = update.managed_source
        if source is None:
            applied.append(
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
            applied.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_path_missing",
                    detail="MCP registry entry not found",
                    current_revision=source.artifact_digest or source.installed_revision,
                    managed_source=source,
                )
            )
            continue

        current_digest = source.artifact_digest or source.installed_revision
        if current_digest == skill.digest:
            applied.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="up_to_date",
                    detail="already up to date",
                    current_revision=current_digest,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue

        is_dirty = compute_skill_content_fingerprint(update.skill_dir) != source.content_fingerprint
        if is_dirty and not force:
            applied.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="skipped_dirty",
                    detail="local modifications detected; rerun with --force",
                    current_revision=current_digest,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue

        try:
            await update_mcp_registry_skill(
                cast("Any", agent.aggregator),
                skill,
                skill_dir=update.skill_dir,
            )
        except ValueError as exc:
            applied.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="integrity_error",
                    detail=str(exc),
                    current_revision=current_digest,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            applied.append(
                SkillUpdateInfo(
                    index=update.index,
                    name=update.name,
                    skill_dir=update.skill_dir,
                    status="source_unreachable",
                    detail=str(exc),
                    current_revision=current_digest,
                    available_revision=skill.digest,
                    managed_source=source,
                )
            )
            continue

        refreshed_source, _error = read_installed_skill_source(update.skill_dir)
        applied.append(
            SkillUpdateInfo(
                index=update.index,
                name=update.name,
                skill_dir=update.skill_dir,
                status="updated",
                detail="updated with --force (local changes overwritten)" if is_dirty else "updated",
                current_revision=current_digest,
                available_revision=skill.digest,
                managed_source=refreshed_source or source,
            )
        )
    return applied


def _is_mcp_update(update: SkillUpdateInfo) -> bool:
    source = update.managed_source
    return source is not None and source.source_origin == "mcp"


def _find_mcp_skill_for_update(
    registries: Sequence[McpSkillRegistry], update: SkillUpdateInfo
) -> McpRegistrySkill | None:
    source = update.managed_source
    if source is None:
        return None
    for registry in registries:
        if registry.server_name != source.mcp_server_name:
            continue
        for skill in registry.skills:
            if skill.source_url == source.source_url or skill.name == update.name:
                return skill
    return None


async def handle_update_skill(
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

    management_scope = resolve_skills_management_scope(ctx.resolve_settings())
    managed_skills_dir = management_scope.managed_directory
    updates = await _enrich_mcp_update_infos(
        ctx,
        agent_name=agent_name,
        updates=check_skill_updates(destination_root=managed_skills_dir),
    )

    if selector is None:
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

    selected = select_skill_updates(updates, selector)
    if not selected:
        outcome.add_message(f"Skill not found: {selector}", channel="error")
        return outcome

    if len(selected) > 1 and not yes:
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
        *await asyncio.to_thread(apply_skill_updates, regular_selected, force=force),
        *await _apply_mcp_skill_updates(
            ctx,
            agent_name=agent_name,
            updates=mcp_selected,
            force=force,
        ),
    ]
    outcome.add_message(
        _format_update_results(applied, title="Skill update results:"),
        right_info="skills",
        agent_name=agent_name,
    )

    if any(result.status == "updated" for result in applied):
        await _refresh_agent_skills(ctx, agent_name)

    return outcome


async def handle_list_skill_templates(
    ctx: CommandContext,
    *,
    agent_name: str,
) -> CommandOutcome:
    """List `mcp-resource-template` entries discovered from connected servers.

    These are RFC 6570 URI templates the user must fill before the skill
    is registered. Without this surface they'd be invisible — the loader
    collects them, but until a `/skills resolve` walks the variables
    they never enter the active manifest set or the model's context.
    """
    outcome = CommandOutcome()

    agent_obj = ctx.agent_provider._agent(agent_name)
    templates = getattr(agent_obj, "skill_template_entries", None)
    if not templates:
        outcome.add_message(
            "No skill templates discovered on this agent's connected MCP servers.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    content = Text()
    append_heading(content, "Skill templates:")
    content.append_text(
        Text(
            "Each entry describes a parameterized skill namespace. "
            "Resolve one with `/skills resolve <number>` to register it.",
            style="dim",
        )
    )
    content.append("\n\n")

    for index, template in enumerate(templates, 1):
        entry = Text()
        entry.append(f"[{index:2}] ", style="dim cyan")
        entry.append(template.url_template, style="bright_blue bold")
        content.append_text(entry)
        content.append("\n")
        if template.description:
            append_wrapped_text(content, template.description, indent="     ")
        content.append("     ", style="dim")
        content.append(f"server: {template.server_name}", style="dim green")
        content.append("\n")
        variables = template.variable_names()
        if variables:
            content.append("     ", style="dim")
            content.append(
                f"variables: {', '.join(variables)}",
                style="dim",
            )
            content.append("\n")
        content.append("\n")

    outcome.add_message(content, right_info="skills", agent_name=agent_name)
    return outcome


async def handle_resolve_skill_template(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
    interactive: bool = True,
) -> CommandOutcome:
    """Walk a template's variables (via MCP completion) and register the result.

    Argument is the 1-based index from `/skills templates`. Optional
    `var=value` overrides may be appended (space-separated) to skip the
    completion prompt for specific variables — useful for scripted /
    non-interactive runs.
    """
    outcome = CommandOutcome()

    agent_obj = ctx.agent_provider._agent(agent_name)
    templates = list(getattr(agent_obj, "skill_template_entries", None) or [])
    if not templates:
        outcome.add_message(
            "No skill templates available on this agent.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    if not argument or not argument.strip():
        outcome.add_message(
            "Usage: /skills resolve <number> [var=value ...]",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    tokens = argument.strip().split()
    selector = tokens[0]
    overrides: dict[str, str] = {}
    for tok in tokens[1:]:
        if "=" not in tok:
            outcome.add_message(
                f"Override must be var=value, got: {tok}",
                channel="error",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome
        key, value = tok.split("=", 1)
        overrides[key.strip()] = value.strip()

    if not selector.isdigit():
        outcome.add_message(
            f"Template selector must be a number from `/skills templates`, got: {selector}",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome
    index = int(selector)
    if not (1 <= index <= len(templates)):
        outcome.add_message(
            f"No template at index {index}. Use `/skills templates` to list.",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome
    template = templates[index - 1]

    values: dict[str, str] = {}
    for var_name in template.variable_names():
        if var_name in overrides:
            values[var_name] = overrides[var_name]
            continue

        try:
            candidates = await agent_obj.complete_skill_template_argument(
                template,
                argument_name=var_name,
                value="",
                context_args=values or None,
            )
        except Exception as exc:  # noqa: BLE001
            outcome.add_message(
                f"Completion failed for variable '{var_name}': {exc}",
                channel="error",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome

        if interactive:
            picked: str | None
            if candidates:
                picked = await ctx.io.prompt_selection(
                    f"Value for `{var_name}` (server-suggested):",
                    options=candidates,
                    allow_cancel=True,
                )
            else:
                # Server returned no suggestions; fall back to free-form
                # input — the SEP allows servers to publish templates
                # without exhaustive completion support, in which case
                # the user has to type the value.
                picked = await ctx.io.prompt_text(
                    f"Value for `{var_name}` (no server suggestions; type or empty to cancel):",
                    allow_empty=True,
                )
            if not picked:
                outcome.add_message(
                    "Resolution cancelled.",
                    channel="info",
                    right_info="skills",
                    agent_name=agent_name,
                )
                return outcome
            values[var_name] = picked
        else:
            # Non-interactive: bail rather than guess.
            outcome.add_message(
                (
                    f"Variable `{var_name}` not provided. Re-run interactively "
                    "or pass `var=value` overrides."
                ),
                channel="error",
                right_info="skills",
                agent_name=agent_name,
            )
            return outcome

    try:
        manifest = await agent_obj.register_resolved_skill_template(template, values)
    except Exception as exc:  # noqa: BLE001
        outcome.add_message(
            f"Failed to register resolved skill: {exc}",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    if manifest is None:
        outcome.add_message(
            (
                "Resolved skill could not be loaded (server returned no "
                "SKILL.md, parse failed, or a same-named skill is already "
                "registered). See the log for details."
            ),
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        f"Registered skill: {manifest.name} (from {template.server_name})",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


async def handle_preview_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    """Show a skill's SKILL.md content without the model touching it.

    Satisfies SEP-2640's "SHOULD let users inspect a skill's content
    before it is loaded into model context." The model decides
    autonomously when to call `read_skill`, so this is the only
    pre-load surface for users.

    The lookup re-uses the agent's SkillReader so the read goes through
    the same trust boundary and (for MCP-backed skills) aggregator
    dispatch as a model-driven read. The output is rendered to the user,
    not added to model context, so a preview never plants skill text in
    the conversation.
    """
    outcome = CommandOutcome()
    if not argument or not argument.strip():
        outcome.add_message(
            "Usage: /skills preview <name>",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    name = argument.strip()
    agent_obj = ctx.agent_provider._agent(agent_name)
    manifests = list(getattr(agent_obj, "_skill_manifests", None) or [])
    match = next(
        (m for m in manifests if m.name.lower() == name.lower()),
        None,
    )
    if match is None:
        outcome.add_message(
            f"No skill named '{name}' on this agent. Run `/skills` to list.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    reader = getattr(agent_obj, "_skill_reader", None)
    if reader is None:
        outcome.add_message(
            "Skill reader is not initialized on this agent.",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    # Use the same `path` argument shape the model uses, so a disabled
    # skill is also unreadable here — preview honors the toggle state.
    if match.path is not None:
        location = str(match.path)
    elif match.uri is not None:
        location = match.uri
    else:
        outcome.add_message(
            f"Skill '{name}' has no readable location.",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    result = await reader.execute({"path": location})
    body = "".join(
        block.text for block in result.content if hasattr(block, "text")
    )

    if result.isError:
        outcome.add_message(
            f"Preview failed: {body}",
            channel="error",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    header = Text()
    append_heading(header, f"Skill preview: {match.name}")
    source = (
        f"mcp-server: {match.server_name}"
        if match.uri
        else (f"filesystem: {match.path.parent}" if match.path else "unknown")
    )
    header.append_text(Text(f"source: {source}\n", style="dim"))
    header.append_text(Text(f"location: {location}\n\n", style="dim"))
    outcome.add_message(header, right_info="skills", agent_name=agent_name)
    # Render the body as markdown so headings, code blocks, etc. survive.
    outcome.add_message(
        body,
        right_info="skills",
        agent_name=agent_name,
        render_markdown=True,
    )
    return outcome


async def handle_disable_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    """Hide a skill from this session.

    Disabled skills don't appear in the model's `<available_skills>`
    block on subsequent renderings and the SkillReader's allow-list no
    longer admits their paths/URIs, so the model can't read them either.
    The disable list is in-process and resets when the session ends.
    """
    outcome = CommandOutcome()
    if not argument or not argument.strip():
        outcome.add_message(
            "Usage: /skills disable <name>",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    name = argument.strip()
    agent_obj = ctx.agent_provider._agent(agent_name)
    if not hasattr(agent_obj, "disable_skill"):
        outcome.add_message(
            "This agent does not support per-skill toggles.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    changed = agent_obj.disable_skill(name)
    if not changed:
        outcome.add_message(
            (
                f"Skill '{name}' is not active on this agent (or already disabled). "
                "Run `/skills` to see the current list."
            ),
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        f"Disabled skill: {name}",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    outcome.add_message(
        (
            "Note: the model's existing context still mentions disabled "
            "skills, but the read_skill tool will refuse to load them."
        ),
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


async def handle_enable_skill(
    ctx: CommandContext,
    *,
    agent_name: str,
    argument: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    if not argument or not argument.strip():
        outcome.add_message(
            "Usage: /skills enable <name>",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    name = argument.strip()
    agent_obj = ctx.agent_provider._agent(agent_name)
    if not hasattr(agent_obj, "enable_skill"):
        outcome.add_message(
            "This agent does not support per-skill toggles.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    changed = agent_obj.enable_skill(name)
    if not changed:
        outcome.add_message(
            f"Skill '{name}' is not currently disabled.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    outcome.add_message(
        f"Enabled skill: {name}",
        channel="info",
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


async def handle_skills_command(
    ctx: CommandContext,
    *,
    agent_name: str,
    action: str | None,
    argument: str | None,
) -> CommandOutcome:
    normalized = str(action or "list").lower()

    if _is_help_flag(action) or _is_help_flag(argument):
        return handle_skills_help(agent_name=agent_name)

    if normalized in {"help"}:
        return handle_skills_help(agent_name=agent_name)

    if normalized in {"list", ""}:
        return await handle_list_skills(ctx, agent_name=agent_name)
    if normalized in {"available", "marketplace", "browse"}:
        return await handle_list_marketplace_skills(ctx, agent_name=agent_name, query=None)
    if normalized in {"search", "find"}:
        query = argument.strip() if argument else ""
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
    if normalized in {"add", "install"}:
        return await handle_add_skill(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"registry", "source"}:
        return await handle_set_skills_registry(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"remove", "rm", "delete", "uninstall"}:
        return await handle_remove_skill(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"update", "refresh", "upgrade"}:
        return await handle_update_skill(ctx, agent_name=agent_name, argument=argument)
    if normalized in {"templates", "template"}:
        return await handle_list_skill_templates(ctx, agent_name=agent_name)
    if normalized in {"resolve"}:
        return await handle_resolve_skill_template(
            ctx, agent_name=agent_name, argument=argument
        )
    if normalized in {"disable", "off"}:
        return await handle_disable_skill(
            ctx, agent_name=agent_name, argument=argument
        )
    if normalized in {"enable", "on"}:
        return await handle_enable_skill(
            ctx, agent_name=agent_name, argument=argument
        )
    if normalized in {"preview", "inspect", "show"}:
        return await handle_preview_skill(
            ctx, agent_name=agent_name, argument=argument
        )

    outcome = CommandOutcome()
    outcome.add_message(
        (
            f"Unknown /skills action: {normalized}. "
            "Use list/available/search/add/remove/update/registry/"
            "templates/resolve/enable/disable/preview/help."
        ),
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
