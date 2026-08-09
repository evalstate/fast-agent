"""Shared skills command handlers."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from rich.text import Text

from fast_agent.commands.command_catalog import (
    format_unknown_command_action,
    normalize_command_action,
)
from fast_agent.commands.handlers import skills_registry as skills_registry_handlers
from fast_agent.commands.handlers._marketplace_argument_parsing import (
    parse_add_argument,
    parse_update_argument,
)
from fast_agent.commands.handlers._text_formatting import (
    append_detail_line,
    append_heading,
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
    SkillsCatalogFormat,
    marketplace_search_tokens,
    parse_skills_catalog_options,
    skills_usage_lines,
)
from fast_agent.skills.configuration import (
    format_marketplace_display_url,
)
from fast_agent.skills.mcp_registry import McpRegistrySkill
from fast_agent.skills.models import MarketplaceSkill, SkillUpdateInfo
from fast_agent.skills.operations import (
    check_skill_updates,
    fetch_marketplace_skills_with_source,
    reload_skill_manifests,
    remove_local_skill,
    select_manifest_by_name_or_index,
    select_skill_updates,
)
from fast_agent.skills.provenance import (
    format_revision_short,
    format_skill_provenance_details,
)
from fast_agent.skills.registry import SkillManifest, SkillRegistry
from fast_agent.skills.scope import (
    order_skill_directories_for_display,
    resolve_skill_directories,
    resolve_skills_management_scope,
)
from fast_agent.skills.source_resolver import SkillSourceResolver, mcp_registry_source
from fast_agent.utils.action_normalization import is_help_flag
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.commandline import join_commandline
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.skills.sources import SkillCatalogEntry, SkillInstallSource
    from fast_agent.tools.execution_environment import ShellRuntimeInfo


REMOTE_ENVIRONMENT_SKILLS_WARNING = (
    "These are local skills and may not be available in your configured environment. "
    "Use /system to check"
)
MODEL_SKILLS_CATALOG_LIMIT = 50
COMPACT_SKILL_DESCRIPTION_LIMIT = 180


@dataclass(frozen=True, slots=True)
class _SkillCatalogPage:
    entries: list["SkillCatalogEntry"]
    source_indices: list[int]
    page: int
    limit: int
    total: int
    result_start: int

    @property
    def result_end(self) -> int:
        return self.result_start + len(self.entries) - 1

    @property
    def next_page(self) -> int | None:
        if not self.entries:
            return None
        return self.page + 1 if self.result_end < self.total else None


@runtime_checkable
class _ShellRuntimeProvider(Protocol):
    @property
    def shell_runtime(self) -> object | None: ...


@runtime_checkable
class _ShellRuntimeInfoProvider(Protocol):
    def runtime_info(self) -> "ShellRuntimeInfo": ...


def _uses_remote_shell_environment(agent: object) -> bool:
    if not isinstance(agent, _ShellRuntimeProvider):
        return False
    shell_runtime = agent.shell_runtime
    if not isinstance(shell_runtime, _ShellRuntimeInfoProvider):
        return False
    return shell_runtime.runtime_info().kind != "local"


async def handle_set_skills_registry(
    ctx: "CommandContext", *, argument: str | None, agent_name: str | None = None
) -> CommandOutcome:
    return await skills_registry_handlers.handle_set_skills_registry(
        ctx,
        argument=argument,
        agent_name=agent_name,
        fetch_skills_with_source=fetch_marketplace_skills_with_source,
    )


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

    content.append_text(Text("Browse available skills with /skills available", style="dim"))
    if total_skills > 0:
        content.append("\n")
        content.append_text(
            Text("Search available skills with /skills search <query>", style="dim")
        )
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
    marketplace: Sequence[SkillCatalogEntry],
    *,
    start_index: int = 1,
    source_indices: Sequence[int] | None = None,
) -> Text:
    content = Text()
    current_bundle = None

    indices = source_indices or range(start_index, start_index + len(marketplace))
    for index, entry in zip(indices, marketplace, strict=True):
        bundle_name = None
        bundle_description = None
        revision = None
        if isinstance(entry, MarketplaceSkill):
            bundle_name = entry.bundle_name
            bundle_description = entry.bundle_description
        if isinstance(entry, McpRegistrySkill):
            revision = entry.revision

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
        if revision:
            content.append("     ", style="dim")
            content.append(
                "integrity: SHA-256 manifest; checked on install",
                style="dim green",
            )
            content.append("\n")
        content.append("\n")

    return content


def _format_install_result(
    skill_name: str,
    install_path: Path,
    *,
    mcp_integrity: bool = False,
) -> Text:
    content = Text()
    content.append(f"Installed skill: {skill_name}", style="green")
    content.append("\n")
    content.append(f"location: {format_display_path(install_path)}", style="dim green")
    if mcp_integrity:
        content.append("\n")
        content.append(
            "integrity: SHA-256 digests matched the server-supplied manifest",
            style="dim green",
        )
        content.append("\n")
        content.append(
            "trust: this does not authenticate the server or publisher, or establish "
            "that the skill is safe",
            style="dim yellow",
        )
    return content


def _format_skill_source_list(
    source: SkillInstallSource,
    entries: list[SkillCatalogEntry],
    *,
    query: str | None = None,
    start_index: int = 1,
    source_indices: Sequence[int] | None = None,
) -> Text:
    content = Text()
    append_heading(content, source.list_heading(query=query))
    repo_hint = source.repository_hint(entries)
    if repo_hint:
        content.append_text(
            Text(
                f"Repository: {format_marketplace_display_url(repo_hint)}",
                style="dim",
            )
        )
        content.append("\n\n")
    content.append_text(
        _format_marketplace_skills(
            entries,
            start_index=start_index,
            source_indices=source_indices,
        )
    )
    if source.ref.kind == "mcp":
        content.append(
            "Integrity checks compare downloaded files with server-supplied digests. "
            "They do not verify the server or publisher; review skills and use only "
            "MCP servers you trust.",
            style="dim yellow",
        )
        content.append("\n")
    return content


def _compact_description(description: str | None) -> str | None:
    if description is None:
        return None
    normalized = re.sub(r"\s+", " ", description).strip()
    if not normalized:
        return None
    if len(normalized) <= COMPACT_SKILL_DESCRIPTION_LIMIT:
        return normalized
    return normalized[: COMPACT_SKILL_DESCRIPTION_LIMIT - 1].rstrip() + "…"


def _catalog_page(
    indexed_entries: list[tuple[int, SkillCatalogEntry]],
    *,
    page: int,
    limit: int,
) -> _SkillCatalogPage:
    start = (page - 1) * limit
    selected = indexed_entries[start : start + limit]
    return _SkillCatalogPage(
        entries=[entry for _index, entry in selected],
        source_indices=[index for index, _entry in selected],
        page=page,
        limit=limit,
        total=len(indexed_entries),
        result_start=start + 1,
    )


def _skills_catalog_command(
    *,
    action: str,
    query: str | None,
    registry: str | None,
    page: int,
    limit: int,
    output: SkillsCatalogFormat,
) -> str:
    argv = ["/skills", action]
    if query is not None:
        argv.append(query)
    if registry is not None:
        argv.extend(("--registry", registry))
    if page > 1:
        argv.extend(("--page", str(page)))
    argv.extend(("--limit", str(limit)))
    if output != "auto":
        argv.append(f"--{output}")
    return join_commandline(argv, syntax="posix")


def _skills_install_command(registry: str | None) -> str:
    placeholder = "SKILL_SELECTOR"
    argv = ["/skills", "add", placeholder]
    if registry is not None:
        argv.extend(("--registry", registry))
    return join_commandline(argv, syntax="posix").replace(
        placeholder,
        "<number|name>",
    )


def skills_available_command(registry: str | None) -> str:
    argv = ["/skills", "available"]
    if registry is not None:
        argv.extend(("--registry", registry))
    return join_commandline(argv, syntax="posix")


def _format_compact_skill_catalog(
    source: SkillInstallSource,
    catalog_page: _SkillCatalogPage,
    *,
    query: str | None,
    registry: str | None,
    action: str,
    output: SkillsCatalogFormat,
) -> Text:
    content = Text()
    content.append(f"Source: {source.ref.display_name}", style="bold")
    content.append("\n")
    if query is not None:
        content.append(f"Search: {query}", style="dim")
        content.append("\n")
    content.append(
        f"Showing {catalog_page.result_start}-{catalog_page.result_end} "
        f"of {catalog_page.total} skills.",
        style="dim",
    )
    content.append("\n\n")
    for index, entry in zip(
        catalog_page.source_indices,
        catalog_page.entries,
        strict=True,
    ):
        content.append(f"{index}. ", style="dim cyan")
        content.append(entry.name, style="bright_blue bold")
        description = _compact_description(entry.description)
        if description is not None:
            content.append(f" — {description}")
        content.append("\n")

    if catalog_page.next_page is not None:
        content.append("\n")
        next_command = _skills_catalog_command(
            action=action,
            query=query,
            registry=registry,
            page=catalog_page.next_page,
            limit=catalog_page.limit,
            output=output,
        )
        content.append(f"Next: {next_command}", style="dim")
        content.append("\n")
    content.append("\n")
    content.append(f"Install: {_skills_install_command(registry)}", style="dim")
    return content


def _skill_catalog_json(
    source: SkillInstallSource,
    catalog_page: _SkillCatalogPage,
    *,
    query: str | None,
    registry: str | None,
    action: str,
    output: SkillsCatalogFormat,
) -> str:
    return json.dumps(
        {
            "schema_version": "1",
            "kind": "skill_catalog",
            "source": {
                "kind": source.ref.kind,
                "display_name": source.ref.display_name,
                "source_url": (
                    format_marketplace_display_url(source.ref.source_url)
                    if source.ref.source_url is not None
                    else None
                ),
                "server_name": source.ref.server_name,
            },
            "query": query,
            "page": catalog_page.page,
            "limit": catalog_page.limit,
            "total": catalog_page.total,
            "next_page": catalog_page.next_page,
            "commands": {
                "install": _skills_install_command(registry),
                "next_page": (
                    _skills_catalog_command(
                        action=action,
                        query=query,
                        registry=registry,
                        page=catalog_page.next_page,
                        limit=catalog_page.limit,
                        output=output,
                    )
                    if catalog_page.next_page is not None
                    else None
                ),
            },
            "skills": [
                {
                    "index": index,
                    "name": entry.name,
                    "description": _compact_description(entry.description),
                    "source_url": (
                        format_marketplace_display_url(entry.source_url)
                        if entry.source_url is not None
                        else None
                    ),
                }
                for index, entry in zip(
                    catalog_page.source_indices,
                    catalog_page.entries,
                    strict=True,
                )
            ],
        },
        indent=2,
        sort_keys=True,
    )


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
            "Browse available skills with `/skills available`.",
            "Search available skills with `/skills search <query>`.",
            "Change registry with `/skills registry`.",
        ),
        right_info="skills",
        agent_name=agent_name,
    )


async def _install_skill_from_add_selector(
    ctx: CommandContext,
    *,
    agent_name: str,
    source: SkillInstallSource,
    managed_skills_dir: Path,
    selection: str,
    managed_directory_override: str | Path | None = None,
) -> CommandOutcome:
    outcome = CommandOutcome()
    try:
        installed = await source.install_skill(
            selection,
            destination_root=managed_skills_dir,
        )
    except Exception as exc:
        outcome.add_message(f"Failed to install skill: {exc}", channel="error")
        return outcome

    outcome.add_message(
        _format_install_result(
            installed.name,
            installed.skill_dir,
            mcp_integrity=source.ref.kind == "mcp",
        ),
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
    source: SkillInstallSource,
    entries: list[SkillCatalogEntry],
    agent_name: str,
    interactive: bool,
) -> str | None:
    content = _format_skill_source_list(source, entries)
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
        options=unique_selection_options(source.selection_options(entries)),
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


async def _refresh_agent_skills(
    ctx: CommandContext,
    agent_name: str,
    *,
    managed_directory_override: str | Path | None = None,
) -> None:
    agent = cast("AgentProtocol", ctx.agent_provider._agent(agent_name))
    if agent.config.skills is not SKILLS_DEFAULT:
        return
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

    agent_obj = ctx.agent_provider._agent(agent_name)
    warn_for_remote_environment = _uses_remote_shell_environment(agent_obj)

    outcome.add_message(
        _format_local_skills_by_directory(manifests_by_dir),
        right_info="skills",
        agent_name=agent_name,
    )

    agent_obj = cast("AgentProtocol", agent_obj)
    config = agent_obj.config
    if config.skills is not SKILLS_DEFAULT:
        manifests = list(config.skill_manifests or [])
        sources = _get_agent_skill_override_sources(manifests)
        outcome.add_message(
            _format_agent_skills_override(manifests, source_paths=sources),
            right_info="skills",
            agent_name=agent_name,
        )

    if warn_for_remote_environment:
        outcome.add_message(
            REMOTE_ENVIRONMENT_SKILLS_WARNING,
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )

    if config.skills is SKILLS_DEFAULT:
        return outcome

    return outcome


def handle_skills_help(*, agent_name: str) -> CommandOutcome:
    outcome = CommandOutcome()
    outcome.add_message(
        "\n".join(skills_usage_lines()),
        right_info="skills",
        agent_name=agent_name,
    )
    return outcome


def _skill_matches_query(entry: SkillCatalogEntry, query: str) -> bool:
    values = [entry.name, entry.description or ""]
    if isinstance(entry, MarketplaceSkill):
        values.extend(
            (
                entry.install_dir_name,
                entry.bundle_name or "",
                entry.bundle_description or "",
            )
        )
    haystack = strip_casefold(" ".join(value for value in values if value))
    return all(token in haystack for token in marketplace_search_tokens(query))


def _skill_catalog_error(message: str) -> CommandOutcome:
    payload = json.dumps(
        {
            "schema_version": "1",
            "kind": "error",
            "error": message,
        },
        indent=2,
        sort_keys=True,
    )
    outcome = CommandOutcome(direct_response=payload)
    outcome.add_message(payload, verbatim=True, right_info="skills")
    return outcome


async def handle_list_marketplace_skills(
    ctx: CommandContext,
    *,
    agent_name: str,
    query: str | None = None,
    marketplace_url_override: str | None = None,
    page: int = 1,
    paginate: bool = False,
    limit: int | None = None,
    output: SkillsCatalogFormat = "full",
    action: str = "available",
) -> CommandOutcome:
    outcome = CommandOutcome()

    resolver = SkillSourceResolver(ctx, agent_name=agent_name)
    resolution = await resolver.active_source(override=marketplace_url_override)
    if resolution.source is None:
        message = resolution.error or "Skill source is not available."
        if output == "json":
            return _skill_catalog_error(message)
        outcome.add_message(message, channel="error")
        return outcome
    source = resolution.source
    command_registry = (
        mcp_registry_source(source.ref.server_name)
        if source.ref.kind == "mcp" and source.ref.server_name is not None
        else marketplace_url_override
    )
    try:
        all_entries = await source.list_skills()
    except Exception as exc:
        message = f"Failed to load skills: {exc}"
        if output == "json":
            return _skill_catalog_error(message)
        outcome.add_message(message, channel="error")
        return outcome

    indexed_entries = list(enumerate(all_entries, 1))
    if query is not None:
        indexed_entries = [
            (index, entry) for index, entry in indexed_entries if _skill_matches_query(entry, query)
        ]
    entries = [entry for _index, entry in indexed_entries]
    if not entries:
        if output == "json":
            empty_page = _SkillCatalogPage(
                entries=[],
                source_indices=[],
                page=1,
                limit=limit or MODEL_SKILLS_CATALOG_LIMIT,
                total=0,
                result_start=0,
            )
            payload = _skill_catalog_json(
                source,
                empty_page,
                query=query,
                registry=command_registry,
                action=action,
                output=output,
            )
            outcome.direct_response = payload
            outcome.add_message(payload, verbatim=True, right_info="skills")
            return outcome
        outcome.add_message(source.empty_message(), channel="warning")
        return outcome

    effective_limit = limit
    if effective_limit is None and output in {"compact", "json"}:
        effective_limit = MODEL_SKILLS_CATALOG_LIMIT
    if effective_limit is None and (paginate or page > 1):
        effective_limit = MODEL_SKILLS_CATALOG_LIMIT

    if effective_limit is not None:
        catalog_page = _catalog_page(
            indexed_entries,
            page=page,
            limit=effective_limit,
        )
        if not catalog_page.entries:
            message = f"Page {page} is out of range for {len(entries)} available skills."
            if output == "json":
                return _skill_catalog_error(message)
            outcome.add_message(message, channel="warning")
            return outcome
    else:
        catalog_page = _SkillCatalogPage(
            entries=entries,
            source_indices=[index for index, _entry in indexed_entries],
            page=1,
            limit=len(entries),
            total=len(entries),
            result_start=1,
        )

    if output == "json":
        payload = _skill_catalog_json(
            source,
            catalog_page,
            query=query,
            registry=command_registry,
            action=action,
            output=output,
        )
        outcome.direct_response = payload
        outcome.add_message(payload, verbatim=True, right_info="skills", agent_name=agent_name)
        return outcome

    content = (
        _format_compact_skill_catalog(
            source,
            catalog_page,
            query=query,
            registry=command_registry,
            action=action,
            output=output,
        )
        if output == "compact"
        else _format_skill_source_list(
            source,
            catalog_page.entries,
            query=query,
            start_index=catalog_page.result_start,
            source_indices=catalog_page.source_indices,
        )
    )
    outcome.add_message(content, right_info="skills", agent_name=agent_name)
    if output == "full" and catalog_page.next_page is not None:
        next_command = _skills_catalog_command(
            action=action,
            query=query,
            registry=command_registry,
            page=catalog_page.next_page,
            limit=catalog_page.limit,
            output=output,
        )
        outcome.add_message(
            f"Next page: `{next_command}`.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
    if output == "compact":
        return outcome
    add_info_messages(
        outcome,
        (
            f"Install with `{_skills_install_command(command_registry)}`.",
            "Search available skills with `/skills search <query>`.",
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

    selection = parsed.selector
    marketplace_url = parsed.registry or marketplace_url_override
    if selection is None and not interactive:
        browse_command = skills_available_command(marketplace_url)
        outcome.add_message(
            "A skill selector is required for model-facing installation.",
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        outcome.add_message(
            f"Browse with `{browse_command}` or use `/skills search <query>`.",
            channel="info",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome

    management_scope = resolve_skills_management_scope(
        ctx.resolve_settings(),
        managed_directory_override=parsed.skills_dir or managed_directory_override,
    )
    managed_skills_dir = management_scope.managed_directory
    resolver = SkillSourceResolver(ctx, agent_name=agent_name)
    resolution = await resolver.active_source(override=marketplace_url)
    if resolution.source is None:
        outcome.add_message(resolution.error or "Skill source is not available.", channel="error")
        return outcome
    source = resolution.source

    if selection:
        return await _install_skill_from_add_selector(
            ctx,
            agent_name=agent_name,
            source=source,
            managed_skills_dir=managed_skills_dir,
            selection=selection,
            managed_directory_override=parsed.skills_dir or managed_directory_override,
        )

    try:
        entries = await source.list_skills()
    except Exception as exc:
        outcome.add_message(f"Failed to load skills: {exc}", channel="error")
        return outcome

    if not entries:
        outcome.add_message(source.empty_message(), channel="warning")
        return outcome

    selection = await _select_skill_for_add(
        ctx,
        outcome=outcome,
        source=source,
        entries=entries,
        agent_name=agent_name,
        interactive=interactive,
    )
    if selection is None:
        return outcome

    skill = await source.select_skill(selection)
    if skill is None:
        _add_skill_not_found_message(outcome, selection=selection, agent_name=agent_name)
        return outcome

    return await _install_skill_from_add_selector(
        ctx,
        agent_name=agent_name,
        source=source,
        managed_skills_dir=managed_skills_dir,
        selection=selection,
        managed_directory_override=parsed.skills_dir or managed_directory_override,
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


async def _check_skill_update_sources(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: Sequence[SkillUpdateInfo],
) -> list[SkillUpdateInfo]:
    resolver = SkillSourceResolver(ctx, agent_name=agent_name)
    groups = await resolver.update_sources(updates)
    checked_groups = await asyncio.gather(
        *(group.source.check_updates(group.updates) for group in groups)
    )
    return [update for group in checked_groups for update in group]


async def _apply_skill_update_sources(
    ctx: CommandContext,
    *,
    agent_name: str,
    updates: Sequence[SkillUpdateInfo],
    force: bool,
) -> list[SkillUpdateInfo]:
    resolver = SkillSourceResolver(ctx, agent_name=agent_name)
    groups = await resolver.update_sources(updates)
    applied_groups = await asyncio.gather(
        *(group.source.apply_updates(group.updates, force=force) for group in groups)
    )
    return [update for group in applied_groups for update in group]


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
    updates = await _check_skill_update_sources(ctx, agent_name=agent_name, updates=updates)

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

    applied = await _apply_skill_update_sources(
        ctx,
        agent_name=agent_name,
        updates=selected,
        force=parsed.force,
    )
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
    ["CommandContext", str, str | None, bool],
    Awaitable[CommandOutcome],
]


async def _handle_skills_list_action(
    ctx: "CommandContext",
    agent_name: str,
    _argument: str | None,
    _interactive: bool,
) -> CommandOutcome:
    return await handle_list_skills(ctx, agent_name=agent_name)


async def _handle_skills_available_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> CommandOutcome:
    parsed = parse_skills_catalog_options(argument)
    if parsed.error is not None:
        if parsed.output == "json":
            return _skill_catalog_error(parsed.error)
        outcome = CommandOutcome()
        outcome.add_message(parsed.error, channel="warning", right_info="skills")
        return outcome
    if strip_to_none(parsed.argument) is not None:
        message = (
            "Usage: /skills available [--registry source] [--page N] [--limit N] "
            "[--compact|--full|--json]"
        )
        if parsed.output == "json":
            return _skill_catalog_error(message)
        outcome = CommandOutcome()
        outcome.add_message(
            message,
            channel="warning",
            right_info="skills",
        )
        return outcome
    if not interactive and parsed.output == "full":
        return _skill_catalog_error(
            "Full skills listings are unavailable to model-facing commands; "
            "use compact or paginated JSON output."
        )
    output: SkillsCatalogFormat = (
        "compact" if parsed.output == "auto" and not interactive else parsed.output
    )
    if output == "auto":
        output = "full"
    return await handle_list_marketplace_skills(
        ctx,
        agent_name=agent_name,
        query=None,
        marketplace_url_override=parsed.registry,
        page=parsed.page,
        paginate=parsed.page_explicit,
        limit=parsed.limit,
        output=output,
        action="available",
    )


async def _handle_skills_search_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> CommandOutcome:
    parsed = parse_skills_catalog_options(argument)
    query = strip_to_none(parsed.argument)
    if parsed.error is not None or query is None:
        message = parsed.error or (
            "Usage: /skills search <query> [--registry source] [--page N] [--limit N] "
            "[--compact|--full|--json]"
        )
        if parsed.output == "json":
            return _skill_catalog_error(message)
        outcome = CommandOutcome()
        outcome.add_message(
            message,
            channel="warning",
            right_info="skills",
            agent_name=agent_name,
        )
        return outcome
    if not interactive and parsed.output == "full":
        return _skill_catalog_error(
            "Full skills listings are unavailable to model-facing commands; "
            "use compact or paginated JSON output."
        )
    output: SkillsCatalogFormat = (
        "compact" if parsed.output == "auto" and not interactive else parsed.output
    )
    if output == "auto":
        output = "full"
    return await handle_list_marketplace_skills(
        ctx,
        agent_name=agent_name,
        query=query,
        marketplace_url_override=parsed.registry,
        page=parsed.page,
        paginate=parsed.page_explicit,
        limit=parsed.limit,
        output=output,
        action="search",
    )


async def _handle_skills_add_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> CommandOutcome:
    return await handle_add_skill(
        ctx,
        agent_name=agent_name,
        argument=argument,
        interactive=interactive,
    )


async def _handle_skills_registry_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    _interactive: bool,
) -> CommandOutcome:
    return await handle_set_skills_registry(ctx, agent_name=agent_name, argument=argument)


async def _handle_skills_remove_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    interactive: bool,
) -> CommandOutcome:
    return await handle_remove_skill(
        ctx,
        agent_name=agent_name,
        argument=argument,
        interactive=interactive,
    )


async def _handle_skills_update_action(
    ctx: "CommandContext",
    agent_name: str,
    argument: str | None,
    _interactive: bool,
) -> CommandOutcome:
    return await handle_update_skill(ctx, agent_name=agent_name, argument=argument)


_SKILLS_ACTION_HANDLERS: dict[str, _SkillsActionHandler] = {
    "list": _handle_skills_list_action,
    "available": _handle_skills_available_action,
    "search": _handle_skills_search_action,
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
    interactive: bool = True,
) -> CommandOutcome:
    normalized = normalize_command_action("skills", action)

    if is_help_flag(action) or is_help_flag(argument):
        return handle_skills_help(agent_name=agent_name)

    handler = _SKILLS_ACTION_HANDLERS.get(normalized)
    if handler is not None:
        return await handler(ctx, agent_name, argument, interactive)

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
