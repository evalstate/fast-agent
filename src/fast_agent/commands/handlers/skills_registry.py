"""Skills registry command handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.handlers._marketplace_argument_parsing import (
    optional_selector,
    resolve_registry_argument,
)
from fast_agent.commands.handlers._text_formatting import append_indexed_current_line
from fast_agent.commands.results import CommandOutcome
from fast_agent.skills.configuration import (
    format_marketplace_display_url,
    get_marketplace_url,
    resolve_skill_registries,
)
from fast_agent.skills.operations import fetch_marketplace_skills_with_source
from fast_agent.skills.source_resolver import (
    SkillSourceResolver,
    find_mcp_registry,
    mcp_registry_server_name,
    mcp_registry_source,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from fast_agent.commands.context import CommandContext
    from fast_agent.skills.mcp_registry import McpSkillRegistry


def _append_registry_entry(
    content: Text,
    *,
    display_url: str,
    index: int,
    is_current: bool,
) -> None:
    append_indexed_current_line(content, index, display_url, is_current=is_current)


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
    current_mcp_server = mcp_registry_server_name(current_url)
    current_in_configured = current_display in configured_displays or (
        current_mcp_server is not None
        and find_mcp_registry(list(mcp_registries), current_mcp_server) is not None
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


async def _list_mcp_skill_registries(
    ctx: "CommandContext", *, agent_name: str | None
) -> list[McpSkillRegistry]:
    if agent_name is None:
        return []
    resolver = SkillSourceResolver(ctx, agent_name=agent_name)
    return await resolver.mcp_registries()


async def handle_set_skills_registry(
    ctx: "CommandContext",
    *,
    argument: str | None,
    agent_name: str | None = None,
    fetch_skills_with_source: Callable[
        [str], Awaitable[tuple[Sequence[object], str]]
    ] = fetch_marketplace_skills_with_source,
) -> CommandOutcome:
    outcome = CommandOutcome()
    active_agent_name = agent_name or ctx.current_agent_name
    settings = ctx.resolve_settings()
    configured_urls = [
        url
        for url in resolve_skill_registries(settings)
        if mcp_registry_server_name(url) is None
    ]
    mcp_registries = await _list_mcp_skill_registries(ctx, agent_name=agent_name)

    registry_arg = optional_selector(argument)
    if registry_arg is None:
        outcome.add_message(
            _format_skills_registry_overview(
                current_url=(
                    ctx.active_skill_source(active_agent_name) or get_marketplace_url(settings)
                ),
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
            url = mcp_registry_source(selected_mcp.server_name)
        else:
            url = _resolve_skills_registry_argument(registry_arg, configured_urls, outcome)
    else:
        explicit_mcp_server = mcp_registry_server_name(registry_arg) or registry_arg
        selected_mcp = find_mcp_registry(mcp_registries, explicit_mcp_server)
        url = mcp_registry_source(selected_mcp.server_name) if selected_mcp else None
        if url is None:
            url = _resolve_skills_registry_argument(registry_arg, configured_urls, outcome)

    if url is None:
        return outcome

    if selected_mcp is not None:
        ctx.set_active_skill_source(active_agent_name, url)
        content = Text()
        content.append_text(
            Text(f"Registry set to: {selected_mcp.display_name}", style="green")
        )
        content.append("\n")
        content.append_text(Text(f"Skills discovered: {len(selected_mcp.skills)}", style="dim"))
        outcome.add_message(content, right_info="skills", agent_name=agent_name)
        return outcome

    try:
        marketplace, resolved_url = await fetch_skills_with_source(url)
    except Exception as exc:
        outcome.add_message(f"Failed to load registry: {exc}", channel="error")
        return outcome

    if not marketplace:
        _add_empty_skills_registry_warning(outcome, url)
        return outcome

    ctx.clear_active_skill_source(active_agent_name)
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
