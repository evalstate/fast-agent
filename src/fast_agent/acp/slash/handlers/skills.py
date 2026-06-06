"""Skills slash command handlers."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fast_agent.acp.slash.tool_updates import (
    ToolCallStatus,
    send_fetch_tool_call_start,
    send_tool_call_progress,
)
from fast_agent.commands.command_catalog import (
    format_unknown_command_action,
    normalize_command_action,
)
from fast_agent.commands.command_discovery import render_direct_command_help
from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.renderers.skills_markdown import (
    render_marketplace_skills,
    render_skill_list,
    render_skills_by_directory,
    render_skills_remove_list,
)
from fast_agent.config import get_settings
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.skills.command_support import (
    filter_marketplace_skills,
    marketplace_repository_hint,
    parse_skills_slash_options,
    skills_usage_lines,
)
from fast_agent.skills.configuration import format_marketplace_display_url, get_marketplace_url
from fast_agent.skills.operations import (
    fetch_marketplace_skills,
    reload_skill_manifests,
)
from fast_agent.skills.registry import SkillRegistry
from fast_agent.skills.scope import (
    order_skill_directories_for_display,
    resolve_skill_directories,
    resolve_skills_management_scope,
)
from fast_agent.utils.action_normalization import (
    is_cancel_action,
    is_help_flag,
    split_action_arguments,
)
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.markdown import markdown_code_span
from fast_agent.utils.path_display import format_relative_path
from fast_agent.utils.text import strip_to_none

SkillsActionHandler = Callable[["SlashCommandHandler", str], Awaitable[str]]


def _skills_usage_text() -> str:
    return "\n".join(skills_usage_lines())


if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.interfaces import AgentProtocol


async def handle_skills_available(
    handler: "SlashCommandHandler",
    *,
    query: str | None = None,
    registry: str | None = None,
) -> str:
    normalized_query = strip_to_none(query)
    heading = "skills available" if normalized_query is None else "skills search"
    marketplace_url = registry or get_marketplace_url(get_settings())
    display_url = format_marketplace_display_url(marketplace_url)
    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:
        return (
            f"# {heading}\n\n"
            f"Failed to load marketplace: {exc}\n\n"
            f"Repository: {markdown_code_span(display_url)}"
        )

    if not marketplace:
        return f"# {heading}\n\nNo skills found in the marketplace."

    selected_marketplace = list(marketplace)
    if normalized_query is not None:
        selected_marketplace = filter_marketplace_skills(marketplace, normalized_query)
        if not selected_marketplace:
            return (
                "# skills search\n\n"
                f"No skills matched query {markdown_code_span(normalized_query)}.\n\n"
                "Try `/skills available` to browse all skills."
            )

    repository = display_url
    repo_hint = marketplace_repository_hint(marketplace)
    if repo_hint:
        repository = repo_hint

    rendered = render_marketplace_skills(
        selected_marketplace,
        heading=heading,
        repository=repository,
    )
    if normalized_query is not None:
        rendered = "\n".join(
            [
                rendered,
                "",
                "Install filtered results with `/skills add <name>`. ",
            ]
        )
    return rendered


async def handle_skills(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    if is_help_flag(arguments):
        return _skills_usage_text()

    direct_help = render_direct_command_help("skills", arguments)
    if direct_help is not None:
        return direct_help

    action, remainder = _parse_skills_action(arguments)
    action_handler = _SKILLS_ACTION_HANDLERS.get(action, _handle_unknown_skills_action)
    return await action_handler(handler, remainder)


def _parse_skills_action(arguments: str | None) -> tuple[str, str]:
    requested_action_value, remainder = split_action_arguments(arguments, default_action="")
    requested_action = requested_action_value or ""
    action = normalize_command_action("skills", requested_action)
    if action in _SKILLS_ACTION_HANDLERS:
        return action, remainder
    return "unknown", requested_action


async def _handle_skills_help_action(
    _handler: "SlashCommandHandler", _remainder: str
) -> str:
    return _skills_usage_text()


async def _handle_skills_list_action(handler: "SlashCommandHandler", remainder: str) -> str:
    if is_help_flag(remainder):
        return _skills_usage_text()
    parsed = parse_skills_slash_options(remainder)
    if parsed.error:
        return f"# skills\n\n{parsed.error}"
    return handle_skills_list(handler, skills_dir=parsed.skills_dir)


async def _handle_skills_available_action(
    handler: "SlashCommandHandler", remainder: str
) -> str:
    parsed = parse_skills_slash_options(remainder)
    if parsed.error:
        return f"# skills available\n\n{parsed.error}"
    return await handle_skills_available(handler, registry=parsed.registry)


async def _handle_skills_search_action(handler: "SlashCommandHandler", remainder: str) -> str:
    parsed = parse_skills_slash_options(remainder)
    if parsed.error:
        return f"# skills search\n\n{parsed.error}"
    query = strip_to_none(parsed.argument)
    if query is None:
        return "# skills search\n\nUsage: /skills search <query>"
    return await handle_skills_available(handler, query=query, registry=parsed.registry)


async def _handle_unknown_skills_action(
    _handler: "SlashCommandHandler", remainder: str
) -> str:
    return format_unknown_command_action("skills", remainder)


async def handle_skills_registry(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    outcome = await skills_handlers.handle_set_skills_registry(
        ctx,
        agent_name=handler.current_agent_name,
        argument=argument,
    )
    return handler._format_outcome_as_markdown(outcome, "skills registry", io=io)


def handle_skills_list(
    handler: "SlashCommandHandler",
    *,
    skills_dir: str | None = None,
) -> str:
    settings = get_settings()
    management_scope = resolve_skills_management_scope(
        settings,
        managed_directory_override=skills_dir,
    )
    discovered_directories = order_skill_directories_for_display(
        management_scope.discovered_directories,
        settings=settings,
        managed_directory_override=skills_dir,
    )
    all_manifests = {
        directory: SkillRegistry.load_directory(directory) if directory.exists() else []
        for directory in discovered_directories
    }
    response = render_skills_by_directory(all_manifests, heading="skills", cwd=Path.cwd())
    override_section = skills_override_section(handler)
    if override_section:
        return "\n".join([response, "", override_section])
    return response


def skills_override_section(handler: "SlashCommandHandler") -> str | None:
    agent = handler._get_current_agent()
    if agent is None:
        return None
    config = agent.config
    if config.skills is SKILLS_DEFAULT:
        return None
    manifests = list(config.skill_manifests)
    sources: list[str] = []
    for manifest in manifests:
        path = manifest.path
        source_path = path.parent if Path(path).is_file() else Path(path)
        sources.append(format_relative_path(source_path))
    sources = unique_preserve_order(sources)
    lines = [
        "## Active agent skills (override)",
        "",
        "Note: this agent has an explicit skills configuration. `/skills` lists global skills directories from settings, not per-agent overrides.",
        "Update settings.skills.directories or the --skills flag to change this list.",
    ]
    if sources:
        sources_list = ", ".join(markdown_code_span(source) for source in sources)
        lines.extend(["", f"Sources: {sources_list}"])
    lines.append("")
    if not manifests:
        lines.append("No skills configured for this agent.")
    else:
        lines.append("Configured skills:")
        lines.extend(render_skill_list(manifests, cwd=Path.cwd()))
    return "\n".join(lines)


async def handle_skills_add(handler: "SlashCommandHandler", argument: str) -> str:
    if is_cancel_action(argument):
        return "Cancelled."

    parsed = parse_skills_slash_options(argument)
    if parsed.error:
        return f"# skills add\n\n{parsed.error}"

    agent, error = handler._get_current_agent_or_error("# skills add")
    if error:
        return error
    if agent is None:
        return "# skills add\n\nNo agent available for this session."

    argument_value = strip_to_none(parsed.argument)
    if not argument_value:
        return await _render_skills_add_marketplace(registry=parsed.registry)

    tool_call_id = build_tool_call_id()
    await send_skills_update(
        handler,
        agent,
        tool_call_id,
        title="Install skill",
        status="in_progress",
        message="Installing skill…",
        start=True,
    )

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await skills_handlers.handle_add_skill(
            ctx,
            agent_name=handler.current_agent_name,
            argument=argument_value,
            interactive=False,
            marketplace_url_override=parsed.registry,
            managed_directory_override=parsed.skills_dir,
        )
    except Exception as exc:
        await send_skills_update(
            handler,
            agent,
            tool_call_id,
            title="Install failed",
            status="completed",
            message=str(exc),
        )
        return f"# skills add\n\nFailed to install skill: {exc}"

    if any(message.channel == "error" for message in outcome.messages):
        await send_skills_update(
            handler,
            agent,
            tool_call_id,
            title="Install failed",
            status="completed",
            message="Failed to install skill",
        )
    else:
        await send_skills_update(
            handler,
            agent,
            tool_call_id,
            title="Install complete",
            status="completed",
            message="Installed skill",
        )

    return handler._format_outcome_as_markdown(outcome, "skills add", io=io)


async def _render_skills_add_marketplace(*, registry: str | None = None) -> str:
    marketplace_url = registry or get_marketplace_url(get_settings())
    display_url = format_marketplace_display_url(marketplace_url)
    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:
        return (
            "# skills add\n\n"
            f"Failed to load marketplace: {exc}\n\n"
            f"Repository: {markdown_code_span(display_url)}"
        )

    repository = display_url
    if marketplace:
        repo_url = marketplace[0].repo_url
        repo_ref = marketplace[0].repo_ref
        repository = f"{repo_url}@{repo_ref}" if repo_ref else repo_url

    return render_marketplace_skills(
        marketplace,
        heading="skills add",
        repository=repository,
    )


async def handle_skills_remove(handler: "SlashCommandHandler", argument: str) -> str:
    if is_cancel_action(argument):
        return "Cancelled."

    parsed = parse_skills_slash_options(argument)
    if parsed.error:
        return f"# skills remove\n\n{parsed.error}"

    argument_value = strip_to_none(parsed.argument)
    if not argument_value:
        management_scope = resolve_skills_management_scope(
            get_settings(),
            managed_directory_override=parsed.skills_dir,
        )
        managed_skills_dir = management_scope.managed_directory
        manifests = SkillRegistry.load_directory(managed_skills_dir)
        return render_skills_remove_list(
            heading="skills remove",
            manager_dir=managed_skills_dir,
            manifests=manifests,
            cwd=Path.cwd(),
        )

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await skills_handlers.handle_remove_skill(
            ctx,
            agent_name=handler.current_agent_name,
            argument=argument_value,
            interactive=False,
            managed_directory_override=parsed.skills_dir,
        )
    except Exception as exc:
        return f"# skills remove\n\nFailed to remove skill: {exc}"

    return handler._format_outcome_as_markdown(outcome, "skills remove", io=io)


async def handle_skills_update(handler: "SlashCommandHandler", argument: str) -> str:
    parsed = parse_skills_slash_options(argument)
    if parsed.error:
        return f"# skills update\n\n{parsed.error}"

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await skills_handlers.handle_update_skill(
            ctx,
            agent_name=handler.current_agent_name,
            argument=strip_to_none(parsed.argument),
            managed_directory_override=parsed.skills_dir,
        )
    except Exception as exc:
        return f"# skills update\n\nFailed to update skills: {exc}"

    return handler._format_outcome_as_markdown(outcome, "skills update", io=io)


_SKILLS_ACTION_HANDLERS: dict[str, SkillsActionHandler] = {
    "help": _handle_skills_help_action,
    "list": _handle_skills_list_action,
    "available": _handle_skills_available_action,
    "search": _handle_skills_search_action,
    "add": handle_skills_add,
    "registry": handle_skills_registry,
    "remove": handle_skills_remove,
    "update": handle_skills_update,
}


async def refresh_agent_skills(agent: "AgentProtocol") -> None:
    override_dirs = resolve_skill_directories(get_settings())
    registry, manifests = reload_skill_manifests(
        base_dir=Path.cwd(), override_directories=override_dirs
    )

    await rebuild_agent_instruction(
        agent,
        skill_manifests=manifests,
        skill_registry=registry,
    )


def build_tool_call_id() -> str:
    return str(uuid.uuid4())


async def send_skills_update(
    handler: "SlashCommandHandler",
    agent: "AgentProtocol",
    tool_call_id: str,
    *,
    title: str,
    status: ToolCallStatus,
    message: str | None = None,
    start: bool = False,
) -> None:
    from fast_agent.interfaces import ACPAwareProtocol

    if not isinstance(agent, ACPAwareProtocol):
        return
    acp = agent.acp
    if not acp:
        return
    if start:
        started = await send_fetch_tool_call_start(
            acp,
            tool_call_id=tool_call_id,
            title=title,
        )
        if not started:
            return
    await send_tool_call_progress(
        acp,
        tool_call_id=tool_call_id,
        title=title,
        status=status,
        message=message,
    )
