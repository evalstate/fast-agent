"""Skills slash command handlers."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, cast

from acp.helpers import text_block, tool_content
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.commands.handlers import skills as skills_handlers
from fast_agent.commands.renderers.skills_markdown import (
    render_marketplace_skills,
    render_skill_list,
    render_skills_by_directory,
    render_skills_registry_overview,
    render_skills_remove_list,
)
from fast_agent.config import get_settings
from fast_agent.core.instruction_refresh import rebuild_agent_instruction
from fast_agent.skills import SKILLS_DEFAULT
from fast_agent.skills.manager import (
    candidate_marketplace_urls,
    fetch_marketplace_skills,
    fetch_marketplace_skills_with_source,
    format_marketplace_display_url,
    get_manager_directory,
    get_marketplace_url,
    list_local_skills,
    order_skill_directories_for_display,
    reload_skill_manifests,
    resolve_skill_directories,
    resolve_skill_registries,
)
from fast_agent.skills.registry import format_skills_for_prompt

if TYPE_CHECKING:
    from fast_agent.acp.command_io import ACPCommandIO
    from fast_agent.acp.slash_commands import SlashCommandHandler
    from fast_agent.interfaces import AgentProtocol
    from fast_agent.skills.manager import MarketplaceSkill


def _skills_usage_text() -> str:
    return (
        "Usage: /skills [list|available|search|add|remove|update|registry|help] [args]\n\n"
        "Examples:\n"
        "- /skills available\n"
        "- /skills search docker\n"
        "- /skills add <number|name>\n"
        "- /skills registry"
    )


def _marketplace_search_tokens(query: str) -> list[str]:
    return [token.lower() for token in query.split() if token.strip()]


def _filter_marketplace(marketplace: list[MarketplaceSkill], query: str) -> list[MarketplaceSkill]:
    tokens = _marketplace_search_tokens(query)
    if not tokens:
        return marketplace

    filtered: list[MarketplaceSkill] = []
    for entry in marketplace:
        haystack = " ".join(
            str(getattr(entry, attr, ""))
            for attr in ("name", "description", "bundle_name", "bundle_description")
        ).lower()
        if all(token in haystack for token in tokens):
            filtered.append(entry)
    return filtered


async def handle_skills_available(
    handler: "SlashCommandHandler",
    *,
    query: str | None = None,
) -> str:
    heading = "skills available" if not query else "skills search"
    marketplace_url = get_marketplace_url(get_settings())
    display_url = format_marketplace_display_url(marketplace_url)
    try:
        marketplace = await fetch_marketplace_skills(marketplace_url)
    except Exception as exc:  # noqa: BLE001
        return (
            f"# {heading}\n\n"
            f"Failed to load marketplace: {exc}\n\n"
            f"Repository: `{display_url}`"
        )

    if not marketplace:
        return f"# {heading}\n\nNo skills found in the marketplace."

    selected_marketplace: list[MarketplaceSkill] = list(marketplace)
    if query and query.strip():
        selected_marketplace = _filter_marketplace(list(marketplace), query)
        if not selected_marketplace:
            return (
                "# skills search\n\n"
                f"No skills matched query `{query.strip()}`.\n\n"
                "Try `/skills available` to browse all skills."
            )

    repository = display_url
    repo_url = getattr(marketplace[0], "repo_url", None)
    if repo_url:
        repo_ref = getattr(marketplace[0], "repo_ref", None)
        repository = f"{repo_url}@{repo_ref}" if repo_ref else repo_url

    rendered = render_marketplace_skills(
        selected_marketplace,
        heading=heading,
        repository=repository,
    )
    if query and query.strip():
        rendered = "\n".join(
            [
                rendered,
                "",
                "Install filtered results with `/skills add <name>`. ",
            ]
        )
    return rendered


async def handle_skills(handler: "SlashCommandHandler", arguments: str | None = None) -> str:
    tokens = (arguments or "").strip().split(maxsplit=1)
    action = tokens[0].lower() if tokens else "list"
    remainder = tokens[1] if len(tokens) > 1 else ""

    if action in {"help", "--help", "-h"}:
        return _skills_usage_text()

    if action in {"list", ""}:
        if remainder.strip().lower() in {"help", "--help", "-h"}:
            return _skills_usage_text()
        return handle_skills_list(handler)
    if action in {"available", "browse", "marketplace"}:
        return await handle_skills_available(handler)
    if action in {"search", "find"}:
        query = remainder.strip()
        if not query:
            return "# skills search\n\nUsage: /skills search <query>"
        return await handle_skills_available(handler, query=query)
    if action in {"add", "install"}:
        return await handle_skills_add(handler, remainder)
    if action in {"registry", "source"}:
        return await handle_skills_registry(handler, remainder)
    if action in {"remove", "rm", "delete", "uninstall"}:
        return await handle_skills_remove(handler, remainder)
    if action in {"update", "refresh", "upgrade"}:
        return await handle_skills_update(handler, remainder)

    return (
        "Unknown /skills action. "
        "Use `/skills list`, `/skills available`, `/skills search`, `/skills add`, "
        "`/skills remove`, `/skills update`, or `/skills registry`."
    )


async def handle_skills_registry(handler: "SlashCommandHandler", argument: str) -> str:
    heading = "# skills registry"
    argument = argument.strip()

    settings = get_settings()
    configured_urls = resolve_skill_registries(settings)

    if not argument:
        current = get_marketplace_url(settings)
        display_current = format_marketplace_display_url(current)
        display_registries = [format_marketplace_display_url(url) for url in configured_urls]
        return render_skills_registry_overview(
            heading="skills registry",
            current_registry=display_current,
            configured_urls=display_registries,
        )

    if argument.isdigit():
        index = int(argument)
        if not configured_urls:
            return f"{heading}\n\nNo registries configured."
        if 1 <= index <= len(configured_urls):
            url = configured_urls[index - 1]
        else:
            return f"{heading}\n\nInvalid registry number. Use 1-{len(configured_urls)}."
    else:
        url = argument

    candidates = candidate_marketplace_urls(url)
    try:
        marketplace, resolved_url = await fetch_marketplace_skills_with_source(url)
    except Exception as exc:  # noqa: BLE001
        display_url = format_marketplace_display_url(url)
        handler._logger.warning(
            "Failed to load skills registry",
            data={
                "registry": url,
                "candidates": candidates,
                "error": str(exc),
            },
        )
        return "\n".join(
            [
                heading,
                "",
                f"Failed to load registry: {exc}",
                f"Registry: {display_url}",
            ]
        )

    if not marketplace:
        display_url = format_marketplace_display_url(url)
        return "\n".join(
            [
                heading,
                "",
                "No skills found in the registry; registry unchanged.",
                f"Registry: {display_url}",
            ]
        )

    settings.skills.marketplace_url = resolved_url

    display_url = format_marketplace_display_url(resolved_url)
    if candidates:
        handler._logger.debug(
            "Resolved skills registry",
            data={
                "input": url,
                "resolved": resolved_url,
                "candidates": candidates,
            },
        )
    response_lines = [
        heading,
        "",
        f"Registry set to: `{display_url}`",
        "",
        f"Skills discovered: {len(marketplace)}",
    ]

    return "\n".join(response_lines)


def handle_skills_list(handler: "SlashCommandHandler") -> str:
    settings = get_settings()
    directories = order_skill_directories_for_display(
        resolve_skill_directories(settings),
        settings=settings,
    )
    all_manifests = {directory: list_local_skills(directory) if directory.exists() else [] for directory in directories}
    response = render_skills_by_directory(all_manifests, heading="skills", cwd=Path.cwd())
    override_section = skills_override_section(handler)
    if override_section:
        return "\n".join([response, "", override_section])
    return response


def skills_override_section(handler: "SlashCommandHandler") -> str | None:
    agent = handler._get_current_agent()
    if not agent:
        return None
    config = getattr(agent, "config", None)
    if not config:
        return None
    if getattr(config, "skills", SKILLS_DEFAULT) is SKILLS_DEFAULT:
        return None
    manifests = list(getattr(config, "skill_manifests", []) or [])
    sources: list[str] = []
    for manifest in manifests:
        path = getattr(manifest, "path", None)
        if not path:
            continue
        source_path = path.parent if Path(path).is_file() else Path(path)
        try:
            display_path = source_path.relative_to(Path.cwd())
        except ValueError:
            display_path = source_path
        sources.append(str(display_path))
    sources = sorted(set(sources))
    lines = [
        "## Active agent skills (override)",
        "",
        "Note: this agent has an explicit skills configuration. `/skills` lists global skills directories from settings, not per-agent overrides.",
        "Update settings.skills.directories or the --skills flag to change this list.",
    ]
    if sources:
        sources_list = ", ".join(f"`{source}`" for source in sources)
        lines.extend(["", f"Sources: {sources_list}"])
    lines.append("")
    if not manifests:
        lines.append("No skills configured for this agent.")
    else:
        lines.append("Configured skills:")
        lines.extend(render_skill_list(manifests, cwd=Path.cwd()))
    return "\n".join(lines)


async def handle_skills_add(handler: "SlashCommandHandler", argument: str) -> str:
    if argument.strip().lower() in {"q", "quit", "exit"}:
        return "Cancelled."

    agent, error = handler._get_current_agent_or_error("# skills add")
    if error:
        return error
    assert agent is not None

    tool_call_id = build_tool_call_id()
    await send_skills_update(
        handler,
        agent,
        tool_call_id,
        title="Install skill",
        status="in_progress",
        message="Fetching marketplace…",
        start=True,
    )

    argument_value = argument.strip() or None

    if not argument_value:
        marketplace_url = get_marketplace_url(get_settings())
        display_url = format_marketplace_display_url(marketplace_url)
        try:
            marketplace = await fetch_marketplace_skills(marketplace_url)
        except Exception as exc:  # noqa: BLE001
            return (
                "# skills add\n\n"
                f"Failed to load marketplace: {exc}\n\n"
                f"Repository: `{display_url}`"
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

    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await skills_handlers.handle_add_skill(
            ctx,
            agent_name=handler.current_agent_name,
            argument=argument_value,
            interactive=False,
        )
    except Exception as exc:  # noqa: BLE001
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


async def handle_skills_remove(handler: "SlashCommandHandler", argument: str) -> str:
    if argument.strip().lower() in {"q", "quit", "exit"}:
        return "Cancelled."

    argument_value = argument.strip() or None
    if not argument_value:
        manager_dir = get_manager_directory()
        manifests = list_local_skills(manager_dir)
        return render_skills_remove_list(
            heading="skills remove",
            manager_dir=manager_dir,
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
        )
    except Exception as exc:  # noqa: BLE001
        return f"# skills remove\n\nFailed to remove skill: {exc}"

    return handler._format_outcome_as_markdown(outcome, "skills remove", io=io)


async def handle_skills_update(handler: "SlashCommandHandler", argument: str) -> str:
    ctx = handler._build_command_context()
    io = cast("ACPCommandIO", ctx.io)
    try:
        outcome = await skills_handlers.handle_update_skill(
            ctx,
            agent_name=handler.current_agent_name,
            argument=argument.strip() or None,
        )
    except Exception as exc:  # noqa: BLE001
        return f"# skills update\n\nFailed to update skills: {exc}"

    return handler._format_outcome_as_markdown(outcome, "skills update", io=io)


async def refresh_agent_skills(agent: "AgentProtocol") -> None:
    override_dirs = resolve_skill_directories(get_settings())
    registry, manifests = reload_skill_manifests(
        base_dir=Path.cwd(), override_directories=override_dirs
    )
    instruction_context = None
    try:
        skills_text = format_skills_for_prompt(manifests, read_tool_name="read_text_file")
        instruction_context = {"agentSkills": skills_text}
    except Exception:
        instruction_context = None

    await rebuild_agent_instruction(
        agent,
        skill_manifests=manifests,
        context=instruction_context,
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
    status: str,
    message: str | None = None,
    start: bool = False,
) -> None:
    from fast_agent.interfaces import ACPAwareProtocol

    if not isinstance(agent, ACPAwareProtocol):
        return
    acp = agent.acp
    if not acp:
        return
    try:
        if start:
            await acp.send_session_update(
                ToolCallStart(
                    tool_call_id=tool_call_id,
                    title=title,
                    kind="fetch",
                    status="in_progress",
                    session_update="tool_call",
                )
            )
        content = [tool_content(text_block(message))] if message else None
        await acp.send_session_update(
            ToolCallProgress(
                tool_call_id=tool_call_id,
                title=title,
                status=status,  # type: ignore[arg-type]
                content=content,
                session_update="tool_call_update",
            )
        )
    except Exception:
        return
