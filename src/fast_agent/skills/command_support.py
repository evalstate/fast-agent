"""Shared helpers for skills command parsing and marketplace presentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from fast_agent.commands.command_catalog import (
    SKILLS_ADD_SELECTOR,
    command_usage_lines,
)
from fast_agent.commands.option_parsing import ParsedValueOption, ValueOption, read_value_option
from fast_agent.marketplace import formatting as marketplace_formatting
from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.commandline import join_commandline, split_commandline
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.skills.models import MarketplaceSkill

type _SkillsSlashValueName = Literal["registry", "skills_dir"]


SKILLS_ADD_HINT_CLI = f"Install with: fast-agent skills add <{SKILLS_ADD_SELECTOR}>"
SKILLS_ADD_HINT_SLASH = f"Install with `/skills add <{SKILLS_ADD_SELECTOR}>`."


@dataclass(frozen=True, slots=True)
class SkillsSlashOptions:
    argument: str = ""
    registry: str | None = None
    skills_dir: str | None = None
    error: str | None = None


_SKILLS_SLASH_VALUE_OPTIONS: tuple[ValueOption[_SkillsSlashValueName], ...] = (
    ValueOption("registry", ("--registry", "-r"), error_name="--registry"),
    ValueOption("skills_dir", ("--skills-dir", "--skills"), error_name="--skills-dir"),
)


def skills_usage_lines() -> list[str]:
    """Return the shared usage/help text for skills management commands."""
    return command_usage_lines("skills")


def marketplace_search_tokens(query: str) -> list[str]:
    """Split a marketplace search query into normalized search tokens."""
    try:
        tokens = split_commandline(query, syntax="posix")
    except ValueError:
        tokens = query.split()
    return [normalized for token in tokens if (normalized := normalize_action_token(token))]


def parse_skills_slash_options(argument: str | None) -> SkillsSlashOptions:
    """Parse common slash-only skills options from an action remainder."""
    try:
        tokens = split_commandline(argument or "", syntax="posix")
    except ValueError as exc:
        return SkillsSlashOptions(error=f"Invalid /skills arguments: {exc}")

    registry: str | None = None
    skills_dir: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(tokens):
        value_option = _read_skills_slash_value_option(tokens, index)
        if value_option.error is not None:
            return SkillsSlashOptions(error=value_option.error)
        if value_option.next_index != index:
            if value_option.name == "registry":
                if registry is not None:
                    return SkillsSlashOptions(
                        error=f"Duplicate option: {value_option.display_name}"
                    )
                registry = value_option.value
            elif value_option.name == "skills_dir":
                if skills_dir is not None:
                    return SkillsSlashOptions(
                        error=f"Duplicate option: {value_option.display_name}"
                    )
                skills_dir = value_option.value
            index = value_option.next_index
            continue

        remaining.append(tokens[index])
        index += 1

    if len(remaining) == 1:
        argument = remaining[0]
    else:
        argument = join_commandline(remaining, syntax="posix")

    return SkillsSlashOptions(
        argument=argument,
        registry=registry,
        skills_dir=skills_dir,
    )


def _read_skills_slash_value_option(
    tokens: list[str],
    index: int,
) -> ParsedValueOption[_SkillsSlashValueName]:
    return read_value_option(tokens, index, _SKILLS_SLASH_VALUE_OPTIONS)


def filter_marketplace_skills(
    marketplace: "Sequence[MarketplaceSkill]",
    query: str,
) -> list["MarketplaceSkill"]:
    """Filter marketplace skills by query tokens across key descriptive fields."""
    tokens = marketplace_search_tokens(query)
    if not tokens:
        return list(marketplace)

    filtered: list[MarketplaceSkill] = []
    for entry in marketplace:
        haystack = " ".join(
            value
            for value in (
                entry.name,
                entry.install_dir_name,
                entry.description or "",
                entry.bundle_name or "",
                entry.bundle_description or "",
            )
            if value
        )
        haystack = strip_casefold(haystack)
        if all(token in haystack for token in tokens):
            filtered.append(entry)
    return filtered


def marketplace_repository_hint(marketplace: "Sequence[MarketplaceSkill]") -> str | None:
    """Return a concise repository hint for a marketplace listing."""
    if not marketplace:
        return None
    return marketplace_formatting.format_source_location(
        marketplace[0].repo_url,
        marketplace[0].repo_ref,
    )
