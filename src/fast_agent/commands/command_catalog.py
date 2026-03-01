"""Shared command catalog helpers."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CommandActionSpec:
    """Metadata for a canonical command action."""

    action: str
    help: str
    aliases: tuple[str, ...] = ()
    usage: str | None = None
    examples: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Metadata for a command family."""

    command: str
    summary: str
    usage: str
    actions: tuple[CommandActionSpec, ...]
    default_action: str
    examples: tuple[str, ...] = ()


COMMAND_SPECS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command="skills",
        summary="Manage local skills",
        usage="/skills [list|available|search|add|remove|update|registry|help] [args]",
        actions=(
            CommandActionSpec(action="list", help="List local skills"),
            CommandActionSpec(
                action="available",
                aliases=("marketplace", "browse"),
                help="Browse marketplace skills",
            ),
            CommandActionSpec(
                action="search",
                aliases=("find",),
                help="Search marketplace skills",
                usage="/skills search <query>",
                examples=("/skills search docker",),
            ),
            CommandActionSpec(action="add", aliases=("install",), help="Install a skill"),
            CommandActionSpec(
                action="remove",
                aliases=("rm", "delete", "uninstall"),
                help="Remove a local skill",
            ),
            CommandActionSpec(
                action="update",
                aliases=("refresh", "upgrade"),
                help="Check or apply skill updates",
            ),
            CommandActionSpec(
                action="registry",
                aliases=("source",),
                help="Set the skills registry",
                usage="/skills registry [<number|url|path>]",
            ),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show skills command usage",
            ),
        ),
        default_action="list",
        examples=(
            "/skills available",
            "/skills add <number|name>",
            "/skills registry",
        ),
    ),
    CommandSpec(
        command="cards",
        summary="Manage card packs",
        usage="/cards [list|add|remove|update|publish|registry|help] [args]",
        actions=(
            CommandActionSpec(action="list", help="List installed card packs"),
            CommandActionSpec(action="add", aliases=("install",), help="Install a card pack"),
            CommandActionSpec(
                action="remove",
                aliases=("rm", "delete", "uninstall"),
                help="Remove an installed card pack",
            ),
            CommandActionSpec(
                action="update",
                aliases=("refresh", "upgrade"),
                help="Check or apply card pack updates",
            ),
            CommandActionSpec(action="publish", help="Publish local card pack changes"),
            CommandActionSpec(
                action="registry",
                aliases=("marketplace", "source"),
                help="Set the card-pack registry",
                usage="/cards registry [<number|url|path>]",
            ),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show cards command usage",
            ),
        ),
        default_action="list",
        examples=(
            "/cards add <number|name>",
            "/cards update all --yes",
            "/cards registry",
        ),
    ),
    CommandSpec(
        command="models",
        summary="Model onboarding state",
        usage="/models [doctor|aliases|catalog|help] [args]",
        actions=(
            CommandActionSpec(action="doctor", help="Inspect model onboarding readiness"),
            CommandActionSpec(action="aliases", aliases=("alias",), help="List or manage model aliases"),
            CommandActionSpec(action="catalog", help="Show model catalog for a provider"),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show models command usage",
            ),
        ),
        default_action="doctor",
        examples=(
            "/models doctor",
            "/models aliases list",
            "/models catalog openai --all",
        ),
    ),
    CommandSpec(
        command="check",
        summary="Config diagnostics",
        usage="/check [args]",
        actions=(
            CommandActionSpec(
                action="run",
                help="Run fast-agent check diagnostics",
            ),
        ),
        default_action="run",
    ),
)


_COMMAND_SPECS_BY_NAME: Final[dict[str, CommandSpec]] = {
    spec.command: spec for spec in COMMAND_SPECS
}


def get_command_spec(command_name: str) -> CommandSpec | None:
    """Return catalog metadata for a command family."""

    return _COMMAND_SPECS_BY_NAME.get(command_name.strip().lower())


def command_action_names(command_name: str) -> tuple[str, ...]:
    """Return canonical action names for a command family."""

    spec = get_command_spec(command_name)
    if spec is None:
        return ()
    return tuple(action.action for action in spec.actions)


def command_alias_map(command_name: str) -> dict[str, str]:
    """Return alias-to-action mapping for a command family."""

    spec = get_command_spec(command_name)
    if spec is None:
        return {}

    aliases: dict[str, str] = {}
    for action in spec.actions:
        for alias in action.aliases:
            aliases[alias.lower()] = action.action
    return aliases


def suggest_command_name(command_name: str, *, limit: int = 3) -> tuple[str, ...]:
    """Suggest similar top-level command names."""

    normalized = command_name.strip().lower()
    candidates = [spec.command for spec in COMMAND_SPECS] + [
        "commands",
        "mcp",
        "model",
        "tools",
        "prompts",
        "usage",
        "system",
        "markdown",
    ]
    matches = difflib.get_close_matches(normalized, candidates, n=limit, cutoff=0.5)
    return tuple(matches)


def suggest_command_action(command_name: str, action: str, *, limit: int = 3) -> tuple[str, ...]:
    """Suggest similar action names for a command family."""

    normalized = action.strip().lower()
    if not normalized:
        return ()

    spec = get_command_spec(command_name)
    if spec is None:
        return ()

    candidates = [entry.action for entry in spec.actions]
    for entry in spec.actions:
        candidates.extend(entry.aliases)
    matches = difflib.get_close_matches(normalized, candidates, n=limit, cutoff=0.5)
    return tuple(matches)
