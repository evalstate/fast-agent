"""Shared command catalog helpers."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Final

from fast_agent.utils.action_normalization import normalize_action_token
from fast_agent.utils.collections import unique_preserve_order

SKILLS_ADD_SELECTOR: Final = "number|name|github-url|path"


@dataclass(frozen=True, slots=True)
class CommandArgumentSpec:
    """Metadata for a positional action argument."""

    name: str
    summary: str
    value_name: str | None = None
    required: bool = False


@dataclass(frozen=True, slots=True)
class CommandOptionSpec:
    """Metadata for an action option."""

    name: str
    summary: str
    value_name: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandActionSpec:
    """Metadata for a canonical command action."""

    action: str
    help: str
    aliases: tuple[str, ...] = ()
    usage: str | None = None
    examples: tuple[str, ...] = ()
    arguments: tuple[CommandArgumentSpec, ...] = ()
    options: tuple[CommandOptionSpec, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Metadata for a command family."""

    command: str
    summary: str
    usage: str
    actions: tuple[CommandActionSpec, ...]
    default_action: str
    examples: tuple[str, ...] = ()


def _model_catalog_action(command: str, *, example_provider: str) -> CommandActionSpec:
    return CommandActionSpec(
        action="catalog",
        help="Show model catalog for a provider",
        usage=f"/{command} catalog <provider> [--all]",
        examples=(f"/{command} catalog {example_provider} --all",),
        arguments=(
            CommandArgumentSpec(
                name="provider",
                value_name="provider",
                summary="Provider name, for example anthropic or openai.",
                required=True,
            ),
        ),
        options=(
            CommandOptionSpec(
                name="--all",
                summary="Show the full provider catalog instead of the curated default view.",
            ),
        ),
    )


COMMAND_SPECS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command="skills",
        summary="Manage local skills",
        usage="/skills [list|available|search|add|remove|update|registry|help] [args]",
        actions=(
            CommandActionSpec(action="list", help="List local skills", usage="/skills list"),
            CommandActionSpec(
                action="available",
                aliases=("marketplace", "browse"),
                help="Browse marketplace skills",
                usage="/skills available",
                examples=("/skills available",),
            ),
            CommandActionSpec(
                action="search",
                aliases=("find",),
                help="Search marketplace skills",
                usage="/skills search <query>",
                examples=("/skills search docker",),
                arguments=(
                    CommandArgumentSpec(
                        name="query",
                        value_name="query",
                        summary="Search query.",
                        required=True,
                    ),
                ),
            ),
            CommandActionSpec(
                action="add",
                aliases=("install",),
                help="Install a skill",
                usage=f"/skills add [<{SKILLS_ADD_SELECTOR}>] [--registry url] [--skills-dir path]",
                examples=(
                    f"/skills add <{SKILLS_ADD_SELECTOR}>",
                    "/skills add https://github.com/org/repo/blob/main/skills/example/SKILL.md",
                    "/skills add ./skills/example",
                ),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name=SKILLS_ADD_SELECTOR,
                        summary="Skill name, marketplace index, GitHub SKILL.md URL, or local path.",
                    ),
                ),
                options=(
                    CommandOptionSpec(
                        name="--registry",
                        aliases=("-r",),
                        value_name="url|path",
                        summary="Override the skills registry for this invocation.",
                    ),
                    CommandOptionSpec(
                        name="--skills-dir",
                        value_name="path",
                        summary="Override the managed skills directory for this invocation.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="remove",
                aliases=("rm", "delete", "uninstall"),
                help="Remove a local skill",
                usage="/skills remove [<number|name>] [--skills-dir path]",
                examples=("/skills remove <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Installed skill name or index.",
                    ),
                ),
                options=(
                    CommandOptionSpec(
                        name="--skills-dir",
                        value_name="path",
                        summary="Override the managed skills directory for this invocation.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="update",
                aliases=("refresh", "upgrade"),
                help="Check or apply skill updates",
                usage="/skills update [<number|name|all>] [--skills-dir path] [--force] [--yes]",
                examples=("/skills update all --yes",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name|all",
                        summary="Skill name, index, or 'all'. Omit to run an update check.",
                    ),
                ),
                options=(
                    CommandOptionSpec(
                        name="--skills-dir",
                        value_name="path",
                        summary="Override the managed skills directory for this invocation.",
                    ),
                    CommandOptionSpec(
                        name="--force",
                        summary="Overwrite local modifications.",
                    ),
                    CommandOptionSpec(
                        name="--yes",
                        summary="Confirm multi-skill apply.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="registry",
                aliases=("source",),
                help="Set the skills registry",
                usage="/skills registry [<number|url|path>]",
                examples=("/skills registry",),
                arguments=(
                    CommandArgumentSpec(
                        name="target",
                        value_name="number|url|path",
                        summary="Registry selection, URL, or filesystem path.",
                    ),
                ),
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
            "/skills search docker",
            f"/skills add <{SKILLS_ADD_SELECTOR}>",
            "/skills add https://github.com/org/repo/blob/main/skills/example/SKILL.md",
            "/skills add ./skills/example",
            "/skills registry",
        ),
    ),
    CommandSpec(
        command="cards",
        summary="Manage card packs",
        usage="/cards [list|add|remove|readme|update|publish|registry|help] [args]",
        actions=(
            CommandActionSpec(
                action="list",
                help="List installed card packs",
                usage="/cards list",
            ),
            CommandActionSpec(
                action="add",
                aliases=("install",),
                help="Install a card pack",
                usage="/cards add [<number|name>] [--registry url] [--force]",
                examples=("/cards add <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Card pack name or marketplace index.",
                    ),
                ),
                options=(
                    CommandOptionSpec(
                        name="--registry",
                        aliases=("-r",),
                        value_name="url|path",
                        summary="Override the card registry for this invocation.",
                    ),
                    CommandOptionSpec(
                        name="--force",
                        summary="Overwrite files owned by other packs.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="remove",
                aliases=("rm", "delete", "uninstall"),
                help="Remove an installed card pack",
                usage="/cards remove [<number|name>]",
                examples=("/cards remove <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Installed card pack name or index.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="readme",
                aliases=("show", "cat"),
                help="Show an installed card pack README",
                usage="/cards readme [<number|name>]",
                examples=("/cards readme <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Installed card pack name or index.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="update",
                aliases=("refresh", "upgrade"),
                help="Check or apply card pack updates",
                usage="/cards update [<number|name|all>] [--force] [--yes]",
                examples=("/cards update all --yes",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name|all",
                        summary="Card pack name, index, or 'all'. Omit to run an update check.",
                    ),
                ),
                options=(
                    CommandOptionSpec(name="--force", summary="Overwrite local modifications."),
                    CommandOptionSpec(name="--yes", summary="Confirm multi-pack apply."),
                ),
            ),
            CommandActionSpec(
                action="publish",
                help="Publish local card pack changes",
                usage=(
                    "/cards publish [<number|name>] [--no-push] [--message text] "
                    "[--temp-dir path] [--keep-temp]"
                ),
                examples=("/cards publish <number|name> --no-push",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Installed card pack name or index.",
                    ),
                ),
                options=(
                    CommandOptionSpec(name="--no-push", summary="Commit locally but skip git push."),
                    CommandOptionSpec(
                        name="--message",
                        aliases=("-m",),
                        value_name="text",
                        summary="Commit message for published changes.",
                    ),
                    CommandOptionSpec(
                        name="--temp-dir",
                        value_name="path",
                        summary="Directory for temporary clone checkout when the source repo is remote.",
                    ),
                    CommandOptionSpec(
                        name="--keep-temp",
                        summary="Retain the temporary clone checkout on disk.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="registry",
                aliases=("marketplace", "source"),
                help="Set the card-pack registry",
                usage="/cards registry [<number|url|path>]",
                examples=("/cards registry",),
                arguments=(
                    CommandArgumentSpec(
                        name="target",
                        value_name="number|url|path",
                        summary="Registry selection, URL, or filesystem path.",
                    ),
                ),
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
            "/cards readme <number|name>",
            "/cards update all --yes",
            "/cards registry",
        ),
    ),
    CommandSpec(
        command="plugins",
        summary="Manage command plugins",
        usage="/plugins [list|available|add|remove|update|registry|help] [args]",
        actions=(
            CommandActionSpec(
                action="list",
                help="List installed plugins",
                usage="/plugins list",
            ),
            CommandActionSpec(
                action="available",
                aliases=("marketplace", "browse"),
                help="Browse marketplace plugins",
                usage="/plugins available",
                examples=("/plugins available",),
            ),
            CommandActionSpec(
                action="add",
                aliases=("install",),
                help="Install a plugin",
                usage="/plugins add [<number|name>] [--registry url]",
                examples=("/plugins add <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Plugin name or marketplace index.",
                    ),
                ),
                options=(
                    CommandOptionSpec(
                        name="--registry",
                        aliases=("-r",),
                        value_name="url|path",
                        summary="Override the plugin registry for this invocation.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="remove",
                aliases=("rm", "delete", "uninstall"),
                help="Remove an installed plugin",
                usage="/plugins remove [<number|name>]",
                examples=("/plugins remove <number|name>",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name",
                        summary="Installed plugin name or index.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="update",
                aliases=("refresh", "upgrade"),
                help="Check or apply plugin updates",
                usage="/plugins update [<number|name|all>] [--force] [--yes]",
                examples=("/plugins update all --yes",),
                arguments=(
                    CommandArgumentSpec(
                        name="selector",
                        value_name="number|name|all",
                        summary="Plugin name, index, or 'all'. Omit to run an update check.",
                    ),
                ),
                options=(
                    CommandOptionSpec(name="--force", summary="Overwrite local modifications."),
                    CommandOptionSpec(name="--yes", summary="Confirm multi-plugin apply."),
                ),
            ),
            CommandActionSpec(
                action="registry",
                aliases=("source",),
                help="Set the plugin registry",
                usage="/plugins registry [<number|url|path>]",
                examples=("/plugins registry",),
                arguments=(
                    CommandArgumentSpec(
                        name="target",
                        value_name="number|url|path",
                        summary="Registry selection, URL, or filesystem path.",
                    ),
                ),
            ),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show plugins command usage",
            ),
        ),
        default_action="list",
        examples=(
            "/plugins available",
            "/plugins add <number|name>",
            "/plugins remove <number|name>",
            "/plugins update all --yes",
            "/plugins registry",
        ),
    ),
    CommandSpec(
        command="model",
        summary="Model inspection, switching, and runtime settings",
        usage="/model [reasoning|task_budget|verbosity|fast|web_search|x_search|web_fetch|switch|doctor|references|catalog|help] [args]",
        actions=(
            CommandActionSpec(
                action="reasoning",
                help="Inspect or set reasoning effort",
                usage="/model reasoning [adaptive|off|low|medium|high|xhigh|max|<budget>]",
                examples=("/model reasoning high",),
            ),
            CommandActionSpec(
                action="task_budget",
                help="Inspect or set task budget",
                usage="/model task_budget [off|20k|64k|128k|256k]",
                examples=("/model task_budget 64k",),
            ),
            CommandActionSpec(
                action="verbosity",
                help="Inspect or set text verbosity",
                usage="/model verbosity [low|medium|high]",
                examples=("/model verbosity high",),
            ),
            CommandActionSpec(
                action="fast",
                help="Inspect or set service tier",
                usage="/model fast [on|off|flex|status]",
                examples=("/model fast on",),
            ),
            CommandActionSpec(
                action="web_search",
                help="Inspect or set web search state",
                usage="/model web_search [on|off|default]",
                examples=("/model web_search off",),
            ),
            CommandActionSpec(
                action="x_search",
                help="Inspect or set X Search state",
                usage="/model x_search [on|off|default]",
                examples=("/model x_search on",),
            ),
            CommandActionSpec(
                action="web_fetch",
                help="Inspect or set web fetch state",
                usage="/model web_fetch [on|off|default]",
                examples=("/model web_fetch off",),
            ),
            CommandActionSpec(
                action="switch",
                help="Switch model (starts a new session)",
                usage="/model switch [<name>]",
                examples=("/model switch",),
                arguments=(
                    CommandArgumentSpec(
                        name="name",
                        value_name="name",
                        summary="Model reference or provider model identifier.",
                    ),
                ),
                notes=("Switching models starts a new session to avoid mixing histories.",),
            ),
            CommandActionSpec(
                action="doctor",
                help="Inspect model onboarding readiness",
                usage="/model doctor",
                examples=("/model doctor",),
            ),
            CommandActionSpec(
                action="references",
                help="List or manage model references",
                usage=(
                    "/model references [list] | "
                    "/model references set [<token> [<model-spec>]] "
                    "[--target env|project] [--dry-run] | "
                    "/model references unset [<token>] [--target env|project] [--dry-run]"
                ),
                examples=("/model references",),
            ),
            _model_catalog_action("model", example_provider="openai"),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show model command usage",
            ),
        ),
        default_action="reasoning",
        examples=(
            "/model task_budget 64k",
            "/model switch",
            "/model doctor",
            "/model references",
            "/model catalog openai --all",
        ),
    ),
    CommandSpec(
        command="models",
        summary="Model onboarding and reference diagnostics",
        usage="/models [doctor|references|catalog|help] [args]",
        actions=(
            CommandActionSpec(
                action="doctor",
                help="Inspect model onboarding readiness",
                usage="/models doctor",
                examples=("/models doctor",),
            ),
            CommandActionSpec(
                action="references",
                help="List or manage model references",
                usage="/models references",
                examples=("/models references",),
            ),
            _model_catalog_action("models", example_provider="anthropic"),
            CommandActionSpec(
                action="help",
                aliases=("--help", "-h"),
                help="Show models command usage",
            ),
        ),
        default_action="doctor",
        examples=(
            "/models doctor",
            "/models references",
            "/models catalog anthropic --all",
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
                usage="/check [args]",
                examples=("/check", "/check models --for-model gpt-5"),
            ),
        ),
        default_action="run",
    ),
)


_COMMAND_SPECS_BY_NAME: Final[dict[str, CommandSpec]] = {
    spec.command: spec for spec in COMMAND_SPECS
}
_COMMAND_ACTION_SPECS_BY_NAME: Final[dict[str, dict[str, CommandActionSpec]]] = {
    spec.command: {
        name: action
        for action in spec.actions
        for name in (action.action, *action.aliases)
    }
    for spec in COMMAND_SPECS
}
_COMMAND_ACTION_CANDIDATES_BY_NAME: Final[dict[str, tuple[str, ...]]] = {
    command: tuple(actions)
    for command, actions in _COMMAND_ACTION_SPECS_BY_NAME.items()
}
_COMMAND_ACTION_CANONICAL_BY_CANDIDATE: Final[dict[str, dict[str, str]]] = {
    command: {candidate: action.action for candidate, action in actions.items()}
    for command, actions in _COMMAND_ACTION_SPECS_BY_NAME.items()
}


def get_command_spec(command_name: str) -> CommandSpec | None:
    """Return catalog metadata for a command family."""

    return _COMMAND_SPECS_BY_NAME.get(normalize_action_token(command_name))


def get_command_action_spec(command_name: str, action_name: str) -> CommandActionSpec | None:
    """Return metadata for a command action, resolving aliases."""

    normalized = normalize_action_token(action_name)
    if not normalized:
        return None

    actions = _COMMAND_ACTION_SPECS_BY_NAME.get(normalize_action_token(command_name))
    if actions is None:
        return None
    return actions.get(normalized)


def command_action_names(command_name: str) -> tuple[str, ...]:
    """Return canonical action names for a command family."""

    spec = get_command_spec(command_name)
    if spec is None:
        return ()
    return tuple(action.action for action in spec.actions)


def command_action_tokens(
    command_name: str,
    action_name: str,
    *,
    include_action: bool = True,
) -> tuple[str, ...]:
    """Return the accepted tokens for one command action.

    The command catalog owns action aliases; consumers that need to recognize
    typed subcommands should use this instead of mirroring alias sets.
    """

    normalized_action = normalize_action_token(action_name)
    action = get_command_action_spec(command_name, normalized_action)
    if action is None:
        return ()
    return ((action.action,) if include_action else ()) + action.aliases


def command_usage_lines(command_name: str) -> list[str]:
    """Return user-facing usage lines for a catalogued command family."""

    spec = get_command_spec(command_name)
    if spec is None:
        raise ValueError(f"unknown command catalog entry: {command_name}")

    lines = [f"Usage: {spec.usage}"]
    if spec.examples:
        lines.extend(("", "Examples:"))
        lines.extend(f"- {example}" for example in spec.examples)
    return lines


def format_unknown_command_action(command_name: str, action: str) -> str:
    """Return a warning for an unrecognized action on a catalogued command."""

    actions = "/".join(command_action_names(command_name))
    return (
        f"Unknown /{command_name} action: {action}. "
        f"Use {actions}.{_suggestion_suffix(suggest_command_action(command_name, action))}"
    )


def _suggestion_suffix(suggestions: tuple[str, ...]) -> str:
    if not suggestions:
        return ""
    return " Did you mean: " + ", ".join(f"`{name}`" for name in suggestions)


def normalize_command_action(command_name: str, action_name: str | None) -> str:
    """Return a canonical action name for a command family action or alias."""
    raw_action = normalize_action_token(action_name)
    spec = get_command_spec(command_name)
    if spec is None:
        return raw_action
    if not raw_action:
        return spec.default_action
    action_spec = get_command_action_spec(command_name, raw_action)
    return action_spec.action if action_spec is not None else raw_action


def suggest_command_name(command_name: str, *, limit: int = 3) -> tuple[str, ...]:
    """Suggest similar top-level command names."""

    normalized = normalize_action_token(command_name)
    candidates = [spec.command for spec in COMMAND_SPECS] + [
        "commands",
        "mcp",
        "model",
        "session",
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

    normalized = normalize_action_token(action)
    if not normalized:
        return ()

    spec = get_command_spec(command_name)
    if spec is None:
        return ()

    candidates = _COMMAND_ACTION_CANDIDATES_BY_NAME.get(spec.command, ())
    matches = difflib.get_close_matches(normalized, candidates, n=limit, cutoff=0.5)
    canonical_by_candidate = _COMMAND_ACTION_CANONICAL_BY_CANDIDATE[spec.command]
    return tuple(unique_preserve_order(canonical_by_candidate[match] for match in matches))
