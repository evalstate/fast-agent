"""Shared command catalog and smart-operation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class CommandActionSpec:
    """Metadata for a canonical command action."""

    action: str
    help: str
    operation_name: str | None = None


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Metadata for a command family."""

    command: str
    actions: tuple[CommandActionSpec, ...]
    default_action: str
    tool_exposed: bool = False


SmartCommandOperation = Literal[
    "skills.list",
    "skills.add",
    "skills.remove",
    "skills.update",
    "skills.registry",
    "cards.list",
    "cards.add",
    "cards.remove",
    "cards.update",
    "cards.publish",
    "cards.registry",
    "models.doctor",
    "models.aliases",
    "models.catalog",
    "check",
]


COMMAND_SPECS: Final[tuple[CommandSpec, ...]] = (
    CommandSpec(
        command="skills",
        actions=(
            CommandActionSpec(action="list", help="List local skills"),
            CommandActionSpec(action="add", help="Install a skill"),
            CommandActionSpec(action="remove", help="Remove a local skill"),
            CommandActionSpec(action="update", help="Check or apply skill updates"),
            CommandActionSpec(action="registry", help="Set the skills registry"),
        ),
        default_action="list",
        tool_exposed=True,
    ),
    CommandSpec(
        command="cards",
        actions=(
            CommandActionSpec(action="list", help="List installed card packs"),
            CommandActionSpec(action="add", help="Install a card pack"),
            CommandActionSpec(action="remove", help="Remove an installed card pack"),
            CommandActionSpec(action="update", help="Check or apply card pack updates"),
            CommandActionSpec(action="publish", help="Publish local card pack changes"),
            CommandActionSpec(action="registry", help="Set the card-pack registry"),
        ),
        default_action="list",
        tool_exposed=True,
    ),
    CommandSpec(
        command="models",
        actions=(
            CommandActionSpec(action="doctor", help="Inspect model onboarding readiness"),
            CommandActionSpec(action="aliases", help="List or manage model aliases"),
            CommandActionSpec(action="catalog", help="Show model catalog for a provider"),
        ),
        default_action="doctor",
        tool_exposed=True,
    ),
    CommandSpec(
        command="check",
        actions=(
            CommandActionSpec(
                action="run",
                help="Run fast-agent check diagnostics",
                operation_name="check",
            ),
        ),
        default_action="run",
        tool_exposed=True,
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


def operation_name_for_action(command_name: str, action: str) -> str:
    """Return the smart-operation key for a command/action pair."""

    spec = get_command_spec(command_name)
    normalized_action = action.strip().lower()
    if spec is None:
        return f"{command_name.strip().lower()}.{normalized_action}"

    for action_spec in spec.actions:
        if action_spec.action == normalized_action:
            return action_spec.operation_name or f"{spec.command}.{action_spec.action}"

    return f"{spec.command}.{normalized_action}"


def tool_exposed_operations() -> tuple[str, ...]:
    """Return stable smart-operation keys exposed by tool-enabled commands."""

    operations: list[str] = []
    for spec in COMMAND_SPECS:
        if not spec.tool_exposed:
            continue
        operations.extend(
            action.operation_name or f"{spec.command}.{action.action}" for action in spec.actions
        )
    return tuple(operations)


def smart_operation_help_entries() -> tuple[tuple[str, str], ...]:
    """Return `(operation, help)` entries for tool-exposed operations."""

    entries: list[tuple[str, str]] = []
    for spec in COMMAND_SPECS:
        if not spec.tool_exposed:
            continue
        for action in spec.actions:
            operation = action.operation_name or f"{spec.command}.{action.action}"
            entries.append((operation, action.help))
    return tuple(entries)


_TOOL_OPERATIONS: Final[tuple[str, ...]] = tool_exposed_operations()
_TOOL_OPERATIONS_SET: Final[set[str]] = set(_TOOL_OPERATIONS)


def parse_smart_operation(operation: str) -> tuple[str, str]:
    """Parse and validate a smart-command operation key."""

    normalized = operation.strip().lower()
    if normalized not in _TOOL_OPERATIONS_SET:
        expected = ", ".join(_TOOL_OPERATIONS)
        raise ValueError(
            f"Unknown smart command operation '{operation}'. Expected one of: {expected}."
        )

    if normalized == "check":
        return "check", "run"

    command_name, action = normalized.split(".", 1)
    return command_name, action
