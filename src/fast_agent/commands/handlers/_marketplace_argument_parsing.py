"""Shared argument parsers for marketplace-style command handlers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeVar

from fast_agent.commands.option_parsing import (
    ParsedValueOption,
    ValueOption,
    read_value_option,
)
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

type _PublishValueName = Literal["message", "temp_dir"]
type _AddValueName = Literal["registry", "skills_dir"]
type _UpdateValueName = Literal["skills_dir"]
type _MarketplaceValueName = _PublishValueName | _AddValueName | _UpdateValueName
_ValueName = TypeVar("_ValueName", bound=_MarketplaceValueName)
type _PublishFlagName = Literal["push", "keep_temp"]


@dataclass(frozen=True, slots=True)
class AddArgument:
    selector: str | None = None
    registry: str | None = None
    skills_dir: Path | None = None
    force: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class UpdateArgument:
    selector: str | None = None
    skills_dir: Path | None = None
    force: bool = False
    yes: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PublishArgument:
    selector: str | None = None
    push: bool = True
    message: str | None = None
    temp_dir: Path | None = None
    keep_temp: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RegistryArgument:
    url: str | None = None
    warning: str | None = None


@dataclass(slots=True)
class _MarketplaceParseState:
    selector: str | None = None
    flags: dict[str, bool] = field(default_factory=dict)
    values: dict[str, str | Path] = field(default_factory=dict)
    error: str | None = None

    def flag(self, name: str) -> bool:
        return self.flags.get(name, False)

    def string_value(self, name: str) -> str | None:
        value = self.values.get(name)
        return value if isinstance(value, str) else None

    def path_value(self, name: str) -> Path | None:
        value = self.values.get(name)
        return value if isinstance(value, Path) else None


_ADD_VALUE_OPTIONS: tuple[ValueOption[_AddValueName], ...] = (
    ValueOption("registry", ("--registry", "-r"), error_name="--registry"),
    ValueOption("skills_dir", ("--skills-dir",)),
)
_ADD_FLAG_OPTIONS = {
    "--force": ("force", True),
}
_PUBLISH_VALUE_OPTIONS: tuple[ValueOption[_PublishValueName], ...] = (
    ValueOption("message", ("--message", "-m"), error_name="--message"),
    ValueOption("temp_dir", ("--temp-dir",)),
)
_PUBLISH_FLAG_OPTIONS: dict[str, tuple[_PublishFlagName, bool]] = {
    "--no-push": ("push", False),
    "--push": ("push", True),
    "--keep-temp": ("keep_temp", True),
}
_UPDATE_FLAG_OPTIONS = {
    "--force": ("force", True),
    "--yes": ("yes", True),
}
_UPDATE_VALUE_OPTIONS: tuple[ValueOption[_UpdateValueName], ...] = (
    ValueOption("skills_dir", ("--skills-dir",)),
)


def _assign_selector(current: str | None, token: str) -> tuple[str | None, str | None]:
    selector = strip_to_none(token)
    if selector is None:
        return current, "Selector cannot be empty."
    if current is not None:
        return current, "Only one selector is allowed."
    return selector, None


def optional_selector(argument: str | None) -> str | None:
    """Return a stripped selector, treating empty/whitespace arguments as omitted."""
    return strip_to_none(argument)


def resolve_registry_argument(
    registry_arg: str,
    configured_urls: Sequence[str],
) -> RegistryArgument:
    normalized_registry_arg = strip_to_none(registry_arg)
    if normalized_registry_arg is None:
        return RegistryArgument(warning="Registry URL or number is required.")

    if not normalized_registry_arg.isdigit():
        return RegistryArgument(url=normalized_registry_arg)

    if not configured_urls:
        return RegistryArgument(warning="No registries configured.")

    index = int(normalized_registry_arg)
    if 1 <= index <= len(configured_urls):
        return RegistryArgument(url=configured_urls[index - 1])

    return RegistryArgument(warning=f"Invalid registry number. Use 1-{len(configured_urls)}.")


def _is_option_token(token: str) -> bool:
    return token.startswith("-") and token != "-"


def _option_error_token(token: str) -> str:
    return token.split("=", maxsplit=1)[0]


def _apply_boolean_flag(
    state: dict[str, bool],
    option: tuple[str, bool],
) -> None:
    name, value = option
    state[name] = value


def _split_argument(argument: str | None, command_name: str) -> tuple[list[str], str | None]:
    if argument is None:
        return [], None
    try:
        return split_commandline(argument, syntax="posix"), None
    except ValueError as exc:
        return [], f"Invalid {command_name} arguments: {exc}"


def _duplicate_value_error(
    state: _MarketplaceParseState,
    name: str,
    display_name: str | None,
) -> str | None:
    if name in state.values:
        return f"Duplicate option: {display_name}"
    return None


def _store_parsed_value(
    state: _MarketplaceParseState,
    consumed: ParsedValueOption[_ValueName],
    value: str | Path,
) -> str | None:
    name = consumed.require_name()
    duplicate_error = _duplicate_value_error(state, name, consumed.display_name)
    if duplicate_error is not None:
        return duplicate_error
    state.values[name] = value
    return None


def _assign_add_value(
    state: _MarketplaceParseState,
    consumed: ParsedValueOption[_AddValueName],
) -> str | None:
    name = consumed.require_name()
    value = consumed.require_value()
    parsed_value: str | Path = value if name == "registry" else Path(value).expanduser()
    return _store_parsed_value(state, consumed, parsed_value)


def _assign_publish_value(
    state: _MarketplaceParseState,
    consumed: ParsedValueOption[_PublishValueName],
) -> str | None:
    name = consumed.require_name()
    value = consumed.require_value()
    parsed_value: str | Path = value if name == "message" else Path(value).expanduser()
    return _store_parsed_value(state, consumed, parsed_value)


def _assign_update_value(
    state: _MarketplaceParseState,
    consumed: ParsedValueOption[_UpdateValueName],
) -> str | None:
    return _store_parsed_value(
        state,
        consumed,
        Path(consumed.require_value()).expanduser(),
    )


def _parse_marketplace_argument(
    argument: str | None,
    *,
    command_name: str,
    initial_flags: Mapping[str, bool],
    flag_options: Mapping[str, tuple[str, bool]],
    value_options: Sequence[ValueOption[_ValueName]],
    assign_value: Callable[["_MarketplaceParseState", ParsedValueOption[_ValueName]], str | None],
) -> _MarketplaceParseState:
    tokens, split_error = _split_argument(argument, command_name)
    state = _MarketplaceParseState(flags=dict(initial_flags))
    if split_error is not None:
        state.error = split_error
        return state

    index = 0
    while index < len(tokens):
        token = tokens[index]
        flag = flag_options.get(token)
        if flag is not None:
            _apply_boolean_flag(state.flags, flag)
            index += 1
            continue
        consumed = read_value_option(tokens, index, value_options)
        if consumed.error is not None:
            state.error = consumed.error
            return state
        if consumed.matched:
            state.error = assign_value(state, consumed)
            if state.error is not None:
                return state
            index = consumed.next_index
            continue
        if _is_option_token(token):
            state.error = f"Unknown option: {_option_error_token(token)}"
            return state
        state.selector, state.error = _assign_selector(state.selector, token)
        if state.error is not None:
            return state
        index += 1

    return state


def parse_add_argument(
    argument: str | None,
    *,
    allow_registry: bool = True,
    allow_skills_dir: bool = True,
    allow_force: bool = True,
) -> AddArgument:
    """Parse marketplace add command arguments into selector and supported overrides."""
    value_option_names = {
        name
        for name, allowed in (
            ("registry", allow_registry),
            ("skills_dir", allow_skills_dir),
        )
        if allowed
    }
    state = _parse_marketplace_argument(
        argument,
        command_name="add",
        initial_flags={"force": False},
        flag_options=_ADD_FLAG_OPTIONS if allow_force else {},
        value_options=tuple(
            option for option in _ADD_VALUE_OPTIONS if option.name in value_option_names
        ),
        assign_value=_assign_add_value,
    )
    if state.error is not None:
        return AddArgument(error=state.error)

    return AddArgument(
        selector=state.selector,
        registry=state.string_value("registry"),
        skills_dir=state.path_value("skills_dir"),
        force=state.flag("force"),
    )


def parse_update_argument(
    argument: str | None,
    *,
    allow_skills_dir: bool = False,
) -> UpdateArgument:
    """Parse update command arguments into a named result."""
    state = _parse_marketplace_argument(
        argument,
        command_name="update",
        initial_flags={"force": False, "yes": False},
        flag_options=_UPDATE_FLAG_OPTIONS,
        value_options=_UPDATE_VALUE_OPTIONS if allow_skills_dir else (),
        assign_value=_assign_update_value,
    )
    if state.error is not None:
        return UpdateArgument(error=state.error)

    return UpdateArgument(
        selector=state.selector,
        skills_dir=state.path_value("skills_dir"),
        force=state.flag("force"),
        yes=state.flag("yes"),
    )


def parse_publish_argument(argument: str | None) -> PublishArgument:
    """Parse card publish command arguments into a named result."""
    state = _parse_marketplace_argument(
        argument,
        command_name="publish",
        initial_flags={"push": True, "keep_temp": False},
        flag_options=_PUBLISH_FLAG_OPTIONS,
        value_options=_PUBLISH_VALUE_OPTIONS,
        assign_value=_assign_publish_value,
    )
    if state.error is not None:
        return PublishArgument(error=state.error)

    return PublishArgument(
        selector=state.selector,
        push=state.flag("push"),
        message=state.string_value("message"),
        temp_dir=state.path_value("temp_dir"),
        keep_temp=state.flag("keep_temp"),
    )
