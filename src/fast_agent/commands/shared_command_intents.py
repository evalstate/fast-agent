"""Shared parsing for session/history command intents across surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal, cast

from fast_agent.commands.command_catalog import command_action_names, normalize_command_action
from fast_agent.commands.option_parsing import ParsedValueOption, ValueOption, read_value_option
from fast_agent.utils.action_normalization import (
    FALSE_WORD_BOOLEAN_ALIASES,
    TRUE_WORD_BOOLEAN_ALIASES,
    is_help_flag,
    normalize_action_token,
    split_action_arguments,
)
from fast_agent.utils.commandline import split_commandline, split_posix_like_preserving_backslashes
from fast_agent.utils.text import strip_to_none

HistoryTurnError = Literal["missing", "invalid"]
HistoryTurnAction = Literal["detail", "rewind"]
HistoryAction = Literal[
    "overview",
    "show",
    "detail",
    "save",
    "load",
    "clear_all",
    "clear_last",
    "rewind",
    "fix",
    "webclear",
    "unknown",
]
ModelCommandAction = Literal[
    "reasoning",
    "task_budget",
    "verbosity",
    "fast",
    "web_search",
    "x_search",
    "web_fetch",
    "switch",
    "doctor",
    "references",
    "catalog",
    "help",
    "unknown",
]
ModelCommandActionCategory = Literal["value", "manager"]
ToolFlagAction = Literal["add_tool", "remove_tool", "dump"]
ToolMutationAction = Literal["add_tool", "remove_tool"]
_ExportValueName = Literal[
    "agent",
    "output",
    "hf_url",
    "hf_dataset",
    "hf_dataset_path",
    "privacy_filter_path",
    "privacy_filter_device",
    "privacy_filter_variant",
]
_ExportFlagName = Literal["privacy_filter", "download_privacy_filter", "show_redactions"]

_MODEL_COMMAND_ACTIONS = frozenset(command_action_names("model"))
ADD_TOOL_ACTION: Final[ToolMutationAction] = "add_tool"
REMOVE_TOOL_ACTION: Final[ToolMutationAction] = "remove_tool"
DUMP_TOOL_ACTION: Final[ToolFlagAction] = "dump"
MODEL_COMMAND_ACTION_CATEGORIES: dict[ModelCommandAction, ModelCommandActionCategory] = {
    "reasoning": "value",
    "task_budget": "value",
    "verbosity": "value",
    "fast": "value",
    "web_search": "value",
    "x_search": "value",
    "web_fetch": "value",
    "switch": "value",
    "doctor": "manager",
    "references": "manager",
    "catalog": "manager",
    "help": "manager",
}
MODEL_VALUE_COMMAND_ACTIONS: frozenset[ModelCommandAction] = frozenset(
    action for action, category in MODEL_COMMAND_ACTION_CATEGORIES.items() if category == "value"
)
MODEL_MANAGER_COMMAND_ACTIONS: frozenset[ModelCommandAction] = frozenset(
    action for action, category in MODEL_COMMAND_ACTION_CATEGORIES.items() if category == "manager"
)
MODEL_DIRECT_HANDLER_ACTIONS: frozenset[ModelCommandAction] = MODEL_VALUE_COMMAND_ACTIONS
TOOL_MUTATION_ACTIONS: frozenset[ToolMutationAction] = frozenset(
    (ADD_TOOL_ACTION, REMOVE_TOOL_ACTION)
)


def is_tool_mutation_action(action: ToolFlagAction | None) -> bool:
    return action in TOOL_MUTATION_ACTIONS


def _normalize_model_command_action(value: str) -> ModelCommandAction | None:
    normalized = normalize_command_action("model", value)
    if normalized in _MODEL_COMMAND_ACTIONS:
        return cast("ModelCommandAction", normalized)
    return None


def _argument_after_first_token(stripped: str, tokens: list[str]) -> str | None:
    if not tokens:
        return None
    if len(tokens) == 2:
        return tokens[1]
    if stripped.startswith(("'", '"')):
        quote = stripped[0]
        escaped = False
        for index, char in enumerate(stripped[1:], start=1):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote:
                return strip_to_none(stripped[index + 1 :])
    return strip_to_none(stripped[len(tokens[0]) :])


@dataclass(frozen=True, slots=True)
class ModelCommandIntent:
    action: ModelCommandAction
    argument: str | None = None
    raw_subcommand: str | None = None
    error: str | None = None


def parse_model_command_intent(
    remainder: str | None,
    *,
    default_action: ModelCommandAction = "reasoning",
) -> ModelCommandIntent:
    stripped = strip_to_none(remainder)
    if stripped is None:
        return ModelCommandIntent(action=default_action)

    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError as exc:
        return ModelCommandIntent(action="unknown", error=str(exc))

    if not tokens:
        return ModelCommandIntent(action=default_action)

    subcmd = normalize_action_token(tokens[0])
    argument = _argument_after_first_token(stripped, tokens)
    action = _normalize_model_command_action(subcmd)
    if action is not None:
        return ModelCommandIntent(action=action, argument=argument)
    return ModelCommandIntent(action="unknown", argument=argument, raw_subcommand=subcmd)


@dataclass(frozen=True, slots=True)
class CardLoadIntent:
    filename: str | None
    tool_action: ToolFlagAction | None = None
    error: str | None = None

    @property
    def add_tool(self) -> bool:
        return is_tool_mutation_action(self.tool_action)

    @property
    def remove_tool(self) -> bool:
        return self.tool_action == REMOVE_TOOL_ACTION


@dataclass(frozen=True, slots=True)
class AgentToolIntent:
    agent_name: str | None
    tool_action: ToolFlagAction | None = None
    error: str | None = None

    @property
    def add_tool(self) -> bool:
        return is_tool_mutation_action(self.tool_action)

    @property
    def remove_tool(self) -> bool:
        return self.tool_action == REMOVE_TOOL_ACTION

    @property
    def dump(self) -> bool:
        return self.tool_action == DUMP_TOOL_ACTION


@dataclass(frozen=True, slots=True)
class _ToolFlagParse:
    subject: str | None
    action: ToolFlagAction | None
    unknown: list[str]
    empty_subject: bool = False
    conflicting_action: bool = False


@dataclass(frozen=True, slots=True)
class _ParsedToolFlagTokens:
    parsed: _ToolFlagParse | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _ExportArgument:
    target: str | None = None
    agent: str | None = None
    output: str | None = None
    hf_url: str | None = None
    hf_dataset: str | None = None
    hf_dataset_path: str | None = None
    privacy_filter: bool = False
    privacy_filter_path: str | None = None
    download_privacy_filter: bool = False
    privacy_filter_device: str | None = None
    privacy_filter_variant: str | None = None
    show_redactions: bool = False
    show_help: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class _PinArgument:
    value: str | None = None
    target: str | None = None


@dataclass(slots=True)
class _ExportParseState:
    target: str | None = None
    option_values: dict[_ExportValueName, str] = field(default_factory=dict)
    enabled_flags: set[_ExportFlagName] = field(default_factory=set)
    show_help: bool = False


@dataclass(frozen=True, slots=True)
class _ExportTokenParseResult:
    next_index: int
    error: str | None = None


_EXPORT_VALUE_OPTIONS: tuple[ValueOption[_ExportValueName], ...] = (
    ValueOption("agent", ("--agent", "-a"), error_name="--agent"),
    ValueOption("output", ("--output", "-o"), error_name="--output"),
    ValueOption("hf_url", ("--hf-url",)),
    ValueOption("hf_dataset", ("--hf-dataset",)),
    ValueOption("hf_dataset_path", ("--hf-dataset-path",)),
    ValueOption("privacy_filter_path", ("--privacy-filter-path",)),
    ValueOption("privacy_filter_device", ("--privacy-filter-device",)),
    ValueOption(
        "privacy_filter_variant",
        ("--privacy-filter-variant", "--privacy-filter-quant"),
    ),
)
_EXPORT_FLAG_OPTIONS: dict[str, _ExportFlagName] = {
    "--privacy-filter": "privacy_filter",
    "--download-privacy-filter": "download_privacy_filter",
    "--show-redactions": "show_redactions",
}
_SIMPLE_HISTORY_ACTIONS: dict[str, HistoryAction] = {
    "list": "overview",
    "show": "show",
    "save": "save",
    "load": "load",
    "fix": "fix",
    "webclear": "webclear",
}
_CLEAR_HISTORY_ACTIONS: dict[str, HistoryAction] = {
    "last": "clear_last",
    "all": "clear_all",
}
HistoryIntentParser = Callable[[str | None], "HistoryActionIntent"]
SessionIntentParser = Callable[[str | None], "SessionCommandIntent"]
_SIMPLE_SESSION_ACTIONS: dict[str, "SessionAction"] = {
    "list": "list",
    "new": "new",
    "resume": "resume",
    "title": "title",
    "fork": "fork",
    "delete": "delete",
    "clear": "delete",
}
_PIN_VALUE_TOKENS = frozenset(
    {
        "toggle",
        *TRUE_WORD_BOOLEAN_ALIASES,
        *FALSE_WORD_BOOLEAN_ALIASES,
    }
)
ADD_TOOL_TOKEN_DESCRIPTIONS: Final[dict[str, str]] = {
    "tool": "Add as tool",
    "--tool": "Add as tool",
    "--as-tool": "Add as tool",
    "-t": "Add as tool",
}
REMOVE_TOOL_TOKEN_DESCRIPTIONS: Final[dict[str, str]] = {
    "remove": "Remove tool",
    "--remove": "Remove tool",
    "--rm": "Remove tool",
}
DUMP_TOOL_TOKEN_DESCRIPTIONS: Final[dict[str, str]] = {
    "dump": "Show agent card",
    "--dump": "Show agent card",
    "-d": "Show agent card",
}
_ADD_TOOL_TOKENS = frozenset(ADD_TOOL_TOKEN_DESCRIPTIONS)
_REMOVE_TOOL_TOKENS = frozenset(REMOVE_TOOL_TOKEN_DESCRIPTIONS)
_DUMP_TOOL_TOKENS = frozenset(DUMP_TOOL_TOKEN_DESCRIPTIONS)
_TOOL_FLAG_ACTIONS: dict[str, ToolFlagAction] = {
    **dict.fromkeys(_ADD_TOOL_TOKENS, ADD_TOOL_ACTION),
    **dict.fromkeys(_REMOVE_TOOL_TOKENS, REMOVE_TOOL_ACTION),
    **dict.fromkeys(_DUMP_TOOL_TOKENS, DUMP_TOOL_ACTION),
}


def _tool_flag_action(token: str, *, allow_dump: bool) -> ToolFlagAction | None:
    action = _TOOL_FLAG_ACTIONS.get(token)
    if action == DUMP_TOOL_ACTION and not allow_dump:
        return None
    return action


def _tool_flag_conflicts(
    current_action: ToolFlagAction | None,
    next_action: ToolFlagAction,
) -> bool:
    return current_action == DUMP_TOOL_ACTION or (
        next_action == DUMP_TOOL_ACTION and current_action is not None
    )


def _tool_flag_subject(token: str, *, strip_agent_prefix: bool) -> str:
    if strip_agent_prefix and token.startswith("@"):
        return token[1:]
    return token


def _parse_tool_flag_tokens(
    tokens: list[str],
    *,
    allow_dump: bool,
    strip_agent_prefix: bool,
) -> _ToolFlagParse:
    action: ToolFlagAction | None = None
    conflicting_action = False
    subject: str | None = None
    unknown: list[str] = []

    for token in tokens:
        next_action = _tool_flag_action(token, allow_dump=allow_dump)
        if next_action is not None:
            conflicting_action = conflicting_action or _tool_flag_conflicts(
                action,
                next_action,
            )
            if action is None or next_action != ADD_TOOL_ACTION:
                action = next_action
            continue
        if subject is None:
            candidate = _tool_flag_subject(token, strip_agent_prefix=strip_agent_prefix)
            if not candidate:
                return _ToolFlagParse(
                    subject=None,
                    action=action,
                    unknown=unknown,
                    empty_subject=True,
                    conflicting_action=conflicting_action,
                )
            subject = candidate
            continue
        unknown.append(token)

    return _ToolFlagParse(
        subject=subject,
        action=action,
        unknown=unknown,
        empty_subject=False,
        conflicting_action=conflicting_action,
    )


def _parse_tool_flag_remainder(
    remainder: str | None,
    *,
    empty_error: str,
    allow_dump: bool,
    strip_agent_prefix: bool,
) -> _ParsedToolFlagTokens:
    stripped = strip_to_none(remainder)
    if stripped is None:
        return _ParsedToolFlagTokens(error=empty_error)

    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError as exc:
        return _ParsedToolFlagTokens(error=f"Invalid arguments: {exc}")

    return _ParsedToolFlagTokens(
        parsed=_parse_tool_flag_tokens(
            tokens,
            allow_dump=allow_dump,
            strip_agent_prefix=strip_agent_prefix,
        )
    )


def parse_agent_tool_intent(
    remainder: str | None,
    *,
    require_tool_agent: bool = False,
) -> AgentToolIntent:
    parse_result = _parse_tool_flag_remainder(
        remainder,
        empty_error="Usage: /agent <name> --tool | /agent [name] --dump",
        allow_dump=True,
        strip_agent_prefix=True,
    )
    if parse_result.error is not None:
        return AgentToolIntent(agent_name=None, error=parse_result.error)

    parsed = parse_result.parsed
    if parsed is None:
        return AgentToolIntent(agent_name=None, error="Invalid arguments")
    error = _validate_agent_tool_intent(
        parsed,
        require_tool_agent=require_tool_agent,
    )
    return AgentToolIntent(
        agent_name=parsed.subject,
        tool_action=parsed.action,
        error=error,
    )


def _validate_agent_tool_intent(
    parsed: _ToolFlagParse,
    *,
    require_tool_agent: bool,
) -> str | None:
    if parsed.unknown:
        return f"Unexpected arguments: {', '.join(parsed.unknown)}"
    if parsed.empty_subject:
        return "Agent name cannot be empty."
    if parsed.conflicting_action:
        return "Use either --tool or --dump, not both."
    if parsed.action is None:
        return "Usage: /agent <name> --tool [remove] | /agent [name] --dump"
    if require_tool_agent and is_tool_mutation_action(parsed.action) and not parsed.subject:
        suffix = " remove" if parsed.action == REMOVE_TOOL_ACTION else ""
        return f"Agent name is required for /agent --tool{suffix}"
    return None


def parse_card_load_intent(remainder: str | None) -> CardLoadIntent:
    parse_result = _parse_tool_flag_remainder(
        remainder,
        empty_error="Filename required for /card",
        allow_dump=False,
        strip_agent_prefix=False,
    )
    if parse_result.error is not None:
        return CardLoadIntent(filename=None, error=parse_result.error)

    parsed = parse_result.parsed
    if parsed is None:
        return CardLoadIntent(filename=None, error="Invalid arguments")
    if parsed.empty_subject or parsed.subject is None:
        return _card_load_intent_from_parse(parsed, error="Filename required for /card")
    if parsed.unknown:
        return _card_load_intent_from_parse(
            parsed,
            error=f"Unexpected arguments: {', '.join(parsed.unknown)}",
        )
    return _card_load_intent_from_parse(parsed)


def _card_load_intent_from_parse(
    parsed: _ToolFlagParse,
    *,
    error: str | None = None,
) -> CardLoadIntent:
    return CardLoadIntent(
        filename=parsed.subject,
        tool_action=parsed.action,
        error=error,
    )


@dataclass(frozen=True, slots=True)
class HistoryActionIntent:
    action: HistoryAction
    argument: str | None = None
    turn_index: int | None = None
    turn_error: HistoryTurnError | None = None
    raw_subcommand: str | None = None


def parse_current_agent_history_intent(remainder: str) -> HistoryActionIntent:
    stripped = strip_to_none(remainder)
    if stripped is None:
        return HistoryActionIntent(action="overview")

    try:
        tokens = split_commandline(stripped, syntax="posix")
        argument = strip_to_none(" ".join(tokens[1:]))
    except ValueError:
        action_token, remainder = split_action_arguments(stripped)
        tokens = [action_token] if action_token is not None else []
        argument = remainder or None

    if not tokens:
        return HistoryActionIntent(action="overview")

    subcmd = normalize_action_token(tokens[0])

    action = _SIMPLE_HISTORY_ACTIONS.get(subcmd)
    if action is not None:
        return HistoryActionIntent(
            action=action,
            argument=argument if action != "overview" else None,
        )
    parser = _HISTORY_ACTION_PARSERS.get(subcmd)
    if parser is not None:
        return parser(argument)

    return HistoryActionIntent(action="unknown", raw_subcommand=subcmd, argument=argument)


def _parse_turn_history_intent(
    action: HistoryTurnAction,
    argument: str | None,
) -> HistoryActionIntent:
    if not argument:
        return HistoryActionIntent(action=action, turn_error="missing")
    try:
        turn_index = int(argument)
    except ValueError:
        return HistoryActionIntent(action=action, turn_error="invalid")
    if turn_index < 1:
        return HistoryActionIntent(action=action, turn_error="invalid")
    return HistoryActionIntent(action=action, turn_index=turn_index)


def _parse_clear_history_intent(argument: str | None) -> HistoryActionIntent:
    if not argument:
        return HistoryActionIntent(action="clear_all")
    action_token, target_agent_value = split_action_arguments(argument)
    if action_token is None:
        return HistoryActionIntent(action="clear_all")
    action = normalize_action_token(action_token)
    target_agent = target_agent_value or None
    history_action = _CLEAR_HISTORY_ACTIONS.get(action)
    if history_action is not None:
        return HistoryActionIntent(action=history_action, argument=target_agent)
    return HistoryActionIntent(action="clear_all", argument=argument)


_HISTORY_ACTION_PARSERS: dict[str, HistoryIntentParser] = {
    "detail": partial(_parse_turn_history_intent, "detail"),
    "review": partial(_parse_turn_history_intent, "detail"),
    "rewind": partial(_parse_turn_history_intent, "rewind"),
    "clear": _parse_clear_history_intent,
}

HISTORY_COMMAND_COMPLETION_DESCRIPTIONS: Final[dict[str, str]] = {
    "list": "Show conversation history overview",
    "show": "Show per-turn timing summaries",
    "detail": "Show a previous user turn in full",
    "review": "Review a previous user turn in full",
    "save": "Save history to a file",
    "load": "Load history from a file",
    "clear": "Clear history (all or last)",
    "rewind": "Rewind to a previous user turn",
    "fix": "Remove the last pending tool call",
    "webclear": "Strip web tool/citation metadata channels",
}


SessionAction = Literal[
    "help",
    "list",
    "new",
    "resume",
    "title",
    "fork",
    "delete",
    "pin",
    "export",
    "error",
    "unknown",
]
SESSION_SIMPLE_PAYLOAD_ACTIONS: frozenset[SessionAction] = frozenset(
    _SIMPLE_SESSION_ACTIONS.values()
)
SESSION_COMMAND_COMPLETION_DESCRIPTIONS: dict[str, str] = {
    "delete": "Delete a session (or all)",
    "pin": "Pin or unpin the current session",
    "clear": "Alias for delete",
    "list": "List recent sessions",
    "new": "Create a new session",
    "resume": "Resume a session",
    "title": "Set session title",
    "fork": "Fork current session",
    "export": "Export a persisted session trace",
}


@dataclass(frozen=True, slots=True)
class SessionCommandIntent:
    action: SessionAction
    argument: str | None = None
    pin_value: str | None = None
    pin_target: str | None = None
    export_target: str | None = None
    export_agent: str | None = None
    export_output: str | None = None
    export_hf_url: str | None = None
    export_hf_dataset: str | None = None
    export_hf_dataset_path: str | None = None
    export_privacy_filter: bool = False
    export_privacy_filter_path: str | None = None
    export_download_privacy_filter: bool = False
    export_privacy_filter_device: str | None = None
    export_privacy_filter_variant: str | None = None
    export_show_redactions: bool = False
    export_help: bool = False
    export_error: str | None = None
    raw_subcommand: str | None = None


def should_default_export_agent(target: str | None, *, current_session_id: str | None) -> bool:
    return target is None and current_session_id is not None


def _parse_session_subcommand_intent(
    *,
    subcmd: str,
    argument: str | None,
) -> SessionCommandIntent:
    action = _SIMPLE_SESSION_ACTIONS.get(subcmd)
    if action is not None:
        return SessionCommandIntent(
            action=action,
            argument=argument,
        )
    parser = _SESSION_SPECIAL_ACTION_PARSERS.get(subcmd)
    if parser is not None:
        return parser(argument)
    return SessionCommandIntent(action="unknown", raw_subcommand=subcmd, argument=argument)


def parse_session_command_intent(remainder: str) -> SessionCommandIntent:
    stripped = strip_to_none(remainder)
    if stripped is None:
        return SessionCommandIntent(action="help")

    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError as exc:
        return SessionCommandIntent(action="error", argument=str(exc))

    if not tokens:
        return SessionCommandIntent(action="help")

    subcmd = normalize_action_token(tokens[0])
    argument = _argument_after_first_token(stripped, tokens)
    return _parse_session_subcommand_intent(subcmd=subcmd, argument=argument)


def _parse_pin_argument(argument: str) -> _PinArgument:
    stripped = strip_to_none(argument)
    if stripped is None:
        return _PinArgument()

    try:
        pin_tokens = split_commandline(stripped, syntax="posix")
    except ValueError:
        action_token, remainder = split_action_arguments(stripped)
        if action_token is None:
            pin_tokens = []
        elif remainder:
            pin_tokens = [action_token, remainder]
        else:
            pin_tokens = [action_token]

    if not pin_tokens:
        return _PinArgument()

    first = normalize_action_token(pin_tokens[0])
    if first in _PIN_VALUE_TOKENS:
        target = strip_to_none(" ".join(pin_tokens[1:]))
        return _PinArgument(value=first, target=target)
    return _PinArgument(target=stripped)


def _parse_pin_session_intent(argument: str | None) -> SessionCommandIntent:
    pin = _parse_pin_argument(argument or "")
    return SessionCommandIntent(
        action="pin",
        pin_value=pin.value,
        pin_target=pin.target,
    )


def _parse_export_session_intent(argument: str | None) -> SessionCommandIntent:
    return _session_intent_from_export(_parse_export_argument(argument))


_SESSION_SPECIAL_ACTION_PARSERS: dict[str, SessionIntentParser] = {
    "pin": _parse_pin_session_intent,
    "export": _parse_export_session_intent,
}


def _session_intent_from_export(export: _ExportArgument) -> SessionCommandIntent:
    return SessionCommandIntent(
        action="export",
        export_target=export.target,
        export_agent=export.agent,
        export_output=export.output,
        export_hf_url=export.hf_url,
        export_hf_dataset=export.hf_dataset,
        export_hf_dataset_path=export.hf_dataset_path,
        export_privacy_filter=export.privacy_filter,
        export_privacy_filter_path=export.privacy_filter_path,
        export_download_privacy_filter=export.download_privacy_filter,
        export_privacy_filter_device=export.privacy_filter_device,
        export_privacy_filter_variant=export.privacy_filter_variant,
        export_show_redactions=export.show_redactions,
        export_help=export.show_help,
        export_error=export.error,
    )


def _parse_export_argument(argument: str | None) -> _ExportArgument:
    stripped = strip_to_none(argument)
    if stripped is None:
        return _ExportArgument()

    try:
        tokens = split_posix_like_preserving_backslashes(stripped)
    except ValueError as exc:
        return _export_parse_error(f"Invalid export arguments: {exc}")

    state = _ExportParseState()
    index = 0
    while index < len(tokens):
        parsed = _parse_export_token(tokens, index, state)
        if parsed.error is not None:
            return _export_parse_error(parsed.error)
        index = parsed.next_index

    return _export_argument_from_parse(
        target=state.target,
        option_values=state.option_values,
        enabled_flags=state.enabled_flags,
        show_help=state.show_help,
    )


def _export_argument_from_parse(
    *,
    target: str | None,
    option_values: dict[_ExportValueName, str],
    enabled_flags: set[_ExportFlagName],
    show_help: bool,
) -> _ExportArgument:
    return _ExportArgument(
        target=target,
        agent=option_values.get("agent"),
        output=option_values.get("output"),
        hf_url=option_values.get("hf_url"),
        hf_dataset=option_values.get("hf_dataset"),
        hf_dataset_path=option_values.get("hf_dataset_path"),
        privacy_filter="privacy_filter" in enabled_flags,
        privacy_filter_path=option_values.get("privacy_filter_path"),
        download_privacy_filter="download_privacy_filter" in enabled_flags,
        privacy_filter_device=option_values.get("privacy_filter_device"),
        privacy_filter_variant=option_values.get("privacy_filter_variant"),
        show_redactions="show_redactions" in enabled_flags,
        show_help=show_help,
    )


def _consume_export_value(
    tokens: list[str],
    index: int,
) -> ParsedValueOption[_ExportValueName]:
    return read_value_option(tokens, index, _EXPORT_VALUE_OPTIONS)


def _apply_export_flag(
    token: str,
    index: int,
    state: _ExportParseState,
) -> _ExportTokenParseResult | None:
    flag_name = _EXPORT_FLAG_OPTIONS.get(token)
    if flag_name is None:
        return None
    state.enabled_flags.add(flag_name)
    return _ExportTokenParseResult(next_index=index + 1)


def _apply_export_target_token(
    token: str,
    index: int,
    state: _ExportParseState,
) -> _ExportTokenParseResult:
    if token.startswith("-"):
        return _ExportTokenParseResult(
            next_index=index,
            error=f"Unknown export option: {token}",
        )
    if state.target is None:
        state.target = _normalize_export_target(token)
        return _ExportTokenParseResult(next_index=index + 1)
    return _ExportTokenParseResult(
        next_index=index,
        error=f"Unexpected export argument: {token}",
    )


def _parse_export_token(
    tokens: list[str],
    index: int,
    state: _ExportParseState,
) -> _ExportTokenParseResult:
    token = tokens[index]
    if is_help_flag(token):
        state.show_help = True
        return _ExportTokenParseResult(next_index=index + 1)

    consumed = _consume_export_value(tokens, index)
    if consumed.error:
        return _ExportTokenParseResult(next_index=index, error=consumed.error)
    if consumed.matched:
        return _apply_export_value(consumed, state)

    flag_result = _apply_export_flag(token, index, state)
    if flag_result is not None:
        return flag_result

    return _apply_export_target_token(token, index, state)


def _apply_export_value(
    consumed: ParsedValueOption[_ExportValueName],
    state: _ExportParseState,
) -> _ExportTokenParseResult:
    name = consumed.require_name()
    value = consumed.require_value()
    if name in state.option_values:
        return _ExportTokenParseResult(
            next_index=consumed.next_index,
            error=f"Duplicate export option: {consumed.display_name or name}",
        )
    state.option_values[name] = value
    return _ExportTokenParseResult(next_index=consumed.next_index)


def _export_parse_error(message: str) -> _ExportArgument:
    return _ExportArgument(error=message)


def _normalize_export_target(target: str) -> str:
    if normalize_action_token(target) == "latest":
        return "latest"
    return target
