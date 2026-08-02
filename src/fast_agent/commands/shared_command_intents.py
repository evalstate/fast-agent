"""Shared parsing for session/history command intents across surfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Final, Literal, cast

from fast_agent.commands.command_catalog import command_action_names, normalize_command_action
from fast_agent.commands.option_parsing import ParsedValueOption, ValueOption, read_value_option
from fast_agent.utils.action_normalization import (
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
    "status",
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
SubagentsCommandAction = Literal["list", "status", "on", "off", "toggle", "help", "unknown"]
_ExportValueName = Literal[
    "agent",
    "output",
    "format",
    "hf_url",
    "hf_dataset",
    "hf_dataset_path",
    "privacy_filter_path",
    "privacy_filter_device",
    "privacy_filter_variant",
]
_ExportFlagName = Literal["privacy_filter", "download_privacy_filter", "show_redactions"]

_MODEL_COMMAND_ACTIONS = frozenset(command_action_names("model"))
MODEL_COMMAND_ACTION_CATEGORIES: dict[ModelCommandAction, ModelCommandActionCategory] = {
    "status": "value",
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
    default_action: ModelCommandAction = "status",
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
        if action == "status" and argument is not None:
            return ModelCommandIntent(
                action="unknown",
                error="Usage: /model status",
            )
        return ModelCommandIntent(action=action, argument=argument)
    return ModelCommandIntent(action="unknown", argument=argument, raw_subcommand=subcmd)


@dataclass(frozen=True, slots=True)
class SubagentsCommandIntent:
    action: SubagentsCommandAction
    error: str | None = None


def parse_subagents_command_intent(remainder: str | None) -> SubagentsCommandIntent:
    tokens, error = _strict_command_tokens(remainder, command_name="subagents")
    if error is not None:
        return SubagentsCommandIntent(action="unknown", error=error)
    if not tokens:
        return SubagentsCommandIntent(action="list")
    if len(tokens) != 1:
        return SubagentsCommandIntent(
            action="unknown",
            error="Usage: /subagents [list|status|on|off|toggle|help]",
        )

    action = normalize_command_action("subagents", tokens[0])
    if action in {"list", "status", "on", "off", "toggle", "help"}:
        return SubagentsCommandIntent(action=cast("SubagentsCommandAction", action))
    return SubagentsCommandIntent(
        action="unknown",
        error=f"Unknown /subagents action: {tokens[0]}",
    )


AgentCommandAction = Literal["status", "list", "use", "tool_add", "tool_remove", "unknown"]
CardCommandAction = Literal["show", "load", "unknown"]


@dataclass(frozen=True, slots=True)
class AgentCommandIntent:
    action: AgentCommandAction
    agent_name: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CardCommandIntent:
    action: CardCommandAction
    source: str | None = None
    agent_name: str | None = None
    as_tool: bool = False
    error: str | None = None


def _strict_command_tokens(
    remainder: str | None,
    *,
    command_name: str,
) -> tuple[list[str], str | None]:
    stripped = strip_to_none(remainder)
    if stripped is None:
        return [], None
    try:
        return split_commandline(stripped, syntax="posix"), None
    except ValueError as exc:
        return [], f"Invalid /{command_name} arguments: {exc}"


def parse_agent_command_intent(remainder: str | None) -> AgentCommandIntent:
    tokens, error = _strict_command_tokens(remainder, command_name="agent")
    if error is not None:
        return AgentCommandIntent(action="unknown", error=error)
    if not tokens:
        return AgentCommandIntent(action="status")

    action = normalize_action_token(tokens[0])
    if action in {"status", "list"}:
        if len(tokens) == 1:
            return AgentCommandIntent(action=cast("AgentCommandAction", action))
    elif action == "use":
        if len(tokens) == 2:
            return AgentCommandIntent(action="use", agent_name=tokens[1])
    elif action == "tool" and len(tokens) == 3:
        operation = normalize_action_token(tokens[1])
        if operation in {"add", "remove"}:
            return AgentCommandIntent(
                action="tool_add" if operation == "add" else "tool_remove",
                agent_name=tokens[2],
            )

    return AgentCommandIntent(
        action="unknown",
        error=("Usage: /agent [status|list|use <name>|tool add <name>|tool remove <name>]"),
    )


def parse_card_command_intent(remainder: str | None) -> CardCommandIntent:
    tokens, error = _strict_command_tokens(remainder, command_name="card")
    if error is not None:
        return CardCommandIntent(action="unknown", error=error)
    if not tokens:
        return CardCommandIntent(action="show")

    action = normalize_action_token(tokens[0])
    if action == "show" and len(tokens) <= 2:
        return CardCommandIntent(
            action="show",
            agent_name=tokens[1] if len(tokens) == 2 else None,
        )
    if action == "load":
        as_tool = tokens[-1:] == ["--as-tool"]
        positional = tokens[1:-1] if as_tool else tokens[1:]
        if len(positional) == 1:
            return CardCommandIntent(
                action="load",
                source=positional[0],
                as_tool=as_tool,
            )

    return CardCommandIntent(
        action="unknown",
        error="Usage: /card [show [agent]|load <path-or-url> [--as-tool]]",
    )


@dataclass(frozen=True, slots=True)
class _ExportArgument:
    target: str | None = None
    agent: str | None = None
    output: str | None = None
    format: str = "codex"
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
    title: str | None = None


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
    ValueOption("format", ("--format", "--export-format"), error_name="--format"),
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
    "unpin": "unpin",
}


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
    if subcmd.isdigit():
        return _parse_turn_history_intent("detail", subcmd)

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
    "unpin",
    "export",
    "error",
    "unknown",
]
SESSION_COMMAND_COMPLETION_DESCRIPTIONS: dict[str, str] = {
    "delete": "Delete a session (or all)",
    "pin": "Set the current session title and pin it",
    "unpin": "Unpin the current session",
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
    pin_title: str | None = None
    export_target: str | None = None
    export_agent: str | None = None
    export_output: str | None = None
    export_format: str = "codex"
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
    return _PinArgument(title=strip_to_none(argument))


def _parse_pin_session_intent(argument: str | None) -> SessionCommandIntent:
    pin = _parse_pin_argument(argument or "")
    return SessionCommandIntent(
        action="pin",
        pin_title=pin.title,
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
        export_format=export.format,
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
        format=option_values.get("format", "codex"),
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
