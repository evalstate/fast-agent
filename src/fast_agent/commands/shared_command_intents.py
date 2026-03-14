"""Shared parsing for session/history command intents across surfaces."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Literal

HistoryTurnError = Literal["missing", "invalid"]
HistoryAction = Literal["overview", "show", "detail", "save", "load", "unknown"]


@dataclass(frozen=True, slots=True)
class HistoryActionIntent:
    action: HistoryAction
    argument: str | None = None
    turn_index: int | None = None
    turn_error: HistoryTurnError | None = None
    raw_subcommand: str | None = None


def parse_current_agent_history_intent(remainder: str) -> HistoryActionIntent:
    stripped = remainder.strip()
    if not stripped:
        return HistoryActionIntent(action="overview")

    try:
        tokens = shlex.split(stripped)
        argument = " ".join(tokens[1:]).strip() or None
    except ValueError:
        tokens = stripped.split(maxsplit=1)
        argument = stripped[len(tokens[0]) :].strip() or None if tokens else None

    if not tokens:
        return HistoryActionIntent(action="overview")

    subcmd = tokens[0].lower()

    simple_actions: dict[str, HistoryAction] = {
        "list": "overview",
        "show": "show",
        "save": "save",
        "load": "load",
    }
    action = simple_actions.get(subcmd)
    if action is not None:
        return HistoryActionIntent(
            action=action,
            argument=argument if action != "overview" else None,
        )
    if subcmd in {"detail", "review"}:
        return _parse_detail_history_intent(argument)

    return HistoryActionIntent(action="unknown", raw_subcommand=subcmd, argument=argument)


def _parse_detail_history_intent(argument: str | None) -> HistoryActionIntent:
    if not argument:
        return HistoryActionIntent(action="detail", turn_error="missing")
    try:
        turn_index = int(argument)
    except ValueError:
        return HistoryActionIntent(action="detail", turn_error="invalid")
    return HistoryActionIntent(action="detail", turn_index=turn_index)


SessionAction = Literal["help", "list", "new", "resume", "title", "fork", "delete", "pin", "unknown"]


@dataclass(frozen=True, slots=True)
class SessionCommandIntent:
    action: SessionAction
    argument: str | None = None
    pin_value: str | None = None
    pin_target: str | None = None
    raw_subcommand: str | None = None


def parse_session_command_intent(remainder: str) -> SessionCommandIntent:
    stripped = remainder.strip()
    if not stripped:
        return SessionCommandIntent(action="help")

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return SessionCommandIntent(action="help")

    if not tokens:
        return SessionCommandIntent(action="help")

    subcmd = tokens[0].lower()
    argument = stripped[len(tokens[0]) :].strip() or None

    simple_actions: dict[str, SessionAction] = {
        "list": "list",
        "new": "new",
        "resume": "resume",
        "title": "title",
        "fork": "fork",
        "delete": "delete",
        "clear": "delete",
    }
    action = simple_actions.get(subcmd)
    if action is not None:
        return SessionCommandIntent(
            action=action,
            argument=argument,
        )
    if subcmd == "pin":
        value, target = _parse_pin_argument(argument or "")
        return SessionCommandIntent(
            action="pin",
            pin_value=value,
            pin_target=target,
        )

    return SessionCommandIntent(action="unknown", raw_subcommand=subcmd, argument=argument)


def _parse_pin_argument(argument: str) -> tuple[str | None, str | None]:
    stripped = argument.strip()
    if not stripped:
        return None, None

    try:
        pin_tokens = shlex.split(stripped)
    except ValueError:
        pin_tokens = stripped.split(maxsplit=1)

    if not pin_tokens:
        return None, None

    first = pin_tokens[0].lower()
    value_tokens = {
        "on",
        "off",
        "toggle",
        "true",
        "false",
        "yes",
        "no",
        "enable",
        "enabled",
        "disable",
        "disabled",
    }
    if first in value_tokens:
        target = " ".join(pin_tokens[1:]).strip() or None
        return first, target
    return None, stripped
