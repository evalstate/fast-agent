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
    "unknown",
]


@dataclass(frozen=True, slots=True)
class SessionCommandIntent:
    action: SessionAction
    argument: str | None = None
    pin_value: str | None = None
    pin_target: str | None = None
    export_target: str | None = None
    export_agent: str | None = None
    export_output: str | None = None
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
    if subcmd == "export":
        (
            target,
            agent,
            output,
            hf_dataset,
            hf_dataset_path,
            privacy_filter,
            privacy_filter_path,
            download_privacy_filter,
            privacy_filter_device,
            privacy_filter_variant,
            show_redactions,
            show_help,
            error,
        ) = _parse_export_argument(argument)
        return SessionCommandIntent(
            action="export",
            export_target=target,
            export_agent=agent,
            export_output=output,
            export_hf_dataset=hf_dataset,
            export_hf_dataset_path=hf_dataset_path,
            export_privacy_filter=privacy_filter,
            export_privacy_filter_path=privacy_filter_path,
            export_download_privacy_filter=download_privacy_filter,
            export_privacy_filter_device=privacy_filter_device,
            export_privacy_filter_variant=privacy_filter_variant,
            export_show_redactions=show_redactions,
            export_help=show_help,
            export_error=error,
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


def _parse_export_argument(
    argument: str | None,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
    str | None,
    bool,
    str | None,
    str | None,
    bool,
    bool,
    str | None,
]:
    stripped = (argument or "").strip()
    if not stripped:
        return None, None, None, None, None, False, None, False, None, None, False, False, None

    try:
        tokens = _split_export_tokens(stripped)
    except ValueError as exc:
        return (
            None,
            None,
            None,
            None,
            None,
            False,
            None,
            False,
            None,
            None,
            False,
            False,
            f"Invalid export arguments: {exc}",
        )

    target: str | None = None
    agent_name: str | None = None
    output_path: str | None = None
    hf_dataset: str | None = None
    hf_dataset_path: str | None = None
    privacy_filter = False
    privacy_filter_path: str | None = None
    download_privacy_filter = False
    privacy_filter_device: str | None = None
    privacy_filter_variant: str | None = None
    show_redactions = False
    show_help = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--help", "-h"}:
            show_help = True
            index += 1
            continue
        if token in {"--agent", "-a"}:
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --agent")
            agent_name = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--agent="):
            agent_name = token.partition("=")[2] or None
            index += 1
            continue
        if token in {"--output", "-o"}:
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --output")
            output_path = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--output="):
            output_path = token.partition("=")[2] or None
            index += 1
            continue
        if token == "--hf-dataset":
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --hf-dataset")
            hf_dataset = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--hf-dataset="):
            hf_dataset = token.partition("=")[2] or None
            index += 1
            continue
        if token == "--hf-dataset-path":
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --hf-dataset-path")
            hf_dataset_path = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--hf-dataset-path="):
            hf_dataset_path = token.partition("=")[2] or None
            index += 1
            continue
        if token == "--privacy-filter":
            privacy_filter = True
            index += 1
            continue
        if token == "--privacy-filter-path":
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --privacy-filter-path")
            privacy_filter_path = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--privacy-filter-path="):
            privacy_filter_path = token.partition("=")[2] or None
            index += 1
            continue
        if token == "--download-privacy-filter":
            download_privacy_filter = True
            index += 1
            continue
        if token == "--privacy-filter-device":
            if index + 1 >= len(tokens):
                return _export_parse_error("Missing value for --privacy-filter-device")
            privacy_filter_device = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--privacy-filter-device="):
            privacy_filter_device = token.partition("=")[2] or None
            index += 1
            continue
        if token in {"--privacy-filter-variant", "--privacy-filter-quant"}:
            if index + 1 >= len(tokens):
                return _export_parse_error(f"Missing value for {token}")
            privacy_filter_variant = tokens[index + 1]
            index += 2
            continue
        if token.startswith("--privacy-filter-variant=") or token.startswith(
            "--privacy-filter-quant="
        ):
            privacy_filter_variant = token.partition("=")[2] or None
            index += 1
            continue
        if token == "--show-redactions":
            show_redactions = True
            index += 1
            continue
        if token.startswith("-"):
            return _export_parse_error(f"Unknown export option: {token}")
        if target is None:
            target = _normalize_export_target(token)
            index += 1
            continue
        return _export_parse_error(f"Unexpected export argument: {token}")

    return (
        target,
        agent_name,
        output_path,
        hf_dataset,
        hf_dataset_path,
        privacy_filter,
        privacy_filter_path,
        download_privacy_filter,
        privacy_filter_device,
        privacy_filter_variant,
        show_redactions,
        show_help,
        None,
    )


def _export_parse_error(
    message: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    bool,
    str | None,
    bool,
    str | None,
    str | None,
    bool,
    bool,
    str | None,
]:
    return None, None, None, None, None, False, None, False, None, None, False, False, message


def _normalize_export_target(target: str) -> str:
    if target.lower() == "latest":
        return "latest"
    return target


def _split_export_tokens(argument: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    token_started = False
    index = 0

    while index < len(argument):
        char = argument[index]
        if quote is None:
            if char.isspace():
                if token_started:
                    tokens.append("".join(current))
                    current = []
                    token_started = False
                index += 1
                continue
            if char in {'"', "'"}:
                quote = char
                token_started = True
                index += 1
                continue
            if char == "\\" and index + 1 < len(argument):
                next_char = argument[index + 1]
                if next_char.isspace() or next_char in {'"', "'"}:
                    current.append(next_char)
                    token_started = True
                    index += 2
                    continue
            current.append(char)
            token_started = True
            index += 1
            continue

        if char == quote:
            quote = None
            index += 1
            continue
        if quote == '"' and char == "\\" and index + 1 < len(argument):
            next_char = argument[index + 1]
            if next_char == '"':
                current.append(next_char)
                token_started = True
                index += 2
                continue
        current.append(char)
        token_started = True
        index += 1

    if quote is not None:
        raise ValueError("No closing quotation")
    if token_started:
        tokens.append("".join(current))
    return tokens
