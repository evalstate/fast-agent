"""Pure command-line parsing for interactive prompt input."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from fast_agent.commands.mcp_command_intents import (
    MCP_TOP_LEVEL_ACTIONS,
    is_mcp_server_name_action,
    parse_mcp_no_args_tokens,
    parse_mcp_server_name_tokens,
)
from fast_agent.commands.shared_command_intents import (
    MODEL_MANAGER_COMMAND_ACTIONS,
    HistoryActionIntent,
    ModelCommandAction,
    SessionCommandIntent,
    parse_agent_tool_intent,
    parse_card_load_intent,
    parse_current_agent_history_intent,
    parse_model_command_intent,
    parse_session_command_intent,
)
from fast_agent.mcp.connect_targets import parse_connect_command_text
from fast_agent.ui.command_payloads import (
    A2ACommand,
    AgentCommand,
    AttachCommand,
    CardsCommand,
    CheckCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandError,
    CommandPayload,
    CommandsCommand,
    CreateSessionCommand,
    ExportSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryViewCommand,
    HistoryWebClearCommand,
    ListPromptsCommand,
    ListSessionsCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadHistoryCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpReconnectCommand,
    ModelFastCommand,
    ModelReasoningCommand,
    ModelsCommand,
    ModelSwitchCommand,
    ModelTaskBudgetCommand,
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
    ModelXSearchCommand,
    PinSessionCommand,
    PluginsCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowMarkdownCommand,
    ShowMcpStatusCommand,
    ShowSystemCommand,
    ShowUsageCommand,
    SkillsCommand,
    SwitchAgentCommand,
    TitleSessionCommand,
    UnknownCommand,
)
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.shell_detection import default_shell_command
from fast_agent.utils.slash_commands import parse_slash_command_line, split_subcommand_and_remainder
from fast_agent.utils.text import strip_casefold, strip_to_none

type _ValueCommandFactory = Callable[[str | None], CommandPayload]
type _ActionArgumentCommandFactory = Callable[[str, str | None], CommandPayload]
type _HistoryTurnErrorFormatter = Callable[[str], str]
type _NoArgumentCommandFactory = Callable[[], str | CommandPayload]
type _RemainderCommandParser = Callable[[str], CommandPayload]
type _PromptSubcommandParser = Callable[[str], CommandPayload]
type _SlashAliasParser = Callable[[str], str | CommandPayload]

_MODEL_VALUE_COMMAND_FACTORIES: dict[str, _ValueCommandFactory] = {
    "reasoning": ModelReasoningCommand,
    "task_budget": ModelTaskBudgetCommand,
    "verbosity": ModelVerbosityCommand,
    "fast": ModelFastCommand,
    "web_search": ModelWebSearchCommand,
    "x_search": ModelXSearchCommand,
    "web_fetch": ModelWebFetchCommand,
    "switch": ModelSwitchCommand,
}


_SESSION_PAYLOAD_FACTORIES: dict[str, _ValueCommandFactory] = {
    "list": lambda _argument: ListSessionsCommand(),
    "new": lambda argument: CreateSessionCommand(session_name=argument),
    "resume": lambda argument: ResumeSessionCommand(session_id=argument),
    "title": lambda argument: TitleSessionCommand(title=argument or ""),
    "fork": lambda argument: ForkSessionCommand(title=argument),
    "delete": lambda argument: ClearSessionsCommand(target=argument),
}

_MCP_SERVER_COMMAND_TYPES = {
    "disconnect": McpDisconnectCommand,
    "reconnect": McpReconnectCommand,
}

_McpTokenParser = Callable[[list[str], str], CommandPayload]


def _parse_mcp_server_name_command(tokens: list[str], _remainder: str) -> CommandPayload:
    subcmd = strip_casefold(tokens[0]) if tokens else ""
    server_command_type = _MCP_SERVER_COMMAND_TYPES[subcmd]
    intent = parse_mcp_server_name_tokens(
        tokens,
        usage=f"Usage: /mcp {subcmd} <server_name>",
    )
    return server_command_type(server_name=intent.server_name, error=intent.error)


def _parse_mcp_list_command(tokens: list[str], _remainder: str) -> CommandPayload:
    intent = parse_mcp_no_args_tokens(tokens, usage="Usage: /mcp list")
    if intent.error:
        return CommandError(intent.error)
    return McpListCommand()


_MCP_TOKEN_PARSERS: dict[str, _McpTokenParser] = {
    "list": _parse_mcp_list_command,
    **dict.fromkeys(_MCP_SERVER_COMMAND_TYPES, _parse_mcp_server_name_command),
}

if set(_MCP_TOKEN_PARSERS) | {"connect"} != set(MCP_TOP_LEVEL_ACTIONS):
    raise RuntimeError("TUI MCP parser table does not match shared MCP actions")

_SLASH_ACTION_FACTORIES: dict[str, _ActionArgumentCommandFactory] = {
    "skills": SkillsCommand,
    "cards": CardsCommand,
    "plugins": PluginsCommand,
}

_SIMPLE_SLASH_FACTORIES: dict[str, _NoArgumentCommandFactory] = {
    "help": lambda: "HELP",
    "system": ShowSystemCommand,
    "usage": ShowUsageCommand,
    "markdown": ShowMarkdownCommand,
    "reload": ReloadAgentsCommand,
    "mcpstatus": ShowMcpStatusCommand,
    "tools": ListToolsCommand,
    "prompts": ListPromptsCommand,
    "exit": lambda: "EXIT",
    "stop": lambda: "STOP",
}


def _parse_quoted_history_target(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError:
        return None

    if len(tokens) != 1:
        return None

    # Allow explicit quoting/escaping to force agent-name parsing for values
    # that would otherwise collide with /history subcommands.
    if stripped == tokens[0]:
        return None
    return tokens[0]


def _parse_hash_agent_command(body: str, *, quiet: bool) -> HashAgentCommand | str:
    stripped = body.strip()
    if not stripped:
        prefix = "##" if quiet else "#"
        return f"{prefix}{body}"

    for index, char in enumerate(stripped):
        if char.isspace():
            agent_name = stripped[:index]
            message = stripped[index:].strip()
            return HashAgentCommand(agent_name=agent_name, message=message, quiet=quiet)

    return HashAgentCommand(agent_name=stripped, message="", quiet=quiet)


def try_parse_hash_agent_command(text: str) -> HashAgentCommand | None:
    prefix = ""
    quiet = False
    if text.startswith("##"):
        prefix = "##"
        quiet = True
    elif text.startswith("#"):
        prefix = "#"
    else:
        return None

    body = text[len(prefix) :]
    if not body or body[0].isspace():
        return None

    parsed = _parse_hash_agent_command(body, quiet=quiet)
    return parsed if isinstance(parsed, HashAgentCommand) else None


def _parse_connect_command(remainder: str, *, usage: str) -> McpConnectCommand:
    if not remainder:
        return McpConnectCommand(request=None, error=usage)
    try:
        return McpConnectCommand(
            request=parse_connect_command_text(remainder),
            error=None,
        )
    except ValueError as exc:
        return McpConnectCommand(request=None, error=str(exc))


def _parse_attach_command(remainder: str) -> AttachCommand:
    if not remainder:
        return AttachCommand(paths=())

    try:
        tokens = split_commandline(remainder)
    except ValueError as exc:
        return AttachCommand(paths=(), error=str(exc))

    if len(tokens) == 1 and strip_casefold(tokens[0]) == "clear":
        return AttachCommand(paths=(), clear=True)

    return AttachCommand(paths=tuple(tokens))


def _parse_history_command(remainder: str) -> CommandPayload:
    if not remainder:
        return HistoryViewCommand(agent=None)

    quoted_target = _parse_quoted_history_target(remainder)
    if quoted_target is not None:
        return HistoryViewCommand(agent=quoted_target)

    try:
        tokens = split_commandline(remainder, syntax="posix")
    except ValueError:
        candidate = remainder.strip()
        return HistoryViewCommand(agent=candidate or None)

    if not tokens:
        return HistoryViewCommand(agent=None)

    intent = parse_current_agent_history_intent(remainder)
    shared_payload = _history_payload_from_shared_intent(intent)
    if shared_payload is not None:
        return shared_payload

    return HistoryViewCommand(agent=remainder)


def _history_load_payload_from_intent(intent: HistoryActionIntent) -> LoadHistoryCommand:
    if not intent.argument:
        return LoadHistoryCommand(
            filename=None,
            error="Filename required for /history load",
        )
    return LoadHistoryCommand(filename=intent.argument, error=None)


def _history_review_payload_from_intent(
    turn_index: int | None,
    turn_error: str | None,
) -> HistoryReviewCommand:
    error = _history_turn_error_message("detail", turn_error)
    return HistoryReviewCommand(
        turn_index=None if error else turn_index,
        error=error,
    )


def _history_rewind_payload_from_intent(
    turn_index: int | None,
    turn_error: str | None,
) -> HistoryRewindCommand:
    error = _history_turn_error_message("rewind", turn_error)
    return HistoryRewindCommand(
        turn_index=None if error else turn_index,
        error=error,
    )


def _history_turn_error_message(action: str, turn_error: str | None) -> str | None:
    if turn_error is None:
        return None
    formatter = _HISTORY_TURN_ERROR_FORMATTERS.get(turn_error)
    return formatter(action) if formatter is not None else None


_HISTORY_TURN_ERROR_FORMATTERS: dict[str, _HistoryTurnErrorFormatter] = {
    "missing": lambda action: f"Turn number required for /history {action}",
    "invalid": lambda _action: "Turn number must be an integer",
}


_HISTORY_PAYLOAD_FACTORIES: dict[
    str,
    "Callable[[HistoryActionIntent], CommandPayload]",
] = {
    "overview": lambda _intent: HistoryViewCommand(agent=None),
    "show": lambda intent: HistoryViewCommand(agent=intent.argument, view="table"),
    "save": lambda intent: SaveHistoryCommand(filename=intent.argument),
    "load": _history_load_payload_from_intent,
    "detail": lambda intent: _history_review_payload_from_intent(
        intent.turn_index,
        intent.turn_error,
    ),
    "rewind": lambda intent: _history_rewind_payload_from_intent(
        intent.turn_index,
        intent.turn_error,
    ),
    "fix": lambda intent: HistoryFixCommand(agent=intent.argument),
    "webclear": lambda intent: HistoryWebClearCommand(agent=intent.argument),
    "clear_last": lambda intent: ClearCommand(kind="clear_last", agent=intent.argument),
    "clear_all": lambda intent: ClearCommand(kind="clear_history", agent=intent.argument),
}


def _history_payload_from_shared_intent(
    intent: HistoryActionIntent,
) -> CommandPayload | None:
    payload_factory = _HISTORY_PAYLOAD_FACTORIES.get(intent.action)
    if payload_factory is None:
        return None
    return payload_factory(intent)


def _parse_session_command(remainder: str) -> CommandPayload:
    intent = parse_session_command_intent(remainder)
    if intent.action == "error":
        return CommandError(
            message=f"Invalid /session arguments: {intent.argument or 'parse error'}"
        )
    if intent.action in {"help", "unknown"}:
        return ListSessionsCommand(show_help=True)

    simple_payload = _simple_session_payload_from_intent(intent)
    if simple_payload is not None:
        return simple_payload

    if intent.action == "export":
        return ExportSessionCommand(
            target=intent.export_target,
            agent_name=intent.export_agent,
            output_path=intent.export_output,
            hf_dataset=intent.export_hf_dataset,
            hf_dataset_path=intent.export_hf_dataset_path,
            privacy_filter=intent.export_privacy_filter,
            privacy_filter_path=intent.export_privacy_filter_path,
            download_privacy_filter=intent.export_download_privacy_filter,
            privacy_filter_device=intent.export_privacy_filter_device,
            privacy_filter_variant=intent.export_privacy_filter_variant,
            show_redactions=intent.export_show_redactions,
            show_help=intent.export_help,
            error=intent.export_error,
        )
    return PinSessionCommand(value=intent.pin_value, target=intent.pin_target)


def _simple_session_payload_from_intent(
    intent: SessionCommandIntent,
) -> CommandPayload | None:
    factory = _SESSION_PAYLOAD_FACTORIES.get(intent.action)
    return factory(intent.argument) if factory is not None else None


def _parse_card_command(remainder: str) -> CommandPayload:
    intent = parse_card_load_intent(remainder)
    return LoadAgentCardCommand(
        filename=intent.filename,
        add_tool=intent.add_tool,
        remove_tool=intent.remove_tool,
        error=intent.error,
    )


def _parse_agent_command(remainder: str) -> CommandPayload:
    intent = parse_agent_tool_intent(remainder, require_tool_agent=True)
    return AgentCommand(
        agent_name=intent.agent_name,
        add_tool=intent.add_tool,
        remove_tool=intent.remove_tool,
        dump=intent.dump,
        error=intent.error,
    )


def _parse_mcp_command(remainder: str) -> CommandPayload:
    if not remainder:
        return ShowMcpStatusCommand()

    subcmd, sub_remainder = split_subcommand_and_remainder(remainder)
    subcmd = strip_casefold(subcmd)
    if subcmd == "connect":
        return _parse_connect_command(
            sub_remainder,
            usage=(
                "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] "
                "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
            ),
        )

    try:
        tokens = split_commandline(remainder, syntax="posix")
    except ValueError as exc:
        return _mcp_invalid_arguments_payload(subcmd, str(exc))

    return _parse_mcp_tokens(tokens, remainder)


def _parse_mcp_tokens(tokens: list[str], remainder: str) -> CommandPayload:
    subcmd = strip_casefold(tokens[0]) if tokens else ""
    parser = _MCP_TOKEN_PARSERS.get(subcmd)
    if parser is not None:
        return parser(tokens, remainder)
    return UnknownCommand(command=f"/mcp {remainder}".strip())


def _mcp_invalid_arguments_payload(subcmd: str, message: str) -> CommandPayload:
    error = f"Invalid arguments: {message}"
    if is_mcp_server_name_action(subcmd):
        server_command_type = _MCP_SERVER_COMMAND_TYPES[subcmd]
        return server_command_type(server_name=None, error=error)
    return CommandError(error)


def _parse_connect_alias_command(remainder: str) -> McpConnectCommand:
    return _parse_connect_command(
        remainder,
        usage="Usage: /connect <target>",
    )


def _single_token_or_raw_argument(remainder: str, tokens: list[str]) -> str:
    raw_argument = remainder[len(tokens[0]) :].strip()
    if len(tokens) == 2:
        return tokens[1]
    return raw_argument


def _prompt_load_payload(argument: str) -> LoadPromptCommand:
    if not argument:
        return LoadPromptCommand(filename=None, error="Filename required for /prompt load")
    return LoadPromptCommand(filename=argument, error=None)


_PROMPT_SUBCOMMAND_PARSERS: dict[str, _PromptSubcommandParser] = {
    "load": _prompt_load_payload,
}


def _prompt_selection_payload(remainder: str) -> CommandPayload:
    if strip_casefold(Path(remainder).suffix) in {".json", ".md"}:
        return LoadPromptCommand(filename=remainder, error=None)
    if remainder.isdigit():
        return SelectPromptCommand(prompt_index=int(remainder), prompt_name=None)
    return SelectPromptCommand(prompt_index=None, prompt_name=remainder)


def _parse_prompt_command(remainder: str) -> CommandPayload:
    if not remainder:
        return SelectPromptCommand(prompt_index=None, prompt_name=None)

    try:
        tokens = split_commandline(remainder, syntax="posix")
    except ValueError as exc:
        return CommandError(message=f"Invalid /prompt arguments: {exc}")

    if tokens:
        subcmd = strip_casefold(tokens[0])
        argument = _single_token_or_raw_argument(remainder, tokens)
        parser = _PROMPT_SUBCOMMAND_PARSERS.get(subcmd)
        if parser is not None:
            return parser(argument)

    return _prompt_selection_payload(remainder)


def _parse_model_command(
    cmd_line: str,
    remainder: str,
    *,
    default_action: ModelCommandAction = "reasoning",
) -> CommandPayload:
    intent = parse_model_command_intent(remainder, default_action=default_action)
    if intent.error is not None:
        return CommandError(message=f"Invalid /model arguments: {intent.error}")
    factory = _MODEL_VALUE_COMMAND_FACTORIES.get(intent.action)
    if factory is not None:
        return factory(intent.argument)
    if intent.action in MODEL_MANAGER_COMMAND_ACTIONS:
        return ModelsCommand(
            action=intent.action,
            argument=intent.argument,
            command_name="model",
        )
    return UnknownCommand(command=cmd_line)


def _parse_models_command(remainder: str) -> CommandPayload:
    intent = parse_model_command_intent(remainder, default_action="doctor")
    if intent.error is not None:
        return CommandError(message=f"Invalid /models arguments: {intent.error}")
    if intent.action in MODEL_MANAGER_COMMAND_ACTIONS:
        return ModelsCommand(
            action=intent.action,
            argument=intent.argument,
            command_name="models",
        )
    invalid_action = intent.raw_subcommand or intent.action
    return CommandError(
        message=f"Invalid /models action '{invalid_action}'. Use /model for runtime model settings."
    )


def _parse_model_slash_command(remainder: str) -> CommandPayload:
    cmd_line = f"/model {remainder}".strip()
    return _parse_model_command(cmd_line, remainder)


def _parse_models_slash_command(remainder: str) -> CommandPayload:
    return _parse_models_command(remainder)


def _parse_a2a_command(remainder: str) -> CommandPayload:
    if not remainder:
        return A2ACommand(action="status", argument=None)
    tokens = remainder.split(maxsplit=1)
    action = tokens[0].lower()
    argument = tokens[1].strip() if len(tokens) > 1 else None
    if action in {
        "list",
        "status",
        "tasks",
        "card",
        "reset",
        "connect",
        "transport",
        "help",
        "?",
        "-h",
        "--help",
        "commands",
    }:
        return A2ACommand(action=action, argument=argument)
    return A2ACommand(action=action, argument=argument, error=f"Unknown /a2a action: {action}")


def _parse_action_argument_command(
    command_name: str,
    remainder: str,
    factory: _ActionArgumentCommandFactory,
) -> CommandPayload:
    stripped = remainder.strip()
    if not stripped:
        return factory("list", None)
    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError as exc:
        return CommandError(message=f"Invalid /{command_name} arguments: {exc}")
    if not tokens:
        return factory("list", None)
    action_end = _first_shell_token_end(stripped)
    return factory(strip_casefold(tokens[0]), strip_to_none(stripped[action_end:]))


def _first_shell_token_end(text: str) -> int:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char.isspace():
            return index
    return len(text)


def _parse_history_alias_filename(remainder: str) -> str | None:
    stripped = remainder.strip()
    if not stripped:
        return None
    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError:
        return stripped
    return strip_to_none(" ".join(tokens))


def _parse_single_alias_value(remainder: str) -> str | None:
    stripped = remainder.strip()
    if not stripped:
        return None
    try:
        tokens = split_commandline(stripped, syntax="posix")
    except ValueError:
        return stripped
    if len(tokens) == 1:
        return tokens[0]
    return stripped


def _parse_load_history_alias(remainder: str) -> LoadHistoryCommand:
    filename = _parse_history_alias_filename(remainder)
    if filename is None:
        return LoadHistoryCommand(filename=None, error="Filename required for /history load")
    return LoadHistoryCommand(filename=filename, error=None)


def _parse_save_history_alias(remainder: str) -> SaveHistoryCommand:
    return SaveHistoryCommand(filename=_parse_history_alias_filename(remainder))


def _parse_resume_alias(remainder: str) -> ResumeSessionCommand:
    return ResumeSessionCommand(session_id=_parse_single_alias_value(remainder))


def _parse_fast_alias(remainder: str) -> ModelFastCommand:
    return ModelFastCommand(value=_parse_single_alias_value(remainder))


_SLASH_ALIAS_PARSERS: dict[str, _SlashAliasParser] = {
    "save_history": _parse_save_history_alias,
    "save": _parse_save_history_alias,
    "load_history": _parse_load_history_alias,
    "load": _parse_load_history_alias,
    "resume": _parse_resume_alias,
    "fast": _parse_fast_alias,
}


def _parse_slash_alias_command(
    cmd: str,
    remainder: str,
) -> str | CommandPayload | None:
    alias_parser = _SLASH_ALIAS_PARSERS.get(cmd)
    if alias_parser is not None:
        return alias_parser(remainder)
    factory = _SLASH_ACTION_FACTORIES.get(cmd)
    if factory is not None:
        return _parse_action_argument_command(cmd, remainder, factory)
    return None


_COMMAND_PARSERS: dict[str, _RemainderCommandParser] = {
    "a2a": _parse_a2a_command,
    "tasks": lambda remainder: A2ACommand(action="tasks", argument=remainder or None),
    "history": _parse_history_command,
    "session": _parse_session_command,
    "card": _parse_card_command,
    "agent": _parse_agent_command,
    "mcp": _parse_mcp_command,
    "connect": _parse_connect_alias_command,
    "prompt": _parse_prompt_command,
    "model": _parse_model_slash_command,
    "models": _parse_models_slash_command,
    "attach": _parse_attach_command,
    "check": lambda remainder: CheckCommand(argument=remainder or None),
    "commands": lambda remainder: CommandsCommand(argument=remainder or None),
}


def _parse_slash_command(cmd_line: str) -> str | CommandPayload:
    parsed = parse_slash_command_line(cmd_line)
    if parsed is None:
        return UnknownCommand(command=cmd_line)
    command_name, remainder = parsed
    if not command_name:
        return ""
    cmd = strip_casefold(command_name)

    simple_factory = _SIMPLE_SLASH_FACTORIES.get(cmd)
    if simple_factory is not None:
        if remainder:
            return CommandError(message=f"Unexpected arguments for /{cmd}: {remainder}")
        return simple_factory()

    parser = _COMMAND_PARSERS.get(cmd)
    if parser is not None:
        return parser(remainder)

    alias_result = _parse_slash_alias_command(cmd, remainder)
    if alias_result is not None:
        return alias_result

    return UnknownCommand(command=cmd_line)


def _parse_slash_input(cmd_line: str) -> str | CommandPayload:
    if not cmd_line[1:].strip():
        return ""
    return _parse_slash_command(cmd_line)


def _parse_shell_input(cmd_line: str) -> ShellCommand:
    command = cmd_line[1:].strip()
    return ShellCommand(command=command or default_shell_command())


def parse_special_input(text: str) -> str | CommandPayload:
    stripped = text.lstrip()
    cmd_line = stripped.splitlines()[0] if stripped.startswith("/") else text

    if cmd_line and cmd_line.startswith("/"):
        return _parse_slash_input(cmd_line)

    if cmd_line and cmd_line.startswith("@"):
        return SwitchAgentCommand(agent_name=cmd_line[1:].strip())

    parsed_hash_command = try_parse_hash_agent_command(cmd_line.lstrip())
    if parsed_hash_command is not None:
        return parsed_hash_command

    if cmd_line and cmd_line.startswith("!"):
        return _parse_shell_input(cmd_line)

    return text
