"""Pure command-line parsing for interactive prompt input."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from fast_agent.commands.mcp_command_intents import parse_mcp_session_tokens
from fast_agent.commands.shared_command_intents import (
    parse_current_agent_history_intent,
    parse_session_command_intent,
)
from fast_agent.mcp.connect_targets import parse_connect_command_text
from fast_agent.ui.command_payloads import (
    AgentCommand,
    AttachCommand,
    CardsCommand,
    ClearCommand,
    ClearSessionsCommand,
    CommandPayload,
    CreateSessionCommand,
    ForkSessionCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryReviewCommand,
    HistoryRewindCommand,
    HistoryShowCommand,
    HistoryWebClearCommand,
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
    ModelVerbosityCommand,
    ModelWebFetchCommand,
    ModelWebSearchCommand,
    PinSessionCommand,
    ReloadAgentsCommand,
    ResumeSessionCommand,
    SaveHistoryCommand,
    SelectPromptCommand,
    ShellCommand,
    ShowHistoryCommand,
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
from fast_agent.utils.slash_commands import split_subcommand_and_remainder


def _default_shell_command() -> str:
    if platform.system() == "Windows":
        for shell_name in ["pwsh", "powershell", "cmd"]:
            shell_path = shutil.which(shell_name)
            if shell_path:
                return shell_path
        return os.environ.get("COMSPEC", "cmd.exe")

    shell_env = os.environ.get("SHELL")
    if shell_env and Path(shell_env).exists():
        return shell_env

    for shell_name in ["bash", "zsh", "sh"]:
        shell_path = shutil.which(shell_name)
        if shell_path:
            return shell_path

    return "sh"


def _parse_quoted_history_target(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None

    if len(tokens) != 1:
        return None

    # Allow explicit quoting/escaping to force agent-name parsing for values
    # that would otherwise collide with /history subcommands.
    if stripped == tokens[0]:
        return None
    return tokens[0]


def _parse_mcp_single_server_name(tokens: list[str], *, usage: str) -> tuple[str | None, str | None]:
    name = tokens[1] if len(tokens) > 1 else None
    error = None if name else usage
    return name, error


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

    if len(tokens) == 1 and tokens[0].lower() == "clear":
        return AttachCommand(paths=(), clear=True)

    return AttachCommand(paths=tuple(tokens))


def _parse_history_command(remainder: str) -> CommandPayload:
    if not remainder:
        return ShowHistoryCommand(agent=None)

    quoted_target = _parse_quoted_history_target(remainder)
    if quoted_target is not None:
        return ShowHistoryCommand(agent=quoted_target)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        candidate = remainder.strip()
        return ShowHistoryCommand(agent=candidate or None)

    if not tokens:
        return ShowHistoryCommand(agent=None)

    subcmd = tokens[0].lower()
    argument = " ".join(tokens[1:]).strip()

    targeted_payload = _parse_targeted_history_action(subcmd, argument)
    if targeted_payload is not None:
        return targeted_payload

    intent = parse_current_agent_history_intent(remainder)
    shared_payload = _history_payload_from_shared_intent(intent)
    if shared_payload is not None:
        return shared_payload

    return ShowHistoryCommand(agent=remainder)


def _parse_targeted_history_action(subcmd: str, argument: str) -> CommandPayload | None:
    if subcmd == "rewind":
        return _parse_history_rewind_command(argument)
    if subcmd == "fix":
        return HistoryFixCommand(agent=argument or None)
    if subcmd == "webclear":
        return HistoryWebClearCommand(agent=argument or None)
    if subcmd == "clear":
        return _parse_history_clear_command(argument)
    return None


def _parse_history_rewind_command(argument: str) -> HistoryRewindCommand:
    stripped = argument.strip()
    if not stripped:
        return HistoryRewindCommand(
            turn_index=None,
            error="Turn number required for /history rewind",
        )
    try:
        turn_index = int(stripped)
    except ValueError:
        return HistoryRewindCommand(
            turn_index=None,
            error="Turn number must be an integer",
        )
    return HistoryRewindCommand(turn_index=turn_index, error=None)


def _history_payload_from_shared_intent(intent) -> CommandPayload | None:
    if intent.action == "overview":
        return ShowHistoryCommand(agent=None)
    if intent.action == "show":
        return HistoryShowCommand(agent=intent.argument)
    if intent.action == "save":
        return SaveHistoryCommand(filename=intent.argument)
    if intent.action == "load":
        if not intent.argument:
            return LoadHistoryCommand(
                filename=None,
                error="Filename required for /history load",
            )
        return LoadHistoryCommand(filename=intent.argument, error=None)
    if intent.action == "detail":
        return _history_review_payload_from_intent(intent.turn_index, intent.turn_error)
    return None


def _history_review_payload_from_intent(
    turn_index: int | None,
    turn_error: str | None,
) -> HistoryReviewCommand:
    if turn_error == "missing":
        return HistoryReviewCommand(
            turn_index=None,
            error="Turn number required for /history detail",
        )
    if turn_error == "invalid":
        return HistoryReviewCommand(
            turn_index=None,
            error="Turn number must be an integer",
        )
    return HistoryReviewCommand(turn_index=turn_index, error=None)


def _parse_history_clear_command(argument: str) -> ClearCommand:
    clear_tokens = argument.split(maxsplit=1) if argument else []
    action = clear_tokens[0].lower() if clear_tokens else "all"
    target_agent = clear_tokens[1].strip() if len(clear_tokens) > 1 else None
    if action == "last":
        return ClearCommand(kind="clear_last", agent=target_agent)
    if action == "all":
        return ClearCommand(kind="clear_history", agent=target_agent)
    return ClearCommand(kind="clear_history", agent=argument or None)


def _parse_session_command(remainder: str) -> CommandPayload:
    intent = parse_session_command_intent(remainder)
    if intent.action in {"help", "unknown"}:
        return ListSessionsCommand(show_help=True)
    if intent.action == "list":
        return ListSessionsCommand()
    if intent.action == "new":
        return CreateSessionCommand(session_name=intent.argument)
    if intent.action == "resume":
        return ResumeSessionCommand(session_id=intent.argument)
    if intent.action == "title":
        return TitleSessionCommand(title=intent.argument or "")
    if intent.action == "fork":
        return ForkSessionCommand(title=intent.argument)
    if intent.action == "delete":
        return ClearSessionsCommand(target=intent.argument)
    return PinSessionCommand(value=intent.pin_value, target=intent.pin_target)


def _parse_card_command(remainder: str) -> CommandPayload:
    if not remainder:
        return LoadAgentCardCommand(
            filename=None,
            add_tool=False,
            remove_tool=False,
            error="Filename required for /card",
        )

    try:
        tokens = shlex.split(remainder)
    except ValueError as exc:
        return LoadAgentCardCommand(
            filename=None,
            add_tool=False,
            remove_tool=False,
            error=f"Invalid arguments: {exc}",
        )

    add_tool = False
    remove_tool = False
    filename = None
    for token in tokens:
        if token in {"tool", "--tool", "--as-tool", "-t"}:
            add_tool = True
            continue
        if token in {"remove", "--remove", "--rm"}:
            remove_tool = True
            add_tool = True
            continue
        if filename is None:
            filename = token

    if not filename:
        return LoadAgentCardCommand(
            filename=None,
            add_tool=add_tool,
            remove_tool=remove_tool,
            error="Filename required for /card",
        )

    return LoadAgentCardCommand(
        filename=filename,
        add_tool=add_tool,
        remove_tool=remove_tool,
        error=None,
    )


def _parse_agent_command(remainder: str) -> CommandPayload:
    if not remainder:
        return AgentCommand(
            agent_name=None,
            add_tool=False,
            remove_tool=False,
            dump=False,
            error="Usage: /agent <name> --tool | /agent [name] --dump",
        )

    try:
        tokens = shlex.split(remainder)
    except ValueError as exc:
        return AgentCommand(
            agent_name=None,
            add_tool=False,
            remove_tool=False,
            dump=False,
            error=f"Invalid arguments: {exc}",
        )

    agent_command = _parse_agent_tokens(tokens)
    return _validate_agent_command(agent_command)


def _parse_agent_tokens(tokens: list[str]) -> AgentCommand:
    add_tool = False
    remove_tool = False
    dump = False
    agent_name = None
    unknown: list[str] = []

    for token in tokens:
        if token in {"tool", "--tool", "--as-tool", "-t"}:
            add_tool = True
            continue
        if token in {"remove", "--remove", "--rm"}:
            remove_tool = True
            add_tool = True
            continue
        if token in {"dump", "--dump", "-d"}:
            dump = True
            continue
        if agent_name is None:
            agent_name = token[1:] if token.startswith("@") else token
            continue
        unknown.append(token)

    error = f"Unexpected arguments: {', '.join(unknown)}" if unknown else None
    return AgentCommand(
        agent_name=agent_name,
        add_tool=add_tool,
        remove_tool=remove_tool,
        dump=dump,
        error=error,
    )


def _validate_agent_command(command: AgentCommand) -> AgentCommand:
    if command.error is not None:
        return command
    if command.add_tool and command.dump:
        return AgentCommand(
            agent_name=command.agent_name,
            add_tool=command.add_tool,
            remove_tool=command.remove_tool,
            dump=command.dump,
            error="Use either --tool or --dump, not both",
        )
    if not command.add_tool and not command.dump:
        return AgentCommand(
            agent_name=command.agent_name,
            add_tool=command.add_tool,
            remove_tool=command.remove_tool,
            dump=command.dump,
            error="Usage: /agent <name> --tool | /agent [name] --dump",
        )
    if command.add_tool and not command.agent_name:
        return AgentCommand(
            agent_name=command.agent_name,
            add_tool=command.add_tool,
            remove_tool=command.remove_tool,
            dump=command.dump,
            error="Agent name is required for /agent --tool",
        )
    return command


def _parse_mcp_command(remainder: str) -> CommandPayload:
    if not remainder:
        return ShowMcpStatusCommand()

    subcmd, sub_remainder = split_subcommand_and_remainder(remainder)
    subcmd = subcmd.lower()
    if subcmd == "connect":
        return _parse_connect_command(
            sub_remainder,
            usage=(
                "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] "
                "[--timeout <seconds>] [--oauth|--no-oauth] [--reconnect|--no-reconnect]"
            ),
        )

    try:
        tokens = shlex.split(remainder)
    except ValueError as exc:
        return McpConnectCommand(request=None, error=f"Invalid arguments: {exc}")

    subcmd = tokens[0].lower() if tokens else ""
    if subcmd == "list":
        return McpListCommand()
    if subcmd == "disconnect":
        name, error = _parse_mcp_single_server_name(
            tokens,
            usage="Usage: /mcp disconnect <server_name>",
        )
        return McpDisconnectCommand(server_name=name, error=error)
    if subcmd == "reconnect":
        name, error = _parse_mcp_single_server_name(
            tokens,
            usage="Usage: /mcp reconnect <server_name>",
        )
        return McpReconnectCommand(server_name=name, error=error)
    if subcmd == "session":
        return parse_mcp_session_tokens(tokens[1:])
    return UnknownCommand(command="mcp")


def _parse_connect_alias_command(remainder: str) -> McpConnectCommand:
    return _parse_connect_command(
        remainder,
        usage="Usage: /connect <target>",
    )


def _parse_prompt_command(remainder: str) -> CommandPayload:
    if not remainder:
        return SelectPromptCommand(prompt_index=None, prompt_name=None)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        tokens = []

    if tokens:
        subcmd = tokens[0].lower()
        argument = remainder[len(tokens[0]) :].strip()
        if subcmd == "load":
            if not argument:
                return LoadPromptCommand(filename=None, error="Filename required for /prompt load")
            return LoadPromptCommand(filename=argument, error=None)

    if remainder.lower().endswith((".json", ".md")):
        return LoadPromptCommand(filename=remainder, error=None)
    if remainder.isdigit():
        return SelectPromptCommand(prompt_index=int(remainder), prompt_name=None)
    return SelectPromptCommand(prompt_index=None, prompt_name=remainder)


def _parse_model_command(cmd_line: str, remainder: str) -> CommandPayload:
    if not remainder:
        return ModelReasoningCommand(value=None)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        tokens = remainder.split(maxsplit=1)

    if not tokens:
        return ModelReasoningCommand(value=None)

    subcmd = tokens[0].lower()
    argument = remainder[len(tokens[0]) :].strip()

    value = argument or None
    value_command_factories: dict[str, Callable[[str | None], CommandPayload]] = {
        "reasoning": ModelReasoningCommand,
        "verbosity": ModelVerbosityCommand,
        "fast": ModelFastCommand,
        "web_search": ModelWebSearchCommand,
        "web_fetch": ModelWebFetchCommand,
        "switch": ModelSwitchCommand,
    }
    factory = value_command_factories.get(subcmd)
    if factory is not None:
        return factory(value)
    if subcmd in {"doctor", "references", "catalog", "help"}:
        return ModelsCommand(action=subcmd, argument=value)
    return UnknownCommand(command=cmd_line)


def _parse_slash_alias_command(
    cmd: str,
    remainder: str,
    *,
    cmd_line: str,
) -> str | CommandPayload | None:
    if cmd in {"save_history", "save"}:
        filename = remainder or None
        return SaveHistoryCommand(filename=filename)
    if cmd in {"load_history", "load"}:
        if not remainder:
            return LoadHistoryCommand(filename=None, error="Filename required for /history load")
        return LoadHistoryCommand(filename=remainder, error=None)
    if cmd == "resume":
        return ResumeSessionCommand(session_id=remainder or None)
    if cmd == "fast":
        return ModelFastCommand(value=remainder or None)
    if cmd == "skills":
        if not remainder:
            return SkillsCommand(action="list", argument=None)
        tokens = remainder.split(maxsplit=1)
        action = tokens[0].lower()
        argument = tokens[1].strip() if len(tokens) > 1 else None
        return SkillsCommand(action=action, argument=argument)
    if cmd == "cards":
        if not remainder:
            return CardsCommand(action="list", argument=None)
        tokens = remainder.split(maxsplit=1)
        action = tokens[0].lower()
        argument = tokens[1].strip() if len(tokens) > 1 else None
        return CardsCommand(action=action, argument=argument)
    return None


def _parse_slash_command(cmd_line: str) -> str | CommandPayload:
    cmd_parts = cmd_line[1:].strip().split(maxsplit=1)
    cmd = cmd_parts[0].lower()
    remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

    simple_factories: dict[str, Callable[[], str | CommandPayload]] = {
        "help": lambda: "HELP",
        "system": ShowSystemCommand,
        "usage": ShowUsageCommand,
        "markdown": ShowMarkdownCommand,
        "reload": ReloadAgentsCommand,
        "mcpstatus": ShowMcpStatusCommand,
        "tools": ListToolsCommand,
        "exit": lambda: "EXIT",
        "stop": lambda: "STOP",
    }
    simple_factory = simple_factories.get(cmd)
    if simple_factory is not None:
        return simple_factory()

    command_parsers: dict[str, Callable[[str], CommandPayload]] = {
        "history": _parse_history_command,
        "session": _parse_session_command,
        "card": _parse_card_command,
        "agent": _parse_agent_command,
        "mcp": _parse_mcp_command,
        "connect": _parse_connect_alias_command,
        "prompt": _parse_prompt_command,
        "attach": _parse_attach_command,
    }
    parser = command_parsers.get(cmd)
    if parser is not None:
        return parser(remainder)
    if cmd == "model":
        return _parse_model_command(cmd_line, remainder)

    alias_result = _parse_slash_alias_command(cmd, remainder, cmd_line=cmd_line)
    if alias_result is not None:
        return alias_result

    return UnknownCommand(command=cmd_line)


def parse_special_input(text: str) -> str | CommandPayload:
    stripped = text.lstrip()
    cmd_line = stripped.splitlines()[0] if stripped.startswith("/") else text

    if cmd_line and cmd_line.startswith("/"):
        if cmd_line == "/":
            return ""
        return _parse_slash_command(cmd_line)

    if cmd_line and cmd_line.startswith("@"):
        return SwitchAgentCommand(agent_name=cmd_line[1:].strip())

    parsed_hash_command = try_parse_hash_agent_command(cmd_line.lstrip())
    if parsed_hash_command is not None:
        return parsed_hash_command

    if cmd_line and cmd_line.startswith("!"):
        command = cmd_line[1:].strip()
        if command:
            return ShellCommand(command=command)
        return ShellCommand(command=_default_shell_command())

    return text
