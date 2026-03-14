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

from fast_agent.commands.handlers import mcp_runtime as mcp_runtime_handlers
from fast_agent.ui.command_payloads import (
    AgentCommand,
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
    McpConnectMode,
    McpDisconnectCommand,
    McpListCommand,
    McpReconnectCommand,
    McpSessionCommand,
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


def _infer_mcp_connect_mode(target_text: str) -> McpConnectMode:
    stripped = target_text.strip().lower()
    if stripped.startswith(("http://", "https://")):
        return "url"
    if stripped.startswith("@"):
        return "npx"
    if stripped.startswith("npx "):
        return "npx"
    if stripped.startswith("uvx "):
        return "uvx"
    return "stdio"


def _rebuild_mcp_target_text(tokens: list[str]) -> str:
    if not tokens:
        return ""

    rebuilt_parts: list[str] = []
    for token in tokens:
        if token == "" or any(char.isspace() for char in token):
            rebuilt_parts.append(shlex.quote(token))
        else:
            rebuilt_parts.append(token)
    return " ".join(rebuilt_parts)


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

    simple_factories: dict[str, Callable[[str | None], CommandPayload]] = {
        "show": HistoryShowCommand,
        "save": SaveHistoryCommand,
        "fix": HistoryFixCommand,
        "webclear": HistoryWebClearCommand,
    }
    factory = simple_factories.get(subcmd)
    if factory is not None:
        return factory(argument or None)

    if subcmd == "load":
        if not argument:
            return LoadHistoryCommand(
                filename=None,
                error="Filename required for /history load",
            )
        return LoadHistoryCommand(filename=argument, error=None)

    turn_payload = _parse_history_turn_command(subcmd, argument)
    if turn_payload is not None:
        return turn_payload
    if subcmd == "clear":
        return _parse_history_clear_command(argument)

    return ShowHistoryCommand(agent=remainder)


def _parse_history_turn_command(subcmd: str, argument: str) -> CommandPayload | None:
    if subcmd not in {"detail", "review", "rewind"}:
        return None

    if subcmd in {"detail", "review"}:
        missing = "Turn number required for /history detail"
        build_payload = HistoryReviewCommand
    else:
        missing = "Turn number required for /history rewind"
        build_payload = HistoryRewindCommand

    if not argument:
        return build_payload(turn_index=None, error=missing)

    try:
        turn_index = int(argument)
    except ValueError:
        return build_payload(turn_index=None, error="Turn number must be an integer")

    return build_payload(turn_index=turn_index, error=None)


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
    if not remainder:
        return ListSessionsCommand(show_help=True)

    try:
        tokens = shlex.split(remainder)
    except ValueError:
        return ListSessionsCommand(show_help=True)

    if not tokens:
        return ListSessionsCommand(show_help=True)

    subcmd = tokens[0].lower()
    argument = remainder[len(tokens[0]) :].strip()

    if subcmd in {"resume", "list", "new", "delete", "clear", "title", "fork"}:
        return _parse_simple_session_command(subcmd, argument)
    if subcmd == "pin":
        return _parse_session_pin_command(argument)

    return ListSessionsCommand(show_help=True)


def _parse_simple_session_command(subcmd: str, argument: str) -> CommandPayload:
    if subcmd == "resume":
        return ResumeSessionCommand(session_id=argument if argument else None)
    if subcmd == "list":
        return ListSessionsCommand()
    if subcmd == "new":
        return CreateSessionCommand(session_name=argument or None)
    if subcmd in {"delete", "clear"}:
        return ClearSessionsCommand(target=argument or None)
    if subcmd == "title":
        return TitleSessionCommand(title=argument if argument else "")
    return ForkSessionCommand(title=argument if argument else None)


def _parse_session_pin_command(argument: str) -> PinSessionCommand:
    if argument:
        try:
            pin_tokens = shlex.split(argument)
        except ValueError:
            pin_tokens = argument.split(maxsplit=1)
    else:
        pin_tokens = []
    if not pin_tokens:
        return PinSessionCommand(value=None, target=None)
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
        return PinSessionCommand(value=first, target=target)
    return PinSessionCommand(value=None, target=argument or None)


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


def _parse_mcp_session_command(session_tokens: list[str]) -> McpSessionCommand:
    if not session_tokens:
        return McpSessionCommand(
            action="list",
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error=None,
        )

    action = session_tokens[0].lower()
    args = session_tokens[1:]

    if action in {"list", "jar"}:
        return _parse_mcp_session_single_optional_arg(
            action=action,
            args=args,
            usage=f"Usage: /mcp session {action} [<server_or_mcp_name>]",
        )
    if action in {"new", "create"}:
        return _parse_mcp_session_new(args)
    if action in {"resume", "use"}:
        return _parse_mcp_session_use(args)
    if action == "clear":
        return _parse_mcp_session_clear(args)

    return McpSessionCommand(
        action="list",
        server_identity=action,
        session_id=None,
        title=None,
        clear_all=False,
        error=(
            None
            if not args
            else "Usage: /mcp session [list [server]|jar [server]|new [server] [--title <title>]|use <server> <session_id>|clear [server|--all]]"
        ),
    )


def _parse_mcp_session_single_optional_arg(
    *,
    action: str,
    args: list[str],
    usage: str,
) -> McpSessionCommand:
    if len(args) > 1:
        return McpSessionCommand(
            action=action,  # ty: ignore[invalid-argument-type]
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error=usage,
        )
    return McpSessionCommand(
        action=action,  # ty: ignore[invalid-argument-type]
        server_identity=args[0] if args else None,
        session_id=None,
        title=None,
        clear_all=False,
        error=None,
    )


def _parse_mcp_session_new(args: list[str]) -> McpSessionCommand:
    server_identity: str | None = None
    title: str | None = None
    parse_error: str | None = None
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token == "--title":
            idx += 1
            if idx >= len(args):
                parse_error = "Missing value for --title"
                break
            title = args[idx]
        elif token.startswith("--title="):
            title = token.split("=", 1)[1] or None
            if title is None:
                parse_error = "Missing value for --title"
                break
        elif token.startswith("--"):
            parse_error = f"Unknown flag: {token}"
            break
        elif server_identity is None:
            server_identity = token
        else:
            parse_error = f"Unexpected argument: {token}"
            break
        idx += 1

    return McpSessionCommand(
        action="new",
        server_identity=server_identity,
        session_id=None,
        title=title,
        clear_all=False,
        error=parse_error,
    )


def _parse_mcp_session_use(args: list[str]) -> McpSessionCommand:
    if len(args) != 2:
        return McpSessionCommand(
            action="use",
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error="Usage: /mcp session use <server_or_mcp_name> <session_id>",
        )
    return McpSessionCommand(
        action="use",
        server_identity=args[0],
        session_id=args[1],
        title=None,
        clear_all=False,
        error=None,
    )


def _parse_mcp_session_clear(args: list[str]) -> McpSessionCommand:
    clear_all = False
    server_identity: str | None = None
    parse_error: str | None = None
    for token in args:
        if token == "--all":
            clear_all = True
            continue
        if token.startswith("--"):
            parse_error = f"Unknown flag: {token}"
            break
        if server_identity is None:
            server_identity = token
        else:
            parse_error = f"Unexpected argument: {token}"
            break

    if parse_error is None and clear_all and server_identity is not None:
        parse_error = "Use either --all or a specific server, not both"

    if parse_error is None and not clear_all and server_identity is None:
        clear_all = True

    return McpSessionCommand(
        action="clear",
        server_identity=server_identity,
        session_id=None,
        title=None,
        clear_all=clear_all,
        error=parse_error,
    )


def _parse_mcp_connect_command(tokens: list[str]) -> McpConnectCommand:
    if len(tokens) < 2:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error=(
                "Usage: /mcp connect <target> [--name <server>] [--auth <token-value>] [--timeout <seconds>] "
                "[--oauth|--no-oauth] [--reconnect|--no-reconnect]"
            ),
        )

    connect_text = _rebuild_mcp_target_text(tokens[1:]).strip()
    try:
        parsed = mcp_runtime_handlers.parse_connect_input(connect_text)
    except ValueError as exc:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error=str(exc),
        )

    return McpConnectCommand(
        target_text=parsed.target_text,
        parsed_mode=_infer_mcp_connect_mode(parsed.target_text),
        server_name=parsed.server_name,
        auth_token=parsed.auth_token,
        timeout_seconds=parsed.timeout_seconds,
        trigger_oauth=parsed.trigger_oauth,
        reconnect_on_disconnect=parsed.reconnect_on_disconnect,
        force_reconnect=parsed.force_reconnect,
        error=None,
    )


def _parse_mcp_command(remainder: str) -> CommandPayload:
    if not remainder:
        return ShowMcpStatusCommand()

    try:
        tokens = shlex.split(remainder)
    except ValueError as exc:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error=f"Invalid arguments: {exc}",
        )

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
        return _parse_mcp_session_command(tokens[1:])
    if subcmd == "connect":
        return _parse_mcp_connect_command(tokens)
    return UnknownCommand(command="mcp")


def _parse_connect_alias_command(remainder: str) -> McpConnectCommand:
    parsed_mode = _infer_mcp_connect_mode(remainder)
    if not remainder:
        return McpConnectCommand(
            target_text="",
            parsed_mode="stdio",
            server_name=None,
            auth_token=None,
            timeout_seconds=None,
            trigger_oauth=None,
            reconnect_on_disconnect=None,
            force_reconnect=False,
            error="Usage: /connect <target>",
        )
    return McpConnectCommand(
        target_text=remainder,
        parsed_mode=parsed_mode,
        server_name=None,
        auth_token=None,
        timeout_seconds=None,
        trigger_oauth=None,
        reconnect_on_disconnect=None,
        force_reconnect=False,
        error=None,
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
    if subcmd in {"doctor", "aliases", "catalog", "help"}:
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

    if cmd_line and cmd_line.startswith("##"):
        quiet_body = cmd_line[2:]
        if quiet_body and not quiet_body[0].isspace():
            return _parse_hash_agent_command(quiet_body, quiet=True)
        return text

    if cmd_line and cmd_line.startswith("#"):
        return _parse_hash_agent_command(cmd_line[1:], quiet=False)

    if cmd_line and cmd_line.startswith("!"):
        command = cmd_line[1:].strip()
        if command:
            return ShellCommand(command=command)
        return ShellCommand(command=_default_shell_command())

    return text
