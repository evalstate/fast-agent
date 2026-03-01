"""Pure command-line parsing for interactive prompt input."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
from pathlib import Path

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
    ModelReasoningCommand,
    ModelsCommand,
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


def _parse_mcp_single_server_name(tokens: list[str], *, usage: str) -> tuple[str | None, str | None]:
    name = tokens[1] if len(tokens) > 1 else None
    error = None if name else usage
    return name, error


def parse_special_input(text: str) -> str | CommandPayload:
    stripped = text.lstrip()
    cmd_line = stripped.splitlines()[0] if stripped.startswith("/") else text

    if cmd_line and cmd_line.startswith("/"):
        if cmd_line == "/":
            return ""
        cmd_parts = cmd_line[1:].strip().split(maxsplit=1)
        cmd = cmd_parts[0].lower()

        if cmd == "help":
            return "HELP"
        if cmd == "system":
            return ShowSystemCommand()
        if cmd == "usage":
            return ShowUsageCommand()
        if cmd == "history":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return ShowHistoryCommand(agent=None)
            try:
                tokens = shlex.split(remainder)
            except ValueError:
                candidate = remainder.strip()
                return ShowHistoryCommand(agent=candidate or None)
            if not tokens:
                return ShowHistoryCommand(agent=None)
            subcmd = tokens[0].lower()
            argument = " ".join(tokens[1:]).strip()
            if subcmd == "show":
                return ShowHistoryCommand(agent=argument or None)
            if subcmd == "save":
                return SaveHistoryCommand(filename=argument or None)
            if subcmd == "load":
                if not argument:
                    return LoadHistoryCommand(
                        filename=None,
                        error="Filename required for /history load",
                    )
                return LoadHistoryCommand(filename=argument, error=None)
            if subcmd == "review":
                if not argument:
                    return HistoryReviewCommand(
                        turn_index=None,
                        error="Turn number required for /history review",
                    )
                try:
                    turn_index = int(argument)
                except ValueError:
                    return HistoryReviewCommand(turn_index=None, error="Turn number must be an integer")
                return HistoryReviewCommand(turn_index=turn_index, error=None)
            if subcmd == "fix":
                return HistoryFixCommand(agent=argument or None)
            if subcmd == "webclear":
                return HistoryWebClearCommand(agent=argument or None)
            if subcmd == "rewind":
                if not argument:
                    return HistoryRewindCommand(
                        turn_index=None,
                        error="Turn number required for /history rewind",
                    )
                try:
                    turn_index = int(argument)
                except ValueError:
                    return HistoryRewindCommand(turn_index=None, error="Turn number must be an integer")
                return HistoryRewindCommand(turn_index=turn_index, error=None)
            if subcmd == "clear":
                tokens = argument.split(maxsplit=1) if argument else []
                action = tokens[0].lower() if tokens else "all"
                target_agent = tokens[1].strip() if len(tokens) > 1 else None
                if action == "last":
                    return ClearCommand(kind="clear_last", agent=target_agent)
                if action == "all":
                    return ClearCommand(kind="clear_history", agent=target_agent)
                return ClearCommand(kind="clear_history", agent=argument or None)
            return ShowHistoryCommand(agent=remainder)
        if cmd == "markdown":
            return ShowMarkdownCommand()
        if cmd in ("save_history", "save"):
            filename = cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            return SaveHistoryCommand(filename=filename)
        if cmd in ("load_history", "load"):
            filename = cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            if not filename:
                return LoadHistoryCommand(filename=None, error="Filename required for /history load")
            return LoadHistoryCommand(filename=filename, error=None)
        if cmd == "resume":
            session_id = cmd_parts[1].strip() if len(cmd_parts) > 1 and cmd_parts[1].strip() else None
            return ResumeSessionCommand(session_id=session_id)
        if cmd == "session":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
            if subcmd == "resume":
                return ResumeSessionCommand(session_id=argument if argument else None)
            if subcmd == "list":
                return ListSessionsCommand()
            if subcmd == "new":
                return CreateSessionCommand(session_name=argument or None)
            if subcmd in {"delete", "clear"}:
                return ClearSessionsCommand(target=argument or None)
            if subcmd == "pin":
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
            if subcmd == "title":
                return TitleSessionCommand(title=argument if argument else "")
            if subcmd == "fork":
                return ForkSessionCommand(title=argument if argument else None)
            return ListSessionsCommand(show_help=True)
        if cmd == "card":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
        if cmd == "agent":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
            if unknown:
                return AgentCommand(
                    agent_name=agent_name,
                    add_tool=add_tool,
                    remove_tool=remove_tool,
                    dump=dump,
                    error=f"Unexpected arguments: {', '.join(unknown)}",
                )
            if add_tool and dump:
                return AgentCommand(
                    agent_name=agent_name,
                    add_tool=add_tool,
                    remove_tool=remove_tool,
                    dump=dump,
                    error="Use either --tool or --dump, not both",
                )
            if not add_tool and not dump:
                return AgentCommand(
                    agent_name=agent_name,
                    add_tool=add_tool,
                    remove_tool=remove_tool,
                    dump=dump,
                    error="Usage: /agent <name> --tool | /agent [name] --dump",
                )
            if add_tool and not agent_name:
                return AgentCommand(
                    agent_name=agent_name,
                    add_tool=add_tool,
                    remove_tool=remove_tool,
                    dump=dump,
                    error="Agent name is required for /agent --tool",
                )
            return AgentCommand(
                agent_name=agent_name,
                add_tool=add_tool,
                remove_tool=remove_tool,
                dump=dump,
                error=None,
            )
        if cmd == "reload":
            return ReloadAgentsCommand()
        if cmd == "mcpstatus":
            return ShowMcpStatusCommand()
        if cmd == "mcp":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
                session_tokens = tokens[1:]
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

                if action == "list":
                    if len(args) > 1:
                        return McpSessionCommand(
                            action="list",
                            server_identity=None,
                            session_id=None,
                            title=None,
                            clear_all=False,
                            error="Usage: /mcp session list [<server_or_mcp_name>]",
                        )
                    return McpSessionCommand(
                        action="list",
                        server_identity=args[0] if args else None,
                        session_id=None,
                        title=None,
                        clear_all=False,
                        error=None,
                    )

                if action == "jar":
                    if len(args) > 1:
                        return McpSessionCommand(
                            action="jar",
                            server_identity=None,
                            session_id=None,
                            title=None,
                            clear_all=False,
                            error="Usage: /mcp session jar [<server_or_mcp_name>]",
                        )
                    return McpSessionCommand(
                        action="jar",
                        server_identity=args[0] if args else None,
                        session_id=None,
                        title=None,
                        clear_all=False,
                        error=None,
                    )

                if action in {"new", "create"}:
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

                if action == "resume":
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

                if action == "use":
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

                if action == "clear":
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
            if subcmd == "connect":
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
                connect_args = tokens[1:]
                target_tokens: list[str] = []
                server_name: str | None = None
                auth_token: str | None = None
                timeout_seconds: float | None = None
                trigger_oauth: bool | None = None
                reconnect_on_disconnect: bool | None = None
                force_reconnect = False
                parse_error: str | None = None
                idx = 0
                while idx < len(connect_args):
                    token = connect_args[idx]
                    if token in {"--name", "-n"}:
                        idx += 1
                        if idx >= len(connect_args):
                            parse_error = "Missing value for --name"
                            break
                        server_name = connect_args[idx]
                    elif token == "--timeout":
                        idx += 1
                        if idx >= len(connect_args):
                            parse_error = "Missing value for --timeout"
                            break
                        try:
                            timeout_seconds = float(connect_args[idx])
                        except ValueError:
                            parse_error = "--timeout must be a number"
                            break
                    elif token == "--auth":
                        idx += 1
                        if idx >= len(connect_args):
                            parse_error = "Missing value for --auth"
                            break
                        auth_token = connect_args[idx]
                    elif token.startswith("--auth="):
                        auth_token = token.split("=", 1)[1]
                        if not auth_token:
                            parse_error = "Missing value for --auth"
                            break
                    elif token == "--oauth":
                        trigger_oauth = True
                    elif token == "--no-oauth":
                        trigger_oauth = False
                    elif token == "--reconnect":
                        force_reconnect = True
                    elif token == "--no-reconnect":
                        reconnect_on_disconnect = False
                    else:
                        target_tokens.append(token)
                    idx += 1

                target_text = _rebuild_mcp_target_text(target_tokens).strip()
                parsed_mode = _infer_mcp_connect_mode(target_text)
                if not parse_error and not target_text:
                    parse_error = "Connection target is required"

                return McpConnectCommand(
                    target_text=target_text,
                    parsed_mode=parsed_mode,
                    server_name=server_name,
                    auth_token=auth_token,
                    timeout_seconds=timeout_seconds,
                    trigger_oauth=trigger_oauth,
                    reconnect_on_disconnect=reconnect_on_disconnect,
                    force_reconnect=force_reconnect,
                    error=parse_error,
                )
            return UnknownCommand(command=cmd)
        if cmd == "connect":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
        if cmd == "prompt":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
        if cmd == "tools":
            return ListToolsCommand()
        if cmd == "model":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
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
            if subcmd == "reasoning":
                return ModelReasoningCommand(value=argument or None)
            if subcmd == "verbosity":
                return ModelVerbosityCommand(value=argument or None)
            if subcmd == "web_search":
                return ModelWebSearchCommand(value=argument or None)
            if subcmd == "web_fetch":
                return ModelWebFetchCommand(value=argument or None)
            return UnknownCommand(command=cmd_line)
        if cmd == "skills":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return SkillsCommand(action="list", argument=None)
            tokens = remainder.split(maxsplit=1)
            action = tokens[0].lower()
            argument = tokens[1].strip() if len(tokens) > 1 else None
            return SkillsCommand(action=action, argument=argument)
        if cmd == "cards":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return CardsCommand(action="list", argument=None)
            tokens = remainder.split(maxsplit=1)
            action = tokens[0].lower()
            argument = tokens[1].strip() if len(tokens) > 1 else None
            return CardsCommand(action=action, argument=argument)
        if cmd == "models":
            remainder = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
            if not remainder:
                return ModelsCommand(action="doctor", argument=None)
            tokens = remainder.split(maxsplit=1)
            action = tokens[0].lower()
            argument = tokens[1].strip() if len(tokens) > 1 else None
            return ModelsCommand(action=action, argument=argument)
        if cmd == "exit":
            return "EXIT"
        if cmd.lower() == "stop":
            return "STOP"

        return UnknownCommand(command=cmd_line)

    if cmd_line and cmd_line.startswith("@"):
        return SwitchAgentCommand(agent_name=cmd_line[1:].strip())

    if cmd_line and cmd_line.startswith("#"):
        rest = cmd_line[1:].strip()
        if " " in rest:
            agent_name, message = rest.split(" ", 1)
            return HashAgentCommand(agent_name=agent_name.strip(), message=message.strip())
        if rest:
            return HashAgentCommand(agent_name=rest.strip(), message="")

    if cmd_line and cmd_line.startswith("!"):
        command = cmd_line[1:].strip()
        if command:
            return ShellCommand(command=command)
        return ShellCommand(command=_default_shell_command())

    return text
