"""Canonical MCP connect target parsing and normalization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, cast

import mslex

from fast_agent.cli.commands.url_parser import generate_server_name as generate_url_server_name
from fast_agent.cli.commands.url_parser import parse_server_url, parse_server_urls
from fast_agent.utils.commandline import (
    CommandLineSyntax,
    join_commandline,
    resolve_commandline_syntax,
    split_commandline,
)

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings

McpConnectMode = Literal["url", "stdio", "npx", "uvx"]
McpTransport = Literal["http", "sse", "stdio"]

_FAST_AGENT_CONNECT_FLAG_NAMES: tuple[str, ...] = (
    "--auth",
    "--oauth",
    "--no-oauth",
    "--timeout",
    "--name",
    "-n",
    "--reconnect",
    "--no-reconnect",
)

_WHOLE_SINGLE_QUOTED_ARG_PATTERN = re.compile(r"(^|\s)'([^']+)'(?=\s|$)")


def _rewrite_shell_single_quotes_for_windows(text: str) -> str:
    return _WHOLE_SINGLE_QUOTED_ARG_PATTERN.sub(
        lambda match: f"{match.group(1)}{mslex.quote(match.group(2))}",
        text,
    )


def _split_connect_command_text(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> list[str]:
    if syntax == "auto" and _WHOLE_SINGLE_QUOTED_ARG_PATTERN.search(text):
        # Preserve shell-style single-quoted arguments on Windows without
        # misparsing apostrophes inside ordinary path/token text.
        if resolve_commandline_syntax(syntax) == "windows":
            return split_commandline(
                _rewrite_shell_single_quotes_for_windows(text),
                syntax="windows",
            )
        return split_commandline(text, syntax="posix")
    return split_commandline(text, syntax=syntax)


@dataclass(frozen=True, slots=True)
class NormalizedMcpTarget:
    mode: McpConnectMode
    transport: McpTransport | None
    url: str | None
    command: str | None
    args: tuple[str, ...]
    server_name: str | None


@dataclass(frozen=True, slots=True)
class McpConnectOptions:
    auth_token: str | None
    timeout_seconds: float | None
    trigger_oauth: bool | None
    reconnect_on_disconnect: bool | None
    force_reconnect: bool


@dataclass(frozen=True, slots=True)
class ParsedMcpConnectRequest:
    target: NormalizedMcpTarget
    options: McpConnectOptions


def _slugify_server_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-_").lower()
    return normalized or "mcp-server"


def _basenameish(value: str) -> str:
    return re.split(r"[/\\]", value.strip())[-1]


def _flag_name(token: str) -> str | None:
    if token in _FAST_AGENT_CONNECT_FLAG_NAMES:
        return token
    if token.startswith("--auth="):
        return "--auth"
    if token.startswith("--timeout="):
        return "--timeout"
    if token.startswith("--name="):
        return "--name"
    return None


def _build_url_target_flag_error(*, source_path: str, flag: str) -> str:
    if flag == "--auth":
        return (
            f"`{source_path}` must be a pure target string. "
            "Move --auth to `headers` or `auth` settings."
        )
    return (
        f"`{source_path}` must be a pure target string. "
        "Move fast-agent flags to structured server settings."
    )


def _validate_timeout(value: str) -> float:
    timeout_seconds = float(value)
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "Invalid value for --timeout: expected a finite number greater than 0"
        )
    return timeout_seconds


def infer_connect_mode_from_tokens(tokens: Sequence[str]) -> McpConnectMode:
    if not tokens:
        raise ValueError("Connection target is required")

    first = tokens[0].strip()
    lowered = first.lower()
    if lowered.startswith(("http://", "https://")):
        return "url"
    if first.startswith("@"):
        return "npx"
    if lowered == "npx":
        return "npx"
    if lowered == "uvx":
        return "uvx"
    return "stdio"


def infer_connect_mode_from_text(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> McpConnectMode:
    return infer_connect_mode_from_tokens(split_commandline(text, syntax=syntax))


def infer_connect_mode(target_text: str) -> McpConnectMode:
    return infer_connect_mode_from_text(target_text)


def infer_transport(target: NormalizedMcpTarget) -> McpTransport | None:
    if target.transport is not None:
        return target.transport
    if target.mode == "url":
        return None
    return "stdio"


def _normalize_target_tokens(
    tokens: Sequence[str],
    *,
    server_name: str | None = None,
) -> NormalizedMcpTarget:
    if not tokens:
        raise ValueError("Connection target is required")

    mode = infer_connect_mode_from_tokens(tokens)
    resolved_server_name = server_name.strip() if isinstance(server_name, str) and server_name.strip() else None

    if mode == "url":
        if len(tokens) != 1:
            raise ValueError("URL connect targets do not accept extra arguments")
        if len(parse_server_urls(tokens[0])) != 1:
            raise ValueError("Singular MCP connect targets do not support multiple URLs")
        _generated_name, transport, parsed_url = parse_server_url(tokens[0])
        return NormalizedMcpTarget(
            mode="url",
            transport=transport,
            url=parsed_url,
            command=None,
            args=(),
            server_name=resolved_server_name,
        )

    if mode == "npx":
        if tokens[0].startswith("@"):
            command = "npx"
            args = tuple(tokens)
        else:
            command = "npx"
            args = tuple(tokens[1:])
        if not args:
            raise ValueError("Connection target is required")
        return NormalizedMcpTarget(
            mode="npx",
            transport="stdio",
            url=None,
            command=command,
            args=args,
            server_name=resolved_server_name,
        )

    if mode == "uvx":
        args = tuple(tokens[1:])
        if not args:
            raise ValueError("Connection target is required")
        return NormalizedMcpTarget(
            mode="uvx",
            transport="stdio",
            url=None,
            command="uvx",
            args=args,
            server_name=resolved_server_name,
        )

    return NormalizedMcpTarget(
        mode="stdio",
        transport="stdio",
        url=None,
        command=tokens[0],
        args=tuple(tokens[1:]),
        server_name=resolved_server_name,
    )


def normalize_connect_target_text(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
    server_name: str | None = None,
) -> NormalizedMcpTarget:
    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("Connection target is required")
    return _normalize_target_tokens(
        _split_connect_command_text(normalized_text, syntax=syntax),
        server_name=server_name,
    )


def infer_server_name(target: str | NormalizedMcpTarget) -> str:
    normalized_target = (
        normalize_connect_target_text(target) if isinstance(target, str) else target
    )
    if normalized_target.server_name:
        return normalized_target.server_name

    if normalized_target.mode == "url":
        url = normalized_target.url
        if not url:
            return "mcp-server"
        return generate_url_server_name(url)

    if normalized_target.mode in {"npx", "uvx"}:
        package = normalized_target.args[0] if normalized_target.args else normalized_target.command or ""
        if package.startswith("@") and package.count("@") > 1:
            package = package.rsplit("@", 1)[0]
        elif not package.startswith("@"):
            package = package.split("@", 1)[0]
        return _slugify_server_name(package.rsplit("/", 1)[-1])

    command = normalized_target.command or ""
    if command:
        return _slugify_server_name(_basenameish(command))

    return "mcp-server"


def parse_connect_command_tokens(tokens: Sequence[str]) -> ParsedMcpConnectRequest:
    if not tokens:
        raise ValueError("Connection target is required")

    target_tokens: list[str] = []
    server_name: str | None = None
    auth_token: str | None = None
    timeout_seconds: float | None = None
    trigger_oauth: bool | None = None
    reconnect_on_disconnect: bool | None = None
    force_reconnect = False

    idx = 0
    token_list = list(tokens)
    while idx < len(token_list):
        token = token_list[idx]
        if token in {"--name", "-n"}:
            idx += 1
            if idx >= len(token_list):
                raise ValueError("Missing value for --name")
            server_name = token_list[idx]
        elif token.startswith("--name="):
            server_name = token.split("=", 1)[1]
            if not server_name:
                raise ValueError("Missing value for --name")
        elif token == "--auth":
            idx += 1
            if idx >= len(token_list):
                raise ValueError("Missing value for --auth")
            auth_token = token_list[idx]
        elif token.startswith("--auth="):
            auth_token = token.split("=", 1)[1]
            if not auth_token:
                raise ValueError("Missing value for --auth")
        elif token == "--timeout":
            idx += 1
            if idx >= len(token_list):
                raise ValueError("Missing value for --timeout")
            timeout_seconds = _validate_timeout(token_list[idx])
        elif token.startswith("--timeout="):
            timeout_seconds = _validate_timeout(token.split("=", 1)[1])
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

    return ParsedMcpConnectRequest(
        target=_normalize_target_tokens(target_tokens, server_name=server_name),
        options=McpConnectOptions(
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            trigger_oauth=trigger_oauth,
            reconnect_on_disconnect=reconnect_on_disconnect,
            force_reconnect=force_reconnect,
        ),
    )


def parse_connect_command_text(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> ParsedMcpConnectRequest:
    return parse_connect_command_tokens(_split_connect_command_text(text, syntax=syntax))


def render_normalized_target(
    target: NormalizedMcpTarget,
    *,
    syntax: CommandLineSyntax = "auto",
) -> str:
    return join_commandline(_render_target_argv(target), syntax=syntax)


def _render_target_argv(target: NormalizedMcpTarget) -> list[str]:
    if target.mode == "url":
        return [target.url] if target.url else []

    if target.mode == "npx" and target.command == "npx" and target.args:
        if target.args[0].startswith("@"):
            return list(target.args)

    argv: list[str] = []
    if target.command:
        argv.append(target.command)
    argv.extend(target.args)
    return argv


def render_connect_request(
    request: ParsedMcpConnectRequest,
    *,
    redact_auth: bool = False,
    syntax: CommandLineSyntax = "auto",
) -> str:
    argv = _render_target_argv(request.target)
    if request.target.server_name:
        argv.extend(["--name", request.target.server_name])
    if request.options.auth_token:
        argv.extend(["--auth", "[REDACTED]" if redact_auth else request.options.auth_token])
    if request.options.timeout_seconds is not None:
        argv.extend(["--timeout", str(request.options.timeout_seconds)])
    if request.options.trigger_oauth is True:
        argv.append("--oauth")
    elif request.options.trigger_oauth is False:
        argv.append("--no-oauth")
    if request.options.reconnect_on_disconnect is False:
        argv.append("--no-reconnect")
    if request.options.force_reconnect:
        argv.append("--reconnect")
    return join_commandline(argv, syntax=syntax)


def normalize_connect_config_target(
    *,
    target: str | None = None,
    transport: str | None = None,
    url: str | None = None,
    command: str | None = None,
    args: Sequence[str] | None = None,
    server_name: str | None = None,
    headers: Mapping[str, str] | None = None,
    auth: Mapping[str, Any] | None = None,
    reconnect_on_disconnect: bool | None = None,
    source_path: str = "target",
) -> tuple[NormalizedMcpTarget, dict[str, Any]]:
    overrides: dict[str, Any] = {}
    if transport is not None:
        overrides["transport"] = transport
    if url is not None:
        overrides["url"] = url
    if command is not None:
        overrides["command"] = command
    if args is not None:
        overrides["args"] = list(args)
    if headers is not None:
        overrides["headers"] = dict(headers)
    if auth is not None:
        overrides["auth"] = dict(auth)
    if reconnect_on_disconnect is not None:
        overrides["reconnect_on_disconnect"] = reconnect_on_disconnect

    if target is not None:
        normalized_target_text = target.strip()
        if not normalized_target_text:
            raise ValueError(f"`{source_path}` must be a non-empty string")
        tokens = _split_connect_command_text(normalized_target_text)
        if infer_connect_mode_from_tokens(tokens) == "url":
            for token in tokens:
                flag = _flag_name(token)
                if flag is not None:
                    raise ValueError(_build_url_target_flag_error(source_path=source_path, flag=flag))
        return _normalize_target_tokens(tokens, server_name=server_name), overrides

    if url is not None:
        return _normalize_target_tokens([url], server_name=server_name), overrides

    if command is not None:
        command_tokens = [command, *(list(args) if args else [])]
        return _normalize_target_tokens(command_tokens, server_name=server_name), overrides

    raise ValueError(f"`{source_path}` must be a non-empty string")


def build_server_config_from_target(
    target: str | NormalizedMcpTarget,
    *,
    server_name: str | None = None,
    auth_token: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> tuple[str, MCPServerSettings]:
    from fast_agent.config import MCPServerSettings

    normalized_target = (
        normalize_connect_target_text(target, server_name=server_name)
        if isinstance(target, str)
        else target
    )
    effective_target = (
        normalized_target
        if server_name is None or normalized_target.server_name == server_name
        else NormalizedMcpTarget(
            mode=normalized_target.mode,
            transport=normalized_target.transport,
            url=normalized_target.url,
            command=normalized_target.command,
            args=normalized_target.args,
            server_name=server_name,
        )
    )

    resolved_name = infer_server_name(effective_target)
    payload: dict[str, Any] = {"name": resolved_name}

    if effective_target.mode == "url":
        url_value = effective_target.url
        if not url_value:
            raise ValueError("Connection target is required")
        parsed_urls = parse_server_urls(url_value, auth_token=auth_token)
        if len(parsed_urls) != 1:
            raise ValueError("Singular MCP connect targets do not support multiple URLs")
        _generated_name, transport, parsed_url, headers = parsed_urls[0]
        payload.update(
            {
                "transport": transport,
                "url": parsed_url,
                "headers": headers,
            }
        )
    else:
        if not effective_target.command:
            raise ValueError("Connection target is required")
        payload.update(
            {
                "transport": "stdio",
                "command": effective_target.command,
                "args": list(effective_target.args),
            }
        )

    if overrides:
        payload.update(dict(overrides))

    resolved_settings = MCPServerSettings.model_validate(payload)
    final_name: str = resolved_settings.name or resolved_name
    return final_name, resolved_settings


def resolve_target_entry(
    target: str,
    *,
    default_name: str | None,
    overrides: Mapping[str, Any],
    source_path: str,
) -> tuple[str, MCPServerSettings]:
    normalized_target, _normalized_overrides = normalize_connect_config_target(
        target=target,
        server_name=default_name,
        transport=cast("str | None", overrides.get("transport")),
        url=cast("str | None", overrides.get("url")),
        command=cast("str | None", overrides.get("command")),
        args=cast("Sequence[str] | None", overrides.get("args")),
        headers=cast("Mapping[str, str] | None", overrides.get("headers")),
        auth=cast("Mapping[str, Any] | None", overrides.get("auth")),
        reconnect_on_disconnect=cast("bool | None", overrides.get("reconnect_on_disconnect")),
        source_path=source_path,
    )

    resolved_name, resolved_settings = build_server_config_from_target(
        normalized_target,
        auth_token=None,
        overrides=dict(overrides),
    )
    return resolved_name, resolved_settings


__all__ = [
    "McpConnectMode",
    "McpConnectOptions",
    "McpTransport",
    "NormalizedMcpTarget",
    "ParsedMcpConnectRequest",
    "build_server_config_from_target",
    "infer_connect_mode",
    "infer_connect_mode_from_text",
    "infer_connect_mode_from_tokens",
    "infer_server_name",
    "infer_transport",
    "normalize_connect_config_target",
    "normalize_connect_target_text",
    "parse_connect_command_text",
    "parse_connect_command_tokens",
    "render_connect_request",
    "render_normalized_target",
    "resolve_target_entry",
]
