"""Canonical MCP connect target parsing and normalization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import mslex

from fast_agent.cli.commands.url_parser import generate_server_name as generate_url_server_name
from fast_agent.cli.commands.url_parser import parse_server_url, parse_server_urls
from fast_agent.commands.option_parsing import matches_option_token, read_option_token_value
from fast_agent.utils.commandline import (
    CommandLineSyntax,
    join_commandline,
    resolve_commandline_syntax,
    split_commandline,
)
from fast_agent.utils.text import strip_casefold, strip_to_none

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from fast_agent.config import MCPServerSettings

McpConnectMode = Literal["url", "stdio", "npx", "uvx"]
McpTransport = Literal["http", "sse", "stdio"]

MCP_CONNECT_FLAG_DESCRIPTIONS: dict[str, str] = {
    "--name": "set attached server name",
    "-n": "set attached server name",
    "--auth": "set bearer token for URL servers",
    "--timeout": "set startup timeout in seconds",
    "--oauth": "enable oauth flow",
    "--no-oauth": "disable oauth flow",
    "--reconnect": "force reconnect and refresh tools",
    "--no-reconnect": "disable reconnect-on-disconnect",
}
_NPX_PACKAGE_VALUE_OPTIONS = frozenset({"--package", "-p"})
_UVX_PACKAGE_VALUE_OPTIONS = frozenset({"--from"})

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


@dataclass(frozen=True, slots=True)
class NormalizedConnectConfigTarget:
    target: NormalizedMcpTarget
    overrides: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BuiltMcpServerConfig:
    server_name: str
    settings: MCPServerSettings


def _slugify_server_name(value: str) -> str:
    normalized = strip_casefold(re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-_"))
    return normalized or "mcp-server"


def _launcher_name_argument(mode: McpConnectMode, args: Sequence[str]) -> str:
    if not args:
        return ""

    value_options = (
        _NPX_PACKAGE_VALUE_OPTIONS
        if mode == "npx"
        else _UVX_PACKAGE_VALUE_OPTIONS
        if mode == "uvx"
        else frozenset()
    )
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in value_options:
            skip_next = True
            continue
        if any(arg.startswith(f"{option}=") for option in value_options):
            continue
        if not arg.startswith("-"):
            return arg
    return args[0]


def _basenameish(value: str) -> str:
    return re.split(r"[/\\]", value.strip())[-1]


def _build_url_target_flag_error(*, source_path: str, flag: str) -> str:
    if flag == "--auth":
        return (
            f"`{source_path}` must be a pure target string. "
            "Move --auth to `access_token`, `headers`, or `auth` settings."
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


@dataclass(slots=True)
class _ConnectOptionState:
    server_name: str | None = None
    auth_token: str | None = None
    timeout_seconds: float | None = None
    trigger_oauth: bool | None = None
    reconnect_on_disconnect: bool | None = None
    force_reconnect: bool = False


@dataclass(frozen=True, slots=True)
class _ConnectTargetSlice:
    tokens: list[str]
    delimited: bool


@dataclass(frozen=True, slots=True)
class _ConnectSwitchOption:
    apply: Callable[[_ConnectOptionState], None]
    already_set: Callable[[_ConnectOptionState], bool]


@dataclass(frozen=True, slots=True)
class _ConnectValueOption:
    option_names: tuple[str, ...]
    error_name: str
    apply: Callable[[_ConnectOptionState, str], None]
    already_set: Callable[[_ConnectOptionState], bool]

    def matches(self, token: str) -> bool:
        return matches_option_token(token, self.option_names)


def _consume_connect_option_value(
    token_list: Sequence[str],
    idx: int,
    *,
    option_names: tuple[str, ...],
    error_name: str,
) -> tuple[str, int]:
    parsed = read_option_token_value(token_list, idx, option_names, error_name=error_name)
    if parsed.error is not None:
        raise ValueError(parsed.error)
    return parsed.require_value(), parsed.next_index


def _set_trigger_oauth_enabled(options: _ConnectOptionState) -> None:
    options.trigger_oauth = True


def _set_trigger_oauth_disabled(options: _ConnectOptionState) -> None:
    options.trigger_oauth = False


def _set_force_reconnect(options: _ConnectOptionState) -> None:
    options.force_reconnect = True


def _set_reconnect_on_disconnect_disabled(options: _ConnectOptionState) -> None:
    options.reconnect_on_disconnect = False


def _trigger_oauth_already_set(options: _ConnectOptionState) -> bool:
    return options.trigger_oauth is not None


def _force_reconnect_already_set(options: _ConnectOptionState) -> bool:
    return options.force_reconnect


def _reconnect_on_disconnect_already_set(options: _ConnectOptionState) -> bool:
    return options.reconnect_on_disconnect is not None


def _set_server_name(options: _ConnectOptionState, value: str) -> None:
    options.server_name = value


def _set_auth_token(options: _ConnectOptionState, value: str) -> None:
    options.auth_token = value


def _set_timeout_seconds(options: _ConnectOptionState, value: str) -> None:
    options.timeout_seconds = _validate_timeout(value)


def _server_name_already_set(options: _ConnectOptionState) -> bool:
    return options.server_name is not None


def _auth_token_already_set(options: _ConnectOptionState) -> bool:
    return options.auth_token is not None


def _timeout_seconds_already_set(options: _ConnectOptionState) -> bool:
    return options.timeout_seconds is not None


_CONNECT_VALUE_OPTIONS: tuple[_ConnectValueOption, ...] = (
    _ConnectValueOption(
        option_names=("--name", "-n"),
        error_name="--name",
        apply=_set_server_name,
        already_set=_server_name_already_set,
    ),
    _ConnectValueOption(
        option_names=("--auth",),
        error_name="--auth",
        apply=_set_auth_token,
        already_set=_auth_token_already_set,
    ),
    _ConnectValueOption(
        option_names=("--timeout",),
        error_name="--timeout",
        apply=_set_timeout_seconds,
        already_set=_timeout_seconds_already_set,
    ),
)


_CONNECT_SWITCH_OPTIONS: dict[str, _ConnectSwitchOption] = {
    "--oauth": _ConnectSwitchOption(_set_trigger_oauth_enabled, _trigger_oauth_already_set),
    "--no-oauth": _ConnectSwitchOption(_set_trigger_oauth_disabled, _trigger_oauth_already_set),
    "--reconnect": _ConnectSwitchOption(_set_force_reconnect, _force_reconnect_already_set),
    "--no-reconnect": _ConnectSwitchOption(
        _set_reconnect_on_disconnect_disabled,
        _reconnect_on_disconnect_already_set,
    ),
}


def _connect_value_option_for_token(token: str) -> _ConnectValueOption | None:
    return next((option for option in _CONNECT_VALUE_OPTIONS if option.matches(token)), None)


def _flag_name(token: str) -> str | None:
    value_option = _connect_value_option_for_token(token)
    if value_option is not None:
        return value_option.error_name
    return token if token in _CONNECT_SWITCH_OPTIONS else None


def connect_flag_name(token: str) -> str | None:
    """Return the canonical fast-agent MCP connect flag name for a token."""
    return _flag_name(token)


def connect_flag_requires_value_token(token: str) -> bool:
    return _connect_value_option_for_token(token) is not None and "=" not in token


def mcp_connect_flag_descriptions() -> dict[str, str]:
    return dict(MCP_CONNECT_FLAG_DESCRIPTIONS)


def _consume_connect_option(
    token_list: Sequence[str],
    idx: int,
    options: _ConnectOptionState,
) -> int | None:
    token = token_list[idx]
    value_option = _connect_value_option_for_token(token)
    if value_option is not None:
        value, next_idx = _consume_connect_option_value(
            token_list,
            idx,
            option_names=value_option.option_names,
            error_name=value_option.error_name,
        )
        value_option.apply(options, value)
        return next_idx

    switch_option = _CONNECT_SWITCH_OPTIONS.get(token)
    if switch_option is not None:
        switch_option.apply(options)
        return idx + 1
    return None


def _connect_option_already_set(token: str, options: _ConnectOptionState) -> bool:
    value_option = _connect_value_option_for_token(token)
    if value_option is not None:
        return value_option.already_set(options)

    switch_option = _CONNECT_SWITCH_OPTIONS.get(token)
    return switch_option.already_set(options) if switch_option is not None else False


def _duplicate_connect_option_error(token: str) -> ValueError:
    flag = connect_flag_name(token) or token
    return ValueError(
        f"Duplicate MCP connect flag {flag}; use -- before server-owned arguments."
    )


def infer_connect_mode_from_tokens(tokens: Sequence[str]) -> McpConnectMode:
    if not tokens:
        raise ValueError("Connection target is required")

    first = tokens[0].strip()
    if not first:
        raise ValueError("Connection target is required")
    normalized_first = strip_casefold(first)
    if normalized_first.startswith(("http://", "https://")):
        return "url"
    if first.startswith("@"):
        return "npx"
    if normalized_first == "npx":
        return "npx"
    if normalized_first == "uvx":
        return "uvx"
    return "stdio"


def infer_connect_mode_from_text(
    text: str,
    *,
    syntax: CommandLineSyntax = "auto",
) -> McpConnectMode:
    return infer_connect_mode_from_tokens(_split_connect_command_text(text, syntax=syntax))


def infer_connect_mode(target_text: str) -> McpConnectMode:
    return infer_connect_mode_from_text(target_text)


def infer_transport(target: NormalizedMcpTarget) -> McpTransport | None:
    if target.transport is not None:
        return target.transport
    if target.mode == "url":
        return None
    return "stdio"


def _stdio_target(
    *,
    mode: McpConnectMode,
    command: str,
    args: Sequence[str],
    server_name: str | None,
) -> NormalizedMcpTarget:
    return NormalizedMcpTarget(
        mode=mode,
        transport="stdio",
        url=None,
        command=command,
        args=tuple(args),
        server_name=server_name,
    )


def _require_package_args(args: Sequence[str], *, reject_bare_scope: bool = False) -> None:
    if not args or not args[0].strip() or (reject_bare_scope and args[0].strip() == "@"):
        raise ValueError("Connection target is required")


def _normalize_target_tokens(
    tokens: Sequence[str],
    *,
    server_name: str | None = None,
) -> NormalizedMcpTarget:
    if not tokens:
        raise ValueError("Connection target is required")

    mode = infer_connect_mode_from_tokens(tokens)
    resolved_server_name = strip_to_none(server_name)

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
            args = tuple(tokens)
        else:
            args = tuple(tokens[1:])
        _require_package_args(args, reject_bare_scope=True)
        return _stdio_target(
            mode="npx",
            command="npx",
            args=args,
            server_name=resolved_server_name,
        )

    if mode == "uvx":
        args = tuple(tokens[1:])
        _require_package_args(args)
        return _stdio_target(
            mode="uvx",
            command="uvx",
            args=args,
            server_name=resolved_server_name,
        )

    return _stdio_target(
        mode="stdio",
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
        package = _launcher_name_argument(normalized_target.mode, normalized_target.args)
        if not package:
            package = normalized_target.command or ""
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

    options = _ConnectOptionState()
    target_slice = _split_connect_options_from_target(tokens, options)
    target_tokens = _consume_trailing_connect_options(target_slice, options)

    return ParsedMcpConnectRequest(
        target=_normalize_target_tokens(target_tokens, server_name=options.server_name),
        options=McpConnectOptions(
            auth_token=options.auth_token,
            timeout_seconds=options.timeout_seconds,
            trigger_oauth=options.trigger_oauth,
            reconnect_on_disconnect=options.reconnect_on_disconnect,
            force_reconnect=options.force_reconnect,
        ),
    )


def _split_connect_options_from_target(
    tokens: Sequence[str],
    options: _ConnectOptionState,
) -> _ConnectTargetSlice:
    token_list = list(tokens)
    idx = 0
    while idx < len(token_list):
        token = token_list[idx]
        if token == "--":
            return _ConnectTargetSlice(tokens=token_list[idx + 1 :], delimited=True)
        if _connect_option_already_set(token, options):
            raise _duplicate_connect_option_error(token)
        next_idx = _consume_connect_option(token_list, idx, options)
        if next_idx is None:
            break
        idx = next_idx

    return _ConnectTargetSlice(tokens=token_list[idx:], delimited=False)


def _consume_trailing_connect_options(
    target_slice: _ConnectTargetSlice,
    options: _ConnectOptionState,
) -> list[str]:
    if not target_slice.tokens:
        raise ValueError("Connection target is required")
    if target_slice.delimited:
        return target_slice.tokens

    mode = infer_connect_mode_from_tokens(target_slice.tokens)
    if mode == "url":
        return _consume_trailing_url_options(target_slice.tokens, options)
    return _extract_connect_options_from_package_target(target_slice.tokens, options)


def _consume_trailing_url_options(
    target_tokens: list[str],
    options: _ConnectOptionState,
) -> list[str]:
    if len(target_tokens) == 1:
        return target_tokens

    url_target = target_tokens[0]
    trailing_idx = 1
    while trailing_idx < len(target_tokens):
        if _connect_option_already_set(target_tokens[trailing_idx], options):
            raise _duplicate_connect_option_error(target_tokens[trailing_idx])
        next_idx = _consume_connect_option(target_tokens, trailing_idx, options)
        if next_idx is None:
            break
        trailing_idx = next_idx
    if trailing_idx != len(target_tokens):
        raise ValueError("URL connect targets do not accept extra arguments")
    return [url_target]


def _extract_connect_options_from_package_target(
    target_tokens: list[str],
    options: _ConnectOptionState,
) -> list[str]:
    parsed_target = [target_tokens[0]]
    trailing_options = _ConnectOptionState()
    idx = 1
    while idx < len(target_tokens):
        if target_tokens[idx] == "--":
            parsed_target.extend(target_tokens[idx:])
            break
        if _connect_option_already_set(target_tokens[idx], trailing_options):
            raise _duplicate_connect_option_error(target_tokens[idx])
        if _connect_option_already_set(target_tokens[idx], options):
            parsed_target.append(target_tokens[idx])
            idx += 1
            continue
        next_idx = _consume_connect_option(target_tokens, idx, options)
        if next_idx is None:
            parsed_target.append(target_tokens[idx])
            idx += 1
            continue
        _consume_connect_option(target_tokens, idx, trailing_options)
        idx = next_idx
    return parsed_target


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

    if (
        target.mode == "npx"
        and target.command == "npx"
        and target.args
        and target.args[0].startswith("@")
    ):
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
    argv: list[str] = []
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
    argv.extend(_render_target_argv(request.target))
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
) -> NormalizedConnectConfigTarget:
    overrides = _connect_config_overrides(
        transport=transport,
        url=url,
        command=command,
        args=args,
        headers=headers,
        auth=auth,
        reconnect_on_disconnect=reconnect_on_disconnect,
    )

    if target is not None:
        return _normalize_connect_config_target_text(
            target,
            server_name=server_name,
            overrides=overrides,
            source_path=source_path,
        )

    explicit_tokens = _explicit_connect_config_tokens(url=url, command=command, args=args)
    if explicit_tokens is not None:
        return NormalizedConnectConfigTarget(
            target=_normalize_target_tokens(explicit_tokens, server_name=server_name),
            overrides=overrides,
        )

    raise ValueError(f"`{source_path}` must be a non-empty string")


def _connect_config_overrides(
    *,
    transport: str | None,
    url: str | None,
    command: str | None,
    args: Sequence[str] | None,
    headers: Mapping[str, str] | None,
    auth: Mapping[str, Any] | None,
    reconnect_on_disconnect: bool | None,
) -> dict[str, Any]:
    optional_values: tuple[tuple[str, Any], ...] = (
        ("transport", transport),
        ("url", url),
        ("command", command),
        ("headers", dict(headers) if headers is not None else None),
        ("auth", dict(auth) if auth is not None else None),
        ("reconnect_on_disconnect", reconnect_on_disconnect),
    )
    overrides: dict[str, Any] = {
        key: value for key, value in optional_values if value is not None
    }
    if args is not None:
        overrides["args"] = list(args)
    return overrides


def _normalize_connect_config_target_text(
    target: str,
    *,
    server_name: str | None,
    overrides: dict[str, Any],
    source_path: str,
) -> NormalizedConnectConfigTarget:
    normalized_target_text = target.strip()
    if not normalized_target_text:
        raise ValueError(f"`{source_path}` must be a non-empty string")

    tokens = _split_connect_command_text(normalized_target_text)
    _reject_url_target_cli_flags(tokens, source_path=source_path)
    return NormalizedConnectConfigTarget(
        target=_normalize_target_tokens(tokens, server_name=server_name),
        overrides=overrides,
    )


def _reject_url_target_cli_flags(tokens: Sequence[str], *, source_path: str) -> None:
    if not _is_url_config_target_tokens(tokens):
        return

    for token in tokens:
        flag = _flag_name(token)
        if flag is not None:
            raise ValueError(_build_url_target_flag_error(source_path=source_path, flag=flag))


def _is_url_config_target_tokens(tokens: Sequence[str]) -> bool:
    if infer_connect_mode_from_tokens(tokens) == "url":
        return True

    target_tokens = _tokens_after_leading_connect_flags(tokens)
    return bool(target_tokens) and infer_connect_mode_from_tokens(target_tokens) == "url"


def _tokens_after_leading_connect_flags(tokens: Sequence[str]) -> Sequence[str]:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            return tokens[idx + 1 :]
        if token in _CONNECT_SWITCH_OPTIONS:
            idx += 1
            continue
        if _connect_value_option_for_token(token) is None:
            return tokens[idx:]
        idx += 1 if "=" in token else 2
    return ()


def _explicit_connect_config_tokens(
    *,
    url: str | None,
    command: str | None,
    args: Sequence[str] | None,
) -> list[str] | None:
    if url is not None:
        return [url]
    if command is None:
        return None
    return [command, *(list(args) if args else [])]


def _normalized_config_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    management = normalized.get("management")
    if isinstance(management, str):
        normalized["management"] = strip_casefold(management)
    return normalized


def build_server_config_from_target(
    target: str | NormalizedMcpTarget,
    *,
    server_name: str | None = None,
    auth_token: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> BuiltMcpServerConfig:
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
        management = None
        if overrides is not None:
            raw_management = overrides.get("management")
            if isinstance(raw_management, str):
                management = strip_casefold(raw_management)
        _generated_name, transport, parsed_url = parse_server_url(url_value)
        payload.update(
            {
                "transport": transport,
                "url": url_value if management == "provider" else parsed_url,
            }
        )
        if auth_token is not None:
            payload["access_token"] = auth_token
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
        payload.update(_normalized_config_overrides(overrides))

    resolved_settings = MCPServerSettings.model_validate(payload)
    final_name: str = resolved_settings.name or resolved_name
    return BuiltMcpServerConfig(server_name=final_name, settings=resolved_settings)


def resolve_target_entry(
    target: str,
    *,
    default_name: str | None,
    overrides: Mapping[str, Any],
    source_path: str,
) -> BuiltMcpServerConfig:
    normalized = normalize_connect_config_target(
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
    return build_server_config_from_target(
        normalized.target,
        auth_token=None,
        overrides=dict(overrides),
    )


__all__ = [
    "BuiltMcpServerConfig",
    "McpConnectMode",
    "McpConnectOptions",
    "McpTransport",
    "NormalizedConnectConfigTarget",
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
    "connect_flag_name",
    "connect_flag_requires_value_token",
    "mcp_connect_flag_descriptions",
    "parse_connect_command_text",
    "parse_connect_command_tokens",
    "render_connect_request",
    "render_normalized_target",
    "resolve_target_entry",
]
