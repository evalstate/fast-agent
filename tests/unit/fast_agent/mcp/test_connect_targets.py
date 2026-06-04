from __future__ import annotations

import pytest

import fast_agent.mcp.connect_targets as connect_targets_module
from fast_agent.mcp.connect_targets import (
    build_server_config_from_target,
    connect_flag_name,
    connect_flag_requires_value_token,
    infer_server_name,
    mcp_connect_flag_descriptions,
    normalize_connect_config_target,
    parse_connect_command_text,
    render_connect_request,
)
from fast_agent.utils import commandline


def _force_windows_commandline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _resolve_windows_syntax(syntax: commandline.CommandLineSyntax = "auto") -> str:
        return "windows" if syntax == "auto" else syntax

    monkeypatch.setattr(commandline, "resolve_commandline_syntax", _resolve_windows_syntax)
    monkeypatch.setattr(
        connect_targets_module,
        "resolve_commandline_syntax",
        _resolve_windows_syntax,
    )


def test_parse_connect_command_text_preserves_quoted_windows_path() -> None:
    request = parse_connect_command_text('--name docs "C:\\Program Files\\Tool\\tool.exe" --flag')

    assert request.target.mode == "stdio"
    assert request.target.command == "C:\\Program Files\\Tool\\tool.exe"
    assert request.target.args == ("--flag",)
    assert request.target.server_name == "docs"


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("--name", "--name"),
        ("-n", "--name"),
        ("--name=docs", "--name"),
        ("--auth=token", "--auth"),
        ("--timeout=7", "--timeout"),
        ("--oauth", "--oauth"),
        ("--server-owned", None),
    ],
)
def test_connect_flag_name_returns_canonical_fast_agent_flags(
    token: str,
    expected: str | None,
) -> None:
    assert connect_flag_name(token) == expected


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("--name", True),
        ("-n", True),
        ("--auth", True),
        ("--timeout", True),
        ("--name=docs", False),
        ("--oauth", False),
        ("--server-owned", False),
    ],
)
def test_connect_flag_requires_value_token_matches_parser_value_flags(
    token: str,
    expected: bool,
) -> None:
    assert connect_flag_requires_value_token(token) is expected


def test_mcp_connect_flag_descriptions_returns_copy() -> None:
    descriptions = mcp_connect_flag_descriptions()
    descriptions["--name"] = "changed"

    assert mcp_connect_flag_descriptions()["--name"] == "set attached server name"


def test_parse_connect_command_text_accepts_single_quoted_args_on_windows(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text("https://example.com --auth 'Bearer token-from-cli'")

    assert request.options.auth_token == "Bearer token-from-cli"


@pytest.mark.parametrize(
    ("target_text", "mode"),
    [
        ("'https://example.com'", "url"),
        ("'HTTPS://example.com'", "url"),
        ("NPX demo-server", "npx"),
        ("UVX demo-server", "uvx"),
        ("'@scope/server'", "npx"),
    ],
)
def test_infer_connect_mode_uses_windows_single_quote_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    target_text: str,
    mode: str,
) -> None:
    _force_windows_commandline(monkeypatch)

    assert connect_targets_module.infer_connect_mode(target_text) == mode


def test_parse_connect_command_text_preserves_apostrophes_in_windows_path(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text(r"--name docs C:\Users\O'Brien\tool.exe --flag")

    assert request.target.mode == "stdio"
    assert request.target.command == r"C:\Users\O'Brien\tool.exe"
    assert request.target.args == ("--flag",)
    assert request.target.server_name == "docs"


def test_parse_connect_command_text_preserves_apostrophes_in_windows_tokens(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text("https://example.com --auth O'Reilly")

    assert request.options.auth_token == "O'Reilly"


def test_parse_connect_command_text_accepts_mixed_windows_apostrophes_and_single_quotes(
    monkeypatch,
) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text(r"--auth 'Bearer token' C:\Users\O'Brien\tool.exe")

    assert request.target.mode == "stdio"
    assert request.target.command == r"C:\Users\O'Brien\tool.exe"
    assert request.options.auth_token == "Bearer token"


def test_parse_connect_command_text_consumes_documented_trailing_flags_for_stdio() -> None:
    request = parse_connect_command_text("python server.py --timeout 30 --name workspace")

    assert request.target.mode == "stdio"
    assert request.target.command == "python"
    assert request.target.args == ("server.py",)
    assert request.target.server_name == "workspace"
    assert request.options.timeout_seconds == 30.0


def test_parse_connect_command_text_delimits_stdio_server_args() -> None:
    request = parse_connect_command_text("python server.py -- --timeout 30 --name workspace")

    assert request.target.mode == "stdio"
    assert request.target.command == "python"
    assert request.target.args == ("server.py", "--", "--timeout", "30", "--name", "workspace")
    assert request.target.server_name is None
    assert request.options.timeout_seconds is None


def test_parse_connect_command_text_accepts_documented_trailing_name_for_stdio() -> None:
    request = parse_connect_command_text('demo-server --root "My Folder" --name docs')

    assert request.target.mode == "stdio"
    assert request.target.command == "demo-server"
    assert request.target.args == ("--root", "My Folder")
    assert request.target.server_name == "docs"


def test_parse_connect_command_text_accepts_leading_fast_agent_flags_for_stdio() -> None:
    request = parse_connect_command_text("--name docs --timeout 7 python server.py --timeout 30")

    assert request.target.command == "python"
    assert request.target.args == ("server.py", "--timeout", "30")
    assert request.target.server_name == "docs"
    assert request.options.timeout_seconds == 7.0


def test_parse_connect_command_text_accepts_inline_fast_agent_flag_values() -> None:
    request = parse_connect_command_text(
        "--name=docs --auth=secret-token --timeout=7 python server.py"
    )

    assert request.target.command == "python"
    assert request.target.args == ("server.py",)
    assert request.target.server_name == "docs"
    assert request.options.auth_token == "secret-token"
    assert request.options.timeout_seconds == 7.0


@pytest.mark.parametrize("command", ['--name "" python server.py', "python server.py --name ''"])
def test_parse_connect_command_text_rejects_empty_name_value(command: str) -> None:
    with pytest.raises(ValueError, match="Missing value for --name"):
        parse_connect_command_text(command)


@pytest.mark.parametrize("command", ['""', 'npx ""', 'uvx ""', "@"])
def test_parse_connect_command_text_rejects_empty_target_tokens(command: str) -> None:
    with pytest.raises(ValueError, match="Connection target is required"):
        parse_connect_command_text(command)


def test_parse_connect_command_text_accepts_documented_trailing_flags_for_npx() -> None:
    request = parse_connect_command_text(
        "npx demo-server --name docs --timeout 7 --no-oauth --no-reconnect"
    )

    assert request.target.mode == "npx"
    assert request.target.command == "npx"
    assert request.target.args == ("demo-server",)
    assert request.target.server_name == "docs"
    assert request.options.timeout_seconds == 7.0
    assert request.options.trigger_oauth is False
    assert request.options.reconnect_on_disconnect is False


def test_parse_connect_command_text_accepts_positive_switch_flags_for_npx() -> None:
    request = parse_connect_command_text("npx demo-server --oauth --reconnect")

    assert request.target.mode == "npx"
    assert request.target.args == ("demo-server",)
    assert request.options.trigger_oauth is True
    assert request.options.force_reconnect is True


@pytest.mark.parametrize(
    "command",
    [
        "npx demo-server --name docs --name other",
        "npx demo-server --name=docs --name=other",
        "npx demo-server --oauth --no-oauth",
    ],
)
def test_parse_connect_command_text_rejects_duplicate_trailing_fast_agent_flags(
    command: str,
) -> None:
    with pytest.raises(ValueError, match="Duplicate MCP connect flag"):
        parse_connect_command_text(command)


@pytest.mark.parametrize(
    "command",
    [
        "--name first --name second python server.py",
        "--name=first --name=second python server.py",
        "--oauth --no-oauth https://example.com",
    ],
)
def test_parse_connect_command_text_rejects_duplicate_leading_fast_agent_flags(
    command: str,
) -> None:
    with pytest.raises(ValueError, match="Duplicate MCP connect flag"):
        parse_connect_command_text(command)


def test_parse_connect_command_text_delimits_npx_server_args() -> None:
    request = parse_connect_command_text("npx demo-server -- --name server-owned")

    assert request.target.mode == "npx"
    assert request.target.args == ("demo-server", "--", "--name", "server-owned")
    assert request.target.server_name is None


def test_parse_connect_command_text_delimiter_allows_server_args_after_fast_agent_flags() -> None:
    request = parse_connect_command_text("npx demo-server --name docs -- --name server-owned")

    assert request.target.mode == "npx"
    assert request.target.args == ("demo-server", "--", "--name", "server-owned")
    assert request.target.server_name == "docs"


def test_parse_connect_command_text_delimiter_stops_url_option_parsing() -> None:
    with pytest.raises(ValueError, match="extra arguments"):
        parse_connect_command_text("-- https://example.com --auth token")


@pytest.mark.parametrize(
    "command",
    [
        "https://example.com --name docs --name other",
        "https://example.com --oauth --no-oauth",
        "--auth secret https://example.com --auth other",
    ],
)
def test_parse_connect_command_text_rejects_duplicate_url_fast_agent_flags(
    command: str,
) -> None:
    with pytest.raises(ValueError, match="Duplicate MCP connect flag"):
        parse_connect_command_text(command)


@pytest.mark.parametrize(
    ("target_text", "mode"),
    [
        ("https://example.com", "url"),
        ("https://example.com/sse", "url"),
        ("@scope/server", "npx"),
        ("npx demo-server", "npx"),
        ("uvx demo-server", "uvx"),
        ("python demo.py", "stdio"),
    ],
)
def test_parse_connect_command_text_infers_mode(target_text: str, mode: str) -> None:
    request = parse_connect_command_text(target_text)
    assert request.target.mode == mode


def test_render_connect_request_redacts_auth() -> None:
    request = parse_connect_command_text("https://example.com --auth secret-token --name docs")

    rendered = render_connect_request(request, redact_auth=True)

    assert "secret-token" not in rendered
    assert "[REDACTED]" in rendered
    assert "--name docs" in rendered


def test_parse_connect_command_text_rejects_multiple_urls() -> None:
    with pytest.raises(ValueError, match="multiple URLs"):
        parse_connect_command_text("https://one.example,https://two.example")


def test_infer_server_name_handles_localhost_urls() -> None:
    request = parse_connect_command_text("http://localhost:8080/api")
    assert infer_server_name(request.target).startswith("localhost_8080_")


def test_build_server_config_from_target_handles_scoped_package() -> None:
    request = parse_connect_command_text("@modelcontextprotocol/server-filesystem .")

    built_config = build_server_config_from_target(request.target)

    assert built_config.server_name == "server-filesystem"
    assert built_config.settings.transport == "stdio"
    assert built_config.settings.command == "npx"
    assert built_config.settings.args == ["@modelcontextprotocol/server-filesystem", "."]


def test_build_server_config_from_target_slugifies_generated_names_case_insensitively() -> None:
    request = parse_connect_command_text('"Demo Server"')

    built_config = build_server_config_from_target(request.target)

    assert built_config.server_name == "demo-server"


def test_infer_server_name_skips_npx_launcher_flags() -> None:
    request = parse_connect_command_text("npx --yes @modelcontextprotocol/server-filesystem .")

    built_config = build_server_config_from_target(request.target)

    assert built_config.server_name == "server-filesystem"
    assert built_config.settings.command == "npx"
    assert built_config.settings.args == [
        "--yes",
        "@modelcontextprotocol/server-filesystem",
        ".",
    ]


def test_infer_server_name_skips_uvx_launcher_value_options() -> None:
    request = parse_connect_command_text("uvx --from mcp-server server")

    built_config = build_server_config_from_target(request.target)

    assert built_config.server_name == "server"
    assert built_config.settings.command == "uvx"
    assert built_config.settings.args == ["--from", "mcp-server", "server"]


def test_normalize_connect_config_target_rejects_embedded_fast_agent_flags() -> None:
    with pytest.raises(ValueError, match="pure target string"):
        normalize_connect_config_target(
            target="HTTPS://demo.hf.space --auth token",
            source_path="mcp.targets[0].target",
        )


def test_normalize_connect_config_target_rejects_url_with_leading_cli_flags() -> None:
    with pytest.raises(ValueError, match="pure target string"):
        normalize_connect_config_target(
            target="--auth token https://demo.hf.space",
            source_path="mcp.targets[0].target",
        )


def test_normalize_connect_config_target_allows_stdio_flags_in_target_args() -> None:
    normalized = normalize_connect_config_target(
        target="python server.py --timeout 30 --name workspace",
        source_path="mcp.targets[0].target",
    )

    assert normalized.overrides == {}
    assert normalized.target.mode == "stdio"
    assert normalized.target.command == "python"
    assert normalized.target.args == ("server.py", "--timeout", "30", "--name", "workspace")


def test_normalize_connect_config_target_allows_stdio_url_args_with_flag_names() -> None:
    normalized = normalize_connect_config_target(
        target="uvx demo-server --endpoint https://example.com --name workspace",
        source_path="mcp.targets[0].target",
    )

    assert normalized.overrides == {}
    assert normalized.target.mode == "uvx"
    assert normalized.target.command == "uvx"
    assert normalized.target.args == (
        "demo-server",
        "--endpoint",
        "https://example.com",
        "--name",
        "workspace",
    )


def test_normalize_connect_config_target_ignores_blank_configured_server_name() -> None:
    normalized = normalize_connect_config_target(
        target="npx demo-server",
        server_name="  ",
        source_path="mcp.targets[0].target",
    )

    built_config = build_server_config_from_target(normalized.target)

    assert normalized.target.server_name is None
    assert built_config.server_name == "demo-server"


def test_normalize_connect_config_target_accepts_single_quoted_args_on_windows(
    monkeypatch,
) -> None:
    _force_windows_commandline(monkeypatch)

    normalized = normalize_connect_config_target(
        target="python -c 'print(1)'",
        source_path="mcp.targets[0].target",
    )

    assert normalized.target.mode == "stdio"
    assert normalized.target.command == "python"
    assert normalized.target.args == ("-c", "print(1)")


def test_infer_server_name_accepts_single_quoted_args_on_windows(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    assert infer_server_name("python -c 'print(1)'") == "python"


def test_build_server_config_from_target_accepts_single_quoted_args_on_windows(
    monkeypatch,
) -> None:
    _force_windows_commandline(monkeypatch)

    built_config = build_server_config_from_target("python -c 'print(1)'")

    assert built_config.server_name == "python"
    assert built_config.settings.command == "python"
    assert built_config.settings.args == ["-c", "print(1)"]
