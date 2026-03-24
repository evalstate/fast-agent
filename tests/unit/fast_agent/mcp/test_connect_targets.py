from __future__ import annotations

import pytest

import fast_agent.mcp.connect_targets as connect_targets_module
from fast_agent.mcp.connect_targets import (
    build_server_config_from_target,
    infer_server_name,
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
    request = parse_connect_command_text('"C:\\Program Files\\Tool\\tool.exe" --flag --name docs')

    assert request.target.mode == "stdio"
    assert request.target.command == "C:\\Program Files\\Tool\\tool.exe"
    assert request.target.args == ("--flag",)
    assert request.target.server_name == "docs"


def test_parse_connect_command_text_accepts_single_quoted_args_on_windows(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text("https://example.com --auth 'Bearer token-from-cli'")

    assert request.options.auth_token == "Bearer token-from-cli"


def test_parse_connect_command_text_preserves_apostrophes_in_windows_path(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    request = parse_connect_command_text(r"C:\Users\O'Brien\tool.exe --flag --name docs")

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

    request = parse_connect_command_text(r"C:\Users\O'Brien\tool.exe --auth 'Bearer token'")

    assert request.target.mode == "stdio"
    assert request.target.command == r"C:\Users\O'Brien\tool.exe"
    assert request.options.auth_token == "Bearer token"


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

    resolved_name, settings = build_server_config_from_target(request.target)

    assert resolved_name == "server-filesystem"
    assert settings.transport == "stdio"
    assert settings.command == "npx"
    assert settings.args == ["@modelcontextprotocol/server-filesystem", "."]


def test_normalize_connect_config_target_rejects_embedded_fast_agent_flags() -> None:
    with pytest.raises(ValueError, match="pure target string"):
        normalize_connect_config_target(
            target="https://demo.hf.space --auth token",
            source_path="mcp.targets[0].target",
        )


def test_normalize_connect_config_target_allows_stdio_flags_in_target_args() -> None:
    normalized_target, overrides = normalize_connect_config_target(
        target="python server.py --timeout 30 --name workspace",
        source_path="mcp.targets[0].target",
    )

    assert overrides == {}
    assert normalized_target.mode == "stdio"
    assert normalized_target.command == "python"
    assert normalized_target.args == ("server.py", "--timeout", "30", "--name", "workspace")


def test_normalize_connect_config_target_accepts_single_quoted_args_on_windows(
    monkeypatch,
) -> None:
    _force_windows_commandline(monkeypatch)

    normalized_target, _overrides = normalize_connect_config_target(
        target="python -c 'print(1)'",
        source_path="mcp.targets[0].target",
    )

    assert normalized_target.mode == "stdio"
    assert normalized_target.command == "python"
    assert normalized_target.args == ("-c", "print(1)")


def test_infer_server_name_accepts_single_quoted_args_on_windows(monkeypatch) -> None:
    _force_windows_commandline(monkeypatch)

    assert infer_server_name("python -c 'print(1)'") == "python"


def test_build_server_config_from_target_accepts_single_quoted_args_on_windows(
    monkeypatch,
) -> None:
    _force_windows_commandline(monkeypatch)

    resolved_name, settings = build_server_config_from_target("python -c 'print(1)'")

    assert resolved_name == "python"
    assert settings.command == "python"
    assert settings.args == ["-c", "print(1)"]
