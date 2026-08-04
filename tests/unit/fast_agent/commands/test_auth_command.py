from __future__ import annotations

import json
import socketserver
import threading
from typing import TYPE_CHECKING

import keyring
import pytest
from click.utils import strip_ansi
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError
from typer.testing import CliRunner

from fast_agent.auth.credentials import OAuthCredential, save_oauth_credential
from fast_agent.cli.commands import auth as auth_command

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class MemoryKeyring(KeyringBackend):
    priority = 1

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self.values[(service, username)]
        except KeyError as exc:
            raise PasswordDeleteError from exc


@pytest.fixture
def memory_keyring() -> Iterator[MemoryKeyring]:
    original = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(original)


@pytest.fixture
def isolated_auth_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    auth_path = tmp_path / "auth.json"
    codex_home = tmp_path / "codex"
    monkeypatch.setenv("FAST_AGENT_AUTH_FILE", str(auth_path))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.delenv("CODEX_AUTH_JSON_PATH", raising=False)
    monkeypatch.setenv("FAST_AGENT_KEYRING_NOTICE", "false")
    return auth_path


def _write_mcp_config(path: Path) -> Path:
    path.write_text(
        """
mcp:
  servers:
    docs:
      transport: http
      url: https://example.test/api/mcp
    docs-ro:
      transport: http
      url: https://example.test/api/mcp
    docs-disabled:
      transport: http
      url: https://example.test/api/mcp
      auth:
        oauth: false
    docs-memory:
      transport: http
      url: https://example.test/api/mcp
      auth:
        persist: memory
    forced:
      transport: http
      url: https://forced.test/mcp
      auth:
        oauth: true
    disabled:
      transport: http
      url: https://disabled.test/mcp
      auth:
        oauth: false
    bearer:
      transport: http
      url: https://bearer.test/mcp
      headers:
        Authorization: Bearer test
    forwarded:
      transport: http
      url: https://demo.hf.space/mcp
      auth:
        forward: huggingface
    forwarded-bearer:
      transport: http
      url: https://demo.hf.space/mcp
      headers:
        Authorization: Bearer explicit
      auth:
        forward: huggingface
    memory:
      transport: http
      url: https://memory.test/mcp
      auth:
        persist: memory
    local:
      transport: stdio
      command: echo
    managed:
      management: provider
      transport: http
      url: https://managed.test/mcp
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _store_mcp_credential(
    backend: MemoryKeyring,
    resource: str,
    *,
    token: bool = True,
    client_info: bool = True,
) -> None:
    service = "fast-agent-mcp"
    if token:
        backend.set_password(service, f"oauth:tokens:{resource}", "{}")
    if client_info:
        backend.set_password(service, f"oauth:client_info:{resource}", "{}")
    resources = sorted(
        {
            resource,
            *json.loads(backend.get_password(service, "oauth:index") or "[]"),
        }
    )
    backend.set_password(service, "oauth:index", json.dumps(resources))


def test_auth_help_exposes_only_domain_groups() -> None:
    result = CliRunner().invoke(auth_command.app, ["--help"])

    assert result.exit_code == 0, result.output
    output = strip_ansi(result.output)
    assert "provider" in output
    assert "mcp" in output
    assert " login " not in output
    assert " status " not in output


@pytest.mark.parametrize(
    ("args", "replacement"),
    [
        (["login", "codex"], "auth provider login codex"),
        (["logout", "codex", "--yes"], "auth provider logout codex"),
        (
            ["export", "codex", "codex.json", "--force"],
            "auth provider export codex codex.json",
        ),
        (["status", "codex"], "auth provider show codex"),
        (
            ["mcp", "status", "docs", "--config-path", "fast-agent.yaml"],
            "auth mcp show docs",
        ),
        (
            ["mcp", "logout", "docs", "--config-path", "fast-agent.yaml"],
            "auth mcp forget docs",
        ),
    ],
)
def test_removed_commands_return_migration_guidance(
    args: list[str],
    replacement: str,
) -> None:
    result = CliRunner().invoke(auth_command.app, args)

    assert result.exit_code == 2
    assert "removed in 0.10" in result.output
    assert replacement in result.output


def test_provider_nested_token_export_logout_and_json_status(
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    export_path = tmp_path / "xai.auth.json"
    save_oauth_credential(
        "xai",
        OAuthCredential(
            access_token="xai-access",
            refresh_token="xai-refresh",
        ),
    )
    runner = CliRunner()

    token_result = runner.invoke(auth_command.app, ["provider", "token", "xai"])
    show_result = runner.invoke(auth_command.app, ["provider", "show", "xai", "--json"])
    export_result = runner.invoke(
        auth_command.app,
        ["provider", "export", "xai", str(export_path)],
    )
    logout_result = runner.invoke(
        auth_command.app,
        ["provider", "logout", "xai", "--yes"],
    )

    assert token_result.exit_code == 0
    assert token_result.output.strip() == "xai-access"
    assert json.loads(show_result.output)["provider"]["state"] == "ready"
    assert export_result.exit_code == 0
    exported = json.loads(export_path.read_text())
    assert exported["providers"]["xai"]["refresh_token"] == "xai-refresh"
    assert logout_result.exit_code == 0
    assert "removed" in logout_result.output
    assert not isolated_auth_environment.exists() or (
        "xai" not in json.loads(isolated_auth_environment.read_text())["providers"]
    )


def test_provider_show_rejects_unknown_provider_without_traceback() -> None:
    result = CliRunner().invoke(auth_command.app, ["provider", "show", "unknown"])

    assert result.exit_code == 1
    assert "Unsupported OAuth provider" in result.output
    assert "Traceback" not in result.output


def test_mcp_list_json_reports_effective_modes_and_shared_resource(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    _store_mcp_credential(memory_keyring, "https://example.test/api")

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "list", "--config-path", str(config_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    servers = {server["name"]: server for server in payload["servers"]}
    assert servers["docs"]["auth_mode"] == "auto"
    assert servers["docs"]["credential"] == "ready"
    assert servers["docs"]["resource"] == "https://example.test/api"
    assert servers["docs"]["shared_with"] == ["docs-ro"]
    assert servers["docs-disabled"]["shared_with"] == []
    assert servers["docs-memory"]["shared_with"] == []
    assert servers["forced"]["auth_mode"] == "oauth"
    assert servers["disabled"]["auth_mode"] == "none"
    assert servers["bearer"]["auth_mode"] == "bearer"
    assert servers["forwarded"]["auth_mode"] == "forwarded"
    assert servers["forwarded-bearer"]["auth_mode"] == "bearer"
    assert servers["local"]["auth_mode"] == "not_applicable"
    assert servers["managed"]["auth_mode"] == "provider_managed"
    assert servers["memory"]["credential"] == "memory"


def test_mcp_show_requires_configured_server_name(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del memory_keyring, isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    runner = CliRunner()

    url_result = runner.invoke(
        auth_command.app,
        ["mcp", "show", "https://example.test/mcp", "-c", str(config_path)],
    )
    unknown_result = runner.invoke(
        auth_command.app,
        ["mcp", "show", "unknown", "-c", str(config_path)],
    )

    assert url_result.exit_code == 2
    assert "auth mcp credentials" in url_result.output
    assert unknown_result.exit_code == 1
    assert "was not found" in unknown_result.output


def test_exact_endpoint_login_target_never_appends_transport_path() -> None:
    target = auth_command._validated_endpoint(
        "https://example.test/custom/resource",
        None,
    )
    sse_target = auth_command._validated_endpoint(
        "https://example.test/events/sse",
        None,
    )

    assert target.server.url == "https://example.test/custom/resource"
    assert target.transport == "http"
    assert sse_target.server.url == "https://example.test/events/sse"
    assert sse_target.transport == "sse"


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://[::1]:bad/mcp",
        "https://user:secret@example.test/mcp",
        "https:///missing-host",
    ],
)
def test_exact_endpoint_rejects_malformed_or_secret_bearing_urls(endpoint: str) -> None:
    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "login", "--endpoint", endpoint],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["mcp", "login"],
        ["mcp", "login", "docs", "--endpoint", "https://example.test/mcp"],
        ["mcp", "login", "https://example.test/mcp"],
        ["mcp", "login", "docs", "--transport", "sse"],
    ],
)
def test_mcp_login_rejects_ambiguous_targets(args: list[str], tmp_path: Path) -> None:
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    result = CliRunner().invoke(auth_command.app, [*args, "-c", str(config_path)])

    assert result.exit_code == 2


def test_mcp_login_rejects_nonpersistent_server(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del memory_keyring, isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "login", "memory", "-c", str(config_path)],
    )

    assert result.exit_code == 1
    assert "auth.persist: memory" in result.output


def test_mcp_login_reports_its_outer_timeout(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
) -> None:
    del memory_keyring, isolated_auth_environment
    release = threading.Event()

    class HangingHandler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            release.wait(5)

    with socketserver.TCPServer(("127.0.0.1", 0), HangingHandler) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            host = str(server.server_address[0])
            port = int(server.server_address[1])
            result = CliRunner().invoke(
                auth_command.app,
                [
                    "mcp",
                    "login",
                    "--endpoint",
                    f"http://{host}:{port}/custom/path",
                    "--timeout",
                    "1",
                ],
            )
        finally:
            release.set()
            server.shutdown()
            thread.join()

    assert result.exit_code == 1
    assert "timed out after 1 seconds" in result.output
    assert "Increase --timeout" in result.output


def test_destructive_commands_require_yes_when_noninteractive(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    save_oauth_credential("xai", OAuthCredential(access_token="xai-access"))
    _store_mcp_credential(memory_keyring, "https://example.test/api")
    runner = CliRunner()

    provider_result = runner.invoke(auth_command.app, ["provider", "logout", "xai"])
    mcp_result = runner.invoke(
        auth_command.app,
        ["mcp", "forget", "docs", "-c", str(config_path)],
    )

    assert provider_result.exit_code == 2
    assert mcp_result.exit_code == 2
    assert "--yes" in provider_result.output
    assert "--yes" in mcp_result.output


def test_mcp_credentials_and_forget_show_shared_impact(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    resource = "https://example.test/api"
    _store_mcp_credential(memory_keyring, resource)
    _store_mcp_credential(memory_keyring, "https://orphan.test")
    runner = CliRunner()

    credentials_result = runner.invoke(
        auth_command.app,
        ["mcp", "credentials", "-c", str(config_path), "--json"],
    )
    forget_result = runner.invoke(
        auth_command.app,
        ["mcp", "forget", "docs", "-c", str(config_path), "--yes"],
    )

    assert credentials_result.exit_code == 0, credentials_result.output
    credentials = {
        item["resource"]: item for item in json.loads(credentials_result.output)["credentials"]
    }
    assert credentials[resource]["configured_servers"] == ["docs", "docs-ro"]
    assert credentials["https://orphan.test"]["orphaned"] is True

    assert forget_result.exit_code == 0, forget_result.output
    assert resource in forget_result.output
    assert "docs, docs-ro" in forget_result.output
    assert "runtime connections are unchanged" in forget_result.output
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{resource}") is None
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:client_info:{resource}") is None
    assert json.loads(memory_keyring.get_password("fast-agent-mcp", "oauth:index") or "[]") == [
        "https://orphan.test"
    ]


def test_mcp_forget_removes_client_registration_without_tokens(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    resource = "https://example.test/api"
    memory_keyring.set_password(
        "fast-agent-mcp",
        f"oauth:client_info:{resource}",
        "{}",
    )
    runner = CliRunner()

    credentials_result = runner.invoke(
        auth_command.app,
        ["mcp", "credentials", "-c", str(config_path), "--json"],
    )
    forget_result = runner.invoke(
        auth_command.app,
        ["mcp", "forget", "--resource", resource, "-c", str(config_path), "--yes"],
    )

    assert resource in credentials_result.output
    assert forget_result.exit_code == 0, forget_result.output
    assert "Forgot 1" in forget_result.output
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:client_info:{resource}") is None


@pytest.mark.parametrize("suffix", ["mcp", "sse"])
def test_mcp_forget_resource_preserves_trailing_transport_segment(
    suffix: str,
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    base_resource = "https://example.test/path"
    selected_resource = f"{base_resource}/{suffix}"
    _store_mcp_credential(memory_keyring, base_resource)
    _store_mcp_credential(memory_keyring, selected_resource)

    result = CliRunner().invoke(
        auth_command.app,
        [
            "mcp",
            "forget",
            "--resource",
            selected_resource,
            "-c",
            str(config_path),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (
        memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{selected_resource}") is None
    )
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{base_resource}") == "{}"
    assert (
        memory_keyring.get_password("fast-agent-mcp", f"oauth:client_info:{selected_resource}")
        is None
    )
    assert (
        memory_keyring.get_password("fast-agent-mcp", f"oauth:client_info:{base_resource}") == "{}"
    )


def test_mcp_forget_selectors_are_mutually_exclusive(tmp_path: Path) -> None:
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    result = CliRunner().invoke(
        auth_command.app,
        [
            "mcp",
            "forget",
            "docs",
            "--resource",
            "https://example.test/api",
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    assert "exactly one" in result.output


@pytest.mark.parametrize(
    "resource",
    [
        "https:///missing-host",
        "http://[::1]:bad",
        "https://user:secret@example.test",
    ],
)
def test_mcp_forget_rejects_invalid_resource(resource: str, tmp_path: Path) -> None:
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "forget", "--resource", resource, "-c", str(config_path), "--yes"],
    )

    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_combined_auth_json_is_secret_free(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    _store_mcp_credential(memory_keyring, "https://example.test/api")

    result = CliRunner().invoke(
        auth_command.app,
        ["--config-path", str(config_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"providers", "mcp"}
    assert "access_token" not in result.output
    assert "refresh_token" not in result.output


def test_legacy_identity_migration_redacts_url_secrets() -> None:
    result = CliRunner().invoke(
        auth_command.app,
        [
            "mcp",
            "logout",
            "--identity",
            "https://user:password@example.test/api?token=secret",
        ],
    )

    assert result.exit_code == 2
    assert "password" not in result.output
    assert "secret" not in result.output
    assert "[REDACTED]" in result.output


def test_auth_root_options_are_rejected_before_subcommands(tmp_path: Path) -> None:
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    runner = CliRunner()

    json_result = runner.invoke(auth_command.app, ["--json", "provider", "list"])
    config_result = runner.invoke(
        auth_command.app,
        ["--config-path", str(config_path), "mcp", "list"],
    )

    assert json_result.exit_code == 2
    assert config_result.exit_code == 2
    assert "Place the option after" in json_result.output
    assert "Place the option after" in config_result.output


def test_mcp_json_redacts_endpoint_query_and_userinfo(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del memory_keyring, isolated_auth_environment
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        """
mcp:
  servers:
    query:
      transport: http
      url: https://example.test/mcp?access_token=TOPSECRET
    userinfo:
      transport: http
      url: https://user:PASSSECRET@example.test/mcp
""".lstrip(),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "list", "-c", str(config_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    assert "TOPSECRET" not in result.output
    assert "PASSSECRET" not in result.output
    assert "%5BREDACTED%5D" in result.output
    assert "[REDACTED]@" in result.output


def test_mcp_list_reads_legacy_noncanonical_resource_key_without_deleting_it(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        """
mcp:
  servers:
    docs:
      transport: http
      url: HTTPS://EXAMPLE.TEST:443/api/mcp
""".lstrip(),
        encoding="utf-8",
    )
    legacy = "https://EXAMPLE.TEST:443/api"
    canonical = "https://example.test/api"
    _store_mcp_credential(memory_keyring, legacy)

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "list", "-c", str(config_path), "--json"],
    )

    assert result.exit_code == 0, result.output
    server = json.loads(result.output)["servers"][0]
    assert server["resource"] == canonical
    assert server["credential"] == "ready"
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{legacy}") == "{}"
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{canonical}") is None

    forget_result = CliRunner().invoke(
        auth_command.app,
        [
            "mcp",
            "forget",
            "--resource",
            legacy,
            "-c",
            str(config_path),
            "--yes",
        ],
    )
    assert forget_result.exit_code == 0, forget_result.output
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{legacy}") is None


def test_mcp_forget_accepts_displayed_legacy_orphan_resource(
    memory_keyring: MemoryKeyring,
    isolated_auth_environment: Path,
    tmp_path: Path,
) -> None:
    del isolated_auth_environment
    config_path = _write_mcp_config(tmp_path / "fast-agent.yaml")
    legacy = "https://ORPHAN.TEST:443/api"
    _store_mcp_credential(memory_keyring, legacy)
    runner = CliRunner()

    credentials_result = runner.invoke(
        auth_command.app,
        ["mcp", "credentials", "-c", str(config_path), "--json"],
    )
    forget_result = runner.invoke(
        auth_command.app,
        [
            "mcp",
            "forget",
            "--resource",
            legacy,
            "-c",
            str(config_path),
            "--yes",
        ],
    )

    assert legacy in credentials_result.output
    assert forget_result.exit_code == 0, forget_result.output
    assert memory_keyring.get_password("fast-agent-mcp", f"oauth:tokens:{legacy}") is None


def test_mcp_json_keeps_stdout_parseable_when_config_is_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "list", "-c", str(missing), "--json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert set(payload["keyring"]) == {"name", "available", "writable"}
    assert payload["servers"] == []
    assert "does not exist" in result.stderr


def test_auth_mcp_list_reports_invalid_settings_yaml_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "mcp:\n  targets:\n    - name: openai\n        target: https://example.test/mcp\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        auth_command.app,
        ["mcp", "list", "--config-path", str(config_path)],
    )

    assert result.exit_code == 1, result.output
    assert "Error loading fast-agent settings:" in result.output
    assert "Traceback" not in result.output
