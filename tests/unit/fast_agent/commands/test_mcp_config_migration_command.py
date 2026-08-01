from __future__ import annotations

import shlex
import stat
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fast_agent.cli.commands import config as config_command
from fast_agent.config import get_settings


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "fast-agent.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _run(path: Path, *, write: bool = False):
    arguments = ["migrate-mcp", str(path)]
    if write:
        arguments.append("--write")
    return CliRunner().invoke(config_command.app, arguments)


def test_show_mcp_source_and_effective_views_are_redacted(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
mcp:
  defaults:
    protocol_mode: modern
  servers:
    docs:
      target: https://example.com/mcp
      access_token: secret
""".lstrip(),
    )

    source = CliRunner().invoke(config_command.app, ["show-mcp", str(path)])
    effective = CliRunner().invoke(
        config_command.app,
        ["show-mcp", str(path), "--view", "effective"],
    )

    assert source.exit_code == 0, source.output
    assert "target: https://example.com/mcp" in source.output
    assert "protocol_mode: modern" in source.output
    assert "access_token: '[REDACTED]'" in source.output
    assert "transport:" not in source.output
    assert effective.exit_code == 0, effective.output
    assert "protocol_mode: modern" in effective.output
    assert "_provenance:" in effective.output
    assert "secret" not in effective.output

    roundtrip_path = tmp_path / "roundtrip.yaml"
    roundtrip_path.write_text(source.output, encoding="utf-8")
    roundtrip = get_settings(roundtrip_path)
    assert roundtrip.mcp is not None
    assert roundtrip.mcp.servers["docs"].protocol_mode == "modern"


def test_show_mcp_reports_legacy_targets_without_traceback(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
mcp:
  targets:
    - name: docs
      target: https://alice:secret@example.com/mcp?token=topsecret
""",
    )

    result = CliRunner().invoke(config_command.app, ["show-mcp", str(path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Error loading fast-agent settings:" in result.output
    assert "`mcp.targets` is no longer supported" in result.output
    command = shlex.join(["fast-agent", "config", "migrate-mcp", str(path.resolve()), "--write"])
    assert f"`{command}`" in result.output
    assert "Traceback" not in result.output
    assert "input_value" not in result.output
    assert "alice" not in result.output
    assert "secret" not in result.output
    assert "topsecret" not in result.output


def test_show_mcp_does_not_echo_values_from_custom_validation_errors(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
mcp:
  servers:
    public:
      name: alice:secret@example.com?token=topsecret
      url: https://example.com
""",
    )

    result = CliRunner().invoke(config_command.app, ["show-mcp", str(path)])

    assert result.exit_code == 1
    assert result.output == (
        "Error loading fast-agent settings: 1 validation error for Settings\n"
        "mcp: Invalid configuration value\n"
    )
    assert "alice" not in result.output
    assert "secret" not in result.output
    assert "topsecret" not in result.output


def test_migrate_mcp_dry_run_prints_diff_without_mutation(tmp_path: Path) -> None:
    original = """\
# retained root comment
auto_sampling: false
mcp:
  targets: # legacy collection
    - name: "docs" # retained name comment
      target: 'https://example.com/mcp' # retained target comment
      reconnect_on_disconnect: false
"""
    path = _write(tmp_path, original)

    result = _run(path)

    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"--- {path}\n+++ {path}\n")
    assert "-  targets: # legacy collection" in result.output
    assert "+  servers: # legacy collection" in result.output
    assert "docs:" in result.output
    assert "target: 'https://example.com/mcp' # retained target comment" in result.output
    assert path.read_text(encoding="utf-8") == original
    assert not Path(f"{path}.bak").exists()


def test_migrate_mcp_write_backup_settings_load_and_idempotence(tmp_path: Path) -> None:
    original = b"""\
auto_sampling: false
mcp_timeline:
  steps: 7
  step_seconds: 30
mcp:
  targets:
    - name: fetch
      target: "uvx mcp-server-fetch"
"""
    path = tmp_path / "fast-agent.yaml"
    path.write_bytes(original)
    path.chmod(0o640)

    result = _run(path, write=True)

    assert result.exit_code == 0, result.output
    assert Path(f"{path}.bak").read_bytes() == original
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    migrated = path.read_bytes()
    settings = get_settings(path, no_home=True)
    assert settings.mcp is not None
    assert settings.mcp.client.auto_sampling is False
    assert settings.mcp.diagnostics.timeline.steps == 7
    assert settings.mcp.servers["fetch"].command == "uvx"
    assert settings.mcp.servers["fetch"].args == ["mcp-server-fetch"]

    second = _run(path, write=True)

    assert second.exit_code == 0, second.output
    assert second.output == "No MCP migration needed.\n"
    assert path.read_bytes() == migrated
    assert Path(f"{path}.bak").read_bytes() == original


def test_migrate_mcp_does_not_validate_unmerged_settings(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """\
auto_sampling: false
mcp:
  servers:
    provider:
      management: provider
      connector_id: supplied-by-secrets
""",
    )

    result = _run(path, write=True)

    assert result.exit_code == 0, result.output
    assert "auto_sampling: false" in path.read_text(encoding="utf-8")
    assert "client:" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "targets",
    [
        """
    - name: duplicate
      target: uvx first
    - name: duplicate
      target: uvx second
""",
        """
    - https://example.com/mcp
    - https://example.com/mcp
""",
    ],
)
def test_migrate_mcp_refuses_duplicate_or_inferred_collisions(tmp_path: Path, targets: str) -> None:
    path = _write(tmp_path, f"mcp:\n  targets:{targets}")
    original = path.read_bytes()

    result = _run(path, write=True)

    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "duplicate server name" in result.output
    assert path.read_bytes() == original
    assert not Path(f"{path}.bak").exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            """\
mcp:
  targets: []
  servers: {}
""",
            "`mcp.targets` and `mcp.servers` cannot both be set",
        ),
        (
            """\
auto_sampling: false
mcp:
  client:
    auto_sampling: true
""",
            "`auto_sampling` and `mcp.client.auto_sampling` cannot both be set",
        ),
        (
            """\
mcp_timeline: {}
mcp:
  diagnostics:
    timeline: {}
""",
            "`mcp_timeline` and `mcp.diagnostics.timeline` cannot both be set",
        ),
    ],
)
def test_migrate_mcp_refuses_collection_and_legacy_path_conflicts(
    tmp_path: Path, text: str, message: str
) -> None:
    path = _write(tmp_path, text)

    result = _run(path)

    assert result.exit_code == 1
    assert message in result.output


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("- not-a-mapping\n", "`configuration` must be a mapping"),
        ("mcp: invalid\n", "`mcp` must be a mapping"),
        ("mcp:\n  targets: invalid\n", "`mcp.targets` must be a list"),
        ("mcp:\n  targets:\n    - 42\n", "`mcp.targets[0]` must be a string or mapping"),
        (
            "mcp:\n  targets:\n    - name: missing-target\n",
            "`mcp.targets[0].target` is required",
        ),
        (
            "mcp:\n  targets:\n    - name: '  '\n      target: uvx demo\n",
            "`mcp.targets[0].name` must be a non-empty string",
        ),
        (
            """\
mcp:
  targets:
    - name: demo
      target: https://example.com
      command: echo
""",
            "cannot be combined with source fields: command",
        ),
    ],
)
def test_migrate_mcp_reports_malformed_structures(tmp_path: Path, text: str, message: str) -> None:
    path = _write(tmp_path, text)

    result = _run(path)

    assert result.exit_code == 1
    assert result.output.startswith("Error: ")
    assert message in result.output
