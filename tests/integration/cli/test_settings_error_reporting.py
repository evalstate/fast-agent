from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
def test_acp_reports_implicit_legacy_mcp_targets_without_traceback(
    tmp_path: Path,
) -> None:
    (tmp_path / "fast-agent.yaml").write_text(
        """\
mcp:
  targets:
    - name: docs
      target: https://alice:secret@example.com/mcp?token=topsecret
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("FAST_AGENT_HOME", None)
    env.pop("FAST_AGENT_RUNTIME_HOME", None)

    result = subprocess.run(
        ["uv", "run", "fast-agent", "--no-color", "acp"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Error loading fast-agent settings:" in output
    assert "`mcp.targets` is no longer supported" in output
    command = shlex.join(
        [
            "fast-agent",
            "config",
            "migrate-mcp",
            str(tmp_path / "fast-agent.yaml"),
            "--write",
        ]
    )
    assert f"`{command}`" in output
    assert "Traceback" not in output
    assert "input_value" not in output
    assert "alice" not in output
    assert "secret" not in output
    assert "topsecret" not in output
