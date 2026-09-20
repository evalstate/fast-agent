"""Basic go startup must not load unused network clients."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_passthrough_go_defers_unused_network_clients(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from typer.testing import CliRunner
from fast_agent.cli.main import app

result = CliRunner().invoke(
    app, ['go', '--model', 'passthrough', '--no-home', '--message', 'import-smoke']
)
assert result.exit_code == 0, (result.output, result.exception)
assert 'import-smoke' in result.output, result.output

# Loading interactive command dispatch must also leave remote SDKs deferred.
import fast_agent.ui.interactive.command_dispatch

unused = {'a2a', 'requests', 'opentelemetry.exporter.otlp.proto.http.trace_exporter'}
loaded = unused.intersection(sys.modules)
assert not loaded, loaded
""",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
