import os
import subprocess
import sys
from pathlib import Path

import pytest

PROBE_SERVER = """
from fastmcp import FastMCP

mcp = FastMCP("probe")


@mcp.tool()
def probe(text: str) -> str:
    return f"probe:{text}"


mcp.run()
"""


def _snapshot(*roots: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for root in roots for path in root.rglob("*") if path.is_file()}


@pytest.mark.integration
def test_go_isolated_reads_home_config_but_writes_nothing(tmp_path: Path) -> None:
    work = tmp_path / "work"
    home = work / ".fast-agent"
    user_home = tmp_path / "user"
    (home / "agent-cards").mkdir(parents=True)
    (work / ".agents" / "skills" / "demo").mkdir(parents=True)
    user_home.mkdir()

    (work / "server.py").write_text(PROBE_SERVER, encoding="utf-8")
    (home / "fast-agent.yaml").write_text(
        "default_model: passthrough\n"
        "session_history: true\n"
        "logger:\n  type: file\n  level: debug\n"
        "mcp:\n  servers:\n    probe:\n"
        f"      command: {sys.executable}\n"
        "      args: [server.py]\n",
        encoding="utf-8",
    )
    # Would fail the run if card loading were not disabled.
    (home / "agent-cards" / "broken.md").write_text("---\nname: [broken\n---\n", encoding="utf-8")
    (work / ".agents" / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    before = _snapshot(work, user_home)

    env = {k: v for k, v in os.environ.items() if not k.startswith("FAST_AGENT_")}
    env["HOME"] = str(user_home)
    results = tmp_path / "results.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fast_agent.cli",
            "go",
            "--isolated",
            "--servers",
            "probe",
            "--message",
            '***CALL_TOOL probe {"text": "hi"}',
            "--results",
            str(results),
        ],
        capture_output=True,
        check=False,
        text=True,
        cwd=work,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "probe:hi" in results.read_text(encoding="utf-8")
    after = _snapshot(work, user_home)
    # The probe server keeps its own cache under HOME; everything else must be untouched.
    assert {path: data for path, data in after.items() if "fastmcp" not in path.parts} == before
