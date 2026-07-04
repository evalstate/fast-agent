import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_check_uses_home_for_config(tmp_path: Path) -> None:
    home = tmp_path / "env"
    home.mkdir()
    env_config = home / "fastagent.config.yaml"
    env_secrets = home / "fastagent.secrets.yaml"
    env_config.write_text("default_model: gpt-4.1\n", encoding="utf-8")
    env_secrets.write_text("openai:\n  api_key: sk-env-test\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "fastagent.config.yaml").write_text(
        "default_model: gpt-5-mini?reasoning=low\n", encoding="utf-8"
    )
    (work_dir / "fastagent.secrets.yaml").write_text(
        "openai:\n  api_key: sk-cwd-test\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.pop("FAST_AGENT_HOME", None)
    env["COLUMNS"] = "200"
    env["RICH_WIDTH"] = "200"

    result = subprocess.run(
        ["uv", "run", "fast-agent", "check", "--home", str(home)],
        capture_output=True,
        text=True,
        cwd=work_dir,
        env=env,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"

    output = result.stdout
    assert str(env_config) in output
    assert str(env_secrets) in output
    assert "gpt-4.1" in output
