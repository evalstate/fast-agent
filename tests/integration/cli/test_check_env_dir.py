import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.integration
def test_check_uses_env_dir_for_config(tmp_path: Path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    env_config = env_dir / "fastagent.config.yaml"
    env_secrets = env_dir / "fastagent.secrets.yaml"
    env_config.write_text("default_model: gpt-4.1\n", encoding="utf-8")
    env_secrets.write_text("openai:\n  api_key: sk-env-test\n", encoding="utf-8")

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "fastagent.config.yaml").write_text(
        "default_model: gpt-5-mini.low\n", encoding="utf-8"
    )
    (work_dir / "fastagent.secrets.yaml").write_text(
        "openai:\n  api_key: sk-cwd-test\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env.pop("ENVIRONMENT_DIR", None)
    env["COLUMNS"] = "200"
    env["RICH_WIDTH"] = "200"

    result = subprocess.run(
        ["uv", "run", "fast-agent", "check", "--env", str(env_dir)],
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
