from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fast_agent.cli.commands import quickstart, setup

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def test_setup_creates_preferred_config_and_secrets_filenames(tmp_path: Path) -> None:
    target = tmp_path / "app"

    result = CliRunner().invoke(
        setup.app,
        ["--config-dir", str(target), "--force"],
        input="y\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert (target / "fast-agent.yaml").exists()
    assert (target / "fast-agent.secrets.yaml").exists()
    assert not (target / "fastagent.config.yaml").exists()
    assert not (target / "fastagent.secrets.yaml").exists()
    assert "fast-agent.yaml" in result.output
    assert "fast-agent.secrets.yaml" in result.output
    assert "Created fast-agent home:" in result.output
    assert "Created config file:" in result.output
    assert "Created secrets file:" in result.output
    assert "fastagent.config.yaml" not in result.output


def test_quickstart_copies_preferred_config_filename(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(quickstart, "BASE_EXAMPLES_DIR", Path("examples").resolve())

    created = quickstart.copy_example_files("workflow", tmp_path, force=True)

    assert "workflow/fast-agent.yaml" in created
    assert (tmp_path / "workflow" / "fast-agent.yaml").exists()
    assert not (tmp_path / "workflow" / "fastagent.config.yaml").exists()
