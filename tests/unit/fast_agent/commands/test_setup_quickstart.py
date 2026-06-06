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


def test_setup_template_resource_names_are_table_driven() -> None:
    assert setup._template_resource_name("fast-agent.secrets.yaml") == (
        "fast-agent.secrets.yaml.example"
    )
    assert setup._template_resource_name("pyproject.toml") == "pyproject.toml.tmpl"
    assert setup._template_resource_name("agent.py") == "agent.py"


def test_render_pyproject_uses_python_requirement_helper(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "_python_requires", lambda: ">=3.14")

    rendered = setup._render_pyproject(
        'requires-python = "{{python_requires}}"\ndependencies = [{{fast_agent_dep}}]\n'
    )

    assert 'requires-python = ">=3.14"' in rendered
    assert 'dependencies = ["fast-agent-mcp"]' in rendered


def test_python_requires_falls_back_when_metadata_unavailable(monkeypatch: MonkeyPatch) -> None:
    def _raise_metadata_error(_package: str) -> object:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr("importlib.metadata.metadata", _raise_metadata_error)

    assert setup._python_requires() == ">=3.13.7"


def test_quickstart_copies_preferred_config_filename(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(quickstart, "BASE_EXAMPLES_DIR", Path("examples").resolve())

    created = quickstart.copy_example_files("workflow", tmp_path, force=True)

    assert "workflow/fast-agent.yaml" in created
    assert (tmp_path / "workflow" / "fast-agent.yaml").exists()
    assert not (tmp_path / "workflow" / "fastagent.config.yaml").exists()


def test_quickstart_files_summary_formats_singular_counts() -> None:
    info = quickstart.ExampleConfig(
        description="Example",
        files=["agent.py"],
        create_subdir=True,
        path_in_examples=["example"],
        mount_point_files=["data.csv"],
    )

    assert quickstart._files_summary(info) == "1 file\n+ 1 data file"


def test_quickstart_next_steps_are_defined_for_standard_examples() -> None:
    standard_examples = {
        "workflow",
        "researcher",
        "data-analysis",
        "state-transfer",
        "elicitations",
    }

    assert set(quickstart._NEXT_STEP_MESSAGES) == standard_examples
    assert "   - parallel.py: Run agents in parallel" in quickstart._NEXT_STEP_MESSAGES["workflow"]


def test_toad_cards_completion_message_formats_singular_file_count(
    monkeypatch: MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(quickstart.console, "print", printed.append)

    quickstart._show_toad_cards_completion_message([".fast-agent/skills/demo/SKILL.md"])

    assert "\nCreated 1 file in .fast-agent/" in printed
