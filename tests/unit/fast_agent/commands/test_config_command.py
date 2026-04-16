from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fast_agent.cli.commands import config as config_command

if TYPE_CHECKING:
    from pathlib import Path


def test_config_root_lists_display_subcommand() -> None:
    runner = CliRunner()

    result = runner.invoke(config_command.app, [])

    assert result.exit_code == 0, result.output
    assert "display" in result.output
    assert "markdown rendering" in result.output
    assert "model" not in result.output


def test_config_display_updates_logger_settings(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")
    captured_message: str | None = None

    def _fake_form_sync(*args, **kwargs):  # noqa: ARG001
        nonlocal captured_message
        captured_message = kwargs.get("message")
        return {
            "theme_file": "themes/custom.ini",
            "code_theme": "monokai",
            "streaming": "plain",
            "apply_patch_preview_max_lines": 40,
            "render_fences_with_syntax": False,
            "code_word_wrap": True,
            "progress_display": False,
            "show_chat": False,
            "stream_reprint_banner": False,
            "show_tools": True,
            "truncate_tools": False,
            "enable_markup": False,
            "enable_prompt_marks": False,
        }

    monkeypatch.setattr(config_command, "form_sync", _fake_form_sync)

    runner = CliRunner()
    result = runner.invoke(config_command.app, ["display", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Display settings saved" in result.output
    assert captured_message is not None
    assert "Editing:" in captured_message
    assert str(config_path) in captured_message

    config_data, _ = config_command._load_config(config_path)
    logger = config_data["logger"]
    assert logger["theme_file"] == "themes/custom.ini"
    assert logger["code_theme"] == "monokai"
    assert logger["streaming"] == "plain"
    assert logger["apply_patch_preview_max_lines"] == 40
    assert logger["render_fences_with_syntax"] is False
    assert logger["code_word_wrap"] is True
    assert logger["progress_display"] is False
    assert logger["show_chat"] is False
    assert logger["stream_reprint_banner"] is False
    assert logger["show_tools"] is True
    assert logger["truncate_tools"] is False
    assert logger["enable_markup"] is False
    assert logger["enable_prompt_marks"] is False


def test_config_display_removes_default_theme_and_code_theme(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "logger:\n"
        "  theme_file: themes/custom.ini\n"
        "  code_theme: monokai\n",
        encoding="utf-8",
    )

    def _fake_form_sync(*args, **kwargs):  # noqa: ARG001
        return {
            "theme_file": "",
            "code_theme": "native",
            "streaming": "markdown",
            "apply_patch_preview_max_lines": 120,
            "render_fences_with_syntax": True,
            "code_word_wrap": False,
            "progress_display": True,
            "show_chat": True,
            "stream_reprint_banner": True,
            "show_tools": True,
            "truncate_tools": True,
            "enable_markup": True,
            "enable_prompt_marks": True,
        }

    monkeypatch.setattr(config_command, "form_sync", _fake_form_sync)

    runner = CliRunner()
    result = runner.invoke(config_command.app, ["display", "--config", str(config_path)])

    assert result.exit_code == 0, result.output

    config_data, _ = config_command._load_config(config_path)
    logger = config_data["logger"]
    assert "theme_file" not in logger
    assert "code_theme" not in logger


def test_load_config_defaults_to_environment_config_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)

    expected = workspace / ".fast-agent" / "fastagent.config.yaml"

    config_data, config_path = config_command._load_config()

    assert config_data == {}
    assert config_path == expected


def test_load_config_prefers_cwd_config_before_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    workspace.mkdir()
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    (workspace / "fastagent.config.yaml").write_text(
        "logger:\n"
        "  show_tools: false\n",
        encoding="utf-8",
    )
    (nested / "fastagent.config.yaml").write_text(
        "logger:\n"
        "  show_tools: true\n",
        encoding="utf-8",
    )

    config_data, config_path = config_command._load_config()

    assert config_path == nested / "fastagent.config.yaml"
    assert config_data == {"logger": {"show_tools": True}}


def test_load_config_falls_back_to_legacy_parent_config(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    workspace.mkdir()
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    (workspace / "fastagent.config.yaml").write_text(
        "logger:\n"
        "  show_tools: false\n",
        encoding="utf-8",
    )

    config_data, config_path = config_command._load_config()

    assert config_path == workspace / "fastagent.config.yaml"
    assert config_data == {"logger": {"show_tools": False}}


def test_config_display_updates_legacy_parent_config_when_run_from_nested_dir(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    nested = workspace / "child"
    workspace.mkdir()
    nested.mkdir()
    monkeypatch.chdir(nested)
    monkeypatch.delenv("ENVIRONMENT_DIR", raising=False)
    (workspace / "fastagent.config.yaml").write_text(
        "logger:\n"
        "  show_tools: false\n",
        encoding="utf-8",
    )

    def _fake_form_sync(*args, **kwargs):  # noqa: ARG001
        return {
            "theme_file": "",
            "code_theme": "native",
            "streaming": "markdown",
            "apply_patch_preview_max_lines": 120,
            "render_fences_with_syntax": True,
            "code_word_wrap": False,
            "progress_display": True,
            "show_chat": False,
            "stream_reprint_banner": True,
            "show_tools": False,
            "truncate_tools": True,
            "enable_markup": True,
            "enable_prompt_marks": True,
        }

    monkeypatch.setattr(config_command, "form_sync", _fake_form_sync)

    runner = CliRunner()
    result = runner.invoke(config_command.app, ["display"])

    assert result.exit_code == 0, result.output

    config_data, config_path = config_command._load_config()
    assert config_path == workspace / "fastagent.config.yaml"
    logger = config_data["logger"]
    assert logger["show_tools"] is False
    assert logger["show_chat"] is False
    assert (nested / ".fast-agent" / "fastagent.config.yaml").exists() is False


def test_config_display_zero_patch_preview_lines_means_unlimited(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")

    def _fake_form_sync(*args, **kwargs):  # noqa: ARG001
        return {
            "theme_file": "",
            "code_theme": "native",
            "streaming": "markdown",
            "apply_patch_preview_max_lines": 0,
            "render_fences_with_syntax": True,
            "code_word_wrap": False,
            "progress_display": True,
            "show_chat": True,
            "stream_reprint_banner": True,
            "show_tools": True,
            "truncate_tools": True,
            "enable_markup": True,
            "enable_prompt_marks": True,
        }

    monkeypatch.setattr(config_command, "form_sync", _fake_form_sync)

    runner = CliRunner()
    result = runner.invoke(config_command.app, ["display", "--config", str(config_path)])

    assert result.exit_code == 0, result.output

    config_data, _ = config_command._load_config(config_path)
    logger = config_data["logger"]
    assert logger["apply_patch_preview_max_lines"] is None
