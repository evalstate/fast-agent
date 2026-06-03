from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from fast_agent.cli.commands import config as config_command
from fast_agent.types.streaming import STREAMING_MODE_HELP

if TYPE_CHECKING:
    from pathlib import Path


def test_config_root_lists_display_subcommand() -> None:
    runner = CliRunner()

    result = runner.invoke(config_command.app, [])

    assert result.exit_code == 0, result.output
    assert "display" in result.output
    assert "markdown rendering" in result.output
    assert "model" not in result.output


def test_build_display_form_uses_streaming_mode_help() -> None:
    schema = config_command._build_display_form(config_command.LoggerSettings())

    streaming_field = schema.fields["streaming"]
    assert streaming_field.description is not None
    assert STREAMING_MODE_HELP in streaming_field.description


def test_display_bool_defaults_match_logger_settings() -> None:
    defaults = config_command.LoggerSettings().model_dump()

    assert {
        key: defaults[key]
        for key in config_command.DISPLAY_BOOL_DEFAULTS
    } == config_command.DISPLAY_BOOL_DEFAULTS


def test_build_shell_form_uses_model_values() -> None:
    current = config_command.ShellSettings(
        timeout_seconds=42,
        show_bash=False,
        output_display_lines=None,
        output_byte_limit=None,
        write_text_file_mode="apply_patch",
    )

    schema = config_command._build_shell_form(current)

    assert schema.fields["timeout_seconds"].default == 42
    assert schema.fields["show_bash"].default is False
    assert schema.fields["output_display_lines"].default == -1
    assert schema.fields["output_byte_limit"].default == 0
    assert schema.fields["write_text_file_mode"].default == "apply_patch"


def test_normalize_display_updates_trims_theme_values() -> None:
    updates = config_command._normalize_display_updates(
        {
            "theme_file": "  themes/custom.ini  ",
            "code_theme": "   ",
        }
    )

    assert updates["theme_file"] == "themes/custom.ini"
    assert updates["code_theme"] == "native"


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


def test_config_display_normalizes_invalid_streaming_mode(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")

    def _fake_form_sync(*args, **kwargs):  # noqa: ARG001
        return {
            "theme_file": "",
            "code_theme": "native",
            "streaming": "sideways",
            "apply_patch_preview_max_lines": 120,
            "render_fences_with_syntax": True,
            "code_word_wrap": True,
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
    assert config_data["logger"]["streaming"] == "markdown"


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

    expected = workspace / ".fast-agent" / "fast-agent.yaml"

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


def test_load_config_ignores_parent_config(
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

    assert config_path == nested / ".fast-agent" / "fast-agent.yaml"
    assert config_data == {}


def test_config_display_writes_selected_home_config_when_parent_config_exists(
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
    assert config_path == nested / ".fast-agent" / "fast-agent.yaml"
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
