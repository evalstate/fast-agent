from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from rich.style import Style

import fast_agent.config as config_module
from fast_agent.config import LoggerSettings, Settings, get_settings
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay


def test_console_display_applies_theme_file_relative_to_config(tmp_path: Path) -> None:
    theme_dir = tmp_path / "themes"
    theme_dir.mkdir()
    theme_file = theme_dir / "yellow.ini"
    theme_file.write_text(
        "[styles]\nmarkdown.h2 = yellow underline\nmarkdown.link = bright_cyan underline\n",
        encoding="utf-8",
    )

    settings = Settings(logger=LoggerSettings(theme_file="themes/yellow.ini"))
    settings._config_file = str(tmp_path / "fastagent.config.yaml")

    console.configure_console_theme(None)
    try:
        ConsoleDisplay(config=settings)

        assert console.console.get_style("markdown.h2") == Style.parse("yellow underline")
        assert console.console.get_style("markdown.link") == Style.parse("bright_cyan underline")
    finally:
        console.configure_console_theme(None)


def test_console_display_ignores_legacy_theme_when_env_config_is_selected(
    tmp_path: Path, monkeypatch
) -> None:
    theme_dir = tmp_path / "themes"
    theme_dir.mkdir()
    theme_file = theme_dir / "yellow.ini"
    theme_file.write_text(
        "[styles]\nmarkdown.h2 = yellow underline\nmarkdown.link = bright_cyan underline\n",
        encoding="utf-8",
    )
    (tmp_path / "fastagent.config.yaml").write_text(
        "logger:\n  theme_file: themes/yellow.ini\n",
        encoding="utf-8",
    )
    home = tmp_path / ".fast-agent"
    home.mkdir()
    (home / "fastagent.config.yaml").write_text(
        "logger:\n  show_tools: false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)

    previous_settings = config_module._settings
    try:
        config_module._settings = None
        settings = get_settings()
        assert settings.logger.theme_file is None

        console.configure_console_theme(None)
        ConsoleDisplay(config=settings)

        assert console.console.get_style("markdown.h2") == Style.parse("yellow underline")
    finally:
        config_module._settings = previous_settings
        console.configure_console_theme(None)


def test_console_display_without_theme_file_restores_default_theme() -> None:
    theme_file = Path(__file__).resolve().parents[4] / "examples" / "markdown" / "high-contrast.ini"

    console.configure_console_theme(None)
    try:
        console.configure_console_theme(theme_file)
        assert console.console.get_style("markdown.h3") == Style.parse("bold bright_cyan")
        assert console.console.get_style("markdown.block_quote") == Style.parse("bright_blue")
        assert console.console.get_style("markdown.code") == Style.parse(
            "bold bright_green on black"
        )

        ConsoleDisplay(config=Settings(logger=LoggerSettings()))

        assert console.console.get_style("markdown.h3") == Style.parse("bold yellow")
        assert console.console.get_style("markdown.block_quote") == Style.parse("blue")
        assert console.console.get_style("markdown.code") == Style.parse("bright_green on black")
    finally:
        console.configure_console_theme(None)


def test_configless_console_display_preserves_existing_shared_theme() -> None:
    theme_file = Path(__file__).resolve().parents[4] / "examples" / "markdown" / "high-contrast.ini"

    console.configure_console_theme(None)
    try:
        console.configure_console_theme(theme_file)
        assert console.console.get_style("markdown.h3") == Style.parse("bold bright_cyan")

        ConsoleDisplay(config=None)

        assert console.console.get_style("markdown.h3") == Style.parse("bold bright_cyan")
        assert console.console.get_style("markdown.block_quote") == Style.parse("bright_blue")
    finally:
        console.configure_console_theme(None)


def test_console_display_normalizes_partial_logger_config() -> None:
    config = SimpleNamespace(logger=SimpleNamespace(show_chat=False))

    display = ConsoleDisplay(config=config)

    assert display.logger_settings.show_chat is False
    assert display.logger_settings.code_word_wrap is True
    assert display.logger_settings.render_fences_with_syntax is True
    assert config.logger is display.logger_settings


def test_update_logger_settings_does_not_mutate_shared_config() -> None:
    shared_config = Settings(logger=LoggerSettings(show_chat=True, show_tools=True))
    first = ConsoleDisplay(config=shared_config)

    first_original = first.logger_settings
    first.update_logger_settings(
        first_original.model_copy(update={"show_chat": False, "show_tools": False})
    )

    # A second display initialized while the first is suppressed must still see
    # the unmodified shared config (mirrors overlapping parallel subagents).
    second = ConsoleDisplay(config=shared_config)
    second_original = second.logger_settings
    second.update_logger_settings(
        second_original.model_copy(update={"show_chat": False, "show_tools": False})
    )

    first.update_logger_settings(first_original)
    second.update_logger_settings(second_original)

    assert shared_config.logger.show_chat is True
    assert shared_config.logger.show_tools is True
    assert first.logger_settings.show_chat is True
    assert second.logger_settings.show_chat is True
