from __future__ import annotations

from pathlib import Path

from rich.style import Style

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


def test_console_display_uses_theme_file_source_from_project_config(
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
        "logger:\n"
        "  theme_file: themes/yellow.ini\n",
        encoding="utf-8",
    )
    env_dir = tmp_path / ".fast-agent"
    env_dir.mkdir()
    (env_dir / "fastagent.config.yaml").write_text(
        "logger:\n"
        "  show_tools: false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    settings = get_settings()

    console.configure_console_theme(None)
    try:
        ConsoleDisplay(config=settings)

        assert console.console.get_style("markdown.h2") == Style.parse("yellow underline")
        assert console.console.get_style("markdown.link") == Style.parse("bright_cyan underline")
    finally:
        console.configure_console_theme(None)


def test_console_display_without_theme_file_restores_default_theme() -> None:
    theme_file = Path(__file__).resolve().parents[4] / "examples" / "markdown" / "high-contrast.ini"

    console.configure_console_theme(None)
    try:
        console.configure_console_theme(theme_file)
        assert console.console.get_style("markdown.h3") == Style.parse("bold bright_cyan")
        assert console.console.get_style("markdown.block_quote") == Style.parse("bright_blue")
        assert console.console.get_style("markdown.code") == Style.parse("bold bright_green on black")

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
