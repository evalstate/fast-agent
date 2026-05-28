from pathlib import Path

import pytest
from rich.style import Style

import fast_agent.config as config_module
from fast_agent.cli.commands.demo import (
    DemoScenario,
    _build_scenario_markdown,
    _read_markdown_demo_asset,
    _resolve_demo_scenarios,
    markdown,
)
from fast_agent.ui import console


def test_resolve_demo_scenarios_defaults_to_mixed() -> None:
    assert _resolve_demo_scenarios(scenarios=None, cycle=False) == [DemoScenario.mixed]


def test_cycle_includes_fence_focus_first() -> None:
    scenarios = _resolve_demo_scenarios(scenarios=None, cycle=True)

    assert scenarios[0] == DemoScenario.fence_focus


def test_build_fence_focus_scenario_contains_mixed_fence_cases() -> None:
    markdown = _build_scenario_markdown(
        DemoScenario.fence_focus,
        lines=120,
        scale=2,
        seed=0,
    )

    assert "## Scenario: Fence Focus" in markdown
    assert "#### Case 1 — prose before and after a fence" in markdown
    assert "```python" in markdown
    assert "```json" in markdown
    assert "```bash" in markdown
    assert "```apply_patch" in markdown
    assert "#### Case 5 — reference definitions around a fenced block" in markdown
    assert "[render-docs]: https://example.com/rendering \"Renderer notes\"" in markdown
    assert "Trailing prose marker" in markdown


def test_read_markdown_demo_asset_contains_expected_sections() -> None:
    demo_markdown = _read_markdown_demo_asset("demo_markdown.md")

    assert "# Heading 1" in demo_markdown
    assert "## Heading 2" in demo_markdown
    assert "```python" in demo_markdown
    assert "| Column A | Column B | Column C |" in demo_markdown


def test_read_markdown_demo_theme_assets() -> None:
    yellow_theme = _read_markdown_demo_asset("yellow-headings-soft.ini")
    contrast_theme = _read_markdown_demo_asset("high-contrast.ini")

    assert "markdown.h2 = bright_yellow underline" in yellow_theme
    assert "markdown.h2 = bold bright_cyan underline" in contrast_theme


def test_markdown_demo_applies_theme_override_and_renders_content(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")

    theme_path = tmp_path / "yellow.ini"
    theme_path.write_text("[styles]\nmarkdown.h2 = yellow underline\n", encoding="utf-8")

    previous_settings = config_module._settings
    console.configure_console_theme(None)
    try:
        with console.console.capture() as capture:
            markdown(config_path=str(config_path), theme_file=theme_path)

        rendered = capture.get()
        assert "Heading 1" in rendered
        assert "Heading 2" in rendered
        assert "Bullet item one" in rendered
        assert console.console.get_style("markdown.h2") == Style.parse("yellow underline")
    finally:
        config_module._settings = previous_settings
        console.configure_console_theme(None)


def test_markdown_demo_cli_theme_override_uses_cwd_not_config_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    config_dir = workspace / ".fast-agent"
    config_dir.mkdir(parents=True)
    theme_dir = workspace / "themes"
    theme_dir.mkdir()

    config_path = config_dir / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")
    theme_path = theme_dir / "yellow.ini"
    theme_path.write_text("[styles]\nmarkdown.h2 = yellow underline\n", encoding="utf-8")

    previous_settings = config_module._settings
    console.configure_console_theme(None)
    monkeypatch.chdir(workspace)
    try:
        markdown(config_path=str(config_path), theme_file=Path("themes/yellow.ini"))

        assert console.console.get_style("markdown.h2") == Style.parse("yellow underline")
    finally:
        config_module._settings = previous_settings
        console.configure_console_theme(None)


def test_markdown_demo_renders_selected_sample_file(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("logger: {}\n", encoding="utf-8")
    sample_path = tmp_path / "sample.md"
    sample_path.write_text(
        "# Custom Sample\n\nA unique sentence for the selected markdown sample.\n",
        encoding="utf-8",
    )

    previous_settings = config_module._settings
    console.configure_console_theme(None)
    try:
        with console.console.capture() as capture:
            markdown(config_path=str(config_path), sample_file=str(sample_path))

        rendered = capture.get()
        assert "Custom Sample" in rendered
        assert "selected markdown sample" in rendered
    finally:
        config_module._settings = previous_settings
        console.configure_console_theme(None)
