from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PLUGIN_PATH = ROOT / "examples/plugin-commands/peek_commands.py"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("peek_commands", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_style(path: Path) -> str:
    match = re.search(r"<style>\s*(.*?)\s*</style>", path.read_text(encoding="utf-8"), re.S | re.I)
    assert match is not None
    return match.group(1)


def css_rule(css: str, selector: str) -> str:
    match = re.search(rf"(^|\n)\s*{re.escape(selector)}\s*\{{(.*?)\n\s*\}}", css, re.S)
    assert match is not None, selector
    return re.sub(r"\s+", " ", match.group(2)).strip()


def test_active_config_resolves_to_peek_commands_handler() -> None:
    config = (ROOT / ".cdx/fast-agent.yaml").read_text(encoding="utf-8")

    assert "html-summary:" in config
    assert "../examples/plugin-commands/peek_commands.py:html_summary" in config
    assert "./command-actions/example_commands.py:html_summary" not in config


def test_viewer_css_uses_05_design_system_rules() -> None:
    plugin = _load_plugin()
    css = plugin._viewer_css()
    upstream = extract_style(ROOT / ".cdx/html-effectiveness/upstream/05-design-system.html")

    assert "Generated from upstream/05-design-system.html" in css
    for selector in [
        "body",
        "header h1",
        "header .sub",
        "h2",
        ".t-body",
        ".t-small",
        ".btn",
        ".badge",
        ".component-stage",
    ]:
        assert css_rule(css, selector) == css_rule(upstream, selector)


def test_viewer_css_has_no_stale_extension_overrides() -> None:
    plugin = _load_plugin()
    css = plugin._viewer_css()

    assert ".wrap { max-width: 1040px" not in css
    assert ".shell {" not in css
    assert ".hero-card" not in css
    assert ".widget" not in css
    assert ".glass" not in css
    assert ".tile" not in css


def test_rendered_command_uses_block_code() -> None:
    plugin = _load_plugin()
    command = plugin.CommandItem(label="Run checks", command="uv run scripts/lint.py")

    html = plugin.render_command(command)

    assert '<pre class="code"><code>uv run scripts/lint.py</code></pre>' in html
    assert "<code>uv run scripts/lint.py</code>" in html
    assert "<p><code>uv run scripts/lint.py</code></p>" not in html
