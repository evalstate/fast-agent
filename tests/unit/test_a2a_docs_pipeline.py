import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = ROOT / "scripts" / "a2a_docs_pipeline.py"

spec = importlib.util.spec_from_file_location("a2a_docs_pipeline", PIPELINE_PATH)
assert spec is not None
assert spec.loader is not None
a2a_docs_pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a2a_docs_pipeline)


def test_a2a_docs_snippets_are_current() -> None:
    a2a_docs_pipeline.check()


def test_a2a_getting_started_includes_generated_snippets() -> None:
    page = ROOT / "docs" / "docs" / "a2a" / "getting-started.md"
    text = page.read_text(encoding="utf-8")
    for filename in [
        "start-fake-server.sh",
        "cli-stream-command.sh",
        "cli-stream-output.txt",
        "cli-files-command.sh",
        "cli-files-output.txt",
        "agent-card.yaml",
        "tui-session.txt",
    ]:
        assert f'docs/docs/a2a/snippets/{filename}' in text


def test_a2a_cast_asset_is_present() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-streaming-files.cast"
    assert asset.is_file()
    first_line = asset.read_text(encoding="utf-8").splitlines()[0]
    assert '"version"' in first_line


def test_a2a_getting_started_embeds_asciinema_player() -> None:
    page = ROOT / "docs" / "docs" / "a2a" / "getting-started.md"
    text = page.read_text(encoding="utf-8")
    assert "AsciinemaPlayer.create" in text
    assert "../../assets/a2a/a2a-streaming-files.cast" in text
    assert "../../assets/vendor/asciinema-player/asciinema-player.css" in text
    assert "../../assets/vendor/asciinema-player/catppuccin.css" in text
    assert "../../assets/vendor/asciinema-player/asciinema-player.min.js" in text
    assert "fast-agent-dark" in text
    assert "fast-agent-light" in text
    assert 'data-a2a-terminal-theme="auto"' in text
    assert 'data-a2a-terminal-theme="light"' in text
    assert 'data-a2a-terminal-theme="dark"' in text
    assert "rows: 27" in text


def test_asciinema_player_vendor_assets_are_present() -> None:
    vendor = ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player"
    css = vendor / "asciinema-player.css"
    catppuccin = vendor / "catppuccin.css"
    js = vendor / "asciinema-player.min.js"
    assert css.is_file()
    assert catppuccin.is_file()
    assert js.is_file()
    assert "ap-wrapper" in css.read_text(encoding="utf-8")
    catppuccin_text = catppuccin.read_text(encoding="utf-8")
    assert "asciinema-player-theme-fast-agent-dark" in catppuccin_text
    assert "asciinema-player-theme-fast-agent-light" in catppuccin_text
    assert "a2a-terminal-theme-switch" in catppuccin_text
    assert "AsciinemaPlayer" in js.read_text(encoding="utf-8")[:200]


def test_a2a_cast_contains_ansi_escape_sequences() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-streaming-files.cast"
    assert "\\u001b[" in asset.read_text(encoding="utf-8")


def test_a2a_cast_uses_compact_rows() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-streaming-files.cast"
    first_line = asset.read_text(encoding="utf-8").splitlines()[0]
    assert '"height": 27' in first_line
