import importlib.util
import json
import re
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
    ]:
        assert f'docs/docs/a2a/snippets/{filename}' in text


def test_a2a_client_includes_generated_snippets() -> None:
    page = ROOT / "docs" / "docs" / "a2a" / "client.md"
    text = page.read_text(encoding="utf-8")
    for filename in [
        "start-fake-server.sh",
        "cli-hello-command.sh",
        "cli-hello-output.txt",
        "cli-files-command.sh",
        "cli-files-output.txt",
    ]:
        assert f'docs/docs/a2a/snippets/{filename}' in text


def test_a2a_client_server_cast_assets_are_present() -> None:
    assets = ROOT / "docs" / "docs" / "assets" / "a2a"
    for filename in [
        "a2a-client-input-required.cast",
        "a2a-server-card.cast",
    ]:
        asset = assets / filename
        assert asset.is_file()
        first_line = asset.read_text(encoding="utf-8").splitlines()[0]
        assert '"version"' in first_line


def test_a2a_getting_started_does_not_embed_stale_streaming_recording() -> None:
    page = ROOT / "docs" / "docs" / "a2a" / "getting-started.md"
    text = page.read_text(encoding="utf-8")
    assert 'class="fa-terminal-demo"' not in text
    assert "a2a-streaming-files.cast" not in text
    assert "AsciinemaPlayer.create" not in text
    assert "data-a2a-terminal-theme" not in text


def test_a2a_client_server_pages_embed_recordings() -> None:
    client = (ROOT / "docs" / "docs" / "a2a" / "client.md").read_text(encoding="utf-8")
    server = (ROOT / "docs" / "docs" / "a2a" / "server.md").read_text(encoding="utf-8")
    assert (
        'data-fa-asciinema-cast="../../assets/a2a/a2a-client-input-required.cast"'
        in client
    )
    assert 'data-fa-asciinema-cast="../../assets/a2a/a2a-server-card.cast"' in server
    assert "AsciinemaPlayer.create" not in client
    assert "AsciinemaPlayer.create" not in server


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


def test_a2a_input_required_cast_contains_ansi_escape_sequences() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-client-input-required.cast"
    assert "\\u001b[" in asset.read_text(encoding="utf-8")


def test_a2a_input_required_cast_uses_compact_rows() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-client-input-required.cast"
    first_line = asset.read_text(encoding="utf-8").splitlines()[0]
    assert '"height": 18' in first_line


def test_a2a_turn_continuation_cast_uses_terminal_colours() -> None:
    asset = ROOT / "docs" / "docs" / "assets" / "a2a" / "a2a-client-input-required.cast"
    output = "".join(
        event[2]
        for line in asset.read_text(encoding="utf-8").splitlines()[1:]
        if (event := json.loads(line))[1] == "o"
    )
    assert re.search(r"\x1b\[(?:[0-9;]*;)?3[0-7]m", output)
