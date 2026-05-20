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
