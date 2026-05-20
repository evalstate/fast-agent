#!/usr/bin/env python3
"""Generate and verify A2A getting-started docs assets.

This script keeps docs examples, smoke-test commands, and captured output aligned.
It starts the deterministic fake A2A server, runs the documented CLI examples,
and writes the snippets consumed by docs/docs/a2a/getting-started.md.

Usage:
    uv run scripts/a2a_docs_pipeline.py generate
    uv run scripts/a2a_docs_pipeline.py check
    uv run scripts/a2a_docs_pipeline.py record
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_A2A = ROOT / "docs" / "docs" / "a2a"
SNIPPETS = DOCS_A2A / "snippets"
ASSETS = ROOT / "docs" / "docs" / "assets" / "a2a"
RECORDS = Path.home() / "plan" / "records"
PORT = 41242
BASE_URL = f"http://127.0.0.1:{PORT}"

START_FAKE_SERVER = f"uv run python tests/integration/a2a/fake_server.py --port {PORT}\n"
STREAM_COMMAND = f"""uv run fast-agent -x \\
  --a2a {BASE_URL} \\
  --a2a-transport JSONRPC \\
  --message "please stream" \\
  --quiet
"""
FILES_COMMAND = f"""uv run fast-agent -x \\
  --a2a {BASE_URL} \\
  --a2a-transport HTTP+JSON \\
  --message "respond with files" \\
  --quiet
"""
AGENT_CARD = f"""type: a2a
name: fake_remote
url: {BASE_URL}
transport: JSONRPC
"""
TUI_SESSION = "/a2a status\n/a2a transport\nplease stream\nrespond with files\n"

STATIC_SNIPPETS = {
    "start-fake-server.sh": START_FAKE_SERVER,
    "cli-stream-command.sh": STREAM_COMMAND,
    "cli-files-command.sh": FILES_COMMAND,
    "agent-card.yaml": AGENT_CARD,
    "tui-session.txt": TUI_SESSION,
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(command: str) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {command}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout.strip() + "\n"


def _wait_for_server(process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    url = f"{BASE_URL}/.well-known/agent-card.json"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("fake A2A server exited before it was ready")
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:  # noqa: S310 - local test server
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError(f"fake A2A server did not become ready at {url}")


def _start_server() -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "tests/integration/a2a/fake_server.py",
            "--port",
            str(PORT),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_server(process)
    return process


def _stop_server(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def generate() -> None:
    SNIPPETS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for filename, text in STATIC_SNIPPETS.items():
        _write(SNIPPETS / filename, text)

    server = _start_server()
    try:
        _write(SNIPPETS / "cli-stream-output.txt", _run(STREAM_COMMAND))
        _write(SNIPPETS / "cli-files-output.txt", _run(FILES_COMMAND))
    finally:
        _stop_server(server)

    source_cast = RECORDS / "a2a-streaming-files.cast"
    if source_cast.exists():
        shutil.copyfile(source_cast, ASSETS / "a2a-streaming-files.cast")


def check() -> None:
    expected = dict(STATIC_SNIPPETS)
    expected["cli-stream-output.txt"] = "stream chunk one\nstream chunk two\n"
    expected["cli-files-output.txt"] = (
        "file response\n"
        "[report.pdf](https://example.com/report.pdf) (application/pdf)\n"
        "```json\n"
        "{\n"
        "  \"ok\": true,\n"
        "  \"source\": \"fake-a2a-server\"\n"
        "}\n"
        "```\n"
        "[note.txt: 3 bytes text/plain]\n"
    )
    missing_or_changed: list[str] = []
    for filename, text in expected.items():
        path = SNIPPETS / filename
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            missing_or_changed.append(str(path.relative_to(ROOT)))
    required_assets = [
        ASSETS / "a2a-streaming-files.cast",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "asciinema-player.css",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "catppuccin.css",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "asciinema-player.min.js",
    ]
    for asset in required_assets:
        if not asset.exists():
            missing_or_changed.append(str(asset.relative_to(ROOT)))

    page = DOCS_A2A / "getting-started.md"
    page_text = page.read_text(encoding="utf-8") if page.exists() else ""
    for required_text in [
        "AsciinemaPlayer.create",
        "../../assets/a2a/a2a-streaming-files.cast",
        "../../assets/vendor/asciinema-player/asciinema-player.css",
        "../../assets/vendor/asciinema-player/catppuccin.css",
        "../../assets/vendor/asciinema-player/asciinema-player.min.js",
        "catppuccin-mocha",
        "catppuccin-latte",
    ]:
        if required_text not in page_text:
            missing_or_changed.append(f"{page.relative_to(ROOT)} missing {required_text}")

    if missing_or_changed:
        raise SystemExit(
            "A2A docs snippets/assets are stale; run `uv run scripts/a2a_docs_pipeline.py generate`.\n"
            + "\n".join(missing_or_changed)
        )


def record() -> None:
    generate()
    if not shutil.which("asciinema"):
        print("asciinema is not installed; generated text snippets only", file=sys.stderr)
        return
    if not shutil.which("tmux"):
        print("tmux is not installed; generated text snippets only", file=sys.stderr)
        return

    driver = Path("/tmp/a2a-docs-record.sh")
    driver.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
SESSION=a2a_docs_cast
ROOT={ROOT}
BASE_URL={BASE_URL}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 104 -y 27 \
  "cd '$ROOT' && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_MODEL=passthrough uv run fast-agent -x --a2a '$BASE_URL' --a2a-transport JSONRPC"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 4
  tmux send-keys -t "$SESSION" 'please stream' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'respond with files' Enter
  sleep 4
  tmux send-keys -t "$SESSION" '/exit' Enter
  sleep 1
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)

    server = _start_server()
    try:
        command = [
            "asciinema",
            "rec",
            "--overwrite",
            "--cols",
            "104",
            "--rows",
            "27",
            "--idle-time-limit",
            "1.3",
            "-t",
            "fast-agent A2A streaming and files demo",
            "-c",
            str(driver),
            str(ASSETS / "a2a-streaming-files.cast"),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", "a2a_docs_cast"], check=False)
        _stop_server(server)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "check", "record"])
    args = parser.parse_args()
    if args.command == "generate":
        generate()
    elif args.command == "check":
        check()
    else:
        record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
