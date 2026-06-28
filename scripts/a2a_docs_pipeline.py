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
import importlib.util
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def _load_docs_asset_helpers():
    try:
        from docs_assets import (
            asciinema_index_problems,
            record_asciinema_cast,
            require_recording_tools,
            write_asciinema_index,
        )

        return (
            record_asciinema_cast,
            require_recording_tools,
            write_asciinema_index,
            asciinema_index_problems,
        )
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parent / "docs_assets.py"
        spec = importlib.util.spec_from_file_location("docs_assets", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return (
            module.record_asciinema_cast,
            module.require_recording_tools,
            module.write_asciinema_index,
            module.asciinema_index_problems,
        )


(
    record_asciinema_cast,
    require_recording_tools,
    write_asciinema_index,
    asciinema_index_problems,
) = _load_docs_asset_helpers()

ROOT = Path(__file__).resolve().parent.parent
DOCS_A2A = ROOT / "docs" / "docs" / "a2a"
SNIPPETS = DOCS_A2A / "snippets"
ASSETS = ROOT / "docs" / "docs" / "assets" / "a2a"
PORT = 41242
BASE_URL = f"http://127.0.0.1:{PORT}"

START_FAKE_SERVER = f"uv run python tests/integration/a2a/fake_server.py --port {PORT}\n"
HELLO_COMMAND = f"""uv run fast-agent -x \\
  --a2a {BASE_URL} \\
  --a2a-transport JSONRPC \\
  --message "hello" \\
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
TUI_SESSION = "/a2a help\nhelp\n/a2a status\n/a2a transport\nrespond with files\nneed input\nblue\n"

STATIC_SNIPPETS = {
    "start-fake-server.sh": START_FAKE_SERVER,
    "cli-hello-command.sh": HELLO_COMMAND,
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
    time.sleep(0.2)
    if process.poll() is not None:
        stdout = process.stdout.read() if process.stdout else ""
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError(
            f"fake A2A server exited immediately. Is port {PORT} already in use?\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
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


def _shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def generate() -> None:
    SNIPPETS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    for filename, text in STATIC_SNIPPETS.items():
        _write(SNIPPETS / filename, text)

    server = _start_server()
    try:
        _write(SNIPPETS / "cli-hello-output.txt", _run(HELLO_COMMAND))
        _write(SNIPPETS / "cli-files-output.txt", _run(FILES_COMMAND))
    finally:
        _stop_server(server)


def check() -> None:
    expected = dict(STATIC_SNIPPETS)
    expected["cli-hello-output.txt"] = "fake echo: hello [text]\n"
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
        ASSETS / "a2a-client-input-required.cast",
        ASSETS / "a2a-server-card.cast",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "asciinema-player.css",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "catppuccin.css",
        ROOT / "docs" / "docs" / "assets" / "vendor" / "asciinema-player" / "asciinema-player.min.js",
    ]
    for asset in required_assets:
        if not asset.exists():
            missing_or_changed.append(str(asset.relative_to(ROOT)))

    page = DOCS_A2A / "getting-started.md"
    page_text = page.read_text(encoding="utf-8") if page.exists() else ""
    for stale_text in [
        "AsciinemaPlayer.create",
        "data-a2a-terminal-theme",
        "a2a-terminal-theme-switch",
        "a2a-streaming-files.cast",
    ]:
        if stale_text in page_text:
            missing_or_changed.append(f"{page.relative_to(ROOT)} still contains {stale_text}")

    page_assets = {
        DOCS_A2A / "client.md": {
            'data-fa-asciinema-cast="../../assets/a2a/a2a-client-input-required.cast"',
            'data-fa-asciinema-rows="18"',
        },
        DOCS_A2A / "server.md": {
            'data-fa-asciinema-cast="../../assets/a2a/a2a-server-card.cast"',
            'data-fa-asciinema-cols="104"',
            'data-fa-asciinema-rows="20"',
        },
    }
    for asset_page, required_texts in page_assets.items():
        asset_page_text = asset_page.read_text(encoding="utf-8") if asset_page.exists() else ""
        for required_text in required_texts:
            if required_text not in asset_page_text:
                missing_or_changed.append(
                    f"{asset_page.relative_to(ROOT)} missing {required_text}"
                )
        for stale_text in [
            "AsciinemaPlayer.create",
            "a2a-terminal-demo",
            "a2a-client-cli.cast",
            "a2a-real-llm-hf-streaming.cast",
        ]:
            if stale_text in asset_page_text:
                missing_or_changed.append(
                    f"{asset_page.relative_to(ROOT)} still contains {stale_text}"
                )
    missing_or_changed.extend(asciinema_index_problems())

    if missing_or_changed:
        raise SystemExit(
            "A2A docs snippets/assets are stale; run `uv run scripts/a2a_docs_pipeline.py generate`.\n"
            + "\n".join(missing_or_changed)
        )


def _record_input_required(driver: Path) -> None:
    driver.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '\\033[1;35mfast-agent ▸\\033[0m need input\\r\\n'
printf '\\033[1;33mA2A task TASK_STATE_INPUT_REQUIRED:\\033[0m Please provide the missing value.\\r\\n'
printf '\\033[2m/a2a status a2a_remote\\033[0m\\r\\n'
printf '\\033[1mA2A status: a2a_remote\\033[0m\\r\\n'
printf '  URL: \\033[4;36m{BASE_URL}\\033[0m\\r\\n'
printf '  Transport: \\033[36mJSONRPC\\033[0m\\r\\n'
printf '  Context: \\033[32m7b7c8d9e\\033[0m\\r\\n'
printf '  Task: \\033[33mtask-input-001\\033[0m\\r\\n'
printf '  Last state: \\033[33mTASK_STATE_INPUT_REQUIRED\\033[0m\\r\\n'
printf '  Client transport: \\033[36mJsonRpcTransport\\033[0m\\r\\n'
printf '\\033[1;35mfast-agent ▸\\033[0m blue\\r\\n'
printf '\\033[32minput received: blue\\033[0m\\r\\n'
printf '\\033[2mTask cleared after completion; context preserved for the next turn.\\033[0m\\r\\n'
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    record_asciinema_cast(
        output=ASSETS / "a2a-client-input-required.cast",
        title="fast-agent A2A input-required continuation",
        command=str(driver),
        cols=96,
        rows=18,
        idle_time_limit=1,
    )


def _record_server_card(driver: Path) -> None:
    driver.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={_shell_quote(ROOT)}
LOG=/tmp/a2a-docs-server-card.log
cd "$ROOT"
rm -f "$LOG"
uv run fast-agent serve --transport a2a --host 0.0.0.0 --port 41241 --model passthrough >"$LOG" 2>&1 &
SERVER_PID=$!
cleanup() {{
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}}
trap cleanup EXIT

for _ in $(seq 1 80); do
  if curl -fsS -H 'Host: a2a.example.test:41241' 'http://127.0.0.1:41241/.well-known/agent-card.json' >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$LOG"
    exit 1
  fi
  sleep 0.25
done
curl -fsS -H 'Host: a2a.example.test:41241' 'http://127.0.0.1:41241/.well-known/agent-card.json' >/dev/null

printf '\\033[1;36m$ uv run fast-agent serve --transport a2a --host 0.0.0.0 --port 41241 --model passthrough\\033[0m\\r\\n'
printf 'fast-agent A2A server listening on http://0.0.0.0:41241\\r\\n'
printf '\\033[1;36m$ curl -s -H "Host: a2a.example.test:41241" http://127.0.0.1:41241/.well-known/agent-card.json | python -m json.tool\\033[0m\\r\\n'
curl -fsS -H 'Host: a2a.example.test:41241' 'http://127.0.0.1:41241/.well-known/agent-card.json' \\
  | python -c 'import json, sys; print(json.dumps(json.load(sys.stdin)["supportedInterfaces"], indent=2))'
printf '\\033[1;32mThe served card uses the hostname from the incoming card request.\\033[0m\\r\\n'
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)
    record_asciinema_cast(
        output=ASSETS / "a2a-server-card.cast",
        title="fast-agent A2A server card and transports",
        command=str(driver),
        cols=104,
        rows=20,
        idle_time_limit=1,
    )


def record() -> None:
    generate()
    try:
        require_recording_tools()
    except RuntimeError as exc:
        print(f"{exc}; generated text snippets only", file=sys.stderr)
        return

    with tempfile.TemporaryDirectory(prefix="fast-agent-a2a-docs-") as temp_dir:
        temp_path = Path(temp_dir)
        _record_input_required(temp_path / "a2a-input-required.sh")
        _record_server_card(temp_path / "a2a-server-card.sh")
    write_asciinema_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "check", "record"])
    args = parser.parse_args()
    if args.command == "generate":
        generate()
    elif args.command == "check":
        check()
    elif args.command == "record":
        record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
