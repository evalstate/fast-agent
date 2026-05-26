#!/usr/bin/env python3
"""Generate and verify A2A getting-started docs assets.

This script keeps docs examples, smoke-test commands, and captured output aligned.
It starts the deterministic fake A2A server, runs the documented CLI examples,
and writes the snippets consumed by docs/docs/a2a/getting-started.md.

Usage:
    uv run scripts/a2a_docs_pipeline.py generate
    uv run scripts/a2a_docs_pipeline.py check
    uv run scripts/a2a_docs_pipeline.py record
    uv run scripts/a2a_docs_pipeline.py record-real-llm
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _load_docs_asset_helpers():
    try:
        from docs_assets import record_asciinema_cast, require_recording_tools

        return record_asciinema_cast, require_recording_tools
    except ModuleNotFoundError:
        path = Path(__file__).resolve().parent / "docs_assets.py"
        spec = importlib.util.spec_from_file_location("docs_assets", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module.record_asciinema_cast, module.require_recording_tools


record_asciinema_cast, require_recording_tools = _load_docs_asset_helpers()

ROOT = Path(__file__).resolve().parent.parent
DOCS_A2A = ROOT / "docs" / "docs" / "a2a"
SNIPPETS = DOCS_A2A / "snippets"
ASSETS = ROOT / "docs" / "docs" / "assets" / "a2a"
RECORDS = Path.home() / "plan" / "records"
PORT = 41242
REAL_LLM_PORT = 41243
BASE_URL = f"http://127.0.0.1:{PORT}"
REAL_LLM_BASE_URL = f"http://127.0.0.1:{REAL_LLM_PORT}"
REAL_LLM_MCP_URL = "https://hf.co/mcp"
REAL_LLM_MODEL = "codexresponses.gpt-5.4-mini"
REAL_LLM_CAST = "a2a-real-llm-hf-streaming.cast"
REAL_LLM_SERVER_LOG = Path("/tmp/a2a-real-llm-server.log")
REAL_LLM_READY_TIMEOUT_SECONDS = 90.0

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
TUI_SESSION = "/a2a help\nhelp\n/a2a status\n/a2a transport\nplease stream\nrespond with files\nneed input\nblue\n"

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


def _log_tail(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return f"{path} does not exist"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _wait_for_url(
    url: str,
    *,
    process: subprocess.Popen[str] | None = None,
    log_path: Path | None = None,
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            details = f"\nLOG:\n{_log_tail(log_path)}" if log_path is not None else ""
            raise RuntimeError(
                f"process exited before {url} became ready with status {process.returncode}{details}"
            )
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:  # noqa: S310 - docs smoke URL
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.5)
    details = f"\nLOG:\n{_log_tail(log_path)}" if log_path is not None else ""
    raise TimeoutError(f"{url} did not become ready within {timeout_seconds:.1f}s{details}")


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
        ASSETS / "a2a-client-cli.cast",
        ASSETS / "a2a-client-input-required.cast",
        ASSETS / "a2a-server-card.cast",
        ASSETS / REAL_LLM_CAST,
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
        "fast-agent-dark",
        "fast-agent-light",
    ]:
        if required_text not in page_text:
            missing_or_changed.append(f"{page.relative_to(ROOT)} missing {required_text}")

    page_assets = {
        DOCS_A2A / "client.md": [
            "../../assets/a2a/a2a-client-cli.cast",
            "../../assets/a2a/a2a-client-input-required.cast",
            f"../../assets/a2a/{REAL_LLM_CAST}",
        ],
        DOCS_A2A / "server.md": ["../../assets/a2a/a2a-server-card.cast"],
    }
    for asset_page, required_texts in page_assets.items():
        asset_page_text = asset_page.read_text(encoding="utf-8") if asset_page.exists() else ""
        for required_text in required_texts:
            if required_text not in asset_page_text:
                missing_or_changed.append(
                    f"{asset_page.relative_to(ROOT)} missing {required_text}"
                )

    if missing_or_changed:
        raise SystemExit(
            "A2A docs snippets/assets are stale; run `uv run scripts/a2a_docs_pipeline.py generate`.\n"
            + "\n".join(missing_or_changed)
        )


def record() -> None:
    generate()
    try:
        require_recording_tools()
    except RuntimeError as exc:
        print(f"{exc}; generated text snippets only", file=sys.stderr)
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
  "cd '$ROOT' && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 FAST_AGENT_MODEL=passthrough uv run fast-agent -x --a2a '$BASE_URL' --a2a-transport JSONRPC"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 4
  tmux send-keys -t "$SESSION" '/a2a help' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'help' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'please stream' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'respond with files' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'need input' Enter
  sleep 4
  tmux send-keys -t "$SESSION" 'blue' Enter
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
        record_asciinema_cast(
            output=ASSETS / "a2a-streaming-files.cast",
            title="fast-agent A2A streaming, files, and input-required demo",
            command=str(driver),
            cols=104,
            rows=27,
            cleanup_session="a2a_docs_cast",
        )
    finally:
        _stop_server(server)


def _require_real_llm_recording_tools() -> None:
    try:
        require_recording_tools(("asciinema", "tmux", "curl"))
    except RuntimeError as exc:
        raise SystemExit(str(exc).replace("Cannot record docs assets", "record-real-llm")) from exc
    missing_env = [
        name
        for name in ["HF_TOKEN", "OPENAI_API_KEY"]
        if not os.environ.get(name)
    ]
    if missing_env:
        raise SystemExit(
            "record-real-llm requires environment variables: " + ", ".join(missing_env)
        )


def _start_real_llm_server(instruction: Path) -> subprocess.Popen[str]:
    REAL_LLM_SERVER_LOG.unlink(missing_ok=True)
    log_file = REAL_LLM_SERVER_LOG.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["FAST_AGENT_KEYRING_NOTICE"] = "0"
    model = env.get("A2A_REAL_LLM_MODEL", REAL_LLM_MODEL)
    hf_mcp_url = env.get("A2A_HF_MCP_URL", REAL_LLM_MCP_URL)
    command = [
        "uv",
        "run",
        "fast-agent",
        "serve",
        "a2a",
        "--host",
        "127.0.0.1",
        "--port",
        str(REAL_LLM_PORT),
        "--name",
        "hf-model-research",
        "--model",
        model,
        "--url",
        hf_mcp_url,
        "--instruction",
        str(instruction),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    _wait_for_url(
        f"{REAL_LLM_BASE_URL}/.well-known/agent-card.json",
        process=process,
        log_path=REAL_LLM_SERVER_LOG,
        timeout_seconds=float(
            os.environ.get(
                "A2A_REAL_LLM_READY_TIMEOUT_SECONDS",
                str(REAL_LLM_READY_TIMEOUT_SECONDS),
            )
        ),
    )
    return process


def _stop_real_llm_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def record_real_llm() -> None:
    """Record a provider-backed A2A server/client streaming demo."""
    _require_real_llm_recording_tools()
    ASSETS.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)

    instruction = Path("/tmp/a2a-real-llm-instruction.md")
    instruction.write_text(
        """You are a concise Hugging Face model research assistant.

Use available Hugging Face MCP tools to answer questions about models. When the
user asks about trending models, use markdown with a short heading, 3-5 bullets,
and a brief note about the source or any uncertainty.
""",
        encoding="utf-8",
    )

    server = _start_real_llm_server(instruction)
    driver = Path("/tmp/a2a-real-llm-record.sh")
    driver.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
SESSION=a2a_real_llm_cast
ROOT={ROOT}
BASE_URL={REAL_LLM_BASE_URL}
SERVER_LOG={REAL_LLM_SERVER_LOG}
RECORD_SECONDS="${{A2A_REAL_LLM_RECORD_SECONDS:-70}}"
MODEL="${{A2A_REAL_LLM_MODEL:-{REAL_LLM_MODEL}}}"
HF_MCP_URL="${{A2A_HF_MCP_URL:-{REAL_LLM_MCP_URL}}}"
PROMPT='Use the Hugging Face MCP server if available. Answer in markdown: what models are trending on Hugging Face right now? Include concise bullets and mention any uncertainty.'

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x 120 -y 32 \
  "printf 'fast-agent A2A server ready\\nmodel: %s\\nMCP: %s\\nlog: %s\\n\\n' '$MODEL' '$HF_MCP_URL' '$SERVER_LOG'; tail -n 80 -f '$SERVER_LOG'"
tmux set-option -t "$SESSION" status off >/dev/null
tmux split-window -v -t "$SESSION" -l 20 \
  "cd '$ROOT' && printf 'A2A card: %s/.well-known/agent-card.json\\n' '$BASE_URL'; curl -fsS '$BASE_URL/.well-known/agent-card.json' | python -m json.tool | sed -n '1,22p'; printf '\\ninteractive A2A JSON-RPC client\\n'; TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 FAST_AGENT_MODEL=passthrough uv run fast-agent -x --noenv --a2a '$BASE_URL' --a2a-transport JSONRPC"

(
  for _ in $(seq 1 120); do
    if tmux capture-pane -p -t "$SESSION":0.1 | grep -q 'a2a_remote'; then
      break
    fi
    sleep 0.5
  done
  sleep 1
  tmux send-keys -l -t "$SESSION":0.1 "$PROMPT"
  tmux send-keys -t "$SESSION":0.1 Enter
  sleep "$RECORD_SECONDS"
  tmux send-keys -t "$SESSION":0.1 '/exit' Enter
  sleep 2
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux select-pane -t "$SESSION":0.1
tmux attach-session -t "$SESSION" || true
""",
        encoding="utf-8",
    )
    driver.chmod(0o755)

    try:
        record_asciinema_cast(
            output=ASSETS / REAL_LLM_CAST,
            title="fast-agent A2A real LLM Hugging Face MCP streaming demo",
            command=str(driver),
            cols=120,
            rows=32,
            cleanup_session="a2a_real_llm_cast",
        )
    finally:
        _stop_real_llm_server(server)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["generate", "check", "record", "record-real-llm"])
    args = parser.parse_args()
    if args.command == "generate":
        generate()
    elif args.command == "check":
        check()
    elif args.command == "record":
        record()
    else:
        record_real_llm()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
