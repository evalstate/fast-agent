#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run python - <<'PY'
import asyncio
import logging
import os
import signal
import shlex
import sys
import tempfile
import time
from pathlib import Path

from mcp.types import TextContent

from fast_agent.config import Settings, ShellSettings
from fast_agent.tools.shell_runtime import ShellRuntime


def text_of(result) -> str:
    assert result.content and isinstance(result.content[0], TextContent)
    return result.content[0].text


def kill_pid_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            return
        time.sleep(0.1)


async def run_case(name: str, coro) -> None:
    started = time.monotonic()
    print(f"\n== {name}")
    await coro()
    print(f"ok ({time.monotonic() - started:.2f}s)")


async def main() -> None:
    logger = logging.getLogger("shell-runtime-stress")
    logging.basicConfig(level=logging.WARNING)
    base_config = Settings(shell_execution=ShellSettings(show_bash=False))

    async def inherited_pipe_after_parent_exit() -> None:
        if sys.platform.startswith("win"):
            print("skipped on Windows")
            return
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pid_path = tmp / "descendant.pid"
            script = tmp / "hold_pipe.py"
            script.write_text(
                "\n".join(
                    [
                        "import subprocess, sys",
                        "child = subprocess.Popen(",
                        "    [sys.executable, '-c', 'import time; time.sleep(30)'],",
                        "    stdout=sys.stdout,",
                        "    stderr=sys.stderr,",
                        "    start_new_session=True,",
                        ")",
                        f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(child.pid))",
                        "print('parent exiting', flush=True)",
                    ]
                ),
                encoding="utf-8",
            )
            runtime = ShellRuntime(
                activation_reason="stress",
                logger=logger,
                timeout_seconds=10,
                config=base_config,
            )
            started = time.monotonic()
            try:
                result = await runtime.execute({"command": f'"{sys.executable}" "{script}"'})
            finally:
                kill_pid_file(pid_path)
            elapsed = time.monotonic() - started
            output = text_of(result)
            assert elapsed < 7, elapsed
            assert result.isError is False
            assert "parent exiting" in output
            assert "output collection stopped after" in output

    async def idle_timeout_with_escaped_descendant() -> None:
        if sys.platform.startswith("win"):
            print("skipped on Windows")
            return
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            pid_path = tmp / "descendant.pid"
            script = tmp / "idle_timeout.py"
            script.write_text(
                "\n".join(
                    [
                        "import subprocess, sys, time",
                        "print('before idle timeout', flush=True)",
                        "child = subprocess.Popen(",
                        "    [sys.executable, '-c', 'import time; time.sleep(30)'],",
                        "    stdout=sys.stdout,",
                        "    stderr=sys.stderr,",
                        "    start_new_session=True,",
                        ")",
                        f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(child.pid))",
                        "time.sleep(30)",
                    ]
                ),
                encoding="utf-8",
            )
            runtime = ShellRuntime(
                activation_reason="stress",
                logger=logger,
                timeout_seconds=1,
                warning_interval_seconds=10,
                config=base_config,
            )
            started = time.monotonic()
            try:
                result = await runtime.execute({"command": f'"{sys.executable}" "{script}"'})
            finally:
                kill_pid_file(pid_path)
            elapsed = time.monotonic() - started
            output = text_of(result)
            assert elapsed < 8, elapsed
            assert result.isError is True
            assert "before idle timeout" in output
            assert "timeout after 1s" in output
            assert "output collection stopped after" in output

    async def huge_output_low_retention_limit() -> None:
        runtime = ShellRuntime(
            activation_reason="stress",
            logger=logger,
            timeout_seconds=10,
            output_byte_limit=1024,
            config=base_config,
        )
        code = "import sys; sys.stdout.buffer.write(b'x' * 5_000_000)"
        result = await runtime.execute(
            {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"}
        )
        output = text_of(result)
        assert result.isError is False
        assert "process exit code was 0" in output
        assert "[Output truncated: showing first" in output
        assert len(output.encode("utf-8")) < 5_000

    async def mixed_stderr_truncation_diagnostic() -> None:
        runtime = ShellRuntime(
            activation_reason="stress",
            logger=logger,
            timeout_seconds=10,
            output_byte_limit=1024,
            config=base_config,
        )
        code = (
            "import sys; "
            "sys.stdout.write('HEAD\\n' + 'x' * 1000000); "
            "sys.stderr.write('STDERR_TAIL_MARKER\\n')"
        )
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        result = await runtime.execute({"command": command})
        output = text_of(result)
        assert result.isError is False
        assert "HEAD" in output
        assert "[Output truncated: showing first" in output
        print(f"stderr marker retained: {'STDERR_TAIL_MARKER' in output}")

    await run_case("inherited pipe after parent exit", inherited_pipe_after_parent_exit)
    await run_case("idle timeout with escaped descendant", idle_timeout_with_escaped_descendant)
    await run_case("huge output with low retention limit", huge_output_low_retention_limit)
    await run_case("mixed stderr truncation diagnostic", mixed_stderr_truncation_diagnostic)


asyncio.run(main())
PY
