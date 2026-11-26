"""Measure fresh-process CLI startup to the first rendered prompt on POSIX.

uv run scripts/benchmark_startup.py --output /tmp/startup.json -- \
    .venv/bin/fast-agent --model copilot.gemini-4-neon -x

Sends Ctrl+D at the first prompt; no user message or model request is sent.
Normal startup side effects (session creation, configured servers/hooks) still run.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import pty
import select
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StartupSample:
    ready_seconds: float
    exit_code: int


def measure(command: list[str], *, timeout: float, output_prefix: Path | None) -> StartupSample:
    master, slave = pty.openpty()
    output = bytearray()
    ready: float | None = None
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command, stdin=slave, stdout=slave, stderr=slave, start_new_session=True
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    try:
        deadline = started + timeout
        while time.perf_counter() < deadline:
            if not select.select(
                [master], [], [], min(0.1, max(0, deadline - time.perf_counter()))
            )[0]:
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:  # POSIX PTYs report EOF this way.
                    break
                raise
            if not chunk:
                break
            output.extend(chunk)
            # Require bracketed-paste activation as well as the input glyph so
            # text in the startup banner cannot be mistaken for a ready prompt.
            if ready is None and b"\x1b[?2004h" in output and "❯".encode() in output:
                ready = time.perf_counter() - started
                os.write(master, b"\x04")
                deadline = time.perf_counter() + 10  # Allow normal shutdown.
        if ready is None:
            raise RuntimeError(
                f"No interactive prompt within {timeout:g}s (exit={process.poll()})."
            )
        return StartupSample(ready_seconds=ready, exit_code=process.wait(timeout=10))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        os.close(master)
        if output_prefix is not None:
            output_prefix.write_bytes(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument(
        "--output", type=Path, help="JSON results and adjacent per-run terminal logs"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.runs < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error("Supply a command, positive runs/timeout, and non-negative warmup.")

    samples: list[StartupSample] = []
    for index in range(args.warmup + args.runs):
        label = f"warmup-{index + 1}" if index < args.warmup else f"run-{index - args.warmup + 1}"
        log_path = args.output.with_name(f"{args.output.name}.{label}.log") if args.output else None
        sample = measure(command, timeout=args.timeout, output_prefix=log_path)
        print(
            f"{label}: {sample.ready_seconds * 1000:.1f} ms (exit={sample.exit_code})", flush=True
        )
        if sample.exit_code != 0:
            raise SystemExit(
                "Startup reached a prompt but shutdown failed; inspect the terminal log."
            )
        if index >= args.warmup:
            samples.append(sample)

    timings = [sample.ready_seconds for sample in samples]
    median = statistics.median(timings)
    print(
        f"Median: {median * 1000:.1f} ms; range: {min(timings) * 1000:.1f}–{max(timings) * 1000:.1f} ms"
    )
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "command": command,
                    "cwd": str(Path.cwd()),
                    "warmup_runs": args.warmup,
                    "runs": [asdict(sample) for sample in samples],
                    "median_seconds": median,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
