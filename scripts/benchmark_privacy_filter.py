"""Benchmark the privacy filter on text from a persisted session.

Examples:

    uv run --extra privacy scripts/benchmark_privacy_filter.py \
      --model-dir ~/.cache/fast-agent/privacy-filter-q4f16 \
      --session-dir ~/temp/skills-test/llama.cpp/.fast-agent/sessions/2604242111-39GThH \
      --list

    uv run --extra privacy scripts/benchmark_privacy_filter.py \
      --model-dir ~/.cache/fast-agent/privacy-filter-q4f16 \
      --session-dir ~/temp/skills-test/llama.cpp/.fast-agent/sessions/2604242111-39GThH \
      --candidate 3
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.privacy.privacy_filter_onnx import OpenAIPrivacyFilterOnnxSanitizer

if TYPE_CHECKING:
    from collections.abc import Iterator


DEFAULT_SESSION_DIR = Path(
    "~/temp/skills-test/llama.cpp/.fast-agent/sessions/2604242111-39GThH"
).expanduser()
DEFAULT_MODEL_DIR = Path("~/.cache/fast-agent/privacy-filter-q4f16").expanduser()


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    text: str


def _history_path(session_dir: Path, agent: str) -> Path:
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    agent_info = session.get("continuation", {}).get("agents", {}).get(agent, {})
    history_file = agent_info.get("history_file")
    if not isinstance(history_file, str):
        raise SystemExit(f"No history file for agent {agent!r} in {session_dir}")
    return session_dir / history_file


def _string_values(value: object, *, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        if value:
            yield path, value
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _string_values(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            yield from _string_values(item, path=next_path)


def _load_candidates(session_dir: Path, agent: str) -> list[Candidate]:
    history = json.loads(_history_path(session_dir, agent).read_text(encoding="utf-8"))
    messages = history.get("messages")
    if not isinstance(messages, list):
        raise SystemExit("History file has no messages list")

    candidates: list[Candidate] = []
    for message_index, message in enumerate(messages):
        role = message.get("role") if isinstance(message, dict) else None
        for path, text in _string_values(message):
            if len(text) < 200:
                continue
            # Skip noisy telemetry JSON unless it is itself large enough to matter.
            if "fast-agent-usage" in path and len(text) < 5_000:
                continue
            candidates.append(
                Candidate(
                    label=f"message[{message_index}] {role or '?'} {path}",
                    text=text,
                )
            )
    return sorted(candidates, key=lambda item: len(item.text), reverse=True)


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _run_case(
    *,
    model_dir: Path,
    text: str,
    window_tokens: int,
    overlap_tokens: int,
    intra_threads: int,
    repeat: int,
    show_spans: bool,
) -> None:
    os.environ["FAST_AGENT_PRIVACY_FILTER_MAX_WINDOW_TOKENS"] = str(window_tokens)
    os.environ["FAST_AGENT_PRIVACY_FILTER_WINDOW_OVERLAP_TOKENS"] = str(overlap_tokens)
    os.environ["FAST_AGENT_PRIVACY_FILTER_INTRA_OP_THREADS"] = str(intra_threads)
    os.environ["FAST_AGENT_PRIVACY_FILTER_INTER_OP_THREADS"] = "1"

    progress: list[str] = []
    started = time.perf_counter()
    sanitizer = OpenAIPrivacyFilterOnnxSanitizer(model_dir, progress_callback=progress.append)
    load_elapsed = time.perf_counter() - started

    best: float | None = None
    last = None
    for _ in range(repeat):
        started = time.perf_counter()
        last = sanitizer.sanitize_text(text)
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)

    print(
        f"window={window_tokens:<5} overlap={overlap_tokens:<3} threads={intra_threads:<2} "
        f"load={load_elapsed:6.2f}s best={best or 0:6.2f}s "
        f"rss={_rss_mb():8.1f}MB spans={len(last.spans) if last else 0}"
    )
    for line in progress:
        if "scanning" in line:
            print(f"  {line}")
            break
    if show_spans and last is not None:
        for span in last.spans[:20]:
            snippet = text[span.start : span.end].replace("\n", "\\n")
            print(f"  {span.label} {span.start}:{span.end} {snippet!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=Path, default=DEFAULT_SESSION_DIR)
    parser.add_argument("--agent", default="agent")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--candidate", type=int, default=0)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--show-spans", action="store_true")
    args = parser.parse_args()

    candidates = _load_candidates(args.session_dir.expanduser(), args.agent)
    if args.list:
        for index, candidate in enumerate(candidates[:30]):
            preview = " ".join(candidate.text.split())[:100]
            print(f"{index:2d} len={len(candidate.text):6d} {candidate.label} {preview!r}")
        return

    candidate = candidates[args.candidate]
    print(f"Candidate {args.candidate}: len={len(candidate.text):,} {candidate.label}")
    print()
    for window_tokens, overlap_tokens, intra_threads in [
        (512, 16, 1),
        (1024, 32, 1),
        (2048, 64, 1),
        (1024, 32, 2),
    ]:
        _run_case(
            model_dir=args.model_dir.expanduser(),
            text=candidate.text,
            window_tokens=window_tokens,
            overlap_tokens=overlap_tokens,
            intra_threads=intra_threads,
            repeat=args.repeat,
            show_spans=args.show_spans,
        )


if __name__ == "__main__":
    main()
