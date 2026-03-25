from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_history(
    path: Path,
    *,
    turns: list[dict[str, object]],
) -> None:
    messages: list[dict[str, object]] = []
    for turn in turns:
        messages.append({"role": "user", "content": [{"type": "text", "text": "hi"}]})
        usage_payload = {
            "turn": {
                "input_tokens": turn["input_tokens"],
                "effective_input_tokens": turn["effective_input_tokens"],
                "output_tokens": turn["output_tokens"],
                "total_tokens": turn["total_tokens"],
                "reasoning_tokens": turn["reasoning_tokens"],
                "tool_calls": turn["tool_calls"],
                "cache_usage": {
                    "cache_hit_tokens": turn["cache_hit_tokens"],
                },
            }
        }
        diag_payload = {
            "transport": "websocket",
            "websocket_request_mode": turn["request_mode"],
            "websocket_turn_outcome": turn["turn_outcome"],
        }
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "channels": {
                    "fast-agent-usage": [
                        {"type": "text", "text": json.dumps(usage_payload)}
                    ],
                    "fast-agent-provider-diagnostics": [
                        {"type": "text", "text": json.dumps(diag_payload)}
                    ],
                },
            }
        )

    path.write_text(json.dumps({"messages": messages}), encoding="utf-8")


def _write_session(session_dir: Path, *, turns: list[dict[str, object]]) -> None:
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(
        json.dumps(
            {
                "name": session_dir.name,
                "history_files": ["history_dev.json"],
                "metadata": {"agent_name": "dev"},
            }
        ),
        encoding="utf-8",
    )
    _write_history(session_dir / "history_dev.json", turns=turns)


def test_session_usage_bounce_script_summarizes_recent_sessions(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    _write_session(
        sessions_root / "2603222229-AAAAAA",
        turns=[
            {
                "input_tokens": 2000,
                "effective_input_tokens": 2000,
                "output_tokens": 300,
                "total_tokens": 2300,
                "reasoning_tokens": 100,
                "tool_calls": 2,
                "cache_hit_tokens": 0,
                "request_mode": "create",
                "turn_outcome": "fresh",
            },
            {
                "input_tokens": 18000,
                "effective_input_tokens": 800,
                "output_tokens": 80,
                "total_tokens": 18080,
                "reasoning_tokens": 10,
                "tool_calls": 1,
                "cache_hit_tokens": 17200,
                "request_mode": "continuation",
                "turn_outcome": "reused",
            },
        ],
    )
    _write_session(
        sessions_root / "2603222230-BBBBBB",
        turns=[
            {
                "input_tokens": 2500,
                "effective_input_tokens": 2500,
                "output_tokens": 450,
                "total_tokens": 2950,
                "reasoning_tokens": 75,
                "tool_calls": 0,
                "cache_hit_tokens": 0,
                "request_mode": "create",
                "turn_outcome": "fresh",
            }
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/session_usage_bounce.py",
            str(sessions_root),
            "--last",
            "2",
            "--top-fields",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[3],
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "Session overview" in stdout
    assert "Fresh vs reused websocket effect (averages)" in stdout
    assert "Accounting consistency checks" in stdout
    assert "2603222230-BBBBBB" in stdout
    assert "2603222229-AAAAAA" in stdout
    assert "reused_cache_hit" in stdout
    assert "0/2" in stdout
    assert "output_tokens" in stdout or "effective_input_tokens" in stdout
