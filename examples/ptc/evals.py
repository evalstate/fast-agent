"""Lightweight evals for programmatic tool calling (PTC).

Each case runs the real CLI (`fast-agent go --ptc`) with the `ptc_eval` AgentCard, then checks
the final answer *and* how the model reached it, using the saved history. Every case prints the
exact command, so a failure can be reproduced (and watched) by hand.

    uv run evals.py                      # all cases with the default model from fast-agent.yaml
    uv run evals.py --model sonnet       # another model
    uv run evals.py tool-image subagents # selected cases
    uv run evals.py --list               # print the commands without running them

The media cases name a colour that exists only in image pixels, so they pass only if the image
reached the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

HERE = Path(__file__).parent
PTC_TOOL = "execute_python"
PTC_CALLS_META = "fast-agent-ptc-calls"
TOOL_MEDIA_CHANNEL = "fast-agent-tool-media-message"

History = list[dict[str, Any]]


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    expect: tuple[str, ...]
    """Case-insensitive substrings the final answer must contain."""
    check: Callable[[History], str | None]
    """Return a failure reason, or None when the run took the intended path."""


def tool_calls(history: History) -> list[tuple[str, str, dict[str, Any]]]:
    """(call id, tool name, arguments) for every top-level tool call."""
    return [
        (call_id, call["params"]["name"], call["params"].get("arguments") or {})
        for message in history
        for call_id, call in (message.get("tool_calls") or {}).items()
    ]


def tool_results(history: History) -> dict[str, dict[str, Any]]:
    return {
        call_id: result
        for message in history
        for call_id, result in (message.get("tool_results") or {}).items()
    }


def code_results(history: History) -> list[dict[str, Any]]:
    results = tool_results(history)
    return [results[call_id] for call_id, name, _ in tool_calls(history) if name == PTC_TOOL]


def nested_tools(history: History) -> list[str]:
    """Tool names called from sandbox code, recorded on the execute_python results."""
    return [
        call["tool"]
        for result in code_results(history)
        for call in (result.get("_meta") or {}).get(PTC_CALLS_META, [])
    ]


def used_code(history: History) -> str | None:
    return None if code_results(history) else "execute_python was never called"


def image_in_code_result(history: History) -> str | None:
    if any(block["type"] == "image" for r in code_results(history) for block in r["content"]):
        return None
    return "no execute_python result carried an image back to the model"


def one_selected_image(history: History) -> str | None:
    images = sum(
        block["type"] == "image" for result in code_results(history) for block in result["content"]
    )
    if nested_tools(history).count("media__get_swatch") < 2:
        return "expected the script to fetch several swatches"
    return None if images == 1 else f"expected exactly 1 image returned to the model, saw {images}"


def image_via_resource(history: History) -> str | None:
    if "get_resource" not in nested_tools(history) + [n for _, n, _ in tool_calls(history)]:
        return "get_resource was not used"
    return image_in_code_result(history)


def staged_attachment(history: History) -> str | None:
    if "attach_media" not in nested_tools(history) + [n for _, n, _ in tool_calls(history)]:
        return "attach_media was not used"
    if any(TOOL_MEDIA_CHANNEL in (message.get("channels") or {}) for message in history):
        return None
    return "no staged media message reached the model"


def parallel_subagents(history: History) -> str | None:
    count = nested_tools(history).count("subagent")
    return None if count >= 2 else f"expected >= 2 subagent calls from code, saw {count}"


def direct_slash_command(history: History) -> str | None:
    if any(name == "slash_command" for _, name, _ in tool_calls(history)):
        return None
    return "slash_command was not called directly (it should not be inside execute_python)"


CASES = (
    Case(
        "orders",
        "Which EMEA customer has the highest total revenue from delivered orders? "
        "Give the name and the runner-up.",
        ("stark", "aperture"),
        used_code,
    ),
    Case(
        "tool-image",
        "Fetch the swatch named swatch-north with the media server's get_swatch tool "
        "and tell me its colour in one word.",
        ("green",),
        image_in_code_result,
    ),
    Case(
        "tool-image-select",
        "List the media swatches, fetch all of them, and show yourself only the one named "
        "swatch-south. What colour is it? One word.",
        ("yellow",),
        one_selected_image,
    ),
    Case(
        "resource-image",
        "Read the MCP resource swatch://west from the media server with get_resource and "
        "tell me its colour in one word.",
        ("orange",),
        image_via_resource,
    ),
    Case(
        "attach-media",
        "Attach the local file assets/swatch-attach.png with attach_media and tell me its "
        "colour in one word.",
        ("blue",),
        staged_attachment,
    ),
    Case(
        "subagents",
        "From a single script, run two subagents concurrently: one finds the capital of "
        "France, the other the capital of Japan. Report both capitals.",
        ("paris", "tokyo"),
        parallel_subagents,
    ),
    Case(
        "slash-direct",
        "Use the slash_command tool to run /status and tell me which MCP servers are attached.",
        ("media", "orders"),
        direct_slash_command,
    ),
)


def command(case: Case, results: Path | None, model: str | None) -> list[str]:
    args = ["uv", "run", "fast-agent", "go", "--config-path", "fast-agent.yaml", "--ptc"]
    args += ["--card", "agent-cards/ptc-eval.md"]
    if model:
        args += ["--model", model]
    if results is not None:
        args += ["--quiet", "--results", str(results)]
    return [*args, "--message", case.prompt]


def clean_env() -> dict[str, str]:
    # A surrounding fast-agent session (or shell profile) must not redirect config discovery.
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"FAST_AGENT_HOME", "FAST_AGENT_RUNTIME_HOME", "ENVIRONMENT_DIR"}
    }


async def run_case(case: Case, model: str | None, workdir: Path) -> tuple[bool, str, float]:
    results = workdir / f"{case.name}.json"
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command(case, results, model),
        cwd=HERE,
        env=clean_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    elapsed = time.perf_counter() - started
    text = output.decode(errors="replace").strip()
    if process.returncode != 0 or not results.exists():
        return False, f"run failed (exit {process.returncode}): {text[-800:]}", elapsed

    history: History = json.loads(results.read_text(encoding="utf-8"))["messages"]
    answer = next(
        (
            " ".join(block.get("text", "") for block in message["content"])
            for message in reversed(history)
            if message["role"] == "assistant"
        ),
        "",
    )
    missing = [word for word in case.expect if word not in answer.lower()]
    if missing:
        return False, f"answer lacks {missing}: {answer.strip()[:300]!r}", elapsed
    reason = case.check(history)
    if reason:
        return False, f"{reason}; answer: {answer.strip()[:120]!r}", elapsed
    return True, answer.strip().replace("\n", " ")[:120], elapsed


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cases", nargs="*", help="Case names (default: all)")
    parser.add_argument("--model", help="Model override, e.g. sonnet or passthrough")
    parser.add_argument("--jobs", type=int, default=3, help="Cases to run concurrently")
    parser.add_argument("--list", action="store_true", help="Print commands without running")
    options = parser.parse_args()

    unknown = set(options.cases) - {case.name for case in CASES}
    if unknown:
        parser.error(f"unknown case(s): {', '.join(sorted(unknown))}")
    selected = [case for case in CASES if not options.cases or case.name in options.cases]

    if options.list:
        for case in selected:
            print(f"# {case.name}\n{shlex.join(command(case, None, options.model))}\n")
        return 0

    semaphore = asyncio.Semaphore(max(1, options.jobs))
    with tempfile.TemporaryDirectory(prefix="ptc-evals-") as temp:

        async def bounded(case: Case) -> tuple[bool, str, float]:
            async with semaphore:
                return await run_case(case, options.model, Path(temp))

        outcomes = await asyncio.gather(*(bounded(case) for case in selected))

    failed = 0
    for case, (passed, detail, elapsed) in zip(selected, outcomes, strict=True):
        failed += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {case.name:<18} {elapsed:6.1f}s  {detail}")
        if not passed:
            print(f"      reproduce: {shlex.join(command(case, None, options.model))}")
    print(f"\n{len(selected) - failed}/{len(selected)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
