"""Opt-in, billable HF GLM-5.3 / Flash smoke; only synthetic inputs.

uv run scripts/hf_glm53_smoke.py --live
FAST_AGENT_LLM_TRACE=1 uv run scripts/hf_glm53_smoke.py --live --backend together
uv run scripts/hf_glm53_smoke.py --case capabilities

Uses HF_TOKEN or the HF login cache, not home/workspace config, tools or skills.
Each case has a 120s deadline, at most three inference iterations, and a token cap.
Traces and library diagnostics stay in a temporary workspace and are deleted.
Only sanitized evidence is retained; authentication failure cancels outstanding cases.
Passing reasoning cases proves acceptance, not the backend's effective effort level.
Native video/PDF support is deliberately not inferred from upstream vendor support.
"""

import argparse
import asyncio
import base64
import json
import re
import struct
import tempfile
import time
import uuid
import zlib
from contextlib import chdir, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from mcp.types import ImageContent
from openai import APIStatusError
from pydantic import BaseModel, ConfigDict

from fast_agent import FastAgent
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.interfaces import AgentProtocol
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import LlmStopReason

BACKENDS = ("together", "novita", "zai-org")
MODELS = ("GLM-5.3", "GLM-5.3-Flash")
CASES = ("capabilities", "low", "high", "max", "default", "image", "tools", "structured")


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: Literal[5]


class AuthenticationStopped(Exception):
    """Cancel sibling requests without retaining a provider error message."""


@dataclass
class Result:
    model: str
    backend: str
    case: str
    status: str = "failed"
    stop_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    seconds: float = 0
    usage_present: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = None
    reasoning_channel: bool = False
    tool_executions: int = 0
    inference_turns: int = 0
    checks: dict[str, bool] = field(default_factory=dict)


def synthetic_image() -> ImageContent:
    """A solid red 64x64 PNG; no external image or file access."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
    png += chunk(b"IEND", b"")
    return ImageContent(type="image", mime_type="image/png", data=base64.b64encode(png).decode())


def error_status(text: str) -> int | None:
    # Streaming errors can arrive as response messages rather than exceptions.
    match = re.search(r"(?:status=|Error code: |HTTP/1\.[01] )(\d{3})", text)
    return int(match[1]) if match else None


def auth_failed(status: int | None, text: str = "") -> bool:
    return status in (401, 403) or any(
        word in text.casefold()
        for word in (
            "unauthorized",
            "authentication",
            "invalid api key",
            "invalid token",
            "forbidden",
        )
    )


def offline_capabilities(model: str, backend: str) -> Result:
    route = f"hf.zai-org/{model}:{backend}"
    # Negative assertions only: do not send unsupported attachments to the text model.
    unsupported = ["video/mp4", "application/pdf", "audio/wav"]
    if model == "GLM-5.3":
        unsupported.append("image/png")
    checks = {
        mime: not ModelDatabase.supports_mime(route, mime, provider=Provider.HUGGINGFACE)
        for mime in unsupported
    }
    return Result(
        model,
        backend,
        "capabilities",
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
    )


async def exercise(
    agent: AgentProtocol, result: Result, inputs: list[str], marker: str, max_tokens: int
) -> None:
    params = RequestParams(max_tokens=max_tokens, max_iterations=3)
    case = result.case
    prompt = Prompt.user("What is 2 + 3? Reply with only the digit.")
    if case == "image":
        prompt = Prompt.user("Name the solid color. One word.", synthetic_image())
    elif case == "tools":
        prompt = Prompt.user(
            "Call lookup_marker exactly once for label demo. Return its result verbatim."
        )

    # Check actual tool exposure, not just decorator configuration.
    tools = (await agent.list_tools()).tools
    expected_tools = {"lookup_marker"} if case == "tools" else set()
    result.checks["isolated_tools"] = {tool.name for tool in tools} == expected_tools
    if not result.checks["isolated_tools"]:
        return

    async with asyncio.timeout(120):
        if case == "structured":
            parsed, response = await agent.structured(
                "What is 2 + 3? Return JSON with the integer field answer.",
                Answer,
                request_params=params,
            )
            result.checks["validated_schema"] = isinstance(parsed, Answer) and parsed.answer == 5
        else:
            response = await agent.generate(prompt, request_params=params)

    text = response.all_text().strip()
    result.stop_reason = response.stop_reason
    result.reasoning_channel = bool((response.channels or {}).get("reasoning"))
    usage = agent.usage_accumulator
    turns = usage.turns if usage else []
    result.inference_turns = len(turns)
    result.usage_present = bool(turns) and all(
        turn.prompt.total is not None
        and turn.completion.total is not None
        and turn.raw_usage is not None
        for turn in turns
    )
    result.prompt_tokens = sum(turn.prompt.total or 0 for turn in turns)
    result.completion_tokens = sum(turn.completion.total or 0 for turn in turns)
    reasoning = [turn.completion.reasoning for turn in turns]
    if reasoning and all(count is not None for count in reasoning):
        result.reasoning_tokens = sum(count for count in reasoning if count is not None)
    result.tool_executions = len(inputs)
    if response.stop_reason == LlmStopReason.ERROR:
        result.http_status = error_status(text)
        result.error_type = "ProviderStreamError"
        if auth_failed(result.http_status, text):
            raise AuthenticationStopped
        return
    if case == "tools":
        result.checks["executed"] = inputs == ["demo"]
        result.checks["continuation"] = len(turns) >= 2 and text == marker
    elif case != "structured":
        result.checks["answer"] = text.casefold().strip(".!") == ("red" if case == "image" else "5")
    result.checks["usage"] = result.usage_present
    result.checks["end_turn"] = response.stop_reason == LlmStopReason.END_TURN
    result.status = "passed" if all(result.checks.values()) else "failed"


def report(path: Path, results: list[Result], state: str, max_tokens: int) -> None:
    lines = [
        "# HF GLM-5.3 live smoke",
        f"Updated: {datetime.now(UTC).isoformat()}; state: {state}.",
        f"Per inference: {max_tokens} output tokens; per case: 120s, at most 3 iterations.",
        "Synthetic inputs only; no raw responses, credentials or traces retained.",
        "Reasoning success means request acceptance, not proof of effective effort.",
        "Capabilities are offline metadata checks. Text-model images are skipped.",
        "Native video/PDF are not live-tested or assumed HF-supported.",
        "",
        "| Model | Backend | Case | Status | Usage | Turns | Seconds |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row.model} | {row.backend} | {row.case} | {row.status} | "
            f"{row.usage_present} | {row.inference_turns} | {row.seconds} |"
        )
    lines += [
        "",
        "## Sanitized evidence",
        "```json",
        json.dumps([asdict(row) for row in results], indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def smoke(args: argparse.Namespace) -> int:
    results: list[Result] = []
    state = "running"
    report(args.report, results, state, args.max_tokens)
    gate = asyncio.Semaphore(args.concurrency)

    async def entrypoint() -> None:
        pass

    # chdir also contains FAST_AGENT_LLM_TRACE output; never write raw traces in the repo.
    with tempfile.TemporaryDirectory(prefix="hf-glm53-smoke-") as directory, chdir(directory):
        config = Path(directory) / "fast-agent.yaml"
        config.write_text(
            "logger:\n  type: none\n  progress_display: false\n"
            "  show_chat: false\n  show_tools: false\n",
            encoding="utf-8",
        )
        fast = FastAgent(
            "HF GLM-5.3 synthetic smoke",
            config_path=str(config),
            parse_cli_args=False,
            quiet=True,
            no_home=True,
            workspace=directory,
            skills_directory=[],
        )
        jobs: list[tuple[str, Result, list[str], str]] = []

        def register(model: str, backend: str, case: str) -> None:
            name = f"smoke_{len(jobs)}"
            inputs: list[str] = []
            marker = f"GLM-SMOKE-{uuid.uuid4().hex}"

            def lookup_marker(label: str) -> str:
                """Look up the synthetic marker for a label."""
                inputs.append(label)
                return marker

            query = f"?reasoning={case}" if case in ("low", "high", "max") else ""
            fast.agent(
                name=name,
                model=f"hf.zai-org/{model}:{backend}{query}",
                instruction="Answer concisely. Use lookup_marker if asked to look up a marker.",
                servers=[],
                skills=[],
                shell=False,
                harness_tools=False,
                function_tools=[lookup_marker] if case == "tools" else [],
            )(entrypoint)
            jobs.append((name, Result(model, backend, case), inputs, marker))

        for model in args.model or MODELS:
            for backend in args.backend or BACKENDS:
                for case in args.case or CASES:
                    if case == "capabilities":
                        results.append(offline_capabilities(model, backend))
                    elif case == "image" and model == "GLM-5.3":
                        results.append(Result(model, backend, case, status="skipped_text_only"))
                    else:
                        register(model, backend, case)
        report(args.report, results, state, args.max_tokens)
        if jobs:
            async with fast.run() as app:

                async def run(name: str, row: Result, inputs: list[str], marker: str) -> None:
                    async with gate:
                        started = time.monotonic()
                        try:
                            await exercise(app[name], row, inputs, marker, args.max_tokens)
                        except AuthenticationStopped:
                            row.status = "auth_failure"
                            raise
                        except Exception as error:
                            # Never serialize exception messages or provider response bodies.
                            row.error_type = type(error).__name__
                            row.http_status = (
                                error.status_code if isinstance(error, APIStatusError) else None
                            )
                            if isinstance(error, ProviderKeyError) or auth_failed(
                                row.http_status, str(error)
                            ):
                                row.status = "auth_failure"
                                raise AuthenticationStopped from None
                        except asyncio.CancelledError:
                            row.status = "cancelled"
                            raise
                        finally:
                            row.tool_executions = len(inputs)
                            row.seconds = round(time.monotonic() - started, 2)
                            results.append(row)
                            report(args.report, results, state, args.max_tokens)

                try:
                    async with asyncio.TaskGroup() as group:
                        for job in jobs:
                            group.create_task(run(*job))
                except* AuthenticationStopped:
                    state = "stopped_auth"
        if state == "running":
            state = "complete"
        report(args.report, results, state, args.max_tokens)
    return int(state == "stopped_auth" or any(row.status == "failed" for row in results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Authorize billable synthetic inference"
    )
    parser.add_argument("--backend", action="append", choices=BACKENDS)
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--max-tokens", type=int, choices=(256, 512, 1024, 2048), default=1024)
    parser.add_argument("--report", type=Path, default=Path("/tmp/glm53-live-results.md"))
    args = parser.parse_args()
    if not args.live and args.case != ["capabilities"]:
        parser.error(
            "Pass --live to authorize inference (or --case capabilities for offline checks)"
        )
    args.report = args.report.resolve()
    # Libraries can log provider errors themselves; keep these diagnostics private and ephemeral.
    with (
        tempfile.TemporaryFile(mode="w+") as diagnostics,
        redirect_stdout(diagnostics),
        redirect_stderr(diagnostics),
    ):
        try:
            code = asyncio.run(smoke(args))
        except Exception as error:
            with args.report.open("a", encoding="utf-8") as output:
                output.write(
                    f"\n## Stopped during setup/cleanup\nError type: {type(error).__name__}\n"
                )
            code = 1
    print(f"Sanitized evidence: {args.report}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
