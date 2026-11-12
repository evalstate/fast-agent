"""Live, billable Sonnet 5.5 progress demo using the normal harness/tool loop.

Run: uv run examples/copilot/sonnet_progress.py --mode all
Requires Copilot login. Sends only synthetic data; never prints credentials or signatures.
"""

import argparse
import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from fast_agent import FastAgent
from fast_agent.llm.stream_types import StreamChunk
from fast_agent.types import LlmStopReason, RequestParams

type DisplayMode = Literal["default", "summarized", "omitted", "between_tools"]
MODES: tuple[DisplayMode, ...] = ("default", "summarized", "omitted", "between_tools")


@dataclass(frozen=True)
class Observation:
    seconds: float
    completed_tools: int
    kind: str
    text: str


async def demonstrate(mode: DisplayMode) -> None:
    model = "copilot.sonnet55"
    if mode == "between_tools":
        model += "?reasoning=off"
    fast = FastAgent(f"Sonnet progress: {mode}", ignore_unknown_args=True)
    observations: list[Observation] = []
    calls: list[str] = []
    start = monotonic()

    @fast.tool
    def read_station(station: Literal["inventory", "sample", "verification"]) -> str:
        """Read a synthetic station report. Follow the next_station in each result."""
        calls.append(station)
        observations.append(Observation(monotonic() - start, len(calls), "tool", station))
        match station:
            case "inventory":
                return "Batch LANTERN has 12 sealed containers. next_station: sample."
            case "sample":
                return "Sample temperature is 18 C, within the 15-20 C target. next_station: verification."
            case "verification":
                return "All 12 seals passed inspection. Final status: READY."

    def observe(chunk: StreamChunk) -> None:
        if chunk.event == "delta" and chunk.text:
            observations.append(
                Observation(
                    monotonic() - start,
                    len(calls),
                    "thinking" if chunk.is_reasoning else "text",
                    chunk.text,
                )
            )

    @fast.agent(
        "probe",
        model=model,
        instruction=(
            "Run the requested inspection using the provided tool. Before each tool call, "
            "give a public-facing progress update of 80-100 words in four sentences: what "
            "was observed, which station you will check next, and what that check establishes. "
            "Describe actions and observable results, not private reasoning. "
            "Make sequential tool calls, following next_station. Do not skip stations."
        ),
    )
    async def run() -> None:
        async with fast.run() as agent:
            remove = agent.probe.add_stream_listener(observe)
            params = RequestParams(max_tokens=4096, max_iterations=6)
            if mode in {"summarized", "omitted"}:
                params.metadata = {"thinking": {"type": "adaptive", "display": mode}}
            try:
                response = await agent.probe.generate(
                    "Inspect batch LANTERN, starting at inventory. Report the final status.",
                    request_params=params,
                )
            finally:
                remove()
            assert response.stop_reason == LlmStopReason.END_TURN, response.stop_reason
            assert calls == ["inventory", "sample", "verification"], calls
            assert "READY" in response.all_text().upper()

    print(f"\n=== {mode} ===", flush=True)
    await run()
    print(f"\n=== {mode}: observed stream timeline ===")
    # Group deltas by tool boundary and channel. Never display opaque signed blocks.
    for completed in range(4):
        for kind in ("thinking", "text"):
            chunks = [
                item
                for item in observations
                if item.completed_tools == completed and item.kind == kind
            ]
            if chunks:
                text = "".join(item.text for item in chunks)
                print(
                    f"+{chunks[0].seconds:.2f}s after {completed} tools: "
                    f"{kind}, {len(text)} chars: {text[:300]!r}"
                )
        if completed < len(calls):
            tool = next(
                item
                for item in observations
                if item.kind == "tool" and item.completed_tools == completed + 1
            )
            print(f"+{tool.seconds:.2f}s tool: {tool.text}")
    between = [
        item for item in observations if item.kind == "thinking" and 0 < item.completed_tools < 3
    ]
    print(f"Between-tool thinking-summary characters: {sum(len(item.text) for item in between)}")
    visible = [
        item
        for item in observations
        if item.kind in {"thinking", "text"} and 0 < item.completed_tools < 3
    ]
    print(f"All between-tool visible characters: {sum(len(item.text) for item in visible)}")
    if mode != "omitted":
        assert visible, f"No between-tool progress observed for {mode}"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    args = parser.parse_args()
    for mode in MODES:
        if args.mode in ("all", mode):
            await demonstrate(mode)


if __name__ == "__main__":
    asyncio.run(main())
