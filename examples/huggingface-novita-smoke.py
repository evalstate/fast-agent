"""Opt-in live HF Novita V4.1 smoke via FastAgent (billable synthetic inference).

Run: HF_TOKEN=... uv run examples/huggingface-novita-smoke.py --live
Uses only HF_TOKEN or the HF login cache, never workspace config, skills, or tools.
Each request is capped at 2048 output tokens; authentication failure stops the run.
"""

import argparse
import asyncio
import base64
import json
import re
import tempfile
from pathlib import Path

from mcp.types import ImageContent
from openai import APIStatusError

from fast_agent import FastAgent
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import LlmStopReason, PromptMessageExtended

MODEL = "hf.deepseek-ai/DeepSeek-V4.1-Flash:novita"


def synthetic_image() -> ImageContent:
    """Generate a solid red PNG without external files or imaging dependencies."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack("!2I5B", 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 64) * 64))
    png += chunk(b"IEND", b"")
    return ImageContent(type="image", mime_type="image/png", data=base64.b64encode(png).decode())


async def smoke() -> int:
    async def entrypoint() -> None:
        pass

    tool_inputs: list[str] = []

    def lookup_marker(label: str) -> str:
        """Look up the synthetic marker for the supplied label."""
        tool_inputs.append(label)
        return "NOVITA-SMOKE-729"

    with tempfile.TemporaryDirectory(prefix="hf-novita-smoke-") as directory:
        config = Path(directory) / "fast-agent.yaml"
        config.write_text(
            "logger:\n  type: none\n  progress_display: false\n"
            "  show_chat: false\n  show_tools: false\n",
            encoding="utf-8",
        )
        fast = FastAgent(
            "HF Novita synthetic smoke",
            config_path=str(config),
            parse_cli_args=False,
            quiet=True,
            no_home=True,
            workspace=directory,
            skills_directory=[],
        )
        for name, query in (("default", ""), ("max", "?reasoning=max"), ("off", "?reasoning=off")):
            fast.agent(name=name, model=MODEL + query, instruction="Answer concisely.")(entrypoint)
        fast.agent(
            name="tools",
            model=MODEL,
            instruction="Use lookup_marker when asked for a marker. Return its result verbatim.",
            function_tools=[lookup_marker],
        )(entrypoint)
        fast.agent(name="image", model=MODEL, instruction="Describe only the image.")(entrypoint)

        cases: list[tuple[str, PromptMessageExtended, str]] = [
            (name, Prompt.user("What is 2 + 3? Reply with only the digit."), "5")
            for name in ("default", "max", "off")
        ]
        cases += [
            ("tools", Prompt.user("Look up the marker for label demo."), "NOVITA-SMOKE-729"),
            ("image", Prompt.user("Name the solid color. One word.", synthetic_image()), "red"),
        ]
        failed = False
        async with fast.run() as app:
            for name, prompt, expected in cases:
                try:
                    async with asyncio.timeout(180):
                        response = await app[name].generate(
                            prompt,
                            request_params=RequestParams(max_tokens=2048, max_iterations=3),
                        )
                    text = response.all_text().strip()
                    if response.stop_reason == LlmStopReason.ERROR:
                        # Streaming failures are returned as messages, not necessarily raised.
                        status_match = re.search(r"\(status=(\d{3})\)", text)
                        print(
                            json.dumps(
                                {
                                    "case": name,
                                    "passed": False,
                                    "stop_reason": response.stop_reason,
                                    "http_status": int(status_match[1]) if status_match else None,
                                }
                            )
                        )
                        print("Stopped: provider error; remaining cases not attempted.")
                        return 1
                    passed = (
                        response.stop_reason == LlmStopReason.END_TURN
                        and text.casefold() == expected.casefold()
                    )
                    if name == "tools":
                        passed = passed and tool_inputs == ["demo"]
                    failed |= not passed
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "passed": passed,
                                "text": text,
                                "stop_reason": response.stop_reason,
                                "channels": list(response.channels or {}),
                                "tool_executions": len(tool_inputs) if name == "tools" else 0,
                            }
                        )
                    )
                except (ProviderKeyError, APIStatusError) as error:
                    status = error.status_code if isinstance(error, APIStatusError) else None
                    print(
                        json.dumps(
                            {
                                "case": name,
                                "passed": False,
                                "http_status": status,
                                "error_type": type(error).__name__,
                            }
                        )
                    )
                    print("Stopped: provider error; remaining cases not attempted.")
                    return 1
                except Exception as error:
                    # Provider exception messages may include request details: never print them.
                    print(
                        json.dumps(
                            {"case": name, "passed": False, "error_type": type(error).__name__}
                        )
                    )
                    failed = True
        return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Authorize billable synthetic inference"
    )
    if not parser.parse_args().live:
        parser.error("Pass --live to authorize live inference")
    raise SystemExit(asyncio.run(smoke()))
