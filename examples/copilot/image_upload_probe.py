"""Live, billable Copilot image smoke; requires normal Copilot authentication.

Run from the repo: uv run examples/copilot/image_upload_probe.py --transport both
Claude always uses SSE; --transport selects GPT transports. No session replay.
Only synthetic data is sent. Provider output, URLs and exception text are suppressed.
"""

import argparse
import asyncio
import base64
import logging
import os
import secrets
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from typing import Any, Literal, TextIO

from mcp.types import ImageContent
from PIL import Image, ImageDraw, ImageFont

from fast_agent.config import LoggerSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.images import CopilotImageUploads
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM
from fast_agent.llm.provider.copilot.responses import CopilotResponsesLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import LlmStopReason, RequestParams


class ObservedUploads(CopilotImageUploads):
    """Observe real normalization, including cache reuse; never expose URLs."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[tuple[str, ...]] = []

    # Any is restricted to the adapter's heterogeneous SDK payload boundary.
    async def normalize(self, payload: dict[str, Any], endpoint: CopilotEndpoint) -> dict[str, Any]:
        result = await super().normalize(payload, endpoint)
        urls: list[str] = []
        for item in result.get("messages", result.get("input", [])):
            blocks: list[dict[str, Any]] = []
            self._content(item.get("content", []), blocks)
            for block in blocks:
                if self._is_url(block):
                    url = block["source"]["url"] if block["type"] == "image" else block["image_url"]
                    urls.append(url)
        self.urls.append(tuple(urls))
        return result


def synthetic_png(code: str) -> ImageContent:
    image = Image.new("RGB", (800, 240), "white")
    ImageDraw.Draw(image).text((40, 65), code, fill="black", font=ImageFont.load_default(size=80))
    output = BytesIO()
    image.save(output, format="PNG")
    return ImageContent(
        type="image", mimeType="image/png", data=base64.b64encode(output.getvalue()).decode()
    )


async def probe(model: str, transport: Literal["sse", "websocket"]) -> None:
    context = Context(
        config=Settings(
            logger=LoggerSettings(
                type="none", progress_display=False, show_chat=False, show_tools=False
            )
        )
    )
    llm = (
        CopilotMessagesLLM(context=context, model=model, transport=transport)
        if model.startswith("claude")
        else CopilotResponsesLLM(context=context, model=model, transport=transport)
    )
    uploads = ObservedUploads()
    llm._image_uploads = uploads
    code = "".join(secrets.choice("23456789") for _ in range(6))
    original = Prompt.user(
        "Read the six-digit code in the image. Reply only with the code.", synthetic_png(code)
    )
    snapshot = original.model_dump()
    params = RequestParams(max_tokens=4096)
    try:
        async with asyncio.timeout(240):
            first = await llm.generate([original], request_params=params)
            if first.stop_reason == LlmStopReason.ERROR or first.all_text().strip() != code:
                raise AssertionError("first turn failed")
            first_urls = uploads.urls[-1]
            if len(first_urls) != 1:
                raise AssertionError("image was not uploaded")
            count = len(uploads.urls)
            second = await llm.generate(
                [
                    original,
                    first,
                    Prompt.user("Read the original image again. Reply only with the code."),
                ],
                request_params=params,
            )
            if second.stop_reason == LlmStopReason.ERROR or second.all_text().strip() != code:
                raise AssertionError("second turn failed")
            if len(uploads.urls) <= count or any(
                urls != first_urls for urls in uploads.urls[count:]
            ):
                raise AssertionError("history URL reuse failed")
            if original.model_dump() != snapshot:
                raise AssertionError("canonical history changed")
    finally:
        if isinstance(llm, CopilotResponsesLLM):
            await llm.close()


async def main(sink: TextIO) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transport",
        choices=("sse", "websocket", "both"),
        default="both",
        help="GPT transport(s); Claude always uses SSE (default: both)",
    )
    args = parser.parse_args()
    cases: list[tuple[str, Literal["sse", "websocket"]]] = [("claude-opus-5.5", "sse")]
    for transport in ("sse", "websocket"):
        if args.transport in (transport, "both"):
            cases.append(("gpt-6-astra", transport))
    logging.disable(logging.CRITICAL)
    failed = False
    for model, transport in cases:
        try:
            with redirect_stdout(sink), redirect_stderr(sink):
                await probe(model, transport)
            print(f"{model} {transport}: PASS (code read; history URL reused)")
        except Exception as error:
            failed = True
            print(f"{model} {transport}: FAIL ({type(error).__name__})")
    return int(failed)


if __name__ == "__main__":
    with open(os.devnull, "w") as sink:
        raise SystemExit(asyncio.run(main(sink)))
