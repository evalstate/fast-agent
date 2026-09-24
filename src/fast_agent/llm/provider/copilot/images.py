"""Request-local Copilot attachment uploads; canonical histories stay inline."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from io import BytesIO
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from PIL import Image

if TYPE_CHECKING:
    from fast_agent.llm.provider.copilot.broker import CopilotEndpoint

logger = logging.getLogger(__name__)
_MAX_URL_IMAGES = 20
_UPLOAD_TIMEOUT_SECONDS = 60
_UPLOAD_URL = "https://uploads.github.com/copilot/chat/attachments"


class CopilotImageUploads:
    """Normalize SDK dictionaries without inspecting tool arguments or other metadata.

    Cache entries are credential-scoped, bounded, and expire after thirty minutes.
    No client or broker lifecycle is required of callers.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[bytes, str, bytes], tuple[float, str]] = OrderedDict()

    # Any is confined to the external SDK's heterogeneous dict/list payload boundary.
    async def normalize(self, payload: dict[str, Any], endpoint: CopilotEndpoint) -> dict[str, Any]:
        result = deepcopy(payload)
        blocks: list[dict[str, Any]] = []
        for field in ("messages", "input"):
            items = result.get(field)
            if isinstance(items, list):
                for index, item in enumerate(items):
                    if isinstance(item, dict):
                        items[index] = item = deepcopy(item)
                        for slot in ("content", "output"):
                            if slot in item:
                                item[slot] = deepcopy(item[slot])
                                self._content(item[slot], blocks)
        remaining = max(0, _MAX_URL_IMAGES - sum(self._is_url(block) for block in blocks))
        # Repeated history images need only one decode/resize per request, including
        # inline overflow. Keep image bytes request-local rather than in the URL cache.
        resized: dict[tuple[str, str], tuple[bytes, str]] = {}
        deadline = time.monotonic() + _UPLOAD_TIMEOUT_SECONDS
        uploads_enabled = True
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            for block in blocks:
                if block.get("type") == "input_image":
                    # Copilot rejects an omitted detail even though OpenAI accepts it.
                    block.setdefault("detail", "auto")
                was_url = self._is_url(block)
                try:
                    await self._image(
                        block,
                        endpoint,
                        client,
                        uploads_enabled and remaining > 0,
                        resized,
                        deadline,
                    )
                except (httpx.HTTPError, TimeoutError, ValueError, OSError):
                    uploads_enabled = False
                    # Exceptions can include credentials or signed URLs; never log their text.
                    logger.warning("Copilot image upload failed; retaining inline image.")
                if not was_url and self._is_url(block):
                    remaining -= 1
        return result

    def _content(self, value: Any, blocks: list[dict[str, Any]]) -> None:
        if isinstance(value, list):
            for index, block in enumerate(value):
                # Break shared SDK-object aliases across content and tool arguments.
                value[index] = deepcopy(block)
                self._content(value[index], blocks)
        elif isinstance(value, dict):
            if value.get("type") == "tool_result":
                for slot in ("content", "output"):
                    if slot in value:
                        value[slot] = deepcopy(value[slot])
                        self._content(value[slot], blocks)
            elif value.get("type") in ("image", "input_image"):
                if isinstance(value.get("source"), dict):
                    value["source"] = deepcopy(value["source"])
                blocks.append(value)

    @staticmethod
    def _is_url(block: dict[str, Any]) -> bool:
        if block.get("type") == "image":
            source = block.get("source")
            return isinstance(source, dict) and source.get("type") == "url"
        url = block.get("image_url")
        return isinstance(url, str) and not url.startswith("data:")

    async def _image(
        self,
        block: dict[str, Any],
        endpoint: CopilotEndpoint,
        client: httpx.AsyncClient,
        allow_upload: bool,
        resized: dict[tuple[str, str], tuple[bytes, str]],
        deadline: float,
    ) -> None:
        source = block.get("source")
        anthropic = block.get("type") == "image"
        if anthropic:
            if not isinstance(source, dict) or source.get("type") != "base64":
                return
            mime, encoded = source.get("media_type"), source.get("data")
        else:
            inline = block.get("image_url")
            if not isinstance(inline, str) or not inline.startswith("data:"):
                return
            header, encoded = inline[5:].split(",", 1)
            mime, encoding = header.split(";", 1)
            if encoding != "base64":
                raise ValueError("Unsupported encoding")
        if not isinstance(mime, str) or not mime.startswith("image/"):
            raise ValueError("Invalid image media type")
        if not isinstance(encoded, str):
            raise ValueError("Invalid image data")
        if endpoint.wire_api == "messages":
            key = (mime, encoded)
            cached = resized.get(key)
            if cached is None:
                cached = self._resize(base64.b64decode(encoded, validate=True), mime)
                resized[key] = cached
            data, mime = cached
            # Update the copy before attempting upload so failures also use safe dimensions.
            encoded = base64.b64encode(data).decode("ascii")
            if anthropic:
                assert isinstance(source, dict)
                source.update(data=encoded, media_type=mime)
            else:
                block["image_url"] = f"data:{mime};base64,{encoded}"
        else:
            if not allow_upload:
                return
            data = base64.b64decode(encoded, validate=True)
        if not allow_upload:
            return
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError("Copilot image upload deadline expired")
        async with asyncio.timeout(timeout):
            url = await self._upload(data, mime, endpoint, client)
        if anthropic:
            assert isinstance(source, dict)
            replacement = dict(source)
            replacement.pop("data", None)
            replacement.pop("media_type", None)
            replacement.update(type="url", url=url)
            block["source"] = replacement
        else:
            block["image_url"] = url

    @staticmethod
    def _resize(data: bytes, mime: str) -> tuple[bytes, str]:
        with Image.open(BytesIO(data)) as image:
            if max(image.size) <= 2000:
                return data, mime
            # Small animations pass through unchanged; resizing flattens to the
            # first frame. Preserve JPEG compression; use PNG for other formats.
            format = "JPEG" if image.format == "JPEG" else "PNG"
            image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format=format)
            return output.getvalue(), "image/jpeg" if format == "JPEG" else "image/png"

    async def _upload(
        self, data: bytes, mime: str, endpoint: CopilotEndpoint, client: httpx.AsyncClient
    ) -> str:
        credential = "\n".join(
            f"{key.lower()}:{value}"
            for key, value in sorted(endpoint.headers.items())
            if key.lower() == "authorization"
        )
        key = (hashlib.sha256(credential.encode()).digest(), mime, hashlib.sha256(data).digest())
        now = time.monotonic()
        for expired in [key for key, (until, _) in self._cache.items() if until <= now]:
            del self._cache[expired]
        cached = self._cache.get(key)
        if cached:
            self._cache.move_to_end(key)
            return cached[1]
        headers = {k: v for k, v in endpoint.headers.items() if k.lower() != "content-type"}
        headers["Content-Type"] = "application/octet-stream"
        response = await client.post(
            _UPLOAD_URL,
            params={"name": "image." + mime.split("/", 1)[1], "content_type": mime},
            content=data,
            headers=headers,
        )
        response.raise_for_status()
        body = response.json()
        url = body.get("url") if isinstance(body, dict) else None
        if not isinstance(url, str):
            raise ValueError("Missing attachment URL")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(character.isspace() or ord(character) < 32 for character in url)
        ):
            raise ValueError("Invalid attachment URL")
        self._cache[key] = (time.monotonic() + 1800, url)
        if len(self._cache) > 128:
            self._cache.popitem(last=False)
        return url
