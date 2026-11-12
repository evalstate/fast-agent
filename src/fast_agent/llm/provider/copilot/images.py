"""Request-local Copilot attachment uploads; canonical histories stay inline."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
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

UploadProgress = Callable[[int, int], None]


class CopilotImageUploads:
    """Normalize SDK dictionaries without inspecting tool arguments or other metadata.

    Cache entries are credential-scoped, bounded, and expire after thirty minutes.
    No client or broker lifecycle is required of callers.
    """

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[bytes, str, bytes], tuple[float, str]] = OrderedDict()

    # Any is confined to the external SDK's heterogeneous dict/list payload boundary.
    async def normalize(
        self,
        payload: dict[str, Any],
        endpoint: CopilotEndpoint,
        on_progress: UploadProgress | None = None,
    ) -> dict[str, Any]:
        """Return a copy with inline images uploaded; report ``(n, total)`` before each POST."""
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
        planned: list[tuple[dict[str, Any], bytes, str]] = []
        uploads_enabled = True
        for block in blocks:
            if block.get("type") == "input_image":
                # Copilot rejects an omitted detail even though OpenAI accepts it.
                block.setdefault("detail", "auto")
            try:
                prepared = self._inline_image(
                    block, endpoint, resized, uploads_enabled and len(planned) < remaining
                )
            except (ValueError, OSError):
                uploads_enabled = False
                logger.warning("Copilot image upload failed; retaining inline image.")
                continue
            if prepared is not None and uploads_enabled and len(planned) < remaining:
                planned.append((block, *prepared))
        if planned:
            await self._upload_planned(planned, endpoint, on_progress)
        return result

    async def _upload_planned(
        self,
        planned: list[tuple[dict[str, Any], bytes, str]],
        endpoint: CopilotEndpoint,
        on_progress: UploadProgress | None,
    ) -> None:
        now = time.monotonic()
        for expired in [key for key, (until, _) in self._cache.items() if until <= now]:
            del self._cache[expired]
        keys = [self._cache_key(data, mime, endpoint) for _, data, mime in planned]
        # Repeated history images upload once; later copies hit the cache.
        total = len({key for key in keys if key not in self._cache})
        uploaded = 0
        deadline = time.monotonic() + _UPLOAD_TIMEOUT_SECONDS
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            for (block, data, mime), key in zip(planned, keys, strict=True):
                cached = self._cache.get(key)
                if cached:
                    self._cache.move_to_end(key)
                    self._apply_url(block, cached[1])
                    continue
                uploaded += 1
                if on_progress is not None:
                    on_progress(uploaded, total)
                try:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise TimeoutError("Copilot image upload deadline expired")
                    async with asyncio.timeout(timeout):
                        url = await self._upload(data, mime, endpoint, client)
                except (httpx.HTTPError, TimeoutError, ValueError, OSError):
                    # Exceptions can include credentials or signed URLs; never log their text.
                    logger.warning("Copilot image upload failed; retaining inline image.")
                    return
                self._cache[key] = (time.monotonic() + 1800, url)
                if len(self._cache) > 128:
                    self._cache.popitem(last=False)
                self._apply_url(block, url)

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

    def _inline_image(
        self,
        block: dict[str, Any],
        endpoint: CopilotEndpoint,
        resized: dict[tuple[str, str], tuple[bytes, str]],
        needs_data: bool,
    ) -> tuple[bytes, str] | None:
        """Validate an inline image, resize it for Messages, and return upload bytes."""
        source = block.get("source")
        if block.get("type") == "image":
            if not isinstance(source, dict) or source.get("type") != "base64":
                return None
            mime, encoded = source.get("media_type"), source.get("data")
        else:
            inline = block.get("image_url")
            if not isinstance(inline, str) or not inline.startswith("data:"):
                return None
            header, encoded = inline[5:].split(",", 1)
            mime, encoding = header.split(";", 1)
            if encoding != "base64":
                raise ValueError("Unsupported encoding")
        if not isinstance(mime, str) or not mime.startswith("image/"):
            raise ValueError("Invalid image media type")
        if not isinstance(encoded, str):
            raise ValueError("Invalid image data")
        if endpoint.wire_api != "messages":
            return (base64.b64decode(encoded, validate=True), mime) if needs_data else None
        key = (mime, encoded)
        cached = resized.get(key)
        if cached is None:
            cached = self._resize(base64.b64decode(encoded, validate=True), mime)
            resized[key] = cached
        data, mime = cached
        # Update the copy before attempting upload so failures also use safe dimensions.
        encoded = base64.b64encode(data).decode("ascii")
        if isinstance(source, dict):
            source.update(data=encoded, media_type=mime)
        else:
            block["image_url"] = f"data:{mime};base64,{encoded}"
        return data, mime

    @staticmethod
    def _apply_url(block: dict[str, Any], url: str) -> None:
        if block.get("type") == "image":
            replacement = dict(block["source"])
            replacement.pop("data", None)
            replacement.pop("media_type", None)
            replacement.update(type="url", url=url)
            block["source"] = replacement
        else:
            block["image_url"] = url

    @staticmethod
    def _cache_key(data: bytes, mime: str, endpoint: CopilotEndpoint) -> tuple[bytes, str, bytes]:
        credential = "\n".join(
            f"{key.lower()}:{value}"
            for key, value in sorted(endpoint.headers.items())
            if key.lower() == "authorization"
        )
        return hashlib.sha256(credential.encode()).digest(), mime, hashlib.sha256(data).digest()

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
        return url
