from __future__ import annotations

import base64
import binascii
import hashlib
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from openai import AsyncOpenAI
    from openai.types.file_create_params import ExpiresAfter

XAI_IMAGE_UPLOAD_MIN_TTL_SECONDS = 3_600
XAI_IMAGE_UPLOAD_MAX_TTL_SECONDS = 2_592_000
XAI_IMAGE_UPLOAD_DEFAULT_TTL_SECONDS = 86_400
XAI_IMAGE_MAX_BYTES = 20 * 1024 * 1024
_XAI_IMAGE_MAX_BASE64_LENGTH = 4 * ((XAI_IMAGE_MAX_BYTES + 2) // 3)
_XAI_IMAGE_CACHE_EXPIRY_MARGIN_SECONDS = 60
_XAI_IMAGE_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
}


class _XAIPublicURLResponse(BaseModel):
    public_url: str


@dataclass(frozen=True, slots=True)
class _CachedPublicImage:
    url: str
    usable_until: float


class XAIImageUploadManager:
    """Upload inline Grok images once and reuse temporary xAI public URLs."""

    def __init__(
        self,
        ttl_seconds: int = XAI_IMAGE_UPLOAD_DEFAULT_TTL_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._cache: dict[str, _CachedPublicImage] = {}

    @staticmethod
    def _decode_supported_image(data_url: str) -> tuple[bytes, str, str] | None:
        if not data_url.startswith("data:"):
            return None

        header, separator, payload = data_url.partition(",")
        if not separator or ";base64" not in header or len(payload) > _XAI_IMAGE_MAX_BASE64_LENGTH:
            return None

        mime_type = header[5:].split(";", 1)[0].lower()
        if mime_type == "image/jpg":
            mime_type = "image/jpeg"
        extension = _XAI_IMAGE_TYPES.get(mime_type)
        if extension is None:
            return None

        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            return None
        if len(data) > XAI_IMAGE_MAX_BYTES:
            return None
        return data, mime_type, extension

    @staticmethod
    def _cache_key(data: bytes, mime_type: str) -> str:
        digest = hashlib.sha256()
        digest.update(mime_type.encode("ascii"))
        digest.update(b"\0")
        digest.update(data)
        return digest.hexdigest()

    async def public_url(self, client: AsyncOpenAI, data_url: str) -> str | None:
        image = self._decode_supported_image(data_url)
        if image is None:
            return None

        data, mime_type, extension = image
        cache_key = self._cache_key(data, mime_type)
        now = self._clock()
        self._cache = {
            key: cached for key, cached in self._cache.items() if cached.usable_until > now
        }
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.url

        expires_after: ExpiresAfter = {
            "anchor": "created_at",
            "seconds": self._ttl_seconds,
        }
        uploaded = await client.files.create(
            file=(f"image-{cache_key[:12]}.{extension}", data, mime_type),
            purpose="assistants",
            expires_after=expires_after,
        )
        public_url = await client.post(
            f"/files/{uploaded.id}/public-url",
            cast_to=_XAIPublicURLResponse,
            body={},
        )
        url = str(public_url.public_url)
        self._cache[cache_key] = _CachedPublicImage(
            url=url,
            usable_until=now + self._ttl_seconds - _XAI_IMAGE_CACHE_EXPIRY_MARGIN_SECONDS,
        )
        return url
