"""Fit inline images in an outgoing Chat Completions request to a route's size limits.

Only the request is rewritten; conversation history is never mutated. Every image is
normalized deterministically (bounded longest side; large payloads re-encoded to the
smallest of their own format and JPEG), so successive requests keep a byte-stable
prefix for prompt caching. If the serialized request still exceeds the byte budget,
the oldest images are replaced with a short text placeholder until it fits.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from PIL import Image, ImageOps

IMAGE_OMITTED_PLACEHOLDER = (
    "[image omitted from this request: including it would exceed the provider's request "
    "size limit. Attach it again (or a crop/smaller version) if you still need to see it.]"
)
_JPEG_QUALITY = 85
_CACHE_ENTRIES = 256
_KEEP_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


@dataclass(frozen=True, slots=True)
class ImageBudget:
    """Route limits. ``None`` disables that step."""

    max_dimension: int | None
    reencode_min_bytes: int | None
    max_request_bytes: int | None


@dataclass(slots=True)
class ImageFitReport:
    images: int = 0
    resized: int = 0
    reencoded: int = 0
    omitted: int = 0
    request_bytes_before: int = 0
    request_bytes_after: int = 0
    dimensions: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.resized or self.reencoded or self.omitted)


@dataclass(frozen=True, slots=True)
class _Normalized:
    url: str
    resized: bool
    reencoded: bool
    dimension_note: str | None


def request_bytes(arguments: dict[str, Any]) -> int:
    """Size of the JSON body the SDK sends (UTF-8, compact)."""
    return len(
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    )


class ChatRequestImageFitter:
    """Stateful only for its normalization cache (images repeat on every turn)."""

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, int | None, int | None], _Normalized] = OrderedDict()

    def fit(
        self, arguments: dict[str, Any], budget: ImageBudget
    ) -> tuple[dict[str, Any], ImageFitReport]:
        report = ImageFitReport()
        messages = arguments.get("messages")
        if not isinstance(messages, list):
            return arguments, report

        locations: list[tuple[int, int]] = []
        new_messages: list[Any] = list(messages)
        for message_index, message in enumerate(messages):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            new_content: list[Any] | None = None
            for part_index, part in enumerate(content):
                url = _data_url(part)
                if url is None:
                    continue
                report.images += 1
                locations.append((message_index, part_index))
                normalized = self._normalize(url, budget)
                if normalized.url == url:
                    continue
                report.resized += normalized.resized
                report.reencoded += normalized.reencoded
                if normalized.dimension_note:
                    report.dimensions.append(normalized.dimension_note)
                if new_content is None:
                    new_content = list(content)
                new_content[part_index] = {
                    **part,
                    "image_url": {**part["image_url"], "url": normalized.url},
                }
            if new_content is not None:
                new_messages[message_index] = {**message, "content": new_content}

        if not report.images:
            return arguments, report

        fitted = {**arguments, "messages": new_messages}
        report.request_bytes_before = request_bytes(arguments)
        size = request_bytes(fitted)
        limit = budget.max_request_bytes
        if limit is not None and size > limit:
            # Oldest first: the newest image is usually the one the model just asked for.
            for message_index, part_index in locations:
                if size <= limit:
                    break
                message = new_messages[message_index]
                content = list(message["content"])
                removed = content[part_index]
                placeholder = {"type": "text", "text": IMAGE_OMITTED_PLACEHOLDER}
                size -= request_bytes(removed) - request_bytes(placeholder)
                content[part_index] = placeholder
                new_messages[message_index] = {**message, "content": content}
                report.omitted += 1
            size = request_bytes(fitted)
        report.request_bytes_after = size
        return (fitted if report.changed else arguments), report

    def _normalize(self, url: str, budget: ImageBudget) -> _Normalized:
        if budget.max_dimension is None and budget.reencode_min_bytes is None:
            return _Normalized(url, False, False, None)
        key = (
            hashlib.sha256(url.encode()).hexdigest(),
            budget.max_dimension,
            budget.reencode_min_bytes,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        normalized = _normalize_data_url(url, budget)
        self._cache[key] = normalized
        if len(self._cache) > _CACHE_ENTRIES:
            self._cache.popitem(last=False)
        return normalized


def _data_url(part: object) -> str | None:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    image_url = part.get("image_url")
    url = image_url.get("url") if isinstance(image_url, dict) else None
    if isinstance(url, str) and url.startswith("data:"):
        return url
    return None


def _normalize_data_url(url: str, budget: ImageBudget) -> _Normalized:
    unchanged = _Normalized(url, False, False, None)
    header, separator, payload = url.partition(",")
    if not separator or not header.endswith(";base64"):
        return unchanged
    try:
        data = base64.b64decode(payload, validate=True)
        with Image.open(BytesIO(data)) as opened:
            if _is_animated(opened):
                return unchanged  # leave animations to the provider
            image_format = opened.format or "PNG"
            original = opened.size
            too_large = budget.max_dimension is not None and max(original) > budget.max_dimension
            heavy = budget.reencode_min_bytes is not None and len(data) > budget.reencode_min_bytes
            if not too_large and not heavy:
                return unchanged
            image = ImageOps.exif_transpose(opened)
            if too_large and budget.max_dimension is not None:
                image.thumbnail(
                    (budget.max_dimension, budget.max_dimension), Image.Resampling.LANCZOS
                )
            candidates = _encode_candidates(image, image_format)
    except (OSError, SyntaxError, ValueError, binascii.Error, Image.DecompressionBombError):
        return unchanged  # not decodable here; the provider decides, the budget still applies
    mime, encoded = min(candidates, key=lambda candidate: len(candidate[1]))
    if not too_large and len(encoded) >= len(data):
        return unchanged
    note = f"{original[0]}x{original[1]}->{image.width}x{image.height}" if too_large else None
    return _Normalized(
        f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}",
        resized=too_large,
        reencoded=not too_large or mime != Image.MIME.get(image_format),
        dimension_note=note,
    )


def _is_animated(image: Image.Image) -> bool:
    try:
        image.seek(1)
    except EOFError:
        return False
    finally:
        image.seek(0)
    return True


def _encode_candidates(image: Image.Image, image_format: str) -> list[tuple[str, bytes]]:
    candidates: list[tuple[str, bytes]] = []
    keep_mime = _KEEP_FORMATS.get(image_format)
    if keep_mime is not None:
        output = BytesIO()
        image.save(output, format=image_format)
        candidates.append((keep_mime, output.getvalue()))
    if not _has_transparency(image):
        output = BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        candidates.append(("image/jpeg", output.getvalue()))
    if not candidates:
        output = BytesIO()
        image.save(output, format="PNG")
        candidates.append(("image/png", output.getvalue()))
    return candidates


def _has_transparency(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA", "PA"):
        alpha = image.getchannel("A").getextrema()
        return isinstance(alpha, tuple) and alpha[0] != 255
    return "transparency" in image.info
