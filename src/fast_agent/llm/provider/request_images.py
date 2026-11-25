"""Fit inline wire-request images into a provider's request budget.

Long screenshot-heavy sessions can accumulate tens of megabytes of base64 image
payloads well before hitting a model's token context limit, because multimodal
models bill images at ~1,000-1,600 tokens regardless of PNG byte size.

``RequestImageFitter`` operates only on outbound wire payloads (Chat Completions
``messages``, Anthropic/Copilot ``messages`` including nested ``tool_result``
blocks, and OpenAI/Copilot ``responses`` ``input`` items). Canonical agent
history is never mutated.

Fitting is deterministic per image digest so prefix-based prompt caches see
identical image bytes on consecutive turns. When a low-water mark is configured,
request-budget overflow omits oldest images in a single batch down to the
low-water mark and remembers that evicted prefix while conversation history
continues monotonically, avoiding per-turn cache invalidation.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from PIL import Image, ImageOps

IMAGE_OMITTED_PLACEHOLDER: Final = (
    "[Earlier image omitted from this request to stay within the provider's "
    "request size limit. Refer to earlier assistant notes about it or re-attach "
    "the image if you need to inspect it again.]"
)

_JPEG_QUALITY: Final = 85
_URL_WIRE_ALLOWANCE: Final = 512
_CACHE_MAX_ENTRIES: Final = 256
_KEEP_FORMATS: Final[dict[str, str]] = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}

WireImageFormat = Literal["chat_completions", "messages", "responses"]


@dataclass(frozen=True, slots=True)
class ImageBudget:
    """Route limits. ``None`` disables that step."""

    max_dimension: int | None
    reencode_min_bytes: int | None
    max_request_bytes: int | None
    low_water_request_bytes: int | None = None
    max_images: int | None = None
    max_image_bytes: int | None = None
    url_image_slots: int = 0
    max_url_images: int | None = None


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
class _CacheKey:
    digest: str
    mime: str
    max_dimension: int | None
    reencode_min_bytes: int | None
    max_image_bytes: int | None


@dataclass(frozen=True, slots=True)
class _NormalizedImage:
    mime: str
    b64: str
    resized: bool
    reencoded: bool
    dimensions: str | None


@dataclass(frozen=True, slots=True)
class _ImagePath:
    item_index: int
    slot_key: str
    part_index: int
    nested_key: str | None = None
    nested_index: int | None = None


@dataclass(slots=True)
class _ImageSlot:
    path: _ImagePath
    block: dict[str, Any]
    digest: str
    mime: str | None
    b64: str | None
    wire_bytes: int = 0
    orig_index: int = 0

    @property
    def is_url(self) -> bool:
        return self.mime is None or self.b64 is None


class RequestImageFitter:
    """Deterministic image normalizer and batch-evicting budget fitter."""

    def __init__(self, wire_format: WireImageFormat = "chat_completions") -> None:
        self._wire_format: WireImageFormat = wire_format
        self._cache: OrderedDict[_CacheKey, _NormalizedImage] = OrderedDict()
        self._evicted_prefix: tuple[str, ...] = ()
        self._last_input_digests: tuple[str, ...] = ()
        self._last_kept_orig_indices: tuple[int, ...] = ()

    def reset_watermark(self) -> None:
        """Clear batch-eviction history (for example after ``/clear``)."""
        self._evicted_prefix = ()
        self._last_input_digests = ()
        self._last_kept_orig_indices = ()

    def record_extra_evictions(self, extra_omitted: int) -> None:
        """Persist additional leading omissions (e.g. from a 413 resend) against canonical digests."""
        if extra_omitted <= 0 or not self._last_input_digests:
            return
        if extra_omitted >= len(self._last_kept_orig_indices):
            new_prefix_end = len(self._last_input_digests)
        else:
            new_prefix_end = self._last_kept_orig_indices[extra_omitted]
        self._evicted_prefix = self._last_input_digests[:new_prefix_end]
        self._last_kept_orig_indices = self._last_kept_orig_indices[extra_omitted:]

    def fit(
        self,
        arguments: dict[str, Any],
        budget: ImageBudget,
        *,
        update_watermark: bool = True,
    ) -> tuple[dict[str, Any], ImageFitReport]:
        report = ImageFitReport()
        list_key = "input" if self._wire_format == "responses" else "messages"
        items = arguments.get(list_key)
        if not isinstance(items, list):
            return arguments, report

        slots = _scan_images(items, self._wire_format)
        if not slots:
            return arguments, report

        report.images = len(slots)
        report.request_bytes_before = request_bytes(arguments)

        new_items = list(items)
        fitted = {**arguments, list_key: new_items}

        all_digests = tuple(slot.digest for slot in slots)
        pre_evicted = 0
        if update_watermark and self._evicted_prefix:
            if (
                self._last_input_digests
                and len(all_digests) >= len(self._last_input_digests)
                and all_digests[: len(self._last_input_digests)] == self._last_input_digests
            ):
                pre_evicted = len(self._evicted_prefix)
            else:
                self._evicted_prefix = ()

        for slot in slots[:pre_evicted]:
            _replace_at(new_items, slot.path, _build_placeholder(self._wire_format, slot.block))
            report.omitted += 1

        surviving = slots[pre_evicted:]

        for slot in surviving:
            if slot.mime is not None and slot.b64 is not None:
                norm = self._normalize(slot.digest, slot.mime, slot.b64, budget)
                if norm.b64 != slot.b64 or norm.mime != slot.mime:
                    updated_block = _build_inline_block(
                        self._wire_format, slot.block, norm.mime, norm.b64
                    )
                    _replace_at(new_items, slot.path, updated_block)
                    slot.block = updated_block
                    slot.mime = norm.mime
                    slot.b64 = norm.b64
                    if norm.resized:
                        report.resized += 1
                    if norm.reencoded:
                        report.reencoded += 1
                    if norm.dimensions:
                        report.dimensions.append(norm.dimensions)
            slot.wire_bytes = request_bytes(slot.block)

        count_omitted = 0
        if budget.max_images is not None and len(surviving) > budget.max_images:
            count_omitted = len(surviving) - budget.max_images
            for slot in surviving[:count_omitted]:
                _replace_at(new_items, slot.path, _build_placeholder(self._wire_format, slot.block))
                report.omitted += 1
            surviving = surviving[count_omitted:]

        if budget.max_url_images is not None:
            url_slots = [slot for slot in surviving if slot.is_url]
            if len(url_slots) > budget.max_url_images:
                excess_urls = len(url_slots) - budget.max_url_images
                dropped_urls = {id(slot) for slot in url_slots[:excess_urls]}
                for slot in url_slots[:excess_urls]:
                    _replace_at(
                        new_items, slot.path, _build_placeholder(self._wire_format, slot.block)
                    )
                    report.omitted += 1
                surviving = [slot for slot in surviving if id(slot) not in dropped_urls]

        raw_size = request_bytes(fitted)
        limit = budget.max_request_bytes
        budget_omitted = 0
        if limit is not None and surviving:
            placeholder_bytes = request_bytes(_build_placeholder(self._wire_format, {}))

            def projected_wire_bytes(omit_count: int) -> int:
                kept = surviving[omit_count:]
                omitted_delta = sum(
                    max(0, s.wire_bytes - placeholder_bytes) for s in surviving[:omit_count]
                )
                projected = raw_size - omitted_delta
                if budget.url_image_slots > 0 and kept:
                    existing_urls = sum(1 for s in kept if s.is_url)
                    free_url_slots = max(0, budget.url_image_slots - existing_urls)
                    if free_url_slots > 0:
                        inline_kept = [s for s in kept if not s.is_url]
                        upload_savings = sum(
                            max(0, s.wire_bytes - _URL_WIRE_ALLOWANCE)
                            for s in inline_kept[:free_url_slots]
                        )
                        projected -= upload_savings
                return projected

            if projected_wire_bytes(0) > limit:
                target = (
                    min(limit, budget.low_water_request_bytes)
                    if budget.low_water_request_bytes is not None
                    else limit
                )
                max_keep_newest = len(surviving) - 1
                while (
                    budget_omitted < max_keep_newest
                    and projected_wire_bytes(budget_omitted) > target
                ):
                    budget_omitted += 1
                if projected_wire_bytes(budget_omitted) > limit:
                    budget_omitted = len(surviving)

                for slot in surviving[:budget_omitted]:
                    _replace_at(
                        new_items, slot.path, _build_placeholder(self._wire_format, slot.block)
                    )
                    report.omitted += 1

        final_surviving = surviving[budget_omitted:]
        prefix_evicted = final_surviving[0].orig_index if final_surviving else len(slots)
        if update_watermark:
            if budget.low_water_request_bytes is not None and (
                count_omitted > 0 or budget_omitted > 0
            ):
                self._evicted_prefix = all_digests[:prefix_evicted]
            self._last_input_digests = all_digests
            self._last_kept_orig_indices = tuple(slot.orig_index for slot in final_surviving)

        report.request_bytes_after = request_bytes(fitted) if report.changed else raw_size
        return (fitted if report.changed else arguments), report

    def _normalize(
        self,
        digest: str,
        mime: str,
        b64: str,
        budget: ImageBudget,
    ) -> _NormalizedImage:
        if (
            budget.max_dimension is None
            and budget.reencode_min_bytes is None
            and budget.max_image_bytes is None
        ):
            return _NormalizedImage(mime, b64, False, False, None)
        key = _CacheKey(
            digest=digest,
            mime=mime,
            max_dimension=budget.max_dimension,
            reencode_min_bytes=budget.reencode_min_bytes,
            max_image_bytes=budget.max_image_bytes,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        result = _normalize_image_data(mime, b64, budget)
        self._cache[key] = result
        if len(self._cache) > _CACHE_MAX_ENTRIES:
            self._cache.popitem(last=False)
        return result


class ChatRequestImageFitter(RequestImageFitter):
    """Fit Chat Completions ``messages[*].content[*]`` ``image_url`` blocks."""

    def __init__(self) -> None:
        super().__init__("chat_completions")


class MessagesRequestImageFitter(RequestImageFitter):
    """Fit Anthropic/Copilot Messages ``image`` blocks (including ``tool_result``)."""

    def __init__(self) -> None:
        super().__init__("messages")


class ResponsesRequestImageFitter(RequestImageFitter):
    """Fit OpenAI/Copilot Responses ``input_image`` blocks."""

    def __init__(self) -> None:
        super().__init__("responses")


def has_responses_images(items: list[Any]) -> bool:
    """Return whether ``items`` carries any ``input_image`` payload."""
    return bool(_scan_images(items, "responses"))


def _index_slots(slots: list[_ImageSlot]) -> list[_ImageSlot]:
    for idx, slot in enumerate(slots):
        slot.orig_index = idx
    return slots


def _scan_images(items: list[Any], wire_format: WireImageFormat) -> list[_ImageSlot]:
    slots: list[_ImageSlot] = []
    if wire_format == "chat_completions":
        for item_idx, msg in enumerate(items):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else None
                if not isinstance(url, str):
                    continue
                if parsed := _parse_data_url(url):
                    mime, b64 = parsed
                    slots.append(
                        _ImageSlot(
                            path=_ImagePath(item_idx, "content", part_idx),
                            block=part,
                            digest=_sha256_str(b64),
                            mime=mime,
                            b64=b64,
                        )
                    )
                else:
                    slots.append(
                        _ImageSlot(
                            path=_ImagePath(item_idx, "content", part_idx),
                            block=part,
                            digest=_sha256_str(url),
                            mime=None,
                            b64=None,
                        )
                    )
        return _index_slots(slots)

    if wire_format == "messages":
        for item_idx, msg in enumerate(items):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part_idx, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                if slot := _messages_image_slot(part, _ImagePath(item_idx, "content", part_idx)):
                    slots.append(slot)
                elif part.get("type") == "tool_result":
                    for nested_key in ("content", "output"):
                        nested = part.get(nested_key)
                        if not isinstance(nested, list):
                            continue
                        for nested_idx, inner in enumerate(nested):
                            if not isinstance(inner, dict):
                                continue
                            if slot := _messages_image_slot(
                                inner,
                                _ImagePath(item_idx, "content", part_idx, nested_key, nested_idx),
                            ):
                                slots.append(slot)
        return _index_slots(slots)

    for item_idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        for slot_key in ("content", "output"):
            parts = item.get(slot_key)
            if not isinstance(parts, list):
                continue
            for part_idx, part in enumerate(parts):
                if not isinstance(part, dict) or part.get("type") != "input_image":
                    continue
                url = part.get("image_url")
                if not isinstance(url, str):
                    continue
                if parsed := _parse_data_url(url):
                    mime, b64 = parsed
                    slots.append(
                        _ImageSlot(
                            path=_ImagePath(item_idx, slot_key, part_idx),
                            block=part,
                            digest=_sha256_str(b64),
                            mime=mime,
                            b64=b64,
                        )
                    )
                else:
                    slots.append(
                        _ImageSlot(
                            path=_ImagePath(item_idx, slot_key, part_idx),
                            block=part,
                            digest=_sha256_str(url),
                            mime=None,
                            b64=None,
                        )
                    )
    return _index_slots(slots)


def _messages_image_slot(part: dict[str, Any], path: _ImagePath) -> _ImageSlot | None:
    if part.get("type") != "image":
        return None
    source = part.get("source")
    if not isinstance(source, dict):
        return None
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if isinstance(media_type, str) and isinstance(data, str):
            return _ImageSlot(
                path=path,
                block=part,
                digest=_sha256_str(data),
                mime=media_type,
                b64=data,
            )
    elif source_type == "url":
        url = source.get("url")
        if isinstance(url, str):
            return _ImageSlot(
                path=path,
                block=part,
                digest=_sha256_str(url),
                mime=None,
                b64=None,
            )
    return None


def _replace_at(items: list[Any], path: _ImagePath, new_block: dict[str, Any]) -> None:
    item = dict(items[path.item_index])
    outer = list(item[path.slot_key])
    if path.nested_key is None or path.nested_index is None:
        outer[path.part_index] = new_block
    else:
        parent = dict(outer[path.part_index])
        inner = list(parent[path.nested_key])
        inner[path.nested_index] = new_block
        parent[path.nested_key] = inner
        outer[path.part_index] = parent
    item[path.slot_key] = outer
    items[path.item_index] = item


def _build_inline_block(
    wire_format: WireImageFormat,
    original_block: dict[str, Any],
    mime: str,
    b64: str,
) -> dict[str, Any]:
    if wire_format == "messages":
        source = original_block.get("source")
        base_source = dict(source) if isinstance(source, dict) else {"type": "base64"}
        return {
            **original_block,
            "source": {**base_source, "type": "base64", "media_type": mime, "data": b64},
        }
    data_url = f"data:{mime};base64,{b64}"
    if wire_format == "responses":
        return {**original_block, "image_url": data_url}
    image_url = original_block.get("image_url")
    base_url_obj = dict(image_url) if isinstance(image_url, dict) else {}
    return {**original_block, "image_url": {**base_url_obj, "url": data_url}}


def _build_placeholder(
    wire_format: WireImageFormat,
    original_block: dict[str, Any],
) -> dict[str, Any]:
    if wire_format == "responses":
        return {"type": "input_text", "text": IMAGE_OMITTED_PLACEHOLDER}
    placeholder: dict[str, Any] = {"type": "text", "text": IMAGE_OMITTED_PLACEHOLDER}
    if wire_format == "messages" and "cache_control" in original_block:
        placeholder["cache_control"] = original_block["cache_control"]
    return placeholder


def _sha256_str(value: str) -> str:
    return hashlib.sha256(value.encode("ascii", errors="ignore")).hexdigest()


def _normalize_image_data(mime: str, b64: str, budget: ImageBudget) -> _NormalizedImage:
    unchanged = _NormalizedImage(
        mime=mime, b64=b64, resized=False, reencoded=False, dimensions=None
    )
    try:
        raw = base64.b64decode(b64, validate=True)
        with Image.open(io.BytesIO(raw)) as opened:
            if _is_animated(opened):
                return unchanged
            image_format = opened.format or "PNG"
            too_large = budget.max_dimension is not None and max(opened.size) > budget.max_dimension
            over_cap = budget.max_image_bytes is not None and len(raw) > budget.max_image_bytes
            heavy = budget.reencode_min_bytes is not None and len(raw) > budget.reencode_min_bytes
            if not too_large and not over_cap and not heavy:
                return unchanged

            image = ImageOps.exif_transpose(opened)
            original = image.size
            resized = False
            if budget.max_dimension is not None and max(image.size) > budget.max_dimension:
                image.thumbnail(
                    (budget.max_dimension, budget.max_dimension), Image.Resampling.LANCZOS
                )
                resized = True
            candidates = _encode_candidates(image, image_format)
            out_mime, encoded = min(candidates, key=lambda candidate: len(candidate[1]))
            while (
                budget.max_image_bytes is not None
                and len(encoded) > budget.max_image_bytes
                and max(image.size) > 64
            ):
                next_side = max(64, int(max(image.size) * 0.75))
                image.thumbnail((next_side, next_side), Image.Resampling.LANCZOS)
                resized = True
                candidates = _encode_candidates(image, image_format)
                out_mime, encoded = min(candidates, key=lambda candidate: len(candidate[1]))
    except (OSError, SyntaxError, ValueError, binascii.Error, Image.DecompressionBombError):
        return unchanged

    if not resized and len(encoded) >= len(raw):
        return unchanged

    note = f"{original[0]}x{original[1]}->{image.width}x{image.height}" if resized else None
    return _NormalizedImage(
        mime=out_mime,
        b64=base64.b64encode(encoded).decode("ascii"),
        resized=resized,
        reencoded=not resized or out_mime != Image.MIME.get(image_format, mime),
        dimensions=note,
    )


def _encode_candidates(image: Image.Image, image_format: str) -> list[tuple[str, bytes]]:
    candidates: list[tuple[str, bytes]] = []
    keep_mime = _KEEP_FORMATS.get(image_format)
    if keep_mime is not None:
        output = io.BytesIO()
        image.save(output, format=image_format, optimize=True)
        candidates.append((keep_mime, output.getvalue()))
    if not _has_transparency(image):
        output = io.BytesIO()
        image.convert("RGB").save(output, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        candidates.append(("image/jpeg", output.getvalue()))
    if not candidates:
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        candidates.append(("image/png", output.getvalue()))
    return candidates


def _is_animated(image: Image.Image) -> bool:
    try:
        image.seek(1)
    except EOFError:
        return False
    finally:
        image.seek(0)
    return True


def _has_transparency(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA", "PA"):
        alpha = image.getchannel("A").getextrema()
        return isinstance(alpha, tuple) and isinstance(alpha[0], (int, float)) and alpha[0] != 255
    return "transparency" in image.info


def _parse_data_url(url: str) -> tuple[str, str] | None:
    if not url.startswith("data:"):
        return None
    header, sep, b64 = url.partition(",")
    if not sep or not header.endswith(";base64"):
        return None
    mime = header.removeprefix("data:").split(";", 1)[0].strip().lower() or "image/png"
    if not mime.startswith("image/"):
        return None
    return mime, b64


def request_bytes(arguments: Any) -> int:
    """Compact UTF-8 JSON byte length, matching what httpx serializes."""
    return len(json.dumps(arguments, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
