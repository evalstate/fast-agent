"""Request-wide DeepSeek vision limits, applied without mutating conversation history."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from fast_agent.core.logging.logger import Logger


def prepare_deepseek_images(items: list[dict[str, Any]], logger: Logger) -> list[dict[str, Any]]:
    # These limits also apply to Responses; count the entire outgoing request.
    # https://api-docs.deepseek.com/guides/vision
    images = [
        part
        for item in items
        for field in ("content", "output")
        if isinstance(parts := item.get(field), list)
        for part in parts
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    max_dimension = 4096 if len(images) >= 15 else 8192
    replacements: dict[int, dict[str, Any]] = {}
    unchecked = 0
    for index, part in enumerate(images, 1):
        url = part.get("image_url")
        if part.get("file_id") or not isinstance(url, str) or not url.startswith("data:"):
            unchecked += 1
            continue
        header, separator, payload = url.partition(",")
        try:
            if not separator or not header.endswith(";base64"):
                raise ValueError("expected a base64 image data URL")
            data = base64.b64decode(payload, validate=True)
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                if max(width, height) <= max_dimension:
                    continue  # Preserve MIME, encoded bytes and pixels exactly.
                if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                    raise ValueError("resize locally to PNG, JPEG, WebP or GIF before attaching")
                try:
                    image.seek(1)
                except EOFError:
                    image.seek(0)
                else:
                    raise ValueError("resize animated images locally before attaching")
                image_format = image.format
                mime = Image.MIME[image_format]
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                output = BytesIO()
                image.save(output, format=image_format)
                resized = base64.b64encode(output.getvalue()).decode("ascii")
                replacements[id(part)] = {**part, "image_url": f"data:{mime};base64,{resized}"}
                logger.info(
                    "Resized DeepSeek inline image",
                    image_index=index,
                    image_count=len(images),
                    original_size=f"{width}x{height}",
                    resized_size=f"{image.width}x{image.height}",
                    max_dimension=max_dimension,
                )
        except (OSError, SyntaxError, ValueError, binascii.Error) as exc:
            # Never include the URL or payload in diagnostics.
            raise ValueError(
                f"DeepSeek inline image {index} could not be prepared; provide a valid static "
                f"PNG, JPEG, WebP or GIF no larger than {max_dimension}px per side."
            ) from exc
    if unchecked:
        logger.warning(
            "DeepSeek image dimensions unchecked for remote URLs/file IDs; "
            "ensure each side is within the request limit or attach inline images.",
            unchecked_images=unchecked,
            image_count=len(images),
            max_dimension=max_dimension,
        )
    if not replacements:
        return items
    result = []
    for item in items:
        copied = dict(item)
        for field in ("content", "output"):
            parts = item.get(field)
            if isinstance(parts, list):
                copied[field] = [replacements.get(id(part), part) for part in parts]
        result.append(copied)
    return result
