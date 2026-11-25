"""Built-in image generation and editing tools for OpenAI, Codex OAuth, xAI, and Google."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import parse_qs, urlparse

import httpx
from mcp_types import CallToolResult, ContentBlock, ImageContent, TextContent, Tool
from PIL import Image, UnidentifiedImageError

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.logging.progress_payloads import build_progress_payload
from fast_agent.event_progress import ProgressAction
from fast_agent.io.path_uri import local_file_url_path
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.tool_result_metadata import set_tool_result_media_preview
from fast_agent.paths import resolve_home_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.config import Settings
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler

logger = get_logger(__name__)


def get_codex_access_token() -> str | None:
    from fast_agent.llm.provider.openai.codex_oauth import (
        get_codex_access_token as _get_codex_access_token,
    )

    return _get_codex_access_token()


def get_xai_access_token() -> str | None:
    from fast_agent.llm.provider.openai.xai_oauth import (
        get_xai_access_token as _get_xai_access_token,
    )

    return _get_xai_access_token()


GENERATE_IMAGE_TOOL_NAME = "generate_image"
EDIT_IMAGE_TOOL_NAME = "edit_image"
IMAGE_GEN_TOOL_NAMES = frozenset({GENERATE_IMAGE_TOOL_NAME, EDIT_IMAGE_TOOL_NAME})
IMAGE_GEN_BUILTIN_OWNER = "image_gen"

DEFAULT_OPENAI_IMAGE_MODEL = "gpt-image-2.6-flare"
DEFAULT_XAI_IMAGE_MODEL = "grok-imagine-image-2.0"
DEFAULT_GOOGLE_IMAGE_MODEL = "gemini-nano-banana-2.1"
DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

MAX_IMAGE_GEN_COUNT = 4
MAX_XAI_EDIT_IMAGES = 5
MAX_GOOGLE_EDIT_IMAGES = 14
DEFAULT_PREVIEW_MAX_EDGE = 1568
DEFAULT_PREVIEW_MAX_BYTES = 2 * 1024 * 1024
INLINE_SVG_MAX_BYTES = 16 * 1024
HTTP_TIMEOUT_SECONDS = 180.0

type ImageGenBackend = Literal["openai", "codexresponses", "xai", "google"]
type ComplexityLevel = Literal["low", "medium", "high"]
type ResolutionLevel = Literal["1k", "2k", "4k"]

OPENAI_ASPECT_RATIO_TO_SIZE: dict[str, str] = {
    "auto": "auto",
    "1:1": "1024x1024",
    "3:2": "1536x1024",
    "2:3": "1024x1536",
}

XAI_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "16:9",
    "9:16",
    "4:3",
    "3:4",
    "3:2",
    "2:3",
    "2:1",
    "1:2",
    "19.5:9",
    "9:19.5",
    "20:9",
    "9:20",
    "21:9",
    "5:2",
)

GOOGLE_ASPECT_RATIOS: tuple[str, ...] = (
    "auto",
    "1:1",
    "1:4",
    "1:8",
    "2:3",
    "3:2",
    "3:4",
    "4:1",
    "4:3",
    "4:5",
    "5:4",
    "8:1",
    "9:16",
    "16:9",
    "21:9",
)

_EXT_TO_OPENAI_FORMAT: dict[str, str] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".webp": "webp",
    ".svg": "svg",
    ".gif": "gif",
}

_FORMAT_TO_MIME: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "svg": "image/svg+xml",
    "gif": "image/gif",
}

_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/svg+xml": ".svg",
    "image/gif": ".gif",
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mime_type": {"type": "string"},
                    "width": {"type": ["integer", "null"]},
                    "height": {"type": ["integer", "null"]},
                    "bytes": {"type": "integer"},
                },
                "required": ["path", "mime_type", "bytes"],
            },
        },
        "model": {"type": "string"},
        "revised_prompt": {"type": ["string", "null"]},
    },
    "required": ["images", "model"],
}


@dataclass(frozen=True, slots=True)
class ResolvedImageGenModel:
    """Resolved image generation backend, model ID, and optional backend defaults."""

    backend: ImageGenBackend
    model: str
    complexity: ComplexityLevel | None = None
    resolution: ResolutionLevel | None = None
    default_quality: str | None = None

    @property
    def spec(self) -> str:
        return f"{self.backend}.{self.model}"

    @property
    def is_openai_family(self) -> bool:
        return self.backend in {"openai", "codexresponses"}

    @property
    def supports_svg_gif(self) -> bool:
        return self.is_openai_family and self.model.startswith("gpt-image-2.6")

    @property
    def supports_complexity(self) -> bool:
        return self.supports_svg_gif

    @property
    def supports_quality(self) -> bool:
        return self.is_openai_family or self.model == "grok-imagine-image-2.0"

    @property
    def quality_values(self) -> tuple[str, ...]:
        if self.is_openai_family:
            return ("auto", "low", "medium", "high")
        if self.model == "grok-imagine-image-2.0":
            return ("auto", "low", "medium")
        return ()

    @property
    def aspect_ratio_values(self) -> tuple[str, ...]:
        if self.is_openai_family:
            return tuple(OPENAI_ASPECT_RATIO_TO_SIZE.keys())
        if self.backend == "google":
            return GOOGLE_ASPECT_RATIOS
        return XAI_ASPECT_RATIOS


@dataclass(frozen=True, slots=True)
class _GeneratedImageData:
    data: bytes
    mime_type: str
    revised_prompt: str | None = None


def _has_openai_api_key(settings: Settings | None) -> bool:
    config_dict = settings.model_dump() if settings is not None else {}
    try:
        return bool(
            ProviderKeyManager.get_optional_api_key(Provider.OPENAI.config_name, config_dict)
        )
    except Exception:
        return False


def _has_codex_credentials(settings: Settings | None) -> bool:
    config_dict = settings.model_dump() if settings is not None else {}
    try:
        return bool(
            ProviderKeyManager.get_optional_api_key(
                Provider.CODEX_RESPONSES.config_name, config_dict
            )
        )
    except Exception:
        return False


def _has_xai_credentials(settings: Settings | None) -> bool:
    config_dict = settings.model_dump() if settings is not None else {}
    try:
        return bool(ProviderKeyManager.get_optional_api_key(Provider.XAI.config_name, config_dict))
    except Exception:
        return False


def _has_google_api_key(settings: Settings | None) -> bool:
    config_dict = settings.model_dump() if settings is not None else {}
    try:
        return bool(
            ProviderKeyManager.get_optional_api_key(Provider.GOOGLE.config_name, config_dict)
        )
    except Exception:
        return False


def _resolve_openai_family_backend(
    settings: Settings | None,
    main_provider: Provider | None,
) -> ImageGenBackend:
    if main_provider == Provider.CODEX_RESPONSES:
        return "codexresponses"
    if main_provider in {Provider.OPENAI, Provider.RESPONSES}:
        return "openai"
    if _has_codex_credentials(settings):
        return "codexresponses"
    if _has_openai_api_key(settings):
        return "openai"
    raise ProviderKeyError(
        "Image generation requires OpenAI or Codex credentials.",
        "Sign in with `fast-agent auth provider login codex` or set OPENAI_API_KEY.",
    )


def _parse_spec_and_query(raw_spec: str) -> tuple[str, dict[str, str]]:
    text = raw_spec.strip()
    if "?" not in text:
        return text, {}
    base, query = text.split("?", 1)
    parsed = parse_qs(query, keep_blank_values=False)
    params = {k.strip().lower(): v[-1].strip().lower() for k, v in parsed.items() if v}
    return base.strip(), params


def _parse_complexity(value: str | None) -> ComplexityLevel | None:
    if value is None:
        return None
    if value in ("low", "medium", "high"):
        return value
    raise ValueError(f"Unsupported image complexity '{value}'. Expected low, medium, or high.")


def _parse_resolution(value: str | None) -> ResolutionLevel | None:
    if value is None:
        return None
    if value in ("1k", "2k", "4k"):
        return value
    raise ValueError(f"Unsupported image resolution '{value}'. Expected 1k, 2k, or 4k.")


def resolve_image_gen_model(
    spec: str | None = None,
    *,
    settings: Settings | None = None,
    main_provider: Provider | None = None,
    validate_credentials: bool = True,
) -> ResolvedImageGenModel:
    """Resolve an image generation model specification and validate backend credentials."""
    effective_spec = spec.strip() if isinstance(spec, str) and spec.strip() else None
    if effective_spec is None and settings is not None and settings.image_gen.model:
        effective_spec = settings.image_gen.model.strip() or None

    query_params: dict[str, str] = {}
    if effective_spec is None:
        if main_provider == Provider.CODEX_RESPONSES:
            backend: ImageGenBackend = "codexresponses"
            model = DEFAULT_OPENAI_IMAGE_MODEL
        elif main_provider in {Provider.OPENAI, Provider.RESPONSES}:
            backend = "openai"
            model = DEFAULT_OPENAI_IMAGE_MODEL
        elif main_provider == Provider.XAI:
            backend = "xai"
            model = DEFAULT_XAI_IMAGE_MODEL
        elif main_provider == Provider.GOOGLE:
            backend = "google"
            model = DEFAULT_GOOGLE_IMAGE_MODEL
        elif _has_codex_credentials(settings):
            backend = "codexresponses"
            model = DEFAULT_OPENAI_IMAGE_MODEL
        elif _has_openai_api_key(settings):
            backend = "openai"
            model = DEFAULT_OPENAI_IMAGE_MODEL
        elif _has_xai_credentials(settings):
            backend = "xai"
            model = DEFAULT_XAI_IMAGE_MODEL
        elif _has_google_api_key(settings):
            backend = "google"
            model = DEFAULT_GOOGLE_IMAGE_MODEL
        else:
            raise ProviderKeyError(
                "No image generation credentials found.",
                "Sign in with `fast-agent auth provider login codex` or "
                "`fast-agent auth provider login xai`, or set "
                "OPENAI_API_KEY / XAI_API_KEY / GOOGLE_API_KEY.",
            )
    else:
        base_spec, query_params = _parse_spec_and_query(effective_spec)
        lowered = base_spec.lower()
        if lowered in {"codex", "codexresponses"}:
            backend, model = "codexresponses", DEFAULT_OPENAI_IMAGE_MODEL
        elif lowered == "openai":
            backend, model = "openai", DEFAULT_OPENAI_IMAGE_MODEL
        elif lowered in {"xai", "grok"}:
            backend, model = "xai", DEFAULT_XAI_IMAGE_MODEL
        elif lowered in {"google", "gemini", "nano-banana", "nanobanana"}:
            backend, model = "google", DEFAULT_GOOGLE_IMAGE_MODEL
        elif "." in base_spec and base_spec.split(".", 1)[0].lower() in {
            "openai",
            "codex",
            "codexresponses",
            "xai",
            "grok",
            "google",
            "gemini",
        }:
            prefix, model_part = base_spec.split(".", 1)
            prefix_lower = prefix.lower()
            model = model_part.strip()
            if not model:
                raise ValueError(f"Missing model name in image generation spec '{effective_spec}'.")
            if prefix_lower in {"codex", "codexresponses"}:
                backend = "codexresponses"
            elif prefix_lower == "openai":
                backend = "openai"
            elif prefix_lower in {"google", "gemini"}:
                backend = "google"
            else:
                backend = "xai"
        elif lowered.startswith("grok-imagine"):
            backend, model = "xai", base_spec
        elif lowered.startswith("gemini-nano-banana") or (
            lowered.startswith("gemini-") and "-image" in lowered
        ):
            backend, model = "google", base_spec
        elif lowered.startswith(("gpt-image", "dall-e")):
            backend = _resolve_openai_family_backend(settings, main_provider)
            model = base_spec
        else:
            raise ValueError(
                f"Unrecognized image generation model '{effective_spec}'. "
                "Use e.g. `codexresponses.gpt-image-2.6-flare`, `openai.gpt-image-2.6-sunburst`, "
                "`xai.grok-imagine-image-2.0`, `google.gemini-nano-banana-2.1`, "
                "or a provider shorthand (`codex`, `openai`, `xai`, `google`, `nano-banana`)."
            )

    cfg = settings.image_gen if settings is not None else None
    complexity = _parse_complexity(
        query_params.get("complexity") or (cfg.complexity if cfg is not None else None)
    )
    resolution = _parse_resolution(
        query_params.get("resolution") or (cfg.resolution if cfg is not None else None)
    )
    default_quality = query_params.get("quality")

    if resolution == "4k" and backend == "xai":
        raise ValueError(
            "Unsupported image resolution '4k' for xAI Imagine models. Expected 1k or 2k."
        )

    resolved = ResolvedImageGenModel(
        backend=backend,
        model=model,
        complexity=complexity,
        resolution=resolution,
        default_quality=default_quality,
    )
    if default_quality is not None and resolved.quality_values:
        if default_quality not in resolved.quality_values:
            allowed = ", ".join(resolved.quality_values)
            raise ValueError(
                f"Unsupported default quality '{default_quality}' for {resolved.spec}. "
                f"Allowed values: {allowed}."
            )

    if validate_credentials:
        _resolve_backend_endpoint(resolved, settings)

    return resolved


def _resolve_backend_endpoint(
    resolved: ResolvedImageGenModel,
    settings: Settings | None,
) -> tuple[str, dict[str, str]]:
    """Return ``(base_url, headers)`` for the resolved backend, raising if unauthenticated."""
    config_dict = settings.model_dump() if settings is not None else {}

    if resolved.backend == "codexresponses":
        from fast_agent.llm.provider.openai.codex_oauth import parse_chatgpt_account_id
        from fast_agent.llm.provider.openai.codex_responses import CODEX_BASE_URL

        codex_key = ProviderKeyManager.get_optional_api_key(
            Provider.CODEX_RESPONSES.config_name, config_dict
        )
        token = codex_key or get_codex_access_token()
        if not token:
            raise ProviderKeyError(
                "Codex credentials are missing for image generation.",
                "Sign in with: fast-agent auth provider login codex",
            )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "originator": "codex_cli_rs",
        }
        account_id = parse_chatgpt_account_id(token)
        if account_id:
            headers["chatgpt-account-id"] = account_id
        base_url = (
            settings.codexresponses.base_url
            if settings is not None
            and settings.codexresponses is not None
            and settings.codexresponses.base_url
            else CODEX_BASE_URL
        )
        return base_url.rstrip("/"), headers

    if resolved.backend == "openai":
        api_key = ProviderKeyManager.get_optional_api_key(Provider.OPENAI.config_name, config_dict)
        if not api_key:
            raise ProviderKeyError(
                "OPENAI_API_KEY is required for openai.* image generation.",
                "Set OPENAI_API_KEY or use `codexresponses.gpt-image-2.6-flare` with Codex OAuth.",
            )
        base_url = (
            settings.openai.base_url
            if settings is not None and settings.openai is not None and settings.openai.base_url
            else os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        )
        return base_url.rstrip("/"), {"Authorization": f"Bearer {api_key}"}

    if resolved.backend == "google":
        api_key = ProviderKeyManager.get_optional_api_key(Provider.GOOGLE.config_name, config_dict)
        if not api_key:
            raise ProviderKeyError(
                "GOOGLE_API_KEY is required for google.* image generation.",
                "Set GOOGLE_API_KEY or configure google.api_key in fast-agent.secrets.yaml.",
            )
        base_url = (
            settings.google.base_url
            if settings is not None and settings.google is not None and settings.google.base_url
            else os.getenv("GOOGLE_BASE_URL", DEFAULT_GOOGLE_BASE_URL)
        )
        google_headers: dict[str, str] = {"x-goog-api-key": api_key}
        if settings is not None and settings.google is not None and settings.google.default_headers:
            google_headers = {**settings.google.default_headers, **google_headers}
        return base_url.rstrip("/"), google_headers

    # xAI: API key if configured, otherwise xAI OAuth token.
    xai_key = ProviderKeyManager.get_optional_api_key(Provider.XAI.config_name, config_dict)
    token = xai_key or get_xai_access_token()
    if not token:
        raise ProviderKeyError(
            "xAI credentials are missing for image generation.",
            "Sign in with `fast-agent auth provider login xai` or set XAI_API_KEY.",
        )
    base_url = (
        settings.xai.base_url
        if settings is not None and settings.xai is not None and settings.xai.base_url
        else os.getenv("XAI_BASE_URL", DEFAULT_XAI_BASE_URL)
    )
    return base_url.rstrip("/"), {"Authorization": f"Bearer {token}"}


def build_image_gen_tools(resolved: ResolvedImageGenModel) -> list[Tool]:
    """Build provider-tailored ``generate_image`` and ``edit_image`` tool definitions."""
    builtin_meta = {"fast_agent": {"builtin": IMAGE_GEN_BUILTIN_OWNER}}

    common_props: dict[str, Any] = {
        "aspect_ratio": {
            "type": "string",
            "enum": list(resolved.aspect_ratio_values),
            "default": "auto",
            "description": "Output aspect ratio (default: auto).",
        },
    }
    if resolved.supports_quality:
        common_props["quality"] = {
            "type": "string",
            "enum": list(resolved.quality_values),
            "default": resolved.default_quality or "auto",
            "description": "Rendering quality (default: auto).",
        }
    common_props["n"] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAX_IMAGE_GEN_COUNT,
        "default": 1,
        "description": f"Number of images to generate (1-{MAX_IMAGE_GEN_COUNT}, default: 1).",
    }
    if resolved.is_openai_family:
        common_props["transparent"] = {
            "type": "boolean",
            "default": False,
            "description": "Request a transparent background (incompatible with .jpg/.jpeg).",
        }

    if resolved.is_openai_family:
        ext_hint = (
            ".png, .jpg, .webp, .svg, .gif" if resolved.supports_svg_gif else ".png, .jpg, .webp"
        )
        output_path_desc = (
            f"Optional destination file path. Extension sets format ({ext_hint}; default .png). "
            "Overwrites if it exists. If omitted, auto-saves and returns the path."
        )
    else:
        output_path_desc = (
            "Optional destination file path. Provider chooses the image format (typically .jpg) "
            "and adjusts the saved extension to match if needed. Overwrites if it exists. "
            "If omitted, auto-saves and returns the path."
        )
    output_path_prop: dict[str, Any] = {
        "type": "string",
        "description": output_path_desc,
    }

    ptc_return_note = (
        " Results with image previews arrive wrapped as "
        '{"structured_content": {"images": [...], "model": ..., "revised_prompt": ...}, '
        '"content": [{"media_id": ...}]}; read r.get("structured_content", r)["images"] '
        'and return r.get("content") to view previews.'
    )
    ptc_returns_schema: dict[str, Any] = {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "structured_content": _OUTPUT_SCHEMA,
                    "content": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["structured_content", "content"],
            },
            _OUTPUT_SCHEMA,
        ]
    }
    gen_description = (
        f"Generate image(s) from a text prompt using {resolved.spec}. "
        "Saves full-resolution image(s) to disk and returns the saved path(s) and preview."
    )
    fast_agent_meta = builtin_meta.get("fast_agent")
    base_fast_agent_meta = dict(fast_agent_meta) if isinstance(fast_agent_meta, dict) else {}
    gen_meta: dict[str, Any] = {
        **builtin_meta,
        "fast_agent": {
            **base_fast_agent_meta,
            "ptc": {
                "description": gen_description + ptc_return_note,
                "returns": ptc_returns_schema,
            },
        },
    }

    gen_properties: dict[str, Any] = {
        "prompt": {
            "type": "string",
            "description": "Detailed description of the image to generate.",
        },
        **common_props,
        "output_path": output_path_prop,
    }

    generate_tool = Tool(
        name=GENERATE_IMAGE_TOOL_NAME,
        description=gen_description,
        input_schema={
            "type": "object",
            "properties": gen_properties,
            "required": ["prompt"],
        },
        output_schema=_OUTPUT_SCHEMA,
        meta=gen_meta,
    )

    images_prop: dict[str, Any] = {
        "type": "array",
        "items": {"type": "string"},
        "minItems": 1,
        "description": (
            "Source image path(s) (or URLs/data URIs), e.g. paths returned by generate_image."
        ),
    }
    if resolved.backend == "xai":
        images_prop["maxItems"] = MAX_XAI_EDIT_IMAGES
    elif resolved.backend == "google":
        images_prop["maxItems"] = MAX_GOOGLE_EDIT_IMAGES

    edit_properties: dict[str, Any] = {
        "images": images_prop,
        "prompt": {
            "type": "string",
            "description": "Instructions describing how to edit or combine the source image(s).",
        },
    }
    if resolved.is_openai_family:
        edit_properties["mask"] = {
            "type": "string",
            "description": (
                "Optional path to a PNG mask matching the first image's dimensions; "
                "transparent pixels mark the editable area."
            ),
        }
    edit_properties.update(common_props)
    edit_properties["output_path"] = output_path_prop

    edit_description = (
        f"Edit or combine existing image(s) using {resolved.spec}. "
        "Pass paths returned by generate_image to iterate; set output_path to the source path "
        "to edit in place."
    )
    edit_meta: dict[str, Any] = {
        **builtin_meta,
        "fast_agent": {
            **base_fast_agent_meta,
            "ptc": {
                "description": edit_description + ptc_return_note,
                "returns": ptc_returns_schema,
            },
        },
    }
    edit_tool = Tool(
        name=EDIT_IMAGE_TOOL_NAME,
        description=edit_description,
        input_schema={
            "type": "object",
            "properties": edit_properties,
            "required": ["images", "prompt"],
        },
        output_schema=_OUTPUT_SCHEMA,
        meta=edit_meta,
    )

    return [generate_tool, edit_tool]


def _slugify_prompt(prompt: str, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", prompt.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "image"


def _sniff_image_mime(data: bytes) -> str:
    header = data[:32]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    stripped = data.lstrip()[:256].lower()
    if stripped.startswith(b"<svg") or (stripped.startswith(b"<?xml") and b"<svg" in stripped):
        return "image/svg+xml"
    return "image/png"


def _format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _extract_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip().splitlines()
        detail = text[0] if text else response.reason_phrase
        return f"HTTP {response.status_code}: {detail}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error") or str(err)
            return f"HTTP {response.status_code}: {msg}"
        if isinstance(err, str) and err:
            return f"HTTP {response.status_code}: {err}"
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return f"HTTP {response.status_code}: {detail}"
    return f"HTTP {response.status_code}: {response.text.strip()[:240]}"


def _inspect_image(data: bytes, mime_type: str) -> tuple[int | None, int | None, bool]:
    """Return ``(width, height, is_animated)`` for raster images, or ``(None, None, False)``."""
    if mime_type == "image/svg+xml":
        return None, None, False
    try:
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            is_animated = False
            try:
                img.seek(1)
                is_animated = True
            except EOFError:
                is_animated = False
            return width, height, is_animated
    except (UnidentifiedImageError, OSError):
        return None, None, False


def _build_preview_block(data: bytes, mime_type: str) -> ImageContent | None:
    """Create a size-bounded raster preview block for model/terminal display."""
    if mime_type == "image/svg+xml":
        return None
    if mime_type == "image/gif" and len(data) <= DEFAULT_PREVIEW_MAX_BYTES:
        return ImageContent(
            type="image",
            data=base64.b64encode(data).decode("ascii"),
            mime_type="image/gif",
        )
    try:
        with Image.open(BytesIO(data)) as img:
            width, height = img.size
            needs_resize = max(width, height) > DEFAULT_PREVIEW_MAX_EDGE
            needs_reencode = len(data) > DEFAULT_PREVIEW_MAX_BYTES or mime_type not in {
                "image/png",
                "image/jpeg",
                "image/webp",
            }
            if not needs_resize and not needs_reencode:
                return ImageContent(
                    type="image",
                    data=base64.b64encode(data).decode("ascii"),
                    mime_type=mime_type,
                )

            working = img.copy()
            if needs_resize:
                working.thumbnail(
                    (DEFAULT_PREVIEW_MAX_EDGE, DEFAULT_PREVIEW_MAX_EDGE),
                    Image.Resampling.LANCZOS,
                )
            has_alpha = "A" in working.getbands() or (
                working.mode == "P" and "transparency" in working.info
            )
            buf = BytesIO()
            if has_alpha:
                rgba = working.convert("RGBA")
                rgba.save(buf, format="PNG", optimize=True)
                out_mime = "image/png"
                preview_bytes = buf.getvalue()
                if len(preview_bytes) > DEFAULT_PREVIEW_MAX_BYTES:
                    webp_buf = BytesIO()
                    rgba.save(webp_buf, format="WEBP", quality=85)
                    preview_bytes = webp_buf.getvalue()
                    out_mime = "image/webp"
            else:
                working.convert("RGB").save(buf, format="JPEG", quality=85, optimize=True)
                out_mime = "image/jpeg"
                preview_bytes = buf.getvalue()
            if len(preview_bytes) > DEFAULT_PREVIEW_MAX_BYTES:
                return None
            return ImageContent(
                type="image",
                data=base64.b64encode(preview_bytes).decode("ascii"),
                mime_type=out_mime,
            )
    except (UnidentifiedImageError, OSError):
        return None


class ImageGenRuntime:
    """Executes ``generate_image`` and ``edit_image`` calls for one agent."""

    def __init__(
        self,
        resolved: ResolvedImageGenModel,
        *,
        settings: Settings | None = None,
        working_dir: Path | Callable[[], Path] | None = None,
        supports_vision: Callable[[], bool] | None = None,
        agent_name: str | None = None,
    ) -> None:
        self._resolved = resolved
        self._settings = settings
        self._working_dir: Path | Callable[[], Path] = (
            working_dir if callable(working_dir) else (working_dir or Path.cwd()).resolve()
        )
        self._supports_vision = supports_vision
        self._agent_name = agent_name
        self._tools = build_image_gen_tools(resolved)

    def _current_working_dir(self) -> Path:
        if isinstance(self._working_dir, Path):
            return self._working_dir
        return self._working_dir().resolve()

    @property
    def resolved(self) -> ResolvedImageGenModel:
        return self._resolved

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def owns_tool(self, name: str) -> bool:
        return name in IMAGE_GEN_TOOL_NAMES

    def _default_output_dir(self) -> Path:
        cwd = self._current_working_dir()
        if self._settings is not None and self._settings.image_gen.output_dir:
            configured = Path(self._settings.image_gen.output_dir).expanduser()
            return configured if configured.is_absolute() else (cwd / configured)
        try:
            return resolve_home_dir(self._settings, cwd=cwd) / "images"
        except ValueError:
            return Path(tempfile.gettempdir()) / "fast-agent-images"

    def _resolve_input_path(self, raw_path: str) -> Path:
        local = local_file_url_path(raw_path)
        path = local if local is not None else Path(raw_path).expanduser()
        return path if path.is_absolute() else (self._current_working_dir() / path)

    def _resolve_output_paths(
        self,
        *,
        output_path: str | None,
        count: int,
        ext: str,
        prompt: str,
    ) -> list[Path]:
        if output_path is not None:
            raw = Path(output_path).expanduser()
            target = raw if raw.is_absolute() else (self._current_working_dir() / raw)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.suffix:
                target = target.with_suffix(ext)
            elif target.suffix.lower() != ext and not (
                target.suffix.lower() == ".jpeg" and ext == ".jpg"
            ):
                target = target.with_suffix(ext)
            if count == 1:
                return [target]
            return [
                target.with_name(f"{target.stem}-{idx}{target.suffix}")
                for idx in range(1, count + 1)
            ]

        out_dir = self._default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = _slugify_prompt(prompt)
        stem = f"{stamp}-{slug}"

        if count == 1:
            candidate = out_dir / f"{stem}{ext}"
            counter = 2
            while candidate.exists():
                candidate = out_dir / f"{stem}-{counter}{ext}"
                counter += 1
            return [candidate]

        paths: list[Path] = []
        suffix_offset = 0
        while True:
            prefix = f"{stem}-{suffix_offset}" if suffix_offset else stem
            candidates = [out_dir / f"{prefix}-{idx}{ext}" for idx in range(1, count + 1)]
            if not any(p.exists() for p in candidates):
                paths = candidates
                break
            suffix_offset += 1
        return paths

    def _validate_common_args(
        self,
        arguments: dict[str, Any],
    ) -> tuple[str, str, str | None, int, bool, str | None]:
        prompt_raw = arguments.get("prompt")
        if not isinstance(prompt_raw, str) or not prompt_raw.strip():
            raise ValueError("'prompt' must be a non-empty string.")
        prompt = prompt_raw.strip()

        aspect_raw = arguments.get("aspect_ratio", "auto")
        if aspect_raw is None:
            aspect_ratio = "auto"
        elif isinstance(aspect_raw, str) and aspect_raw in self._resolved.aspect_ratio_values:
            aspect_ratio = aspect_raw
        else:
            allowed = ", ".join(self._resolved.aspect_ratio_values)
            raise ValueError(f"Unsupported aspect_ratio '{aspect_raw}'. Allowed: {allowed}.")

        quality: str | None = None
        if self._resolved.supports_quality:
            quality_raw = arguments.get("quality")
            if quality_raw is None:
                quality = self._resolved.default_quality or "auto"
            elif isinstance(quality_raw, str) and quality_raw in self._resolved.quality_values:
                quality = quality_raw
            else:
                allowed = ", ".join(self._resolved.quality_values)
                raise ValueError(f"Unsupported quality '{quality_raw}'. Allowed: {allowed}.")

        n_raw = arguments.get("n", 1)
        if isinstance(n_raw, bool) or not isinstance(n_raw, int):
            raise ValueError("'n' must be an integer.")
        if n_raw < 1 or n_raw > MAX_IMAGE_GEN_COUNT:
            raise ValueError(f"'n' must be between 1 and {MAX_IMAGE_GEN_COUNT}.")

        transparent_raw = arguments.get("transparent", False)
        if not isinstance(transparent_raw, bool):
            raise ValueError("'transparent' must be a boolean.")
        if transparent_raw and not self._resolved.is_openai_family:
            raise ValueError("'transparent' is only supported by OpenAI/Codex image models.")

        output_path_raw = arguments.get("output_path")
        output_path: str | None = None
        if output_path_raw is not None:
            if not isinstance(output_path_raw, str) or not output_path_raw.strip():
                raise ValueError("'output_path' must be a non-empty string when provided.")
            output_path = output_path_raw.strip()

        return prompt, aspect_ratio, quality, n_raw, transparent_raw, output_path

    def _infer_openai_format(self, output_path: str | None, transparent: bool) -> str:
        if output_path is None:
            return "png"
        ext = Path(output_path).suffix.lower()
        if not ext:
            return "png"
        fmt = _EXT_TO_OPENAI_FORMAT.get(ext)
        if fmt is None:
            raise ValueError(
                f"Unsupported output file extension '{ext}'. "
                "Use .png, .jpg, .webp"
                + (", .svg, or .gif." if self._resolved.supports_svg_gif else ".")
            )
        if fmt in {"svg", "gif"} and not self._resolved.supports_svg_gif:
            raise ValueError(
                f"Output format '{fmt}' requires a gpt-image-2.6 model (active: {self._resolved.spec})."
            )
        if transparent and fmt == "jpeg":
            raise ValueError(
                "Transparent background is not supported with JPEG output (.jpg/.jpeg)."
            )
        return fmt

    async def _load_image_bytes(
        self,
        client: httpx.AsyncClient,
        source: str,
    ) -> tuple[str, bytes, str]:
        """Load an input image from local path, data URI, or HTTP(S) URL."""
        if source.startswith("data:"):
            header, _, b64_payload = source.partition(",")
            if not b64_payload:
                raise ValueError("Invalid data URI for input image.")
            mime = header[5:].split(";", 1)[0] or "image/png"
            raw = base64.b64decode(b64_payload)
            ext = _MIME_TO_EXT.get(mime, ".png")
            return f"input{ext}", raw, mime

        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            resp = await client.get(source, timeout=60.0)
            if resp.status_code != 200:
                raise ValueError(f"Failed to fetch input image '{source}': HTTP {resp.status_code}")
            raw = resp.content
            mime = resp.headers.get("content-type", "").split(";", 1)[
                0
            ].strip() or _sniff_image_mime(raw)
            name = Path(parsed.path).name or f"input{_MIME_TO_EXT.get(mime, '.png')}"
            return name, raw, mime

        path = self._resolve_input_path(source)
        if not path.is_file():
            raise ValueError(f"Input image not found: {source}")
        raw = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or _sniff_image_mime(raw)
        return path.name, raw, mime

    def _source_to_data_or_url(self, source: str) -> str:
        """Convert a local path, data URI, or HTTP(S) URL to a URL or base64 data URI."""
        if source.startswith(("http://", "https://", "data:")):
            return source
        path = self._resolve_input_path(source)
        if not path.is_file():
            raise ValueError(f"Input image not found: {source}")
        raw = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or _sniff_image_mime(raw)
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    async def _decode_api_images(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, Any],
        *,
        fallback_mime: str,
    ) -> list[_GeneratedImageData]:
        items = payload.get("data")
        if not isinstance(items, list) or not items:
            raise ValueError("Image API returned no images.")
        results: list[_GeneratedImageData] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            revised = item.get("revised_prompt")
            revised_str = revised if isinstance(revised, str) and revised.strip() else None
            b64_data = item.get("b64_json")
            if isinstance(b64_data, str) and b64_data:
                raw = base64.b64decode(b64_data)
                mime = item.get("mime_type") if isinstance(item.get("mime_type"), str) else None
                results.append(
                    _GeneratedImageData(
                        data=raw,
                        mime_type=mime or _sniff_image_mime(raw) or fallback_mime,
                        revised_prompt=revised_str,
                    )
                )
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                dl = await client.get(url, timeout=60.0)
                if dl.status_code != 200:
                    raise ValueError(f"Failed to download generated image: HTTP {dl.status_code}")
                raw = dl.content
                mime = (
                    item.get("mime_type")
                    if isinstance(item.get("mime_type"), str)
                    else dl.headers.get("content-type", "").split(";", 1)[0].strip()
                )
                results.append(
                    _GeneratedImageData(
                        data=raw,
                        mime_type=mime or _sniff_image_mime(raw) or fallback_mime,
                        revised_prompt=revised_str,
                    )
                )
        if not results:
            raise ValueError("Image API returned no usable image payloads.")
        return results

    async def _call_openai_generate(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        quality: str | None,
        n: int,
        transparent: bool,
        output_format: str,
    ) -> list[_GeneratedImageData]:
        base_url, headers = _resolve_backend_endpoint(self._resolved, self._settings)
        body: dict[str, Any] = {
            "model": self._resolved.model,
            "prompt": prompt,
            "n": n,
            "size": OPENAI_ASPECT_RATIO_TO_SIZE[aspect_ratio],
            "output_format": output_format,
        }
        if quality and quality != "auto":
            body["quality"] = quality
        if transparent:
            body["background"] = "transparent"
        if self._resolved.complexity and self._resolved.supports_complexity:
            body["complexity"] = self._resolved.complexity

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{base_url}/images/generations", headers=headers, json=body)
            if resp.status_code != 200:
                raise RuntimeError(_extract_error_message(resp))
            return await self._decode_api_images(
                client,
                resp.json(),
                fallback_mime=_FORMAT_TO_MIME.get(output_format, "image/png"),
            )

    async def _call_openai_edit(
        self,
        *,
        images: list[str],
        mask: str | None,
        prompt: str,
        aspect_ratio: str,
        quality: str | None,
        n: int,
        transparent: bool,
        output_format: str,
    ) -> list[_GeneratedImageData]:
        base_url, headers = _resolve_backend_endpoint(self._resolved, self._settings)
        data_fields: dict[str, str] = {
            "model": self._resolved.model,
            "prompt": prompt,
            "n": str(n),
            "size": OPENAI_ASPECT_RATIO_TO_SIZE[aspect_ratio],
            "output_format": output_format,
        }
        if quality and quality != "auto":
            data_fields["quality"] = quality
        if transparent:
            data_fields["background"] = "transparent"
        if self._resolved.complexity and self._resolved.supports_complexity:
            data_fields["complexity"] = self._resolved.complexity

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            if self._resolved.backend == "codexresponses":
                body: dict[str, Any] = {
                    "model": self._resolved.model,
                    "prompt": prompt,
                    "n": n,
                    "size": OPENAI_ASPECT_RATIO_TO_SIZE[aspect_ratio],
                    "output_format": output_format,
                    "images": [{"image_url": self._source_to_data_or_url(src)} for src in images],
                }
                if quality and quality != "auto":
                    body["quality"] = quality
                if transparent:
                    body["background"] = "transparent"
                if self._resolved.complexity and self._resolved.supports_complexity:
                    body["complexity"] = self._resolved.complexity
                if mask is not None:
                    body["mask"] = {"image_url": self._source_to_data_or_url(mask)}
                resp = await client.post(
                    f"{base_url}/images/edits",
                    headers=headers,
                    json=body,
                )
            else:
                files: list[tuple[str, tuple[str, bytes, str]]] = []
                field_name = "image" if len(images) == 1 else "image[]"
                for src in images:
                    fname, raw, mime = await self._load_image_bytes(client, src)
                    files.append((field_name, (fname, raw, mime)))
                if mask is not None:
                    m_name, m_raw, m_mime = await self._load_image_bytes(client, mask)
                    files.append(("mask", (m_name, m_raw, m_mime)))

                resp = await client.post(
                    f"{base_url}/images/edits",
                    headers=headers,
                    data=data_fields,
                    files=files,
                )
            if resp.status_code != 200:
                raise RuntimeError(_extract_error_message(resp))
            return await self._decode_api_images(
                client,
                resp.json(),
                fallback_mime=_FORMAT_TO_MIME.get(output_format, "image/png"),
            )

    async def _call_xai_generate(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        quality: str | None,
        n: int,
    ) -> list[_GeneratedImageData]:
        base_url, headers = _resolve_backend_endpoint(self._resolved, self._settings)
        body: dict[str, Any] = {
            "model": self._resolved.model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        if aspect_ratio != "auto":
            body["aspect_ratio"] = aspect_ratio
        if quality and quality != "auto" and self._resolved.supports_quality:
            body["quality"] = quality
        if self._resolved.resolution:
            body["resolution"] = self._resolved.resolution

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{base_url}/images/generations", headers=headers, json=body)
            if resp.status_code != 200:
                raise RuntimeError(_extract_error_message(resp))
            return await self._decode_api_images(client, resp.json(), fallback_mime="image/jpeg")

    async def _call_xai_edit(
        self,
        *,
        images: list[str],
        prompt: str,
        aspect_ratio: str,
        quality: str | None,
        n: int,
    ) -> list[_GeneratedImageData]:
        if len(images) > MAX_XAI_EDIT_IMAGES:
            raise ValueError(
                f"xAI image editing supports at most {MAX_XAI_EDIT_IMAGES} source images."
            )
        base_url, headers = _resolve_backend_endpoint(self._resolved, self._settings)
        body: dict[str, Any] = {
            "model": self._resolved.model,
            "prompt": prompt,
            "n": n,
            "response_format": "b64_json",
        }
        image_refs = [
            {"url": self._source_to_data_or_url(src), "type": "image_url"} for src in images
        ]
        if len(image_refs) == 1:
            body["image"] = image_refs[0]
        else:
            body["images"] = image_refs
        if aspect_ratio != "auto":
            body["aspect_ratio"] = aspect_ratio
        if quality and quality != "auto" and self._resolved.supports_quality:
            body["quality"] = quality
        if self._resolved.resolution:
            body["resolution"] = self._resolved.resolution

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            resp = await client.post(f"{base_url}/images/edits", headers=headers, json=body)
            if resp.status_code != 200:
                raise RuntimeError(_extract_error_message(resp))
            return await self._decode_api_images(client, resp.json(), fallback_mime="image/jpeg")

    def _decode_google_images(self, payload: dict[str, Any]) -> list[_GeneratedImageData]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Google image API returned no candidates.")
        results: list[_GeneratedImageData] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content")
            if not isinstance(content, dict):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            text_notes: list[str] = []
            candidate_images: list[tuple[bytes, str]] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_notes.append(text.strip())
                inline = part.get("inlineData") or part.get("inline_data")
                if isinstance(inline, dict):
                    b64_data = inline.get("data")
                    if isinstance(b64_data, str) and b64_data:
                        raw = base64.b64decode(b64_data)
                        mime = inline.get("mimeType") or inline.get("mime_type")
                        mime_str = (
                            mime if isinstance(mime, str) and mime else _sniff_image_mime(raw)
                        )
                        candidate_images.append((raw, mime_str or "image/jpeg"))
            note = "\n".join(text_notes) if text_notes else None
            for idx, (raw, mime_str) in enumerate(candidate_images):
                results.append(
                    _GeneratedImageData(
                        data=raw,
                        mime_type=mime_str,
                        revised_prompt=note if idx == 0 else None,
                    )
                )
        if not results:
            raise ValueError("Google image API returned no usable image payloads.")
        return results

    async def _call_google_request(
        self,
        *,
        parts: list[dict[str, Any]],
        aspect_ratio: str,
        n: int,
    ) -> list[_GeneratedImageData]:
        base_url, headers = _resolve_backend_endpoint(self._resolved, self._settings)
        generation_config: dict[str, Any] = {
            "responseModalities": ["TEXT", "IMAGE"],
        }
        image_config: dict[str, str] = {}
        if aspect_ratio != "auto":
            image_config["aspectRatio"] = aspect_ratio
        if self._resolved.resolution:
            image_config["imageSize"] = self._resolved.resolution.upper()
        if image_config:
            generation_config["imageConfig"] = image_config

        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": generation_config,
        }
        url = f"{base_url}/models/{self._resolved.model}:generateContent"

        async def _single(client: httpx.AsyncClient) -> list[_GeneratedImageData]:
            resp = await client.post(url, headers=headers, json=body)
            if resp.status_code != 200:
                raise RuntimeError(_extract_error_message(resp))
            return self._decode_google_images(resp.json())

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            if n == 1:
                return await _single(client)
            batches = await asyncio.gather(*(_single(client) for _ in range(n)))
            return [img for batch in batches for img in batch]

    async def _call_google_generate(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        n: int,
    ) -> list[_GeneratedImageData]:
        return await self._call_google_request(
            parts=[{"text": prompt}],
            aspect_ratio=aspect_ratio,
            n=n,
        )

    async def _call_google_edit(
        self,
        *,
        images: list[str],
        prompt: str,
        aspect_ratio: str,
        n: int,
    ) -> list[_GeneratedImageData]:
        if len(images) > MAX_GOOGLE_EDIT_IMAGES:
            raise ValueError(
                f"Google image editing supports at most {MAX_GOOGLE_EDIT_IMAGES} source images."
            )
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            loaded = [await self._load_image_bytes(client, src) for src in images]
        parts: list[dict[str, Any]] = [
            {
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            }
            for _, raw, mime in loaded
        ]
        parts.append({"text": prompt})
        return await self._call_google_request(
            parts=parts,
            aspect_ratio=aspect_ratio,
            n=n,
        )

    def _build_result(
        self,
        generated: list[_GeneratedImageData],
        *,
        prompt: str,
        output_path: str | None,
        elapsed_seconds: float,
    ) -> CallToolResult:
        actual_ext = _MIME_TO_EXT.get(generated[0].mime_type, ".png")
        requested_ext = Path(output_path).suffix.lower() if output_path else ""
        ext_adjusted = bool(
            requested_ext
            and requested_ext != actual_ext
            and not (requested_ext == ".jpeg" and actual_ext == ".jpg")
        )

        saved_paths = self._resolve_output_paths(
            output_path=output_path,
            count=len(generated),
            ext=actual_ext,
            prompt=prompt,
        )

        image_blocks: list[ImageContent] = []
        structured_images: list[dict[str, Any]] = []
        summary_lines: list[str] = []
        inline_svg_blocks: list[str] = []
        revised_prompt: str | None = None

        for item, dest in zip(generated, saved_paths, strict=True):
            dest.write_bytes(item.data)
            width, height, is_animated = _inspect_image(item.data, item.mime_type)
            fmt_label = item.mime_type.split("/", 1)[-1].replace("+xml", "")
            if is_animated:
                fmt_label = f"animated {fmt_label}"
            dims = f"{width}x{height} " if width and height else ""
            try:
                display_path = str(dest.relative_to(self._current_working_dir()))
            except ValueError:
                display_path = str(dest)
            summary_lines.append(
                f"Saved {display_path} ({dims}{fmt_label}, {_format_bytes(len(item.data))})"
            )
            structured_images.append(
                {
                    "path": str(dest),
                    "mime_type": item.mime_type,
                    "width": width,
                    "height": height,
                    "bytes": len(item.data),
                }
            )
            if item.revised_prompt and revised_prompt is None:
                revised_prompt = item.revised_prompt

            if item.mime_type == "image/svg+xml":
                if len(item.data) <= INLINE_SVG_MAX_BYTES:
                    inline_svg_blocks.append(item.data.decode("utf-8", errors="replace"))
            else:
                preview = _build_preview_block(item.data, item.mime_type)
                if preview is not None:
                    image_blocks.append(preview)

        if summary_lines:
            summary_lines[0] = (
                f"{summary_lines[0]} · {self._resolved.spec} · {elapsed_seconds:.1f}s"
            )
        if ext_adjusted:
            summary_lines.append(
                f"Note: provider returned {generated[0].mime_type}, saved with {actual_ext} extension."
            )
        if revised_prompt:
            summary_lines.append(f"Revised prompt: {revised_prompt}")
        if inline_svg_blocks:
            summary_lines.extend(inline_svg_blocks)

        summary_text = "\n".join(summary_lines)
        vision_ok = self._supports_vision() if self._supports_vision is not None else True

        content: list[ContentBlock] = []
        if vision_ok and image_blocks:
            content.extend(image_blocks)
        content.append(TextContent(type="text", text=summary_text))

        result = CallToolResult(
            content=content,
            structured_content={
                "images": structured_images,
                "model": self._resolved.spec,
                "revised_prompt": revised_prompt,
            },
            is_error=False,
        )
        if image_blocks and not vision_ok:
            set_tool_result_media_preview(result, image_blocks)
        return result

    def _emit_progress(
        self,
        *,
        action: ProgressAction,
        tool_name: str,
        tool_use_id: str | None,
        details: str,
        tool_event: str | None = None,
        tool_state: str | None = None,
        tool_terminal: bool | None = None,
    ) -> None:
        payload = build_progress_payload(
            action=action,
            agent_name=self._agent_name,
            tool_name=tool_name,
            server_name="local",
            tool_use_id=tool_use_id,
            tool_event=tool_event,
            tool_state=tool_state,
            tool_terminal=tool_terminal,
            details=details,
        )
        logger.info(f"Image tool {tool_name}: {details}", data=payload)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
        *,
        tool_handler: ToolExecutionHandler | None = None,
    ) -> CallToolResult:
        args = arguments or {}
        started = time.monotonic()
        progress_started = False
        handler_call_id: str | None = None
        if tool_handler is not None:
            try:
                handler_call_id = await tool_handler.on_tool_start(name, "local", args, tool_use_id)
            except Exception:
                handler_call_id = None

        async def _notify_handler_progress(
            progress: float, total: float | None, message: str
        ) -> None:
            if tool_handler is not None and handler_call_id is not None:
                try:
                    await tool_handler.on_tool_progress(handler_call_id, progress, total, message)
                except Exception:
                    pass

        async def _notify_handler_complete(
            *,
            success: bool,
            content: list[ContentBlock] | None,
            error: str | None,
        ) -> None:
            if tool_handler is not None and handler_call_id is not None:
                try:
                    await tool_handler.on_tool_complete(handler_call_id, success, content, error)
                except Exception:
                    pass

        try:
            prompt, aspect_ratio, quality, n, transparent, output_path = self._validate_common_args(
                args
            )
            if name == GENERATE_IMAGE_TOOL_NAME:
                count_label = f"{n} images " if n > 1 else ""
                start_detail = f"Generating {count_label}with {self._resolved.model}..."
                self._emit_progress(
                    action=ProgressAction.CALLING_TOOL,
                    tool_name=name,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                    details=start_detail,
                )
                progress_started = True
                await _notify_handler_progress(0.0, 2.0, start_detail)
                if self._resolved.is_openai_family:
                    output_format = self._infer_openai_format(output_path, transparent)
                    generated = await self._call_openai_generate(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        quality=quality,
                        n=n,
                        transparent=transparent,
                        output_format=output_format,
                    )
                elif self._resolved.backend == "google":
                    generated = await self._call_google_generate(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        n=n,
                    )
                else:
                    generated = await self._call_xai_generate(
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        quality=quality,
                        n=n,
                    )
            elif name == EDIT_IMAGE_TOOL_NAME:
                images_raw = args.get("images")
                if (
                    not isinstance(images_raw, list)
                    or not images_raw
                    or not all(isinstance(x, str) and x.strip() for x in images_raw)
                ):
                    raise ValueError("'images' must be a non-empty list of image paths or URLs.")
                images = [x.strip() for x in images_raw]
                mask_raw = args.get("mask")
                mask: str | None = None
                if mask_raw is not None:
                    if not self._resolved.is_openai_family:
                        raise ValueError("'mask' is only supported by OpenAI/Codex image models.")
                    if not isinstance(mask_raw, str) or not mask_raw.strip():
                        raise ValueError("'mask' must be a non-empty file path when provided.")
                    mask = mask_raw.strip()

                input_label = f"{len(images)} images" if len(images) > 1 else "image"
                start_detail = f"Editing {input_label} with {self._resolved.model}..."
                self._emit_progress(
                    action=ProgressAction.CALLING_TOOL,
                    tool_name=name,
                    tool_use_id=tool_use_id,
                    tool_event="start",
                    details=start_detail,
                )
                progress_started = True
                await _notify_handler_progress(0.0, 2.0, start_detail)
                if self._resolved.is_openai_family:
                    output_format = self._infer_openai_format(output_path, transparent)
                    generated = await self._call_openai_edit(
                        images=images,
                        mask=mask,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        quality=quality,
                        n=n,
                        transparent=transparent,
                        output_format=output_format,
                    )
                elif self._resolved.backend == "google":
                    generated = await self._call_google_edit(
                        images=images,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        n=n,
                    )
                else:
                    generated = await self._call_xai_edit(
                        images=images,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        quality=quality,
                        n=n,
                    )
            else:
                raise ValueError(f"Unknown image generation tool: {name}")

            img_noun = "image" if len(generated) == 1 else "images"
            saving_detail = f"Saving {len(generated)} {img_noun}..."
            self._emit_progress(
                action=ProgressAction.TOOL_PROGRESS,
                tool_name=name,
                tool_use_id=tool_use_id,
                details=saving_detail,
            )
            await _notify_handler_progress(1.0, 2.0, saving_detail)
            result = self._build_result(
                generated,
                prompt=prompt,
                output_path=output_path,
                elapsed_seconds=time.monotonic() - started,
            )
            self._emit_progress(
                action=ProgressAction.TOOL_PROGRESS,
                tool_name=name,
                tool_use_id=tool_use_id,
                details=f"Saved {len(generated)} {img_noun}",
                tool_state="completed",
                tool_terminal=True,
            )
            await _notify_handler_complete(
                success=True,
                content=result.content or None,
                error=None,
            )
            return result
        except asyncio.CancelledError:
            if progress_started:
                self._emit_progress(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_name=name,
                    tool_use_id=tool_use_id,
                    details="cancelled",
                    tool_state="cancelled",
                    tool_terminal=True,
                )
            await _notify_handler_complete(success=False, content=None, error="cancelled")
            raise
        except Exception as exc:
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            logger.warning(f"Image generation tool {name} failed: {message}")
            if progress_started:
                self._emit_progress(
                    action=ProgressAction.TOOL_PROGRESS,
                    tool_name=name,
                    tool_use_id=tool_use_id,
                    details="failed",
                    tool_state="failed",
                    tool_terminal=True,
                )
            error_text = f"Image generation error: {message}"
            await _notify_handler_complete(success=False, content=None, error=error_text)
            return CallToolResult(
                content=[TextContent(type="text", text=error_text)],
                is_error=True,
            )
