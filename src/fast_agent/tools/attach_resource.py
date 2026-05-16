"""Deprecated compatibility module for the initial resource-oriented attach API."""

from fast_agent.tools.attach_media import (
    ATTACHABLE_MIME_TYPES,
    DEFAULT_ATTACH_MEDIA_MAX_BYTES,
    DEFAULT_ATTACH_RESOURCE_MAX_BYTES,
    AttachMediaResult,
    AttachResourceResult,
    build_attach_media,
    build_attach_resource,
    model_supports_attach_media,
    model_supports_attach_resource,
    supported_attach_media_mime_types,
    supported_attach_resource_mime_types,
)

__all__ = [
    "ATTACHABLE_MIME_TYPES",
    "DEFAULT_ATTACH_MEDIA_MAX_BYTES",
    "DEFAULT_ATTACH_RESOURCE_MAX_BYTES",
    "AttachMediaResult",
    "AttachResourceResult",
    "build_attach_media",
    "build_attach_resource",
    "model_supports_attach_media",
    "model_supports_attach_resource",
    "supported_attach_media_mime_types",
    "supported_attach_resource_mime_types",
]
