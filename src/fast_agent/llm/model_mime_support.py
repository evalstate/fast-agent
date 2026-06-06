from typing import Literal, Sequence

from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.mime_utils import DOCUMENT_MIME_TYPES, normalize_mime_type

ResourceSource = Literal["embedded", "link"]


def provider_document_mime_override(
    provider: Provider | None,
    mime_type: str | None,
    *,
    resource_source: ResourceSource | None = None,
) -> bool | None:
    normalized_mime_type = normalize_mime_type(mime_type or "")
    if not normalized_mime_type or normalized_mime_type not in DOCUMENT_MIME_TYPES:
        return None

    if (
        resource_source == "link"
        and provider in {Provider.ANTHROPIC, Provider.ANTHROPIC_VERTEX}
        and normalized_mime_type != "application/pdf"
    ):
        return False

    if provider in {
        Provider.OPENAI,
        Provider.AZURE,
        Provider.ALIYUN,
        Provider.GOOGLE_OAI,
    }:
        return normalized_mime_type == "application/pdf"

    return None


def tokenizes_support_mime(
    tokenizes: Sequence[str],
    mime_type: str,
    *,
    provider: Provider | None = None,
    resource_source: ResourceSource | None = None,
) -> bool:
    normalized_supported = {
        normalized for mime in tokenizes if (normalized := normalize_mime_type(mime))
    }

    normalized = normalize_mime_type(mime_type)
    candidate = normalized or ""
    if candidate.endswith("/*") and "/" in candidate:
        prefix = candidate.split("/", 1)[0] + "/"
        return any(supported.startswith(prefix) for supported in normalized_supported)

    override = provider_document_mime_override(
        provider,
        mime_type,
        resource_source=resource_source,
    )

    if override is False:
        return False

    return bool(candidate and candidate in normalized_supported)
