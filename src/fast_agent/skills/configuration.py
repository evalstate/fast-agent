"""Settings-aware skills configuration helpers."""

from __future__ import annotations

from fast_agent.config import Settings, get_settings
from fast_agent.marketplace import registry_urls as marketplace_registry_urls
from fast_agent.skills.models import DEFAULT_MARKETPLACE_URL, DEFAULT_SKILL_REGISTRIES
from fast_agent.skills.operations import normalize_marketplace_url


def get_marketplace_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    skills_settings = resolved_settings.skills
    url = skills_settings.marketplace_url
    if not url and skills_settings.marketplace_urls:
        url = skills_settings.marketplace_urls[0]
    return normalize_marketplace_url(url or DEFAULT_MARKETPLACE_URL)


def resolve_skill_registries(settings: Settings | None = None) -> list[str]:
    resolved_settings = settings or get_settings()
    skills_settings = resolved_settings.skills
    return marketplace_registry_urls.resolve_registry_urls(
        skills_settings.marketplace_urls,
        default_urls=DEFAULT_SKILL_REGISTRIES,
        active_url=skills_settings.marketplace_url,
    )


def format_marketplace_display_url(url: str) -> str:
    return marketplace_registry_urls.format_marketplace_display_url(url)
