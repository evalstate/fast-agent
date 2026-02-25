from fast_agent.cards.manager import DEFAULT_CARD_REGISTRIES, resolve_card_registries
from fast_agent.config import CardsSettings, Settings, SkillsSettings
from fast_agent.marketplace.registry_urls import (
    format_marketplace_display_url,
    resolve_registry_urls,
)
from fast_agent.skills.manager import DEFAULT_SKILL_REGISTRIES, resolve_skill_registries


def test_format_marketplace_display_url_for_github_variants() -> None:
    assert (
        format_marketplace_display_url(
            "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json"
        )
        == "https://github.com/huggingface/skills"
    )
    assert (
        format_marketplace_display_url("https://github.com/huggingface/skills")
        == "https://github.com/huggingface/skills"
    )


def test_resolve_registry_urls_dedupes_only_equivalent_sources() -> None:
    resolved = resolve_registry_urls(
        [
            "https://github.com/huggingface/skills/blob/main/marketplace.json",
            "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
            "https://github.com/anthropics/skills",
        ],
        default_urls=["https://github.com/fast-agent-ai/skills"],
    )

    assert resolved == [
        "https://github.com/huggingface/skills/blob/main/marketplace.json",
        "https://github.com/anthropics/skills",
    ]


def test_resolve_registry_urls_preserves_distinct_github_refs() -> None:
    resolved = resolve_registry_urls(
        [
            "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
            "https://raw.githubusercontent.com/huggingface/skills/dev/marketplace.json",
        ],
        default_urls=["https://github.com/fast-agent-ai/skills"],
    )

    assert resolved == [
        "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/huggingface/skills/dev/marketplace.json",
    ]


def test_resolve_skill_registries_keeps_configured_and_active_distinct_sources() -> None:
    settings = Settings(
        skills=SkillsSettings(
            marketplace_urls=list(DEFAULT_SKILL_REGISTRIES),
            marketplace_url="https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
        )
    )

    resolved = resolve_skill_registries(settings)

    assert len(resolved) == 4
    assert resolved == [
        *list(DEFAULT_SKILL_REGISTRIES),
        "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
    ]


def test_resolve_card_registries_keeps_configured_and_active_distinct_sources() -> None:
    settings = Settings(
        cards=CardsSettings(
            marketplace_urls=list(DEFAULT_CARD_REGISTRIES),
            marketplace_url="https://raw.githubusercontent.com/fast-agent-ai/card-packs/main/marketplace.json",
        )
    )

    resolved = resolve_card_registries(settings)

    assert len(resolved) == 2
    assert resolved == [
        *list(DEFAULT_CARD_REGISTRIES),
        "https://raw.githubusercontent.com/fast-agent-ai/card-packs/main/marketplace.json",
    ]
