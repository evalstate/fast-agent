from fast_agent.cards.manager import DEFAULT_CARD_REGISTRIES, resolve_card_registries
from fast_agent.config import CardsSettings, Settings, SkillsSettings
from fast_agent.marketplace.registry_urls import (
    format_marketplace_display_url,
    resolve_registry_urls,
)
from fast_agent.skills.configuration import resolve_skill_registries
from fast_agent.skills.models import DEFAULT_SKILL_REGISTRIES


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
    assert (
        format_marketplace_display_url(
            "https://github.com/huggingface/skills/tree/main/.claude-plugin"
        )
        == "https://github.com/huggingface/skills"
    )
    assert (
        format_marketplace_display_url(
            "https://github.com/huggingface/skills/blob/master/.claude-plugin/marketplace.json"
        )
        == "https://github.com/huggingface/skills"
    )


def test_format_marketplace_display_url_normalizes_github_host_case() -> None:
    assert (
        format_marketplace_display_url(
            "https://GitHub.com/huggingface/skills/tree/main/.claude-plugin"
        )
        == "https://github.com/huggingface/skills"
    )
    assert (
        format_marketplace_display_url(
            "https://RAW.GitHubUserContent.com/huggingface/skills/main/marketplace.json"
        )
        == "https://github.com/huggingface/skills"
    )


def test_format_marketplace_display_url_strips_outer_whitespace() -> None:
    assert (
        format_marketplace_display_url(
            "  https://github.com/huggingface/skills/tree/main/.claude-plugin  "
        )
        == "https://github.com/huggingface/skills"
    )
    assert format_marketplace_display_url("   ") == ""


def test_format_marketplace_display_url_preserves_distinct_github_registry_paths() -> None:
    raw_dev = "https://raw.githubusercontent.com/huggingface/skills/dev/marketplace.json"
    nested = "https://github.com/huggingface/skills/blob/main/examples/marketplace.json"

    assert format_marketplace_display_url(raw_dev) == raw_dev
    assert format_marketplace_display_url(nested) == nested


def test_resolve_registry_urls_dedupes_only_equivalent_sources() -> None:
    resolved = resolve_registry_urls(
        [
            "https://github.com/huggingface/skills/blob/main/marketplace.json",
            "https://github.com/huggingface/skills/tree/main/.claude-plugin",
            "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
            "https://github.com/anthropics/skills",
        ],
        default_urls=["https://github.com/fast-agent-ai/skills"],
    )

    assert resolved == [
        "https://github.com/huggingface/skills/blob/main/marketplace.json",
        "https://github.com/anthropics/skills",
    ]


def test_resolve_registry_urls_dedupes_default_blob_and_tree_registry_paths() -> None:
    resolved = resolve_registry_urls(
        [
            "https://github.com/huggingface/skills/blob/main/marketplace.json",
            "https://github.com/huggingface/skills/tree/main/.claude-plugin",
            "https://github.com/huggingface/skills/blob/master/.claude-plugin/marketplace.json",
            "https://github.com/huggingface/skills",
        ],
        default_urls=[],
    )

    assert resolved == [
        "https://github.com/huggingface/skills/blob/main/marketplace.json",
        "https://github.com/huggingface/skills/blob/master/.claude-plugin/marketplace.json",
        "https://github.com/huggingface/skills",
    ]


def test_resolve_registry_urls_dedupes_mixed_case_github_hosts() -> None:
    resolved = resolve_registry_urls(
        [
            "https://GitHub.com/huggingface/skills",
            "https://github.com/huggingface/skills",
        ],
        default_urls=[],
    )

    assert resolved == ["https://GitHub.com/huggingface/skills"]


def test_resolve_registry_urls_preserves_configured_main_and_master_registries() -> None:
    resolved = resolve_registry_urls(
        [
            "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
            "https://raw.githubusercontent.com/huggingface/skills/master/marketplace.json",
        ],
        default_urls=[],
    )

    assert resolved == [
        "https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
        "https://raw.githubusercontent.com/huggingface/skills/master/marketplace.json",
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


def test_resolve_registry_urls_strips_and_skips_blank_entries() -> None:
    resolved = resolve_registry_urls(
        [
            " https://github.com/huggingface/skills ",
            " ",
        ],
        default_urls=["https://github.com/fast-agent-ai/skills"],
        active_url=" https://github.com/anthropics/skills ",
    )

    assert resolved == [
        "https://github.com/huggingface/skills",
        "https://github.com/anthropics/skills",
    ]


def test_resolve_skill_registries_dedupes_equivalent_active_source() -> None:
    settings = Settings(
        skills=SkillsSettings(
            marketplace_urls=list(DEFAULT_SKILL_REGISTRIES),
            marketplace_url="https://raw.githubusercontent.com/huggingface/skills/main/marketplace.json",
        )
    )

    resolved = resolve_skill_registries(settings)

    assert resolved == list(DEFAULT_SKILL_REGISTRIES)


def test_resolve_card_registries_dedupes_equivalent_active_source() -> None:
    settings = Settings(
        cards=CardsSettings(
            marketplace_urls=list(DEFAULT_CARD_REGISTRIES),
            marketplace_url="https://raw.githubusercontent.com/fast-agent-ai/card-packs/main/marketplace.json",
        )
    )

    resolved = resolve_card_registries(settings)

    assert resolved == list(DEFAULT_CARD_REGISTRIES)
