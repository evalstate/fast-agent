from __future__ import annotations

import pytest

from fast_agent.cards import manager
from fast_agent.config import CardsSettings, Settings


def test_get_marketplace_url_uses_first_configured_card_registry() -> None:
    settings = Settings(
        cards=CardsSettings(
            marketplace_urls=[
                "https://example.com/registry-a.json",
                "https://example.com/registry-b.json",
            ]
        )
    )

    assert manager.get_marketplace_url(settings) == "https://example.com/registry-a.json"


def test_get_marketplace_url_prefers_active_card_registry() -> None:
    settings = Settings(
        cards=CardsSettings(
            marketplace_url="https://example.com/active.json",
            marketplace_urls=["https://example.com/registry-a.json"],
        )
    )

    assert manager.get_marketplace_url(settings) == "https://example.com/active.json"


def test_resolve_card_registries_uses_typed_card_settings() -> None:
    settings = Settings(
        cards=CardsSettings(
            marketplace_url="https://example.com/active.json",
            marketplace_urls=["https://example.com/registry-a.json"],
        )
    )

    assert manager.resolve_card_registries(settings) == [
        "https://example.com/registry-a.json",
        "https://example.com/active.json",
    ]


def test_candidate_marketplace_urls_for_github_repo() -> None:
    urls = manager.candidate_marketplace_urls("https://github.com/example/card-packs")
    assert urls == [
        "https://raw.githubusercontent.com/example/card-packs/main/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/card-packs/main/marketplace.json",
        "https://raw.githubusercontent.com/example/card-packs/master/.claude-plugin/marketplace.json",
        "https://raw.githubusercontent.com/example/card-packs/master/marketplace.json",
    ]


def test_parse_marketplace_payload_normalizes_entries() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "description": "Alpha pack",
                "kind": "bundle",
                "repo_url": "https://github.com/example/cards",
                "repo_ref": "main",
                "repo_path": "packs/alpha",
            },
            {
                "name": "beta",
                "repo": "https://github.com/example/cards",
                "path": "packs/beta",
            },
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload, source_url="https://example.com/marketplace.json"
    )
    assert len(packs) == 2

    first = packs[0]
    assert first.name == "alpha"
    assert first.kind == "bundle"
    assert first.repo_path == "packs/alpha"

    second = packs[1]
    assert second.name == "beta"
    assert second.kind == "card"
    assert second.repo_path == "packs/beta"


def test_parse_marketplace_payload_normalizes_card_pack_kind() -> None:
    packs = manager._parse_marketplace_payload(
        {
            "entries": [
                {
                    "name": "alpha",
                    "kind": " BUNDLE ",
                    "repo_url": "https://github.com/example/cards",
                    "repo_path": "packs/alpha",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(packs) == 1
    assert packs[0].kind == "bundle"


def test_parse_installed_card_pack_source_normalizes_kind() -> None:
    source = manager._parse_installed_card_pack_source(
        {
            "schema_version": 1,
            "installed_via": "marketplace",
            "source_origin": "remote",
            "name": "alpha",
            "kind": " CARD ",
            "repo_url": "https://github.com/example/cards",
            "repo_path": "packs/alpha",
            "installed_revision": "abc123",
            "installed_at": "2026-02-25T00:00:00Z",
            "content_fingerprint": f"sha256:{'0' * 64}",
            "installed_files": ["agent-cards/alpha.md"],
        }
    )

    assert source.kind == "card"


def test_parse_marketplace_payload_names_normalized_manifest_path_from_parent_dir() -> None:
    payload = {
        "entries": [
            {
                "kind": "bundle",
                "repo_url": "https://github.com/example/cards",
                "repo_path": r"packs\alpha\card-pack.yaml",
            }
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload, source_url="https://example.com/marketplace.json"
    )

    assert len(packs) == 1
    assert packs[0].name == "alpha"
    assert packs[0].repo_path == "packs/alpha/card-pack.yaml"
    assert packs[0].repo_subdir == "packs/alpha"


def test_parse_marketplace_payload_does_not_hide_entry_conversion_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_entry(_entry: object) -> None:
        raise RuntimeError("entry conversion failed")

    monkeypatch.setattr(manager, "_card_pack_from_entry_model", fail_entry)

    with pytest.raises(RuntimeError, match="entry conversion failed"):
        manager._parse_marketplace_payload(
            {
                "entries": [
                    {
                        "repo_url": "https://github.com/example/cards",
                        "repo_path": "packs/alpha",
                    }
                ]
            }
        )


def test_parse_marketplace_payload_treats_scp_source_url_as_repo_url() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "kind": "bundle",
                "url": "git@github.com:example/cards.git",
                "repo_path": "packs/alpha",
            }
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload, source_url="https://example.com/marketplace.json"
    )

    assert len(packs) == 1
    assert packs[0].repo_url == "git@github.com:example/cards.git"
    assert packs[0].repo_path == "packs/alpha"


def test_parse_marketplace_payload_treats_local_source_url_as_repo_url() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "kind": "bundle",
                "source_url": "/tmp/card-packs",
                "repo_path": "packs/alpha",
            }
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload,
        source_url="https://example.com/marketplace.json",
    )

    assert len(packs) == 1
    assert packs[0].repo_url == "/tmp/card-packs"
    assert packs[0].repo_path == "packs/alpha"
    assert packs[0].source_url == "/tmp/card-packs"


def test_parse_marketplace_payload_does_not_treat_registry_url_as_repo_url() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "kind": "bundle",
                "repo_path": "packs/alpha",
            }
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload,
        source_url="https://example.com/marketplace.json",
    )

    assert packs == []


def test_parse_marketplace_payload_does_not_parse_github_registry_url_as_pack_path() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "kind": "bundle",
                "repo_url": "https://github.com/example/cards",
            }
        ]
    }

    packs = manager._parse_marketplace_payload(
        payload,
        source_url=(
            "https://raw.githubusercontent.com/example/cards/main/.claude-plugin/marketplace.json"
        ),
    )

    assert packs == []


def test_select_card_pack_by_name_or_index() -> None:
    entries = [
        manager.MarketplaceCardPack(
            name="alpha",
            description=None,
            kind="card",
            repo_url="https://example.com/a.git",
            repo_ref=None,
            repo_path="packs/alpha",
        ),
        manager.MarketplaceCardPack(
            name="beta",
            description=None,
            kind="bundle",
            repo_url="https://example.com/b.git",
            repo_ref=None,
            repo_path="packs/beta",
        ),
    ]

    assert manager.select_card_pack_by_name_or_index(entries, "1") == entries[0]
    assert manager.select_card_pack_by_name_or_index(entries, "beta") == entries[1]
    assert manager.select_card_pack_by_name_or_index(entries, "missing") is None
