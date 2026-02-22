from __future__ import annotations

from fast_agent.cards import manager


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

    packs = manager._parse_marketplace_payload(payload, source_url="https://example.com/marketplace.json")
    assert len(packs) == 2

    first = packs[0]
    assert first.name == "alpha"
    assert first.kind == "bundle"
    assert first.repo_path == "packs/alpha"

    second = packs[1]
    assert second.name == "beta"
    assert second.kind == "card"
    assert second.repo_path == "packs/beta"


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
