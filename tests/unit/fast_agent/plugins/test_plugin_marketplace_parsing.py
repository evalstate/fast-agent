from fast_agent.plugins.marketplace import parse_marketplace_plugins


def test_parse_plugin_marketplace_ignores_card_pack_only_entries() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "hf-codemode",
                    "kind": "card",
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": "packs/hf-codemode",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert plugins == []


def test_parse_plugin_marketplace_reads_command_plugins() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "hf-codemode",
                    "kind": "card",
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": "packs/hf-codemode",
                }
            ],
            "command_plugins": [
                {
                    "name": "finder",
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": "plugins/finder",
                }
            ],
        },
        source_url="https://example.com/marketplace.json",
    )

    assert [plugin.name for plugin in plugins] == ["finder"]


def test_parse_plugin_marketplace_reads_generic_plugin_entries() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "finder",
                    "kind": "plugin",
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": "plugins/finder",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert [plugin.name for plugin in plugins] == ["finder"]
    assert plugins[0].repo_path == "plugins/finder"
