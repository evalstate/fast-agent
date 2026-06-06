from fast_agent.plugins.marketplace import parse_marketplace_plugins


def test_parse_plugin_marketplace_ignores_card_pack_only_entries() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": f"ignored-{kind}",
                    "kind": kind,
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": f"packs/{kind}",
                }
                for kind in ("card", "card_pack", "card-pack", "bundle")
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


def test_parse_plugin_marketplace_does_not_treat_registry_url_as_repo_url() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "finder",
                    "kind": "plugin",
                    "repo_path": "plugins/finder",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert plugins == []


def test_parse_plugin_marketplace_names_manifest_file_path_from_parent_dir() -> None:
    plugins = parse_marketplace_plugins(
        {
            "command_plugins": [
                {
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": "plugins/finder/plugin.yaml",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(plugins) == 1
    assert plugins[0].name == "finder"
    assert plugins[0].install_dir_name == "finder"


def test_parse_plugin_marketplace_names_normalized_manifest_path_from_parent_dir() -> None:
    plugins = parse_marketplace_plugins(
        {
            "command_plugins": [
                {
                    "repo_url": "https://github.com/example/card-packs",
                    "repo_path": r"plugins\finder\plugin.yaml",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(plugins) == 1
    assert plugins[0].name == "finder"
    assert plugins[0].install_dir_name == "finder"
    assert plugins[0].repo_path == "plugins/finder/plugin.yaml"


def test_parse_plugin_marketplace_treats_relative_source_as_repo_path() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "finder",
                    "repo_url": "https://github.com/example/card-packs",
                    "source": "plugins/finder",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(plugins) == 1
    assert plugins[0].repo_url == "https://github.com/example/card-packs"
    assert plugins[0].repo_path == "plugins/finder"
    assert plugins[0].source_url == "https://example.com/marketplace.json"


def test_parse_plugin_marketplace_treats_scp_source_as_repo_url() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "finder",
                    "source": "git@github.com:example/card-packs.git",
                    "repo_path": "plugins/finder",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(plugins) == 1
    assert plugins[0].repo_url == "git@github.com:example/card-packs.git"
    assert plugins[0].repo_path == "plugins/finder"


def test_parse_plugin_marketplace_treats_local_source_url_as_repo_url() -> None:
    plugins = parse_marketplace_plugins(
        {
            "entries": [
                {
                    "name": "finder",
                    "source_url": "/tmp/card-packs",
                    "repo_path": "plugins/finder",
                }
            ]
        },
        source_url="https://example.com/marketplace.json",
    )

    assert len(plugins) == 1
    assert plugins[0].repo_url == "/tmp/card-packs"
    assert plugins[0].repo_path == "plugins/finder"
    assert plugins[0].source_url == "/tmp/card-packs"
