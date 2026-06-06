from __future__ import annotations

import pytest

from fast_agent.skills.marketplace_parsing import parse_marketplace_payload


def test_parse_marketplace_payload_derives_fallback_name_from_repo_path_leaf() -> None:
    payload = {
        "entries": [
            {
                "repo_url": "https://github.com/example/skills",
                "repo_path": "skills/alpha",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].name == "alpha"


def test_parse_marketplace_payload_returns_empty_for_invalid_payload_shape() -> None:
    assert parse_marketplace_payload("not a marketplace") == []


def test_parse_marketplace_payload_does_not_hide_entry_conversion_failures(
    monkeypatch,
) -> None:
    import fast_agent.skills.marketplace_parsing as parser

    def fail_entry(_entry: object) -> None:
        raise RuntimeError("entry conversion failed")

    monkeypatch.setattr(parser, "_skill_from_entry_model", fail_entry)

    with pytest.raises(RuntimeError, match="entry conversion failed"):
        parse_marketplace_payload(
            {
                "entries": [
                    {
                        "repo_url": "https://github.com/example/skills",
                        "repo_path": "skills/alpha",
                    }
                ]
            }
        )


def test_parse_marketplace_payload_derives_fallback_name_from_manifest_parent() -> None:
    payload = {
        "entries": [
            {
                "repo_url": "https://github.com/example/skills",
                "repo_path": r"skills\alpha\SKILL.md",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].name == "alpha"
    assert skills[0].repo_path == "skills/alpha/SKILL.md"


def test_parse_marketplace_payload_expands_plugin_bundle_entries() -> None:
    payload = {
        "metadata": {"pluginRoot": "bundles"},
        "plugins": [
            {
                "name": "Useful Bundle",
                "description": "Helpful tools",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "ref": "main",
                    "path": "  bundle-root  ",
                },
                "skills": ["  alpha  ", " nested/beta "],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert [skill.name for skill in skills] == ["alpha", "beta"]
    assert [skill.repo_path for skill in skills] == [
        "bundles/bundle-root/alpha",
        "bundles/bundle-root/nested/beta",
    ]
    assert all(skill.repo_url == "https://github.com/example/skills" for skill in skills)
    assert all(skill.bundle_name == "Useful Bundle" for skill in skills)


def test_parse_marketplace_payload_preserves_explicit_skills_with_plugins() -> None:
    payload = {
        "skills": [
            {
                "name": "explicit",
                "repo_url": "https://github.com/example/skills",
                "repo_path": "skills/explicit",
            }
        ],
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "bundle-root",
                },
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert [skill.name for skill in skills] == ["explicit", "alpha"]
    assert [skill.repo_path for skill in skills] == ["skills/explicit", "bundle-root/alpha"]


def test_parse_marketplace_payload_names_manifest_file_skills_from_parent_dir() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "bundle-root",
                },
                "skills": ["alpha/SKILL.md", "nested/beta/skill.md"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert [skill.name for skill in skills] == ["alpha", "beta"]
    assert [skill.repo_path for skill in skills] == [
        "bundle-root/alpha/SKILL.md",
        "bundle-root/nested/beta/skill.md",
    ]


def test_parse_marketplace_payload_names_backslash_manifest_paths_from_parent_dir() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "bundle-root",
                },
                "skills": [r"nested\beta\SKILL.md"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].name == "beta"
    assert skills[0].repo_path == "bundle-root/nested/beta/SKILL.md"


def test_parse_marketplace_payload_names_current_dir_skill_from_source_path() -> None:
    payload = {
        "plugins": [
            {
                "name": "alpha",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "skills/alpha",
                },
                "skills": ["./"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].name == "alpha"
    assert skills[0].repo_path == "skills/alpha"


def test_parse_marketplace_payload_keeps_full_github_repo_url_for_plugin_bundle() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": {
                    "source": "github",
                    "repo": "https://github.com/example/skills",
                    "path": "bundle-root",
                },
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_url == "https://github.com/example/skills"
    assert skills[0].repo_path == "bundle-root/alpha"


def test_parse_marketplace_payload_trims_full_github_repo_url_for_plugin_bundle() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": {
                    "source": "github",
                    "repo": "  https://github.com/example/skills  ",
                    "path": "bundle-root",
                },
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_url == "https://github.com/example/skills"


def test_parse_marketplace_payload_treats_scp_plugin_source_as_repo_url() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": "git@github.com:example/skills.git",
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_url == "git@github.com:example/skills.git"
    assert skills[0].repo_path == "alpha"


def test_parse_marketplace_payload_treats_local_source_url_as_repo_url() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "source_url": "/tmp/skills",
                "repo_path": "skills/alpha",
            }
        ],
    }

    skills = parse_marketplace_payload(
        payload,
        source_url="https://example.com/marketplace.json",
    )

    assert len(skills) == 1
    assert skills[0].repo_url == "/tmp/skills"
    assert skills[0].repo_path == "skills/alpha"
    assert skills[0].source_url == "/tmp/skills"


def test_parse_marketplace_payload_preserves_marketplace_source_url() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "repo_url": "https://github.com/example/skills",
                "repo_path": "skills/alpha",
            }
        ],
    }

    skills = parse_marketplace_payload(
        payload,
        source_url="https://example.com/marketplace.json",
    )

    assert len(skills) == 1
    assert skills[0].repo_url == "https://github.com/example/skills"
    assert skills[0].source_url == "https://example.com/marketplace.json"


def test_parse_marketplace_payload_expands_github_tree_plugin_source() -> None:
    payload = {
        "plugins": [
            {
                "name": "Useful Bundle",
                "source": "https://github.com/org/repo/tree/main/plugins/app",
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_url == "https://github.com/org/repo"
    assert skills[0].repo_ref == "main"
    assert skills[0].repo_path == "plugins/app/alpha"


def test_parse_marketplace_payload_rejects_traversal_plugin_root() -> None:
    payload = {
        "metadata": {"pluginRoot": "../outside"},
        "plugins": [
            {
                "name": "Unsafe Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "bundle-root",
                },
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert skills == []


def test_parse_marketplace_payload_rejects_traversal_source_path() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "repo": "https://github.com/example/skills",
                "source": "../outside",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert skills == []


def test_parse_marketplace_payload_normalizes_source_path_with_shared_rules() -> None:
    payload = {
        "entries": [
            {
                "name": "alpha",
                "repo": "https://github.com/example/skills",
                "source": "./bundles\\alpha",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_path == "bundles/alpha/skills/alpha"


def test_parse_marketplace_payload_keeps_root_skills_source_path() -> None:
    payload = {
        "entries": [
            {
                "name": "foo",
                "repo": "https://github.com/example/skills",
                "source": "skills/foo",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_path == "skills/foo"


def test_parse_marketplace_payload_keeps_nested_skills_source_path() -> None:
    payload = {
        "entries": [
            {
                "name": "foo",
                "repo": "https://github.com/example/skills",
                "source": "examples/skills/foo",
            }
        ]
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].repo_path == "examples/skills/foo"


def test_parse_marketplace_payload_rejects_traversal_bundle_skill_path() -> None:
    payload = {
        "plugins": [
            {
                "name": "Unsafe Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "bundle-root",
                },
                "skills": ["../alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert skills == []


def test_parse_marketplace_payload_rejects_traversal_plugin_source_path() -> None:
    payload = {
        "plugins": [
            {
                "name": "Unsafe Bundle",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "../outside",
                },
                "skills": ["alpha"],
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert skills == []


def test_parse_marketplace_payload_does_not_label_unexpanded_plugin_as_bundle() -> None:
    payload = {
        "plugins": [
            {
                "name": "session-investigator",
                "description": "Investigate sessions",
                "source": {
                    "source": "github",
                    "repo": "example/skills",
                    "path": "skills/session-investigator",
                },
            }
        ],
    }

    skills = parse_marketplace_payload(payload)

    assert len(skills) == 1
    assert skills[0].name == "session-investigator"
    assert skills[0].bundle_name is None
