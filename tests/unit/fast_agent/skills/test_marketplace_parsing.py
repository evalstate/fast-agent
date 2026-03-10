from __future__ import annotations

from fast_agent.skills.marketplace_parsing import parse_marketplace_payload


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
                    "path": "bundle-root",
                },
                "skills": ["alpha", "nested/beta"],
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
