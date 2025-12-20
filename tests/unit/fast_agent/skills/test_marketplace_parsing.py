from fast_agent.skills.manager import _parse_marketplace_payload


def test_parse_plugin_marketplace_source_path() -> None:
    payload = {
        "name": "test-market",
        "metadata": {"pluginRoot": "./plugins"},
        "plugins": [
            {
                "name": "alpha",
                "description": "Alpha skill",
                "source": "./alpha",
            }
        ],
    }
    skills = _parse_marketplace_payload(
        payload,
        source_url="https://raw.githubusercontent.com/org/repo/main/.claude-plugin/marketplace.json",
    )
    assert len(skills) == 1
    skill = skills[0]
    assert skill.repo_url == "https://github.com/org/repo"
    assert skill.repo_ref == "main"
    assert skill.repo_path == "plugins/alpha"


def test_parse_plugin_marketplace_skills_list() -> None:
    payload = {
        "name": "test-market",
        "plugins": [
            {
                "name": "bundle",
                "description": "Bundle skills",
                "source": "./",
                "strict": False,
                "skills": [
                    "./skills/xlsx",
                    "./skills/pdf",
                ],
            }
        ],
    }
    skills = _parse_marketplace_payload(
        payload,
        source_url="https://raw.githubusercontent.com/org/repo/main/.claude-plugin/marketplace.json",
    )
    assert [skill.repo_path for skill in skills] == ["skills/xlsx", "skills/pdf"]
    assert [skill.name for skill in skills] == ["xlsx", "pdf"]


def test_resolve_skill_source_dir_prefers_named_skill(tmp_path) -> None:
    from fast_agent.skills.manager import _resolve_skill_source_dir

    source_dir = tmp_path / "plugin"
    named_dir = source_dir / "skills" / "model-trainer"
    named_dir.mkdir(parents=True)
    (named_dir / "SKILL.md").write_text("skill", encoding="utf-8")

    resolved = _resolve_skill_source_dir(source_dir, "model-trainer")
    assert resolved == named_dir
