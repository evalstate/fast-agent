from pathlib import Path

from fast_agent.core.fastagent import FastAgent
from fast_agent.skills import SkillManifest


def _manifest(name: str, path: Path) -> SkillManifest:
    return SkillManifest(name=name, description="", body="", path=path)


def test_deduplicate_skills_normalizes_name_case_and_padding(tmp_path: Path) -> None:
    first = _manifest(" Alpha ", tmp_path / "first" / "SKILL.md")
    duplicate = _manifest("alpha", tmp_path / "second" / "SKILL.md")
    distinct = _manifest("beta", tmp_path / "third" / "SKILL.md")

    deduplicated = FastAgent._deduplicate_skills([first, duplicate, distinct])

    assert deduplicated == [first, distinct]
