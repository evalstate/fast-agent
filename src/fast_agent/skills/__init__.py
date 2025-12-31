"""Skill discovery utilities."""

from .registry import SkillManifest, SkillRegistry, format_skills_for_prompt


class SkillsDefault:
    """Sentinel for default skills resolution (use context/global skills)."""

    def __repr__(self) -> str:
        return "SKILLS_DEFAULT"


SKILLS_DEFAULT = SkillsDefault()

__all__ = [
    "SKILLS_DEFAULT",
    "SkillsDefault",
    "SkillManifest",
    "SkillRegistry",
    "format_skills_for_prompt",
]
