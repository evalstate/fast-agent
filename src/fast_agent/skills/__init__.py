"""Skill discovery and management utilities."""

from .manager import (
    MarketplaceInfo,
    MarketplaceSkill,
    fetch_marketplace,
    format_installed_skills,
    format_marketplace_skills,
    install_skill,
    list_installed_skills,
    remove_skill,
)
from .registry import SkillManifest, SkillRegistry, format_skills_for_prompt

__all__ = [
    # Registry
    "SkillManifest",
    "SkillRegistry",
    "format_skills_for_prompt",
    # Manager
    "MarketplaceInfo",
    "MarketplaceSkill",
    "fetch_marketplace",
    "format_installed_skills",
    "format_marketplace_skills",
    "install_skill",
    "list_installed_skills",
    "remove_skill",
]
