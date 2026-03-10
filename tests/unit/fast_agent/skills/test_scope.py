from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.config import SkillsSettings, get_settings
from fast_agent.skills.scope import (
    order_skill_directories_for_display,
    resolve_skills_management_scope,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_skills_management_scope_keeps_override_visible(tmp_path: Path) -> None:
    configured_dir = (tmp_path / "configured").resolve()
    override_dir = (tmp_path / "override").resolve()
    settings = get_settings().model_copy(
        update={
            "environment_dir": str(tmp_path / ".fast-agent"),
            "skills": SkillsSettings(directories=[str(configured_dir)]),
        }
    )

    scope = resolve_skills_management_scope(
        settings,
        cwd=tmp_path,
        managed_directory_override=override_dir,
    )

    assert scope.management_source == "override"
    assert scope.managed_directory == override_dir
    assert scope.discovered_directories == [configured_dir, override_dir]


def test_order_skill_directories_for_display_puts_managed_directory_last(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"environment_dir": str(tmp_path / ".fast-agent")})
    managed_dir = (tmp_path / ".fast-agent" / "skills").resolve()
    agents_dir = (tmp_path / ".agents" / "skills").resolve()
    claude_dir = (tmp_path / ".claude" / "skills").resolve()

    ordered = order_skill_directories_for_display(
        [managed_dir, agents_dir, claude_dir],
        settings=settings,
        cwd=tmp_path,
    )

    assert ordered == [agents_dir, claude_dir, managed_dir]
