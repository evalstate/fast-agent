"""Tests for URI-backed manifests in format_skills_for_prompt."""

from pathlib import Path

from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt


def _mcp_manifest(name: str = "git-workflow") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name="test-server",
    )


def _fs_manifest(tmp_path: Path, name: str = "git-workflow") -> SkillManifest:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = skill_dir / "SKILL.md"
    manifest_path.write_text(
        f"---\nname: {name}\ndescription: The {name} skill\n---\nBody\n",
        encoding="utf-8",
    )
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="Body",
        path=manifest_path,
    )


def test_mcp_manifest_emits_uri_location() -> None:
    output = format_skills_for_prompt([_mcp_manifest()], include_preamble=False)
    assert "<location>skill://git-workflow/SKILL.md</location>" in output
    assert "<directory>skill://git-workflow</directory>" in output
    # MCP manifests must NOT render filesystem subdir tags
    assert "<scripts>" not in output
    assert "<references>" not in output
    assert "<assets>" not in output


def test_filesystem_manifest_regression(tmp_path: Path) -> None:
    """Filesystem manifests render exactly as before (regression guard)."""
    manifest = _fs_manifest(tmp_path)
    output = format_skills_for_prompt([manifest], include_preamble=False)
    assert f"<location>{manifest.path}</location>" in output
    assert f"<directory>{manifest.path.parent}</directory>" in output
    assert "skill://" not in output


def test_mixed_manifests_preamble_mentions_mcp(tmp_path: Path) -> None:
    output = format_skills_for_prompt(
        [_fs_manifest(tmp_path), _mcp_manifest()],
        include_preamble=True,
    )
    # Preamble note should mention URIs scheme-agnostically per the SEP
    # (skill:// is SHOULD, not MUST; github:// / repo:// also valid).
    assert "URI" in output
    assert "skill://" in output
    assert "<location>skill://git-workflow/SKILL.md</location>" in output


def test_preamble_omits_mcp_note_when_no_uri_skills(tmp_path: Path) -> None:
    output = format_skills_for_prompt(
        [_fs_manifest(tmp_path)],
        include_preamble=True,
    )
    # MCP-specific relative-path guidance only appears when relevant.
    assert "skill://acme" not in output
