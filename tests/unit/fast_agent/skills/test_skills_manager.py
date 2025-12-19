"""Tests for the SkillsManager module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fast_agent.config import SkillsSettings
from fast_agent.skills.skills_manager import (
    Marketplace,
    MarketplaceSkill,
    SkillsManager,
)


def write_skill(
    directory: Path, name: str, description: str = "desc", body: str = "Body"
) -> Path:
    """Create a skill manifest for testing."""
    skill_dir = directory / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        f"""---
name: {name}
description: {description}
---
{body}
""",
        encoding="utf-8",
    )
    return manifest


SAMPLE_MARKETPLACE_JSON = {
    "name": "test-skills",
    "owner": {"name": "Test Owner"},
    "metadata": {"description": "Test skills", "version": "1.0.0"},
    "plugins": [
        {
            "name": "skill-one",
            "source": "./skill_one",
            "description": "First test skill",
        },
        {
            "name": "skill-two",
            "source": "./skill_two",
            "description": "Second test skill",
        },
    ],
}


class TestMarketplaceSkill:
    def test_from_dict(self):
        data = {
            "name": "test-skill",
            "source": "./test",
            "description": "A test skill",
        }
        skill = MarketplaceSkill.from_dict(data)
        assert skill.name == "test-skill"
        assert skill.source == "./test"
        assert skill.description == "A test skill"

    def test_from_dict_missing_fields(self):
        data = {}
        skill = MarketplaceSkill.from_dict(data)
        assert skill.name == ""
        assert skill.source == ""
        assert skill.description == ""


class TestMarketplace:
    def test_from_dict(self):
        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)
        assert marketplace.name == "test-skills"
        assert marketplace.owner == "Test Owner"
        assert marketplace.version == "1.0.0"
        assert len(marketplace.skills) == 2
        assert marketplace.skills[0].name == "skill-one"

    def test_from_dict_minimal(self):
        data = {"plugins": []}
        marketplace = Marketplace.from_dict(data)
        assert marketplace.name == "unknown"
        assert marketplace.owner == "unknown"
        assert marketplace.version == "1.0.0"
        assert marketplace.skills == []


class TestSkillsManager:
    def test_init_with_settings(self, tmp_path: Path):
        settings = SkillsSettings(
            marketplace_url="https://example.com/marketplace.json",
            install_directory=".custom/skills",
        )
        manager = SkillsManager(settings=settings, base_dir=tmp_path)
        assert manager._marketplace_url == "https://example.com/marketplace.json"
        assert manager._install_dir == (tmp_path / ".custom/skills").resolve()

    def test_init_defaults(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        assert "huggingface" in manager._marketplace_url
        assert manager._install_dir == (tmp_path / ".fast-agent/skills").resolve()

    def test_get_installed_skills_empty(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        skills = manager.get_installed_skills()
        assert skills == []

    def test_get_installed_skills(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "test-skill", "Test description")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)
        skills = manager.get_installed_skills()

        assert len(skills) == 1
        assert skills[0].name == "test-skill"
        assert skills[0].description == "Test description"

    def test_get_installed_skill_names(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "Alpha-Skill")
        write_skill(install_dir, "beta-skill")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)
        names = manager.get_installed_skill_names()

        assert names == {"alpha-skill", "beta-skill"}

    def test_format_installed_skills_empty(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        formatted = manager.format_installed_skills()
        assert "No skills installed" in formatted
        assert "/skills add" in formatted

    def test_format_installed_skills(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "test-skill", "A great skill")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)
        formatted = manager.format_installed_skills()

        assert "# Installed Skills" in formatted
        assert "test-skill" in formatted
        assert "A great skill" in formatted
        assert "/skills remove" in formatted

    def test_format_marketplace_skills(self, tmp_path: Path):
        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)
        manager = SkillsManager(base_dir=tmp_path)
        formatted = manager.format_marketplace_skills(marketplace)

        assert "test-skills" in formatted
        assert "Test Owner" in formatted
        assert "skill-one" in formatted
        assert "skill-two" in formatted
        assert "First test skill" in formatted

    def test_format_marketplace_shows_installed(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "skill-one")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)
        manager = SkillsManager(settings=settings, base_dir=tmp_path)
        formatted = manager.format_marketplace_skills(marketplace)

        assert "(installed)" in formatted

    def test_parse_github_url(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        url = "https://raw.githubusercontent.com/owner/repo/main/path/file.json"
        owner, repo, branch = manager._parse_github_url(url)
        assert owner == "owner"
        assert repo == "repo"
        assert branch == "main"

    def test_parse_github_url_invalid(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        with pytest.raises(ValueError, match="Cannot parse GitHub URL"):
            manager._parse_github_url("https://example.com/short")


class TestSkillsManagerRemove:
    def test_remove_skill_not_installed(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        success, message = manager.remove_skill("nonexistent")
        assert not success
        assert "No skills installed" in message

    def test_remove_skill_by_name(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "test-skill")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        # Verify skill exists
        assert len(manager.get_installed_skills()) == 1

        success, message = manager.remove_skill("test-skill")
        assert success
        assert "Successfully removed" in message

        # Verify skill was removed
        assert len(manager.get_installed_skills()) == 0
        assert not (install_dir / "test-skill").exists()

    def test_remove_skill_by_number(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "skill-one")
        write_skill(install_dir, "skill-two")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        success, message = manager.remove_skill("1")
        assert success

        skills = manager.get_installed_skills()
        assert len(skills) == 1

    def test_remove_skill_invalid_number(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "test-skill")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        success, message = manager.remove_skill("99")
        assert not success
        assert "Invalid skill number" in message

    def test_remove_skill_not_found_by_name(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "test-skill")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        success, message = manager.remove_skill("other-skill")
        assert not success
        assert "not found" in message


class TestSkillsManagerFetch:
    @pytest.mark.asyncio
    async def test_fetch_marketplace(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_MARKETPLACE_JSON
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            marketplace = await manager.fetch_marketplace()

        assert marketplace.name == "test-skills"
        assert len(marketplace.skills) == 2

    @pytest.mark.asyncio
    async def test_fetch_marketplace_caches(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_MARKETPLACE_JSON
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await manager.fetch_marketplace()
            await manager.fetch_marketplace()  # Should use cache

        # Only called once due to caching
        assert mock_get.call_count == 1

    @pytest.mark.asyncio
    async def test_fetch_marketplace_force_refresh(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)

        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_MARKETPLACE_JSON
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            await manager.fetch_marketplace()
            await manager.fetch_marketplace(force_refresh=True)

        # Called twice due to force refresh
        assert mock_get.call_count == 2


class TestSkillsManagerInstall:
    @pytest.mark.asyncio
    async def test_install_skill_by_name(self, tmp_path: Path):
        settings = SkillsSettings(
            marketplace_url="https://raw.githubusercontent.com/test/skills/main/marketplace.json",
            install_directory=".fast-agent/skills",
        )
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)

        # Mock the GitHub API responses
        mock_dir_response = MagicMock()
        mock_dir_response.json.return_value = [{"type": "dir", "name": "skill-folder"}]
        mock_dir_response.raise_for_status = MagicMock()

        mock_files_response = MagicMock()
        mock_files_response.json.return_value = [
            {"type": "file", "name": "SKILL.md", "download_url": "https://example.com/SKILL.md"}
        ]
        mock_files_response.raise_for_status = MagicMock()

        mock_content_response = MagicMock()
        mock_content_response.content = b"""---
name: skill-one
description: Test skill
---
Body content
"""
        mock_content_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(
                side_effect=[mock_dir_response, mock_files_response, mock_content_response]
            )
            mock_client.return_value.__aenter__.return_value.get = mock_get

            success, message = await manager.install_skill("skill-one", marketplace)

        assert success
        assert "Successfully installed" in message

        # Verify the skill was installed
        installed = manager.get_installed_skills()
        assert len(installed) == 1
        assert installed[0].name == "skill-one"

    @pytest.mark.asyncio
    async def test_install_skill_by_number(self, tmp_path: Path):
        settings = SkillsSettings(
            marketplace_url="https://raw.githubusercontent.com/test/skills/main/marketplace.json",
            install_directory=".fast-agent/skills",
        )
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)

        mock_dir_response = MagicMock()
        mock_dir_response.json.return_value = [{"type": "dir", "name": "skill-folder"}]
        mock_dir_response.raise_for_status = MagicMock()

        mock_files_response = MagicMock()
        mock_files_response.json.return_value = [
            {"type": "file", "name": "SKILL.md", "download_url": "https://example.com/SKILL.md"}
        ]
        mock_files_response.raise_for_status = MagicMock()

        mock_content_response = MagicMock()
        mock_content_response.content = b"""---
name: skill-two
description: Second skill
---
Body
"""
        mock_content_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_get = AsyncMock(
                side_effect=[mock_dir_response, mock_files_response, mock_content_response]
            )
            mock_client.return_value.__aenter__.return_value.get = mock_get

            success, message = await manager.install_skill("2", marketplace)

        assert success
        assert "skill-two" in message

    @pytest.mark.asyncio
    async def test_install_skill_already_installed(self, tmp_path: Path):
        install_dir = tmp_path / ".fast-agent" / "skills"
        write_skill(install_dir, "skill-one")

        settings = SkillsSettings(install_directory=".fast-agent/skills")
        manager = SkillsManager(settings=settings, base_dir=tmp_path)

        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)
        success, message = await manager.install_skill("skill-one", marketplace)

        assert not success
        assert "already installed" in message

    @pytest.mark.asyncio
    async def test_install_skill_invalid_number(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)

        success, message = await manager.install_skill("99", marketplace)
        assert not success
        assert "Invalid skill number" in message

    @pytest.mark.asyncio
    async def test_install_skill_not_found(self, tmp_path: Path):
        manager = SkillsManager(base_dir=tmp_path)
        marketplace = Marketplace.from_dict(SAMPLE_MARKETPLACE_JSON)

        success, message = await manager.install_skill("nonexistent", marketplace)
        assert not success
        assert "not found" in message
