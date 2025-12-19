"""
Skills Manager - Manages skill installation from marketplaces.

This module provides functionality to:
- Fetch and display available skills from marketplace URLs
- Install skills from GitHub repositories
- Remove installed skills
- List currently installed skills

Usage via slash commands:
- /skills - List installed skills
- /skills add - Show marketplace skills and install
- /skills remove <name|number> - Remove an installed skill
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
from urllib.parse import urlparse

import httpx

from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    from fast_agent.config import SkillsSettings

logger = get_logger(__name__)


@dataclass
class MarketplaceSkill:
    """Represents a skill available in a marketplace."""

    name: str
    source: str
    description: str

    @classmethod
    def from_dict(cls, data: dict) -> "MarketplaceSkill":
        """Create a MarketplaceSkill from a marketplace plugin dict."""
        return cls(
            name=data.get("name", ""),
            source=data.get("source", ""),
            description=data.get("description", ""),
        )


@dataclass
class Marketplace:
    """Represents a skills marketplace."""

    name: str
    owner: str
    description: str
    version: str
    skills: list[MarketplaceSkill]

    @classmethod
    def from_dict(cls, data: dict) -> "Marketplace":
        """Create a Marketplace from the marketplace.json data."""
        owner_data = data.get("owner", {})
        metadata = data.get("metadata", {})
        plugins = data.get("plugins", [])

        return cls(
            name=data.get("name", "unknown"),
            owner=owner_data.get("name", "unknown") if isinstance(owner_data, dict) else str(owner_data),
            description=metadata.get("description", ""),
            version=metadata.get("version", "1.0.0"),
            skills=[MarketplaceSkill.from_dict(p) for p in plugins],
        )


class SkillsManager:
    """Manages skill installation, removal, and listing."""

    def __init__(
        self,
        *,
        settings: "SkillsSettings | None" = None,
        base_dir: Path | None = None,
    ) -> None:
        """
        Initialize the SkillsManager.

        Args:
            settings: Skills settings from config (marketplace_url, install_directory)
            base_dir: Base directory for resolving relative paths (defaults to cwd)
        """
        self._base_dir = base_dir or Path.cwd()
        self._settings = settings
        self._marketplace_url = (
            settings.marketplace_url
            if settings
            else "https://raw.githubusercontent.com/huggingface/skills/main/.claude-plugin/marketplace.json"
        )
        self._install_dir = self._resolve_path(
            settings.install_directory if settings else ".fast-agent/skills"
        )
        self._cached_marketplace: Marketplace | None = None

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to base_dir."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return (self._base_dir / path).resolve()

    async def fetch_marketplace(self, *, force_refresh: bool = False) -> Marketplace:
        """
        Fetch the marketplace data from the configured URL.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            Marketplace data

        Raises:
            httpx.HTTPError: If the fetch fails
            json.JSONDecodeError: If the response is not valid JSON
        """
        if self._cached_marketplace and not force_refresh:
            return self._cached_marketplace

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self._marketplace_url)
            response.raise_for_status()
            data = response.json()

        self._cached_marketplace = Marketplace.from_dict(data)
        return self._cached_marketplace

    def get_installed_skills(self) -> list[SkillManifest]:
        """
        Get list of currently installed skills.

        Returns:
            List of installed skill manifests
        """
        if not self._install_dir.exists():
            return []

        registry = SkillRegistry(
            base_dir=self._base_dir,
            directories=[self._install_dir],
        )
        return registry.load_manifests()

    def get_installed_skill_names(self) -> set[str]:
        """Get the names of all installed skills."""
        manifests = self.get_installed_skills()
        return {m.name.lower() for m in manifests}

    def _parse_github_url(self, marketplace_url: str) -> tuple[str, str, str]:
        """
        Parse a GitHub raw content URL to extract owner, repo, and branch.

        Args:
            marketplace_url: URL like https://raw.githubusercontent.com/owner/repo/branch/path

        Returns:
            Tuple of (owner, repo, branch)
        """
        parsed = urlparse(marketplace_url)
        path_parts = parsed.path.strip("/").split("/")

        if len(path_parts) < 3:
            raise ValueError(f"Cannot parse GitHub URL: {marketplace_url}")

        owner = path_parts[0]
        repo = path_parts[1]
        branch = path_parts[2]

        return owner, repo, branch

    def _get_skill_folder_name(self, source: str) -> str:
        """
        Extract the skill folder name from the source path.

        The source is like "./hf_dataset_creator" and we need to find
        the actual skill folder inside it (e.g., "hugging-face-dataset-creator").

        For now, we'll use the source directory name as a fallback,
        but ideally we'd look at the skills subfolder.
        """
        # Remove leading ./ if present
        source_clean = source.lstrip("./")
        return source_clean

    async def _download_skill_folder(
        self,
        skill: MarketplaceSkill,
        owner: str,
        repo: str,
        branch: str,
    ) -> Path:
        """
        Download a skill folder from GitHub.

        This downloads the skills subfolder from the source path.

        Args:
            skill: The marketplace skill to download
            owner: GitHub repo owner
            repo: GitHub repo name
            branch: Git branch

        Returns:
            Path to the installed skill directory

        Raises:
            httpx.HTTPError: If download fails
            ValueError: If skill structure is invalid
        """
        source_clean = skill.source.lstrip("./")

        # First, get the directory listing to find the skill folder
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{source_clean}/skills"
        headers = {"Accept": "application/vnd.github.v3+json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get the skills subdirectory listing
            response = await client.get(api_url, headers=headers, params={"ref": branch})
            response.raise_for_status()
            contents = response.json()

            # Find the skill folder (should be a directory with SKILL.md)
            skill_folder_name = None
            for item in contents:
                if item["type"] == "dir":
                    skill_folder_name = item["name"]
                    break

            if not skill_folder_name:
                raise ValueError(f"No skill folder found in {source_clean}/skills")

            # Create the install directory
            install_path = self._install_dir / skill_folder_name
            install_path.mkdir(parents=True, exist_ok=True)

            # Get the contents of the skill folder
            skill_api_url = f"{api_url}/{skill_folder_name}"
            response = await client.get(skill_api_url, headers=headers, params={"ref": branch})
            response.raise_for_status()
            skill_contents = response.json()

            # Download each file in the skill folder
            for item in skill_contents:
                if item["type"] == "file":
                    file_url = item["download_url"]
                    file_response = await client.get(file_url)
                    file_response.raise_for_status()

                    file_path = install_path / item["name"]
                    file_path.write_bytes(file_response.content)
                    logger.debug(f"Downloaded: {item['name']} -> {file_path}")

            logger.info(f"Installed skill '{skill.name}' to {install_path}")
            return install_path

    async def install_skill(
        self,
        skill_identifier: str | int,
        marketplace: Marketplace | None = None,
    ) -> tuple[bool, str]:
        """
        Install a skill from the marketplace.

        Args:
            skill_identifier: Skill name or number (1-indexed from marketplace list)
            marketplace: Optional pre-fetched marketplace data

        Returns:
            Tuple of (success, message)
        """
        if marketplace is None:
            try:
                marketplace = await self.fetch_marketplace()
            except Exception as e:
                return False, f"Failed to fetch marketplace: {e}"

        # Find the skill
        skill: MarketplaceSkill | None = None

        if isinstance(skill_identifier, int) or skill_identifier.isdigit():
            idx = int(skill_identifier) - 1  # Convert to 0-indexed
            if 0 <= idx < len(marketplace.skills):
                skill = marketplace.skills[idx]
            else:
                return False, f"Invalid skill number: {skill_identifier}. Valid range: 1-{len(marketplace.skills)}"
        else:
            # Find by name (case-insensitive)
            for s in marketplace.skills:
                if s.name.lower() == skill_identifier.lower():
                    skill = s
                    break

            if not skill:
                available = ", ".join(s.name for s in marketplace.skills)
                return False, f"Skill '{skill_identifier}' not found. Available: {available}"

        # Check if already installed
        installed = self.get_installed_skill_names()
        if skill.name.lower() in installed:
            return False, f"Skill '{skill.name}' is already installed."

        # Parse GitHub URL and download
        try:
            owner, repo, branch = self._parse_github_url(self._marketplace_url)
            await self._download_skill_folder(skill, owner, repo, branch)
            return True, f"Successfully installed skill: {skill.name}"
        except Exception as e:
            logger.exception(f"Failed to install skill {skill.name}")
            return False, f"Failed to install skill '{skill.name}': {e}"

    def remove_skill(self, skill_identifier: str | int) -> tuple[bool, str]:
        """
        Remove an installed skill.

        Args:
            skill_identifier: Skill name or number (1-indexed from installed list)

        Returns:
            Tuple of (success, message)
        """
        installed = self.get_installed_skills()

        if not installed:
            return False, "No skills installed."

        # Find the skill
        manifest: SkillManifest | None = None

        if isinstance(skill_identifier, int) or (
            isinstance(skill_identifier, str) and skill_identifier.isdigit()
        ):
            idx = int(skill_identifier) - 1  # Convert to 0-indexed
            if 0 <= idx < len(installed):
                manifest = installed[idx]
            else:
                return False, f"Invalid skill number: {skill_identifier}. Valid range: 1-{len(installed)}"
        else:
            # Find by name (case-insensitive)
            for m in installed:
                if m.name.lower() == skill_identifier.lower():
                    manifest = m
                    break

            if not manifest:
                available = ", ".join(m.name for m in installed)
                return False, f"Skill '{skill_identifier}' not found. Installed: {available}"

        # Remove the skill directory
        try:
            skill_dir = manifest.path.parent
            if skill_dir.exists():
                shutil.rmtree(skill_dir)
                logger.info(f"Removed skill directory: {skill_dir}")
                return True, f"Successfully removed skill: {manifest.name}"
            else:
                return False, f"Skill directory not found: {skill_dir}"
        except Exception as e:
            logger.exception(f"Failed to remove skill {manifest.name}")
            return False, f"Failed to remove skill '{manifest.name}': {e}"

    def format_installed_skills(self, skills: Sequence[SkillManifest] | None = None) -> str:
        """
        Format the list of installed skills for display.

        Args:
            skills: Optional pre-loaded skills list

        Returns:
            Formatted markdown string
        """
        if skills is None:
            skills = self.get_installed_skills()

        if not skills:
            return "No skills installed.\n\nUse `/skills add` to browse and install skills from the marketplace."

        lines = ["# Installed Skills", ""]

        for idx, skill in enumerate(skills, start=1):
            lines.append(f"**{idx}. {skill.name}**")
            if skill.description:
                lines.append(f"   {skill.description}")
            lines.append(f"   _Location: {skill.path.parent}_")
            lines.append("")

        lines.append("---")
        lines.append("Use `/skills remove <name|number>` to uninstall a skill.")

        return "\n".join(lines)

    def format_marketplace_skills(
        self,
        marketplace: Marketplace,
        installed_names: set[str] | None = None,
    ) -> str:
        """
        Format the marketplace skills for display.

        Args:
            marketplace: The marketplace data
            installed_names: Optional set of installed skill names (lowercase)

        Returns:
            Formatted markdown string
        """
        if installed_names is None:
            installed_names = self.get_installed_skill_names()

        lines = [
            f"# {marketplace.name}",
            f"_by {marketplace.owner} (v{marketplace.version})_",
            "",
        ]

        if marketplace.description:
            lines.append(marketplace.description)
            lines.append("")

        lines.append("## Available Skills")
        lines.append("")

        for idx, skill in enumerate(marketplace.skills, start=1):
            is_installed = skill.name.lower() in installed_names
            status = " _(installed)_" if is_installed else ""
            lines.append(f"**{idx}. {skill.name}**{status}")
            if skill.description:
                # Truncate long descriptions
                desc = skill.description
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                lines.append(f"   {desc}")
            lines.append("")

        lines.append("---")
        lines.append("Enter a skill name or number to install, or 'q' to cancel.")

        return "\n".join(lines)
