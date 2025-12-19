"""
Skills Manager for installing, listing, and removing skills from a marketplace.

This module provides functionality to:
- Fetch available skills from a remote marketplace (JSON file)
- List installed skills using the SkillRegistry
- Install skills from GitHub repositories (sparse checkout)
- Remove installed skills
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from fast_agent.constants import DEFAULT_SKILLS_INSTALL_DIR, DEFAULT_SKILLS_MARKETPLACE_URL
from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.registry import SkillManifest, SkillRegistry

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class MarketplaceSkill:
    """Represents a skill available in the marketplace."""

    name: str
    source: str
    description: str
    installed: bool = False


@dataclass
class MarketplaceInfo:
    """Information about a skills marketplace."""

    name: str
    owner: str | None = None
    description: str | None = None
    version: str | None = None
    skills: list[MarketplaceSkill] = field(default_factory=list)
    repo_url: str | None = None  # The base GitHub repo URL


def _extract_github_info(marketplace_url: str) -> tuple[str | None, str | None]:
    """
    Extract GitHub owner/repo and branch from a raw GitHub URL.

    Returns:
        Tuple of (repo_url, branch) where repo_url is like 'https://github.com/owner/repo'
    """
    parsed = urlparse(marketplace_url)

    # Handle raw.githubusercontent.com URLs
    # Format: https://raw.githubusercontent.com/owner/repo/branch/path
    if parsed.netloc == "raw.githubusercontent.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 3:
            owner, repo, branch = parts[0], parts[1], parts[2]
            return f"https://github.com/{owner}/{repo}", branch

    # Handle github.com blob URLs
    # Format: https://github.com/owner/repo/blob/branch/path
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4:
            owner, repo = parts[0], parts[1]
            branch = parts[3]
            return f"https://github.com/{owner}/{repo}", branch

    return None, None


async def fetch_marketplace(
    marketplace_url: str | None = None,
    *,
    timeout: float = 30.0,
) -> MarketplaceInfo:
    """
    Fetch and parse a skills marketplace JSON file.

    Args:
        marketplace_url: URL to the marketplace JSON. Defaults to DEFAULT_SKILLS_MARKETPLACE_URL.
        timeout: HTTP request timeout in seconds.

    Returns:
        MarketplaceInfo containing the marketplace metadata and available skills.

    Raises:
        httpx.HTTPError: If the HTTP request fails.
        json.JSONDecodeError: If the response is not valid JSON.
    """
    url = marketplace_url or DEFAULT_SKILLS_MARKETPLACE_URL

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    # Extract GitHub repo info from URL
    repo_url, _ = _extract_github_info(url)

    # Parse the marketplace JSON
    owner_data = data.get("owner", {})
    metadata = data.get("metadata", {})

    info = MarketplaceInfo(
        name=data.get("name", "unknown"),
        owner=owner_data.get("name") if isinstance(owner_data, dict) else None,
        description=metadata.get("description") if isinstance(metadata, dict) else None,
        version=metadata.get("version") if isinstance(metadata, dict) else None,
        repo_url=repo_url,
    )

    # Parse plugins/skills
    plugins = data.get("plugins", [])
    for plugin in plugins:
        if isinstance(plugin, dict):
            skill = MarketplaceSkill(
                name=plugin.get("name", "unknown"),
                source=plugin.get("source", ""),
                description=plugin.get("description", ""),
            )
            info.skills.append(skill)

    return info


def list_installed_skills(
    *,
    base_dir: Path | None = None,
    install_dir: str | None = None,
) -> list[SkillManifest]:
    """
    List installed skills using the SkillRegistry.

    Args:
        base_dir: Base directory for resolving relative paths.
        install_dir: The skills installation directory. Defaults to DEFAULT_SKILLS_INSTALL_DIR.

    Returns:
        List of installed skill manifests.
    """
    skills_dir = install_dir or DEFAULT_SKILLS_INSTALL_DIR
    base = base_dir or Path.cwd()

    registry = SkillRegistry(base_dir=base, directories=[skills_dir])
    return registry.load_manifests()


def _normalize_source_path(source: str) -> str:
    """Normalize the source path by removing leading ./ if present."""
    return source.lstrip("./")


async def install_skill(
    skill: MarketplaceSkill,
    marketplace_info: MarketplaceInfo,
    *,
    base_dir: Path | None = None,
    install_dir: str | None = None,
    marketplace_url: str | None = None,
) -> tuple[bool, str]:
    """
    Install a skill from the marketplace.

    The skill is installed by:
    1. Sparse checkout of the skill folder from the GitHub repository
    2. Copying the skill folder to the install directory

    Args:
        skill: The marketplace skill to install.
        marketplace_info: Marketplace info containing repo URL.
        base_dir: Base directory for resolving relative paths.
        install_dir: Installation directory. Defaults to DEFAULT_SKILLS_INSTALL_DIR.
        marketplace_url: The marketplace URL (used to extract repo info if not in marketplace_info).

    Returns:
        Tuple of (success, message).
    """
    base = base_dir or Path.cwd()
    target_dir = base / (install_dir or DEFAULT_SKILLS_INSTALL_DIR)

    # Ensure install directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Get repo URL
    repo_url = marketplace_info.repo_url
    if not repo_url and marketplace_url:
        repo_url, _ = _extract_github_info(marketplace_url)

    if not repo_url:
        return False, "Could not determine GitHub repository URL"

    # Extract branch from marketplace URL
    url = marketplace_url or DEFAULT_SKILLS_MARKETPLACE_URL
    _, branch = _extract_github_info(url)
    branch = branch or "main"

    # Normalize source path
    source_base = _normalize_source_path(skill.source)

    # The skill structure is: source_base/skills/skill_name/SKILL.md
    # We need to clone the skill folder and put it in target_dir/skill_name/
    skill_source_path = f"{source_base}/skills/{skill.name}"

    # Destination path
    dest_path = target_dir / skill.name

    # Check if already installed
    if dest_path.exists():
        return False, f"Skill '{skill.name}' is already installed at {dest_path}"

    try:
        # Use sparse checkout to get only the skill folder
        success, message = await _sparse_checkout_skill(
            repo_url=repo_url,
            branch=branch,
            skill_path=skill_source_path,
            dest_path=dest_path,
        )
        return success, message

    except Exception as e:
        logger.error("Failed to install skill", data={"skill": skill.name, "error": str(e)})
        return False, f"Failed to install skill: {e}"


async def _sparse_checkout_skill(
    repo_url: str,
    branch: str,
    skill_path: str,
    dest_path: Path,
) -> tuple[bool, str]:
    """
    Perform a sparse checkout of a specific skill folder from a GitHub repository.

    Args:
        repo_url: The GitHub repository URL (e.g., https://github.com/owner/repo).
        branch: The branch to checkout.
        skill_path: The path within the repo to the skill folder.
        dest_path: The destination path for the skill.

    Returns:
        Tuple of (success, message).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Initialize sparse checkout
            subprocess.run(
                ["git", "init"],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                ["git", "remote", "add", "origin", repo_url],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                ["git", "config", "core.sparseCheckout", "true"],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Configure sparse checkout path
            sparse_file = temp_path / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text(f"{skill_path}/\n")

            # Fetch and checkout
            subprocess.run(
                ["git", "fetch", "--depth=1", "origin", branch],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                ["git", "checkout", branch],
                cwd=temp_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Check if the skill folder exists
            skill_folder = temp_path / skill_path
            if not skill_folder.exists():
                return False, f"Skill folder not found in repository: {skill_path}"

            # Verify SKILL.md exists
            skill_md = skill_folder / "SKILL.md"
            if not skill_md.exists():
                return False, f"SKILL.md not found in {skill_path}"

            # Copy to destination
            shutil.copytree(skill_folder, dest_path)

            return True, f"Successfully installed skill to {dest_path}"

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            return False, f"Git operation failed: {error_msg}"


def remove_skill(
    skill_name: str,
    *,
    base_dir: Path | None = None,
    install_dir: str | None = None,
) -> tuple[bool, str]:
    """
    Remove an installed skill.

    Args:
        skill_name: Name of the skill to remove.
        base_dir: Base directory for resolving relative paths.
        install_dir: The skills installation directory. Defaults to DEFAULT_SKILLS_INSTALL_DIR.

    Returns:
        Tuple of (success, message).
    """
    base = base_dir or Path.cwd()
    skills_dir = base / (install_dir or DEFAULT_SKILLS_INSTALL_DIR)
    skill_path = skills_dir / skill_name

    if not skill_path.exists():
        return False, f"Skill '{skill_name}' not found at {skill_path}"

    try:
        shutil.rmtree(skill_path)
        return True, f"Successfully removed skill '{skill_name}'"
    except Exception as e:
        logger.error("Failed to remove skill", data={"skill": skill_name, "error": str(e)})
        return False, f"Failed to remove skill: {e}"


def format_installed_skills(manifests: list[SkillManifest]) -> str:
    """Format installed skills for display."""
    if not manifests:
        return "No skills installed."

    lines = ["# Installed Skills", ""]
    for idx, manifest in enumerate(manifests, 1):
        lines.append(f"**{idx}. {manifest.name}**")
        if manifest.description:
            lines.append(f"   {manifest.description}")
        lines.append(f"   Location: `{manifest.path.parent}`")
        lines.append("")

    return "\n".join(lines)


def format_marketplace_skills(
    marketplace: MarketplaceInfo,
    installed_names: set[str] | None = None,
) -> str:
    """Format marketplace skills for display."""
    installed = installed_names or set()

    lines = [f"# Skills Marketplace: {marketplace.name}", ""]

    if marketplace.owner:
        lines.append(f"**Owner:** {marketplace.owner}")
    if marketplace.description:
        lines.append(f"**Description:** {marketplace.description}")
    if marketplace.version:
        lines.append(f"**Version:** {marketplace.version}")
    lines.append("")

    if not marketplace.skills:
        lines.append("No skills available.")
        return "\n".join(lines)

    lines.append("## Available Skills")
    lines.append("")

    for idx, skill in enumerate(marketplace.skills, 1):
        status = " ✓ (installed)" if skill.name in installed else ""
        lines.append(f"**{idx}. {skill.name}**{status}")
        if skill.description:
            # Truncate long descriptions
            desc = skill.description
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"   {desc}")
        lines.append("")

    lines.append("---")
    lines.append("Use `/skills add <number>` or `/skills add <name>` to install a skill.")
    lines.append("Use `q` to exit.")

    return "\n".join(lines)
