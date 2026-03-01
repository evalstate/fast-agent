"""Markdown renderers for skill summaries."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from fast_agent.skills.manager import format_skill_provenance_details

if TYPE_CHECKING:
    from fast_agent.skills.manager import MarketplaceSkill
    from fast_agent.skills.registry import SkillManifest


def _format_skill_entry(
    *,
    index: int,
    name: str,
    description: str | None,
    source: str | None,
    provenance: str | None,
    installed: str | None,
) -> list[str]:
    lines: list[str] = [f"{index}. **{name}**"]
    if description:
        wrapped = textwrap.wrap(description, width=88)
        lines.extend(f"    > {desc_line}" for desc_line in wrapped[:4])
        if len(wrapped) > 4:
            lines.append("    > …")

    if source:
        lines.append("    > **Source:**")
        lines.append(f"    > {source}")
    if provenance:
        lines.append("    > **Provenance:**")
        lines.append(f"    > {provenance}")
    if installed:
        lines.append("    > **Installed:**")
        lines.append(f"    > {installed}")

    lines.append("")
    return lines


def _display_path(path: Path, *, cwd: Path) -> Path:
    try:
        return path.relative_to(cwd)
    except ValueError:
        return path


def render_skill_list(manifests: Sequence[SkillManifest], *, cwd: Path | None = None) -> list[str]:
    lines: list[str] = []
    cwd = cwd or Path.cwd()

    for index, manifest in enumerate(manifests, 1):
        source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
        display_path = _display_path(source_path, cwd=cwd)
        provenance, installed = format_skill_provenance_details(source_path)
        lines.extend(
            _format_skill_entry(
                index=index,
                name=manifest.name,
                description=manifest.description,
                source=f"`{display_path}`",
                provenance=provenance,
                installed=installed,
            )
        )

    return lines


def render_skills_by_directory(
    manifests_by_dir: dict[Path, list[SkillManifest]],
    *,
    heading: str,
    cwd: Path | None = None,
) -> str:
    lines = [f"# {heading}", ""]
    cwd = cwd or Path.cwd()
    total_skills = sum(len(m) for m in manifests_by_dir.values())
    skill_index = 0

    for directory, manifests in manifests_by_dir.items():
        display_path = _display_path(directory, cwd=cwd)
        lines.append(f"## {display_path}")
        lines.append("")

        if not manifests:
            lines.append("No skills in this directory.")
            lines.append("")
            continue

        for manifest in manifests:
            skill_index += 1
            source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
            source_display = _display_path(source_path, cwd=cwd)
            provenance, installed = format_skill_provenance_details(source_path)
            lines.extend(
                _format_skill_entry(
                    index=skill_index,
                    name=manifest.name,
                    description=manifest.description,
                    source=f"`{source_display}`",
                    provenance=provenance,
                    installed=installed,
                )
            )

    if total_skills == 0:
        lines.append("Use `/skills available` to browse marketplace skills.")
        lines.append("")
        lines.append("Search with `/skills search <query>`.")
    else:
        lines.append("Remove a skill with `/skills remove <number|name>`.")
        lines.append("")
        lines.append("Use `/skills available` to browse marketplace skills.")
        lines.append("")
        lines.append("Search with `/skills search <query>`.")
        lines.append("")
        lines.append("Change skills registry with `/skills registry <number|url|path>`.")

    return "\n".join(lines)


def render_skills_remove_list(
    *,
    manager_dir: Path,
    manifests: Sequence[SkillManifest],
    heading: str,
    cwd: Path | None = None,
) -> str:
    lines = [f"# {heading}", ""]
    cwd = cwd or Path.cwd()
    display_dir = _display_path(manager_dir, cwd=cwd)
    lines.append(f"## {display_dir}")
    lines.append("")

    if not manifests:
        lines.append("No local skills to remove.")
        return "\n".join(lines)

    for index, manifest in enumerate(manifests, 1):
        source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
        source_display = _display_path(source_path, cwd=cwd)
        provenance, installed = format_skill_provenance_details(source_path)
        lines.extend(
            _format_skill_entry(
                index=index,
                name=manifest.name,
                description=manifest.description,
                source=f"`{source_display}`",
                provenance=provenance,
                installed=installed,
            )
        )

    lines.append("Remove with `/skills remove <number|name>`.")
    return "\n".join(lines)


def render_marketplace_skills(
    marketplace: Sequence[MarketplaceSkill],
    *,
    heading: str,
    repository: str | None = None,
) -> str:
    lines = [f"# {heading}", ""]
    if repository:
        lines.append(f"Repository: `{repository}`")
        lines.append("")

    if not marketplace:
        lines.append("No skills found in the marketplace.")
        return "\n".join(lines)

    lines.append("Available skills:")
    lines.append("")

    current_bundle: str | None = None
    skill_index = 0
    for entry in marketplace:
        bundle_name = entry.bundle_name
        bundle_description = entry.bundle_description
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            if lines:
                lines.append("")
            lines.append(f"## {bundle_name}")
            if bundle_description:
                wrapped = textwrap.wrap(bundle_description, width=88)
                lines.extend(f"> {desc_line}" for desc_line in wrapped[:4])
                if len(wrapped) > 4:
                    lines.append("> …")
            lines.append("")

        skill_index += 1
        source = f"[link]({entry.source_url})" if entry.source_url else None
        lines.extend(
            _format_skill_entry(
                index=skill_index,
                name=entry.name,
                description=entry.description,
                source=source,
                provenance=None,
                installed=None,
            )
        )

    lines.append("Install with `/skills add <number|name>`. ")
    lines.append("Search with `/skills search <query>`. ")
    lines.append("Change registry with `/skills registry`.")

    return "\n".join(lines)


def render_skills_registry_overview(
    *,
    heading: str,
    current_registry: str,
    configured_urls: Sequence[str],
) -> str:
    lines = [f"# {heading}", "", f"Registry: {current_registry}", ""]
    if configured_urls:
        lines.append("Configured registries:")
        for index, url in enumerate(configured_urls, 1):
            lines.append(f"- [{index}] {url}")
        lines.append("")

    lines.append(
        "Usage: `/skills registry <number|URL|path>`.\n\n"
        "URL should point to a repo with a valid `marketplace.json`."
    )

    return "\n".join(lines)
