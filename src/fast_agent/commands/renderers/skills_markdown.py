"""Markdown renderers for skill summaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.commands.renderers.markdown_blocks import (
    markdown_heading,
    wrapped_quote_lines,
)
from fast_agent.skills.command_support import SKILLS_ADD_HINT_SLASH
from fast_agent.skills.provenance import format_skill_provenance_details
from fast_agent.utils.markdown import escape_markdown_text, markdown_code_span
from fast_agent.utils.path_display import format_relative_path
from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fast_agent.skills.models import MarketplaceSkill
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
    lines: list[str] = [f"{index}. **{escape_markdown_text(name)}**"]
    escaped_description = escape_markdown_text(description) if description else None
    lines.extend(wrapped_quote_lines(escaped_description, prefix="    > "))

    _append_skill_detail(lines, "Source", source)
    _append_skill_detail(lines, "Provenance", provenance)
    _append_skill_detail(lines, "Installed", installed)

    lines.append("")
    return lines


def _append_skill_detail(lines: list[str], label: str, value: str | None) -> None:
    if not value:
        return
    lines.append(f"    > **{label}:**")
    lines.append(f"    > {value}")


def _format_manifest_entry(
    *,
    index: int,
    manifest: "SkillManifest",
    cwd: Path,
) -> list[str]:
    source_path = manifest.path.parent if manifest.path.is_file() else manifest.path
    display_path = format_relative_path(source_path, cwd=cwd)
    provenance, installed = format_skill_provenance_details(source_path)
    return _format_skill_entry(
        index=index,
        name=manifest.name,
        description=manifest.description,
        source=markdown_code_span(display_path),
        provenance=provenance,
        installed=installed,
    )


def render_skill_list(manifests: Sequence[SkillManifest], *, cwd: Path | None = None) -> list[str]:
    lines: list[str] = []
    cwd = cwd or Path.cwd()

    for index, manifest in enumerate(manifests, 1):
        lines.extend(_format_manifest_entry(index=index, manifest=manifest, cwd=cwd))

    return lines


def _skills_browse_guidance() -> list[str]:
    return [
        "Use `/skills available` to browse marketplace skills.",
        "",
        "Search with `/skills search <query>`.",
    ]


def _skills_marketplace_guidance() -> list[str]:
    return [
        SKILLS_ADD_HINT_SLASH,
        "Search with `/skills search <query>`.",
        "Change registry with `/skills registry`.",
    ]


def render_skills_by_directory(
    manifests_by_dir: dict[Path, list[SkillManifest]],
    *,
    heading: str,
    cwd: Path | None = None,
) -> str:
    lines = [markdown_heading(heading), ""]
    cwd = cwd or Path.cwd()
    total_skills = sum(len(m) for m in manifests_by_dir.values())
    skill_index = 0

    for directory, manifests in manifests_by_dir.items():
        display_path = format_relative_path(directory, cwd=cwd)
        lines.append(f"## {markdown_code_span(display_path)}")
        lines.append("")

        if not manifests:
            lines.append("No skills in this directory.")
            lines.append("")
            continue

        for manifest in manifests:
            skill_index += 1
            lines.extend(_format_manifest_entry(index=skill_index, manifest=manifest, cwd=cwd))

    if total_skills == 0:
        lines.extend(_skills_browse_guidance())
    else:
        lines.append("Remove a skill with `/skills remove <number|name>`.")
        lines.append("")
        lines.extend(_skills_browse_guidance())
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
    lines = [markdown_heading(heading), ""]
    cwd = cwd or Path.cwd()
    display_dir = format_relative_path(manager_dir, cwd=cwd)
    lines.append(f"## {markdown_code_span(display_dir)}")
    lines.append("")

    if not manifests:
        lines.append("No local skills to remove.")
        return "\n".join(lines)

    for index, manifest in enumerate(manifests, 1):
        lines.extend(_format_manifest_entry(index=index, manifest=manifest, cwd=cwd))

    lines.append("Remove with `/skills remove <number|name>`.")
    return "\n".join(lines)


def render_marketplace_skills(
    marketplace: Sequence[MarketplaceSkill],
    *,
    heading: str,
    repository: str | None = None,
) -> str:
    lines = [markdown_heading(heading), ""]
    normalized_repository = strip_to_none(repository)
    if normalized_repository is not None:
        lines.append(f"Repository: {markdown_code_span(normalized_repository)}")
        lines.append("")

    if not marketplace:
        lines.append("No skills found in the marketplace.")
        return "\n".join(lines)

    lines.append("Available skills:")
    lines.append("")

    current_bundle: str | None = None
    for skill_index, entry in enumerate(marketplace, start=1):
        bundle_name = entry.bundle_name
        bundle_description = entry.bundle_description
        if bundle_name and bundle_name != current_bundle:
            current_bundle = bundle_name
            if lines:
                lines.append("")
            lines.append(f"## {escape_markdown_text(bundle_name)}")
            if bundle_description:
                lines.extend(wrapped_quote_lines(escape_markdown_text(bundle_description)))
            lines.append("")

        source = markdown_code_span(entry.source_url) if entry.source_url else None
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

    lines.extend(_skills_marketplace_guidance())

    return "\n".join(lines)

