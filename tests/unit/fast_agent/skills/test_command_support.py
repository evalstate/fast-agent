from __future__ import annotations

from fast_agent.skills.command_support import (
    filter_marketplace_skills,
    marketplace_repository_hint,
    marketplace_search_tokens,
    skills_usage_lines,
)
from fast_agent.skills.models import MarketplaceSkill


def _marketplace_skill(
    *,
    name: str,
    description: str | None = None,
    bundle_name: str | None = None,
    bundle_description: str | None = None,
    repo_ref: str | None = None,
) -> MarketplaceSkill:
    return MarketplaceSkill(
        name=name,
        description=description,
        repo_url="https://github.com/example/skills",
        repo_ref=repo_ref,
        repo_path=f"skills/{name}",
        source_url=None,
        bundle_name=bundle_name,
        bundle_description=bundle_description,
    )


def test_marketplace_search_tokens_support_quoted_phrases() -> None:
    tokens = marketplace_search_tokens('docker "image build"')

    assert tokens == ["docker", "image build"]


def test_filter_marketplace_skills_matches_bundle_and_description_fields() -> None:
    marketplace = [
        _marketplace_skill(
            name="docker-build",
            description="Build Docker images from a repo",
            bundle_name="Containers",
            bundle_description="Docker and OCI workflows",
        ),
        _marketplace_skill(
            name="python-test",
            description="Run pytest in a project",
            bundle_name="Python",
            bundle_description="Virtualenv and packaging helpers",
        ),
    ]

    filtered = filter_marketplace_skills(marketplace, "docker containers")

    assert [entry.name for entry in filtered] == ["docker-build"]


def test_marketplace_repository_hint_includes_ref_when_available() -> None:
    hint = marketplace_repository_hint(
        [_marketplace_skill(name="docker-build", repo_ref="main")]
    )

    assert hint == "https://github.com/example/skills@main"


def test_skills_usage_lines_include_registry_command() -> None:
    assert "- /skills registry" in skills_usage_lines()
