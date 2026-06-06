from __future__ import annotations

from fast_agent.commands.command_catalog import normalize_command_action
from fast_agent.skills.command_support import (
    filter_marketplace_skills,
    marketplace_repository_hint,
    marketplace_search_tokens,
    parse_skills_slash_options,
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
    install_dir_name_override: str | None = None,
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
        install_dir_name_override=install_dir_name_override,
    )


def test_marketplace_search_tokens_support_quoted_phrases() -> None:
    tokens = marketplace_search_tokens('docker "image build"')

    assert tokens == ["docker", "image build"]


def test_marketplace_search_tokens_normalizes_and_omits_blank_tokens() -> None:
    tokens = marketplace_search_tokens('  Docker "Image Build" "   " API  ')

    assert tokens == ["docker", "image build", "api"]


def test_parse_skills_slash_options_extracts_common_overrides() -> None:
    parsed = parse_skills_slash_options(
        'alpha --registry ./marketplace.json --skills-dir "My Skills"'
    )

    assert parsed.error is None
    assert parsed.argument == "alpha"
    assert parsed.argument_tokens == ("alpha",)
    assert parsed.registry == "./marketplace.json"
    assert parsed.skills_dir == "My Skills"


def test_parse_skills_slash_options_accepts_registry_short_flag() -> None:
    parsed = parse_skills_slash_options("alpha -r ./marketplace.json")

    assert parsed.error is None
    assert parsed.argument == "alpha"
    assert parsed.registry == "./marketplace.json"


def test_parse_skills_slash_options_does_not_quote_single_path_selector() -> None:
    parsed = parse_skills_slash_options('"My Skills/foo" --registry ./marketplace.json')

    assert parsed.error is None
    assert parsed.argument == "My Skills/foo"
    assert parsed.argument_tokens == ("My Skills/foo",)


def test_parse_skills_slash_options_preserves_quoted_search_argument() -> None:
    parsed = parse_skills_slash_options(
        'docker "image build" --registry=https://example.test/marketplace.json'
    )

    assert parsed.error is None
    assert parsed.argument == "docker 'image build'"
    assert parsed.argument_tokens == ("docker", "image build")
    assert parsed.registry == "https://example.test/marketplace.json"


def test_parse_skills_slash_options_reports_missing_values() -> None:
    parsed = parse_skills_slash_options("alpha --skills-dir")

    assert parsed.error == "Missing value for --skills-dir"


def test_parse_skills_slash_options_rejects_duplicate_value_options() -> None:
    registry = parse_skills_slash_options("alpha --registry one -r two")
    skills_dir = parse_skills_slash_options("alpha --skills-dir one --skills two")

    assert registry.error == "Duplicate option: --registry"
    assert skills_dir.error == "Duplicate option: --skills-dir"


def test_parse_skills_slash_options_does_not_consume_following_option_as_value() -> None:
    parsed = parse_skills_slash_options("alpha --registry --skills-dir ./skills")

    assert parsed.error == "Missing value for --registry"


def test_parse_skills_slash_options_reports_split_errors() -> None:
    parsed = parse_skills_slash_options('alpha "unterminated')

    assert parsed.error == "Invalid /skills arguments: No closing quotation"


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

    filtered = filter_marketplace_skills(marketplace, "DOCKER containers")

    assert [entry.name for entry in filtered] == ["docker-build"]


def test_filter_marketplace_skills_matches_install_dir_name_alias() -> None:
    marketplace = [
        _marketplace_skill(
            name="bundle-entry",
            repo_ref="main",
            install_dir_name_override="canonical-name",
        ),
        _marketplace_skill(name="other"),
    ]

    filtered = filter_marketplace_skills(marketplace, "canonical")

    assert [entry.name for entry in filtered] == ["bundle-entry"]


def test_marketplace_repository_hint_includes_ref_when_available() -> None:
    hint = marketplace_repository_hint(
        [_marketplace_skill(name="docker-build", repo_ref="main")]
    )

    assert hint == "https://github.com/example/skills@main"


def test_skills_action_aliases_normalize_to_canonical_actions() -> None:
    cases = {
        None: "list",
        "": "list",
        "--help": "help",
        "-h": "help",
        "marketplace": "available",
        " MARKETPLACE ": "available",
        "browse": "available",
        "find": "search",
        "install": "add",
        "source": "registry",
        "rm": "remove",
        "delete": "remove",
        "uninstall": "remove",
        "refresh": "update",
        "upgrade": "update",
        "unexpected": "unexpected",
    }

    for action, expected in cases.items():
        assert normalize_command_action("skills", action) == expected


def test_skills_usage_lines_include_registry_command() -> None:
    assert "- /skills registry" in skills_usage_lines()
