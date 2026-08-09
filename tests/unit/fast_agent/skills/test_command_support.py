from __future__ import annotations

from fast_agent.commands.command_catalog import normalize_command_action
from fast_agent.skills.command_support import (
    filter_marketplace_skills,
    marketplace_repository_hint,
    marketplace_search_tokens,
    parse_skills_catalog_options,
    parse_skills_slash_options,
    skills_usage_lines,
)
from fast_agent.skills.configuration import format_marketplace_display_url
from fast_agent.skills.marketplace_source import MarketplaceSkillSource
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


def test_parse_skills_slash_options_preserves_quoted_search_argument() -> None:
    parsed = parse_skills_slash_options(
        'docker "image build" --registry=https://example.test/marketplace.json'
    )

    assert parsed.error is None
    assert parsed.argument == "docker 'image build'"
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


def test_parse_skills_catalog_options_supports_registry_paging_and_json() -> None:
    parsed = parse_skills_catalog_options(
        'image "model trainer" --registry hf --page 2 --limit 25 --json'
    )

    assert parsed.error is None
    assert parsed.argument == "image 'model trainer'"
    assert parsed.registry == "hf"
    assert parsed.page == 2
    assert parsed.page_explicit is True
    assert parsed.limit == 25
    assert parsed.output == "json"


def test_parse_skills_catalog_options_preserves_json_on_split_error() -> None:
    parsed = parse_skills_catalog_options('--json "unterminated')

    assert parsed.output == "json"
    assert parsed.error == "Invalid /skills arguments: No closing quotation"


def test_parse_skills_catalog_options_rejects_invalid_or_conflicting_options() -> None:
    assert parse_skills_catalog_options("--page 0").error is not None
    assert parse_skills_catalog_options("--limit 101").error == (
        "Invalid value for --limit: maximum is 100"
    )
    assert parse_skills_catalog_options("--compact --json").error == (
        "Conflicting output option: --json"
    )
    assert parse_skills_catalog_options("--registry hf -r other").error == (
        "Duplicate option: --registry"
    )
    assert parse_skills_catalog_options("--page 1 --page 2").error == "Duplicate option: --page"
    assert parse_skills_catalog_options("--unknown").error == "Unknown option: --unknown"
    huge_page = "9" * 5_000
    assert parse_skills_catalog_options(f"--page {huge_page}").error == (
        "Invalid value for --page: maximum is 1000000"
    )


def test_parse_skills_catalog_options_preserves_paths_and_dash_queries() -> None:
    windows = parse_skills_catalog_options(r"--registry C:\tmp\registry --compact")
    dashed = parse_skills_catalog_options("-- --starts-with-dash")

    assert windows.registry == r"C:\tmp\registry"
    assert dashed.argument == "--starts-with-dash"


def test_marketplace_source_display_redacts_registry_credentials() -> None:
    url = "https://user:secret@example.test/skills.json?token=secret#private"

    display = format_marketplace_display_url(url)
    source = MarketplaceSkillSource(url)

    assert "secret" not in display
    assert "secret" not in source.ref.display_name
    assert "REDACTED" in source.ref.display_name


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
    hint = marketplace_repository_hint([_marketplace_skill(name="docker-build", repo_ref="main")])

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
