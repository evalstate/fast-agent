from pathlib import Path

from fast_agent.commands.handlers._marketplace_argument_parsing import (
    AddArgument,
    PublishArgument,
    RegistryArgument,
    UpdateArgument,
    optional_selector,
    parse_add_argument,
    parse_publish_argument,
    parse_update_argument,
    resolve_registry_argument,
)


def test_parse_update_argument_handles_selector_and_flags() -> None:
    assert parse_update_argument("--force alpha --yes") == UpdateArgument(
        selector="alpha",
        force=True,
        yes=True,
    )


def test_parse_update_argument_accepts_skills_dir_when_enabled(tmp_path: Path) -> None:
    assert parse_update_argument(
        f'all --skills-dir="{tmp_path}" --force --yes',
        allow_skills_dir=True,
    ) == UpdateArgument(
        selector="all",
        skills_dir=tmp_path,
        force=True,
        yes=True,
    )


def test_parse_update_argument_rejects_skills_dir_by_default(tmp_path: Path) -> None:
    assert (
        parse_update_argument(f'alpha --skills-dir "{tmp_path}"').error
        == "Unknown option: --skills-dir"
    )


def test_parse_update_argument_handles_dry_run_and_all_selector() -> None:
    assert parse_update_argument(None) == UpdateArgument()
    assert parse_update_argument("all --yes") == UpdateArgument(selector="all", yes=True)


def test_parse_update_argument_accepts_repeated_flags() -> None:
    assert parse_update_argument("--force alpha --force --yes --yes") == UpdateArgument(
        selector="alpha",
        force=True,
        yes=True,
    )


def test_parse_update_argument_rejects_duplicate_skills_dir_when_enabled(
    tmp_path: Path,
) -> None:
    parsed = parse_update_argument(
        f'alpha --skills-dir="{tmp_path}" --skills-dir="{tmp_path / "other"}"',
        allow_skills_dir=True,
    )

    assert parsed.error == "Duplicate option: --skills-dir"


def test_optional_selector_strips_and_treats_blank_as_missing() -> None:
    assert optional_selector(None) is None
    assert optional_selector("   ") is None
    assert optional_selector("\n\t") is None
    assert optional_selector(" alpha ") == "alpha"


def test_parse_add_argument_handles_selector_and_overrides(tmp_path: Path) -> None:
    parsed = parse_add_argument(
        f'alpha --registry "./marketplace.json" --skills-dir="{tmp_path}" --force'
    )

    assert parsed == AddArgument(
        selector="alpha",
        registry="./marketplace.json",
        skills_dir=tmp_path,
        force=True,
    )


def test_parse_add_argument_expands_skills_dir_path(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/fast-agent-home")

    parsed = parse_add_argument('alpha --skills-dir="~/skills"')

    assert parsed.skills_dir == Path("/tmp/fast-agent-home/skills")


def test_parse_add_argument_accepts_options_before_selector() -> None:
    assert parse_add_argument("--registry=https://example.test/marketplace.json alpha") == (
        AddArgument(
            selector="alpha",
            registry="https://example.test/marketplace.json",
        )
    )


def test_parse_add_argument_rejects_missing_option_values() -> None:
    cases = {
        "--registry": "Missing value for --registry",
        "--registry=": "Missing value for --registry",
        "--registry --force": "Missing value for --registry",
        "-r --force": "Missing value for --registry",
        "--skills-dir": "Missing value for --skills-dir",
        "--skills-dir=": "Missing value for --skills-dir",
        "--skills-dir --force": "Missing value for --skills-dir",
    }

    for argument, expected_error in cases.items():
        assert parse_add_argument(argument).error == expected_error


def test_parse_add_argument_rejects_duplicate_value_options(tmp_path: Path) -> None:
    cases = {
        "--registry one --registry two": "Duplicate option: --registry",
        f'--skills-dir="{tmp_path}" --skills-dir="{tmp_path / "other"}"': (
            "Duplicate option: --skills-dir"
        ),
    }

    for argument, expected_error in cases.items():
        assert parse_add_argument(argument).error == expected_error


def test_parse_add_argument_rejects_unknown_options() -> None:
    assert parse_add_argument("--bogus alpha").error == "Unknown option: --bogus"
    assert parse_add_argument("-x alpha").error == "Unknown option: -x"


def test_parse_add_argument_can_disable_registry_option() -> None:
    assert (
        parse_add_argument("alpha --registry custom", allow_registry=False).error
        == "Unknown option: --registry"
    )


def test_parse_add_argument_can_disable_skills_dir_option(tmp_path: Path) -> None:
    assert (
        parse_add_argument(
            f'alpha --skills-dir="{tmp_path}"',
            allow_skills_dir=False,
        ).error
        == "Unknown option: --skills-dir"
    )


def test_parse_add_argument_can_disable_force_option() -> None:
    assert (
        parse_add_argument("alpha --force", allow_force=False).error
        == "Unknown option: --force"
    )


def test_parse_add_argument_rejects_empty_or_repeated_selector() -> None:
    assert parse_add_argument('""').error == "Selector cannot be empty."
    assert parse_add_argument("alpha beta").error == "Only one selector is allowed."


def test_parse_add_argument_reports_split_errors() -> None:
    assert parse_add_argument('alpha "unterminated').error == (
        "Invalid add arguments: No closing quotation"
    )


def test_resolve_registry_argument_accepts_raw_url_or_path() -> None:
    assert resolve_registry_argument("https://example.test/marketplace.json", []) == (
        RegistryArgument(url="https://example.test/marketplace.json")
    )
    assert resolve_registry_argument(" https://example.test/marketplace.json ", []) == (
        RegistryArgument(url="https://example.test/marketplace.json")
    )
    assert resolve_registry_argument("../marketplace.json", ["configured"]) == (
        RegistryArgument(url="../marketplace.json")
    )


def test_resolve_registry_argument_selects_one_based_index() -> None:
    assert resolve_registry_argument("2", ["first", "second"]) == RegistryArgument(url="second")
    assert resolve_registry_argument(" 1 ", ["first", "second"]) == RegistryArgument(url="first")


def test_resolve_registry_argument_reports_number_errors() -> None:
    assert resolve_registry_argument(" ", ["first"]) == RegistryArgument(
        warning="Registry URL or number is required."
    )
    assert resolve_registry_argument("1", []) == RegistryArgument(
        warning="No registries configured."
    )
    assert resolve_registry_argument("3", ["first", "second"]) == RegistryArgument(
        warning="Invalid registry number. Use 1-2."
    )


def test_parse_update_argument_rejects_empty_selector() -> None:
    assert parse_update_argument('""').error == "Selector cannot be empty."


def test_parse_update_argument_strips_or_rejects_quoted_selector() -> None:
    assert parse_update_argument('" alpha " --yes') == UpdateArgument(selector="alpha", yes=True)
    assert parse_update_argument('"   "').error == "Selector cannot be empty."


def test_parse_update_argument_rejects_unknown_short_options() -> None:
    assert parse_update_argument("-f").error == "Unknown option: -f"


def test_parse_update_argument_rejects_unknown_long_options() -> None:
    assert parse_update_argument("--bogus").error == "Unknown option: --bogus"


def test_parse_publish_argument_handles_named_fields(tmp_path: Path) -> None:
    parsed = parse_publish_argument(
        f'alpha --no-push --message="publish alpha" --temp-dir="{tmp_path}" --keep-temp'
    )

    assert parsed == PublishArgument(
        selector="alpha",
        push=False,
        message="publish alpha",
        temp_dir=tmp_path,
        keep_temp=True,
    )


def test_parse_publish_argument_last_push_flag_wins() -> None:
    assert parse_publish_argument("alpha --no-push --push").push is True
    assert parse_publish_argument("alpha --push --no-push").push is False


def test_parse_publish_argument_rejects_duplicate_value_options(tmp_path: Path) -> None:
    cases = {
        "--message first --message second": "Duplicate option: --message",
        f'--temp-dir="{tmp_path}" --temp-dir="{tmp_path / "other"}"': (
            "Duplicate option: --temp-dir"
        ),
    }

    for argument, expected_error in cases.items():
        assert parse_publish_argument(argument).error == expected_error


def test_parse_publish_argument_rejects_missing_option_values() -> None:
    cases = {
        "--message": "Missing value for --message",
        "--message=": "Missing value for --message",
        '--message "   "': "Missing value for --message",
        "--message=   ": "Missing value for --message",
        "--message --no-push": "Missing value for --message",
        "-m --no-push": "Missing value for --message",
        "--temp-dir": "Missing value for --temp-dir",
        "--temp-dir=": "Missing value for --temp-dir",
        '--temp-dir "   "': "Missing value for --temp-dir",
        "--temp-dir=   ": "Missing value for --temp-dir",
        "--temp-dir --keep-temp": "Missing value for --temp-dir",
    }

    for argument, expected_error in cases.items():
        assert parse_publish_argument(argument).error == expected_error


def test_parse_publish_argument_accepts_dash_prefixed_value_with_equals() -> None:
    assert parse_publish_argument("--message=-draft").message == "-draft"


def test_parse_publish_argument_rejects_unknown_short_options() -> None:
    assert parse_publish_argument("-x").error == "Unknown option: -x"


def test_parse_publish_argument_rejects_empty_selector() -> None:
    assert parse_publish_argument('""').error == "Selector cannot be empty."


def test_parse_publish_argument_strips_or_rejects_quoted_selector() -> None:
    assert parse_publish_argument('" alpha " --no-push') == PublishArgument(
        selector="alpha",
        push=False,
    )
    assert parse_publish_argument('"   "').error == "Selector cannot be empty."
