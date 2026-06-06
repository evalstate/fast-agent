from fast_agent.marketplace import formatting


def test_format_revision_short_for_commit_hash() -> None:
    assert formatting.format_revision_short("0123456789abcdef") == "0123456"


def test_format_revision_short_for_uppercase_commit_hash() -> None:
    assert formatting.format_revision_short("ABCDEF0123456789") == "ABCDEF0"


def test_format_revision_short_keeps_short_hex_revision() -> None:
    assert formatting.format_revision_short("deadbee") == "deadbee"


def test_format_revision_short_for_named_revision() -> None:
    assert formatting.format_revision_short("main") == "main"


def test_format_installed_at_display_with_z_suffix() -> None:
    assert formatting.format_installed_at_display("2026-02-25T01:02:03Z") == "2026-02-25 01:02:03"


def test_format_installed_at_display_strips_invalid_timestamp_fallback() -> None:
    assert formatting.format_installed_at_display("  not-a-date  ") == "not-a-date"


def test_format_installed_revision_display_uses_short_revision() -> None:
    assert (
        formatting.format_installed_revision_display(
            "2026-02-25T01:02:03Z",
            "0123456789abcdef",
        )
        == "2026-02-25 01:02:03 revision: 0123456"
    )


def test_format_installed_revision_display_supports_table_separator() -> None:
    assert (
        formatting.format_installed_revision_display(
            "2026-02-25T01:02:03Z",
            "main",
            separator=" · ",
            revision_label="",
        )
        == "2026-02-25 01:02:03 · main"
    )


def test_format_source_provenance_includes_optional_ref() -> None:
    assert (
        formatting.format_source_provenance(
            "https://github.com/example/skills",
            "main",
            "skills/alpha",
        )
        == "https://github.com/example/skills@main (skills/alpha)"
    )
    assert (
        formatting.format_source_provenance(
            "https://github.com/example/skills",
            None,
            "skills/alpha",
        )
        == "https://github.com/example/skills (skills/alpha)"
    )


def test_format_source_provenance_strips_blank_source_parts() -> None:
    assert (
        formatting.format_source_provenance(
            " https://github.com/example/skills ",
            "  ",
            " skills/alpha ",
        )
        == "https://github.com/example/skills (skills/alpha)"
    )
    assert (
        formatting.format_source_provenance(
            " https://github.com/example/skills ",
            " main ",
            " skills/alpha ",
        )
        == "https://github.com/example/skills@main (skills/alpha)"
    )
