from fast_agent.marketplace import formatting


def test_format_revision_short_for_commit_hash() -> None:
    assert formatting.format_revision_short("0123456789abcdef") == "0123456"


def test_format_revision_short_for_named_revision() -> None:
    assert formatting.format_revision_short("main") == "main"


def test_format_installed_at_display_with_z_suffix() -> None:
    assert (
        formatting.format_installed_at_display("2026-02-25T01:02:03Z")
        == "2026-02-25 01:02:03"
    )
