from fast_agent.cli.constants import GO_SPECIFIC_OPTIONS, KNOWN_SUBCOMMANDS


def test_known_subcommands_includes_acp() -> None:
    assert "acp" in KNOWN_SUBCOMMANDS


def test_go_specific_options_include_results() -> None:
    assert "--results" in GO_SPECIFIC_OPTIONS


def test_go_specific_options_include_agent_and_noenv() -> None:
    assert "--agent" in GO_SPECIFIC_OPTIONS
    assert "--noenv" in GO_SPECIFIC_OPTIONS
    assert "--no-env" in GO_SPECIFIC_OPTIONS
