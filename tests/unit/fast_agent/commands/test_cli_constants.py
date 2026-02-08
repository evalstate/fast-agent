from fast_agent.cli.constants import KNOWN_SUBCOMMANDS


def test_known_subcommands_includes_acp() -> None:
    assert "acp" in KNOWN_SUBCOMMANDS
