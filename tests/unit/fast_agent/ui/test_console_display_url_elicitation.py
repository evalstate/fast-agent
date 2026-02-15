from fast_agent.ui.console_display import ConsoleDisplay


def test_show_url_elicitation_renders_ordered_output(capsys) -> None:
    display = ConsoleDisplay()

    display.show_url_elicitation(
        message="Please open this link",
        url="https://example.com",
        server_name="localhost",
        elicitation_id="abc-123",
    )

    output = capsys.readouterr().out
    assert output.startswith("\n")

    lines = [line for line in output.splitlines() if line.strip()]
    assert lines[0] == "● URL elicitation required"
    assert lines[1] == "  [localhost] Please open this link"
    assert lines[2] == "  elicitationId: abc-123"
    assert lines[3] == "  example.com → https://example.com"
    assert lines[4] == "  Open URL"


def test_show_url_elicitation_hides_empty_elicitation_id(capsys) -> None:
    display = ConsoleDisplay()

    display.show_url_elicitation(
        message="Please open this link",
        url="https://example.com",
        server_name="localhost",
        elicitation_id=None,
    )

    output = capsys.readouterr().out
    assert "elicitationId:" not in output
