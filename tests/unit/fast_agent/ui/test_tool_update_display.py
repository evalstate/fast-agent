import pytest

from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay


@pytest.mark.asyncio
async def test_tool_update_fallback_escapes_markup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prompt_toolkit.application import current

    def raise_no_app() -> None:
        raise RuntimeError("no running app")

    monkeypatch.setattr(current, "get_app", raise_no_app)
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        await display.show_tool_update("server [draft]", agent_name="agent [dev]")

    rendered = capture.get()
    assert "agent [dev]" in rendered
    assert "server [draft]" in rendered
