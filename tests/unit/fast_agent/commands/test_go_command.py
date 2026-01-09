from pathlib import Path

from fast_agent.cli.commands import go as go_command


def _patch_run_async(monkeypatch, captured: dict) -> None:
    async def fake_run_agent(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(go_command, "_run_agent", fake_run_agent)
    monkeypatch.setattr(go_command, "_set_asyncio_exception_handler", lambda loop: None)


def test_run_async_agent_passes_card_tools(monkeypatch) -> None:
    captured: dict = {}
    _patch_run_async(monkeypatch, captured)

    go_command.run_async_agent(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        agent_cards=["./agents"],
        card_tools=["./tool-cards"],
        model=None,
        message=None,
        prompt_file=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert captured["card_tools"] == ["./tool-cards"]


def test_run_async_agent_merges_default_tool_cards(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}
    _patch_run_async(monkeypatch, captured)

    tool_dir = tmp_path / "tool-cards"
    tool_dir.mkdir()
    (tool_dir / "sizer.md").write_text("---\nname: sizer\n---\n")

    monkeypatch.setattr(go_command, "DEFAULT_AGENT_CARDS_DIR", tmp_path / "agent-cards")
    monkeypatch.setattr(go_command, "DEFAULT_TOOL_CARDS_DIR", tool_dir)

    go_command.run_async_agent(
        name="test-agent",
        instruction="test instruction",
        config_path=None,
        servers=None,
        urls=None,
        auth=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        shell_enabled=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    assert captured["card_tools"] == [str(tool_dir)]
