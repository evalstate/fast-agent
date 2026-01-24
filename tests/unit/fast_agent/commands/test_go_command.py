from pathlib import Path

from fast_agent.cli.commands import go as go_command


def test_run_async_agent_passes_card_tools() -> None:
    run_kwargs = go_command._build_run_agent_kwargs(
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
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        environment_dir=None,
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

    assert run_kwargs["card_tools"] == ["./tool-cards"]


def test_run_async_agent_merges_default_tool_cards(tmp_path: Path) -> None:
    tool_dir = tmp_path / "tool-cards"
    tool_dir.mkdir()
    (tool_dir / "sizer.md").write_text("---\nname: sizer\n---\n")

    run_kwargs = go_command._build_run_agent_kwargs(
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
        resume=None,
        stdio_commands=None,
        agent_name="agent",
        skills_directory=None,
        environment_dir=tmp_path,
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

    assert run_kwargs["card_tools"] == [str(tool_dir)]
