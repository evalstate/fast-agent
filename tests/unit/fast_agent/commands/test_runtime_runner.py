import asyncio

import pytest

from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.cli.runtime.runner import run_request


def _request(*, timeout_seconds: int | None = None) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=None,
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message="hello",
        prompt_file=None,
        result_file=None,
        resume=None,
        url_servers=None,
        stdio_servers=None,
        agent_name="agent",
        target_agent_name=None,
        skills_directory=None,
        environment_dir=None,
        noenv=False,
        force_smart=False,
        shell_runtime=False,
        no_shell=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        timeout_seconds=timeout_seconds,
    )


def test_run_request_exits_124_when_timeout_fires(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def slow_run_agent_request(_request: AgentRunRequest) -> None:
        await asyncio.sleep(10)

    monkeypatch.setattr(
        "fast_agent.cli.runtime.agent_setup.run_agent_request",
        slow_run_agent_request,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_request(_request(timeout_seconds=1))

    captured = capsys.readouterr()
    assert exc_info.value.code == 124
    assert "fast-agent timed out after 1 second." in captured.err
