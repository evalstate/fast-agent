import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.core.exceptions import PromptExitError
from fast_agent.core.fastagent import FastAgent, RunSettings

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol


class _BlockingAgent:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def shutdown(self) -> None:
        self.started.set()
        await asyncio.sleep(3600)


@pytest.mark.asyncio
async def test_finalize_run_limits_shutdown_time_after_exit_request() -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False)
    blocking_agent = _BlockingAgent()
    settings = RunSettings(
        quiet_mode=True,
        cli_model_override=None,
        no_home_mode=False,
        server_mode=False,
        transport=None,
        is_acp_server_mode=False,
        reload_enabled=False,
    )

    await asyncio.wait_for(
        fast._finalize_run(
            None,
            {"agent": cast("AgentProtocol", blocking_agent)},
            had_error=False,
            settings=settings,
            shutdown_timeout=0.01,
        ),
        timeout=0.2,
    )

    assert blocking_agent.started.is_set()


@pytest.mark.asyncio
async def test_direct_run_treats_prompt_exit_as_clean_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("TestAgent", parse_cli_args=False, quiet=True)

    @fast.agent(name="main", model="passthrough", default=True)
    async def main() -> None:
        pass

    handled_errors: list[Exception] = []
    monkeypatch.setattr(fast, "_handle_error", handled_errors.append)

    with pytest.raises(SystemExit) as exc_info:
        async with fast.run():
            raise PromptExitError("exit")

    assert exc_info.value.code == 0
    assert handled_errors == []
