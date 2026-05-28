import asyncio
from typing import TYPE_CHECKING, cast

import pytest

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
        noenv_mode=False,
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
