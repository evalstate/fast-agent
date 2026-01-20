import pytest

import fast_agent.core.fastagent as fastagent_module
import fast_agent.ui.progress_display as progress_display_module
from fast_agent.core.fastagent import FastAgent


@pytest.mark.asyncio
async def test_stdio_server_routes_console_before_progress_stop(monkeypatch):
    agent = FastAgent("TestAgent", parse_cli_args=False)
    agent.args.server = True
    agent.args.transport = "stdio"
    agent.args.quiet = False

    state = {"configured": False}

    def fake_configure(stream):
        assert stream == "stderr"
        state["configured"] = True

    def fake_stop():
        assert state["configured"] is True
        raise RuntimeError("stop")

    monkeypatch.setattr(fastagent_module, "configure_console_stream", fake_configure)
    monkeypatch.setattr(progress_display_module.progress_display, "stop", fake_stop)

    with pytest.raises(RuntimeError, match="stop"):
        async with agent.run():
            pass
