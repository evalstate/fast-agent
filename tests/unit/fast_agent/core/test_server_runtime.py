from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.server_runtime import (
    ServerRuntimeContext,
    run_mcp_server,
)

if TYPE_CHECKING:
    from fast_agent.core.fastagent import ManagedRunState, RunSettings, RuntimeCallbacks


class _Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config


@pytest.mark.asyncio
async def test_run_mcp_server_threads_managed_agent_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_options = []

    async def fake_run_harness_mcp_app_server(**kwargs):
        captured_options.append(kwargs["options"])

    monkeypatch.setattr(
        "fast_agent.mcp.server.harness_app_server.run_harness_mcp_app_server",
        fake_run_harness_mcp_app_server,
    )
    context = ServerRuntimeContext(
        app_name="demo",
        args=SimpleNamespace(
            server_name=None,
            server_description=None,
            agent="reviewer",
            managed_mcp_agent_names=["reviewer"],
            transport="http",
            host="127.0.0.1",
            port=8000,
            instance_scope="shared",
        ),
        callbacks=cast("RuntimeCallbacks", SimpleNamespace(instance_factory=lambda: object())),
        state=cast(
            "ManagedRunState",
            SimpleNamespace(
                primary_instance=SimpleNamespace(
                    app=AgentApp(
                        {
                            "reviewer": cast(
                                "Any",
                                _Agent(
                                    AgentConfig(
                                        name="reviewer",
                                        description="Review code.",
                                        tool_input_schema={
                                            "type": "object",
                                            "properties": {"diff": {"type": "string"}},
                                            "required": ["diff"],
                                        },
                                    )
                                ),
                            )
                        }
                    ),
                    agents={"reviewer": object()},
                ),
                runtime=SimpleNamespace(shell_environment=object()),
            ),
        ),
        config=None,
        skills_directory_override=None,
        settings=cast("RunSettings", SimpleNamespace()),
        acp_server_factory=lambda: object,
    )

    await run_mcp_server(context)

    assert captured_options[0].default_agent == "reviewer"
    assert captured_options[0].managed_agent_tools[0].name == "reviewer"
    assert captured_options[0].managed_agent_tools[0].agent == "reviewer"
    assert captured_options[0].managed_agent_tools[0].description == "Review code."
    assert captured_options[0].managed_agent_tools[0].input_schema["required"] == ["diff"]
