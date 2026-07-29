from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.cli.runtime.agent_setup import _classify_cli_mcp_failure
from fast_agent.cli.runtime.harness_startup import (
    _display_cli_usage_report,
    _run_flow_with_usage_report,
)
from fast_agent.config import MCPServerSettings
from fast_agent.core.exceptions import PromptExitError, ServerInitializationError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)

if TYPE_CHECKING:
    from fast_agent.cli.runtime.run_request import AgentRunRequest


class _AgentProvider:
    def __init__(self, agent: object) -> None:
        self._agents = {"dev": agent}

    def registered_agents(self) -> dict[str, object]:
        return self._agents


def _usage_provider() -> _AgentProvider:
    usage = UsageAccumulator()
    usage.add_turn(
        TurnUsage(
            provider=Provider.RESPONSES,
            usage_schema=UsageSchema.OPENAI_RESPONSES,
            model="gpt-test",
            prompt=PromptTokenUsage(total=100, cache_read=85),
            completion=CompletionTokenUsage(total=20),
            tool_calls=1,
        )
    )
    return _AgentProvider(
        SimpleNamespace(
            usage_accumulator=usage,
            llm=None,
        )
    )


def test_harness_cli_displays_usage_before_session_disposal(capsys) -> None:
    _display_cli_usage_report(_usage_provider(), quiet=False)

    output = capsys.readouterr().out
    assert "Usage Summary" in output
    assert "Cache hit" in output
    assert "85%" in output


def test_harness_cli_omits_usage_in_quiet_mode(capsys) -> None:
    _display_cli_usage_report(_AgentProvider(object()), quiet=True)

    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_harness_cli_displays_usage_on_prompt_exit(capsys) -> None:
    async def exit_flow() -> None:
        raise PromptExitError("exit")

    with pytest.raises(PromptExitError):
        await _run_flow_with_usage_report(
            exit_flow(),
            agent_app=_usage_provider(),
            quiet=False,
        )

    assert "Usage Summary" in capsys.readouterr().out


def test_startup_url_failure_uses_typed_cli_boundary() -> None:
    request = SimpleNamespace(
        startup_mcp_servers={
            "docs": MCPServerSettings(
                transport="http",
                url="https://user:pass@example.com/mcp?token=secret",
            )
        },
        config_path=None,
    )
    fast = SimpleNamespace(
        app=SimpleNamespace(
            context=SimpleNamespace(
                server_registry=SimpleNamespace(get_server_origin=lambda _name: "runtime")
            )
        )
    )
    cause = ServerInitializationError(
        "MCP startup timed out",
        server_name="docs",
    )
    cause.__cause__ = TimeoutError("startup budget expired")

    failure = _classify_cli_mcp_failure(fast, cast("AgentRunRequest", request), cause)

    assert failure is not None
    assert failure.server_name == "docs"
    assert failure.origin == "session"
    assert failure.surface == "startup_url"
    assert failure.kind == "timeout"
    assert "secret" not in failure.input_ref
