from types import SimpleNamespace

import pytest

from fast_agent.cli.runtime.harness_startup import (
    _display_cli_usage_report,
    _run_flow_with_usage_report,
)
from fast_agent.core.exceptions import PromptExitError
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)


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
