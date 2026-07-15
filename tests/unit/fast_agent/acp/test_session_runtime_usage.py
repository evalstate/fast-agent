from typing import cast

from fast_agent.acp.server.session_runtime import ACPServerSessionRuntime, SessionRuntimeHost
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageSchema,
)


def test_build_status_line_meta_uses_canonical_turn_totals() -> None:
    agent = LlmAgent(AgentConfig("status-line"))
    llm = ResponsesLLM(provider=Provider.RESPONSES, model="gpt-5.3-codex")
    llm.usage_accumulator.add_turn(
        TurnUsage(
            provider=Provider.RESPONSES,
            usage_schema=UsageSchema.OPENAI_RESPONSES,
            model="gpt-5.3-codex",
            prompt=PromptTokenUsage(total=120),
            completion=CompletionTokenUsage(total=30),
            tool_calls=2,
        )
    )
    agent._llm = llm
    runtime = ACPServerSessionRuntime(cast("SessionRuntimeHost", object()))

    assert runtime.build_status_line_meta(agent, 0) == {
        "field_meta": {
            "openhands.dev/metrics": {
                "status_line": "120 in, 30 out, 2 tools (0.1%)",
            }
        }
    }
