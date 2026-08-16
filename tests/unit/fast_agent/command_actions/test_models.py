from types import SimpleNamespace

from fast_agent.command_actions.models import PluginCommandActionContext
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
    UserTurnUsage,
)


def _agent(usage: UsageAccumulator):
    return SimpleNamespace(usage_accumulator=usage)


def test_plugin_command_context_exposes_agent_usage() -> None:
    usage = UsageAccumulator()
    attempt = TurnUsage(
        provider=Provider.OPENAI,
        usage_schema=UsageSchema.OPENAI_CHAT,
        model="test",
        prompt=PromptTokenUsage(total=10),
        completion=CompletionTokenUsage(total=2),
    )
    user_turn = UserTurnUsage(agent_name="assistant", attempts=(attempt,))
    context = PluginCommandActionContext(
        command_name="cost",
        arguments="",
        agent=_agent(usage),
        user_turn_usage=(user_turn,),
    )

    assert context.usage is usage
    assert context.user_turn_usage == (user_turn,)
