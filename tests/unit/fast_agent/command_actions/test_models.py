from types import SimpleNamespace

from fast_agent.command_actions.models import PluginCommandActionContext
from fast_agent.llm.usage_tracking import UsageAccumulator


def _agent(usage: UsageAccumulator):
    return SimpleNamespace(usage_accumulator=usage)


def test_plugin_command_context_exposes_agent_usage() -> None:
    usage = UsageAccumulator()
    context = PluginCommandActionContext(
        command_name="cost",
        arguments="",
        agent=_agent(usage),
    )

    assert context.usage is usage
