import asyncio

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.context import Context
from fast_agent.interfaces import ToolRunnerHookCapable
from fast_agent.types import PromptMessageExtended


def get_video_call_transcript(video_id: str) -> str:
    return "Assistant: Hi, how can I assist you today?\n\nCustomer: Hi, I wanted to ask you about last invoice I received..."


class HookedToolAgent(ToolAgent, ToolRunnerHookCapable):
    def __init__(
        self,
        config: AgentConfig,
        context: Context | None = None,
    ):
        tools = [get_video_call_transcript]
        super().__init__(config, tools, context)
        self._hooks = ToolRunnerHooks(
            before_llm_call=self._add_style_hint,
            after_tool_call=self._log_tool_result,
        )

    @property
    def tool_runner_hooks(self) -> ToolRunnerHooks | None:
        return self._hooks

    async def _add_style_hint(self, runner, messages: list[PromptMessageExtended]) -> None:
        if runner.iteration == 0:
            runner.append_messages("Keep the answer to one short sentence.")

    async def _log_tool_result(self, runner, message: PromptMessageExtended) -> None:
        if message.tool_results:
            tool_names = ", ".join(message.tool_results.keys())
            print(f"[hook] tool results received: {tool_names}")


fast = FastAgent("Example Tool Use Application (Hooks)")


@fast.custom(HookedToolAgent)
async def main() -> None:
    async with fast.run() as agent:
        await agent.default.generate(
            "What is the topic of the video call no.1234?",
        )
        await agent.interactive()


if __name__ == "__main__":
    asyncio.run(main())
