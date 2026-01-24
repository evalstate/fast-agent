import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers import prompts as prompt_handlers
from fast_agent.mcp.mcp_aggregator import SEP


class StubPromptArgument:
    def __init__(self, name: str, required: bool, description: str | None = None) -> None:
        self.name = name
        self.required = required
        self.description = description


class StubPrompt:
    def __init__(
        self,
        name: str,
        *,
        title: str | None = None,
        description: str | None = None,
        arguments: list[StubPromptArgument] | None = None,
    ) -> None:
        self.name = name
        self.title = title
        self.description = description
        self.arguments = arguments


class StubPromptResult:
    def __init__(self) -> None:
        self.messages = ["message"]


class StubAgent:
    def __init__(self, prompt_result: StubPromptResult) -> None:
        self._prompt_result = prompt_result
        self.prompt_calls: list[tuple[str, dict[str, str]]] = []
        self.generated_messages: list[object] | None = None

    async def get_prompt(self, namespaced_name: str, arg_values: dict[str, str]) -> StubPromptResult:
        self.prompt_calls.append((namespaced_name, arg_values))
        return self._prompt_result

    async def generate(self, messages, _):  # type: ignore[no-untyped-def]
        self.generated_messages = messages


class StubAgentProvider:
    def __init__(self, prompts: dict[str, list[StubPrompt]], agent: StubAgent) -> None:
        self._prompts = prompts
        self._agent_instance = agent

    def _agent(self, name: str) -> StubAgent:
        return self._agent_instance

    def agent_names(self):
        return ["test-agent"]

    async def list_prompts(self, namespace, agent_name=None):  # type: ignore[no-untyped-def]
        return self._prompts


class StubCommandIO:
    def __init__(self, arg_values: dict[str, str]) -> None:
        self._arg_values = arg_values
        self.prompted_args: list[tuple[str, str | None, bool]] = []
        self.emitted: list[object] = []

    async def emit(self, message):  # type: ignore[no-untyped-def]
        self.emitted.append(message)

    async def prompt_text(
        self,
        prompt: str,
        *,
        default: str | None = None,
        allow_empty: bool = True,
    ) -> str | None:
        return default

    async def prompt_selection(
        self,
        prompt: str,
        *,
        options,
        allow_cancel: bool = False,
        default: str | None = None,
    ) -> str | None:
        return default

    async def prompt_argument(
        self,
        arg_name: str,
        *,
        description: str | None = None,
        required: bool = True,
    ) -> str | None:
        self.prompted_args.append((arg_name, description, required))
        return self._arg_values.get(arg_name)

    async def display_history_turn(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    async def display_history_overview(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    async def display_usage_report(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None

    async def display_system_prompt(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


@pytest.mark.asyncio
async def test_handle_select_prompt_prompts_for_required_args(monkeypatch):
    prompt_args = [
        StubPromptArgument("topic", True, "Choose a topic"),
        StubPromptArgument("style", False, "Optional style"),
    ]
    prompt_obj = StubPrompt("demo", description="demo prompt", arguments=prompt_args)
    prompt_result = StubPromptResult()
    agent = StubAgent(prompt_result)
    provider = StubAgentProvider({"server": [prompt_obj]}, agent)
    io = StubCommandIO({"topic": "cats", "style": "haiku"})
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test-agent",
        io=io,
    )

    monkeypatch.setattr(
        prompt_handlers.PromptMessageExtended,
        "from_get_prompt_result",
        lambda result: ["converted"],
    )
    monkeypatch.setattr(prompt_handlers.progress_display, "resume", lambda: None)
    monkeypatch.setattr(prompt_handlers.progress_display, "pause", lambda: None)

    await prompt_handlers.handle_select_prompt(ctx, agent_name="test-agent", prompt_index=1)

    assert io.prompted_args == [
        ("topic", "Choose a topic", True),
        ("style", "Optional style", False),
    ]
    assert agent.prompt_calls == [
        (f"server{SEP}demo", {"topic": "cats", "style": "haiku"}),
    ]
