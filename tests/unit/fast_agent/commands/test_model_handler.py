import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.model import handle_model_reasoning
from fast_agent.config import Settings, ShellSettings
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec


class _StubIO:
    async def emit(self, message):  # type: ignore[no-untyped-def]
        return None

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):  # type: ignore[no-untyped-def]
        return default

    async def prompt_selection(
        self, prompt: str, *, options, allow_cancel=False, default=None
    ):  # type: ignore[no-untyped-def]
        return default

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):  # type: ignore[no-untyped-def]
        return None

    async def display_history_turn(self, agent_name: str, turn, *, turn_index=None, total_turns=None):  # type: ignore[no-untyped-def]
        return None

    async def display_history_overview(self, agent_name: str, history, usage=None):  # type: ignore[no-untyped-def]
        return None

    async def display_usage_report(self, agents):  # type: ignore[no-untyped-def]
        return None

    async def display_system_prompt(self, agent_name: str, system_prompt: str, *, server_count=0):  # type: ignore[no-untyped-def]
        return None


class _StubLLM:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.reasoning_effort_spec = ReasoningEffortSpec(
            kind="effort",
            allowed_efforts=["low", "medium", "high", "max"],
            allow_auto=True,
            default=ReasoningEffortSetting(kind="effort", value="auto"),
        )
        self.reasoning_effort = None


class _StubShellRuntime:
    def __init__(self, output_byte_limit: int) -> None:
        self.output_byte_limit = output_byte_limit


class _StubAgent:
    def __init__(self, llm: _StubLLM, shell_limit: int | None = None) -> None:
        self.llm = llm
        self._llm = llm
        self.shell_runtime = _StubShellRuntime(shell_limit) if shell_limit is not None else None


class _StubAgentProvider:
    def __init__(self, agent: _StubAgent) -> None:
        self._instance = agent

    def _agent(self, name: str) -> _StubAgent:  # noqa: ARG002
        return self._instance

    def agent_names(self):
        return ["test"]

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):  # noqa: ARG002
        return {}


@pytest.mark.asyncio
async def test_model_reasoning_includes_shell_budget_details() -> None:
    llm = _StubLLM("claude-opus-4-6")
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=21120))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_reasoning(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Resolved model: claude-opus-4-6." in text_messages
    assert "Model max output tokens: 32000." in text_messages
    assert "Shell output budget: 21120 bytes (~6400 tokens, active runtime)." in text_messages


@pytest.mark.asyncio
async def test_model_reasoning_includes_config_override_budget_when_runtime_missing() -> None:
    llm = _StubLLM("claude-opus-4-6")
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(shell_execution=ShellSettings(output_byte_limit=9000)),
    )

    outcome = await handle_model_reasoning(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Shell output budget: 9000 bytes (~2727 tokens, config override)." in text_messages
