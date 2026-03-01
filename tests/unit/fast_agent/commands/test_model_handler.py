import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.model import (
    handle_model_reasoning,
    handle_model_verbosity,
    handle_model_web_fetch,
    handle_model_web_search,
)
from fast_agent.config import Settings, ShellSettings
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.text_verbosity import TextVerbositySpec


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
    def __init__(
        self,
        model_name: str,
        *,
        web_search_supported: bool = False,
        web_fetch_supported: bool = False,
        web_search_default: bool = False,
        web_fetch_default: bool = False,
        sampling_overrides: dict[str, float | int] | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = Provider.RESPONSES
        self.default_request_params = RequestParams()
        if sampling_overrides:
            self.default_request_params = self.default_request_params.model_copy(
                update=sampling_overrides
            )
        self.reasoning_effort_spec = ReasoningEffortSpec(
            kind="effort",
            allowed_efforts=["low", "medium", "high", "max"],
            allow_auto=True,
            default=ReasoningEffortSetting(kind="effort", value="auto"),
        )
        self.reasoning_effort = None
        self.text_verbosity_spec = TextVerbositySpec()
        self.text_verbosity = None
        self.configured_transport = "sse"
        self.active_transport = None
        self.web_search_supported = web_search_supported
        self.web_fetch_supported = web_fetch_supported
        self._web_search_default = web_search_default
        self._web_fetch_default = web_fetch_default
        self._web_search_override: bool | None = None
        self._web_fetch_override: bool | None = None

    @property
    def web_tools_enabled(self) -> tuple[bool, bool]:
        search = (
            self._web_search_override
            if self._web_search_override is not None
            else self._web_search_default
        )
        fetch = (
            self._web_fetch_override
            if self._web_fetch_override is not None
            else self._web_fetch_default
        )
        return bool(search), bool(fetch)

    @property
    def web_search_enabled(self) -> bool:
        search_enabled, _ = self.web_tools_enabled
        return search_enabled

    @property
    def web_fetch_enabled(self) -> bool:
        _, fetch_enabled = self.web_tools_enabled
        return fetch_enabled

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value is not None and not self.web_search_supported:
            raise ValueError("Current model does not support web search configuration.")
        self._web_search_override = value

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        if value is not None and not self.web_fetch_supported:
            raise ValueError("Current model does not support web fetch configuration.")
        self._web_fetch_override = value


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

    assert "Provider: responses." in text_messages
    assert "Resolved model: claude-opus-4-6." in text_messages
    assert "Model max output tokens: 128000." in text_messages
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


@pytest.mark.asyncio
async def test_model_reasoning_includes_transport_details_for_configurable_models() -> None:
    llm = _StubLLM("gpt-5.3-codex")
    llm.configured_transport = "websocket"
    llm.active_transport = "sse"
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_reasoning(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Model transports: sse, websocket." in text_messages
    assert "Configured transport: websocket." in text_messages
    assert (
        "Active transport: sse (websocket fallback was used for this turn)." in text_messages
    )


@pytest.mark.asyncio
async def test_model_reasoning_shows_model_details_when_reasoning_unsupported() -> None:
    llm = _StubLLM("gpt-4.1")
    llm.reasoning_effort_spec = None
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_reasoning(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Provider: responses." in text_messages
    assert "Resolved model: gpt-4.1." in text_messages
    assert "Current model does not support reasoning effort configuration." in text_messages


@pytest.mark.asyncio
async def test_model_reasoning_displays_sampling_overrides() -> None:
    llm = _StubLLM(
        "Qwen/Qwen3.5-397B-A17B",
        sampling_overrides={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    )
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_reasoning(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert (
        "Sampling overrides: temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, "
        "presence_penalty=0.0, repetition_penalty=1.0."
    ) in text_messages


@pytest.mark.asyncio
async def test_model_verbosity_shows_model_details_when_unsupported() -> None:
    llm = _StubLLM("o1")
    llm.text_verbosity_spec = None
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_verbosity(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Provider: responses." in text_messages
    assert "Resolved model: o1." in text_messages
    assert "Current model does not support text verbosity configuration." in text_messages


@pytest.mark.asyncio
async def test_model_web_search_reports_when_unsupported() -> None:
    llm = _StubLLM("gpt-4.1", web_search_supported=False)
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_web_search(ctx, agent_name="test", value=None)
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Current model does not support web_search configuration." in text_messages


@pytest.mark.asyncio
async def test_model_web_search_set_and_reset_to_default() -> None:
    llm = _StubLLM(
        "gpt-5",
        web_search_supported=True,
        web_search_default=False,
    )
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    set_outcome = await handle_model_web_search(ctx, agent_name="test", value="on")
    set_text = [str(m.text) for m in set_outcome.messages]
    assert "Web search: set to enabled." in set_text

    reset_outcome = await handle_model_web_search(ctx, agent_name="test", value="default")
    reset_text = [str(m.text) for m in reset_outcome.messages]
    assert "Web search: set to default (disabled)." in reset_text


@pytest.mark.asyncio
async def test_model_web_fetch_set_when_supported() -> None:
    llm = _StubLLM(
        "claude-sonnet-4-6",
        web_fetch_supported=True,
        web_fetch_default=False,
    )
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_web_fetch(ctx, agent_name="test", value="on")
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Web fetch: set to enabled." in text_messages


@pytest.mark.asyncio
async def test_model_web_search_rejects_invalid_value() -> None:
    llm = _StubLLM("gpt-5", web_search_supported=True)
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_web_search(ctx, agent_name="test", value="maybe")
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Invalid web_search value 'maybe'. Allowed values: on, off, default." in text_messages
