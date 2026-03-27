from pathlib import Path

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.model import (
    handle_model_fast,
    handle_model_reasoning,
    handle_model_switch,
    handle_model_verbosity,
    handle_model_web_fetch,
    handle_model_web_search,
)
from fast_agent.config import Settings, ShellSettings
from fast_agent.core.exceptions import ModelConfigError
from fast_agent.llm.model_database import ModelParameters
from fast_agent.llm.model_factory import ModelConfig
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.model_overlays import LoadedModelOverlay, ModelOverlayManifest
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec
from fast_agent.llm.request_params import RequestParams
from fast_agent.llm.resolved_model import ResolvedModelSpec, resolve_base_model_params
from fast_agent.llm.text_verbosity import TextVerbositySpec


def _build_overlay(
    name: str,
    *,
    provider: Provider,
    model_name: str,
) -> LoadedModelOverlay:
    return LoadedModelOverlay(
        manifest=ModelOverlayManifest.model_validate(
            {
                "name": name,
                "provider": provider.value,
                "model": model_name,
                "picker": {"label": name},
            }
        ),
        manifest_path=Path(f"/tmp/{name}.yaml"),
    )


class _StubIO:
    def __init__(self, *, model_selection_response: str | None = None) -> None:
        self._model_selection_response = model_selection_response
        self.last_initial_provider: str | None = None
        self.last_default_model: str | None = None

    async def emit(self, message):
        return None

    async def prompt_text(self, prompt: str, *, default=None, allow_empty=True):
        return default

    async def prompt_selection(
        self, prompt: str, *, options, allow_cancel=False, default=None
    ):
        return default

    async def prompt_model_selection(
        self,
        *,
        initial_provider=None,
        default_model=None,
    ):
        self.last_initial_provider = initial_provider
        self.last_default_model = default_model
        return self._model_selection_response

    async def prompt_argument(self, arg_name: str, *, description=None, required=True):
        return None

    async def display_history_turn(self, agent_name: str, turn, *, turn_index=None, total_turns=None):
        return None

    async def display_history_overview(self, agent_name: str, history, usage=None):
        return None

    async def display_usage_report(self, agents):
        return None

    async def display_system_prompt(self, agent_name: str, system_prompt: str, *, server_count=0):
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
        service_tier_supported: bool = False,
        service_tier_default: str | None = None,
        available_service_tiers: tuple[str, ...] | None = None,
        sampling_overrides: dict[str, float | int] | None = None,
        provider: Provider = Provider.RESPONSES,
        selected_model_name: str | None = None,
        model_info_override: ModelInfo | None = None,
        overlay_name: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.provider = provider
        self._model_info_override = model_info_override
        model_params = resolve_base_model_params(provider=provider, model_name=model_name)
        if model_info_override is not None:
            model_params = ModelParameters(
                context_window=model_info_override.context_window or 0,
                max_output_tokens=model_info_override.max_output_tokens or 0,
                tokenizes=model_info_override.tokenizes,
                json_mode=model_info_override.json_mode,
                reasoning=model_info_override.reasoning,
                default_provider=provider,
            )
        self.resolved_model = ResolvedModelSpec(
            raw_input=selected_model_name or model_name,
            selected_model_name=selected_model_name or model_name,
            source="overlay" if overlay_name is not None else "direct",
            model_config=ModelConfig(provider=provider, model_name=model_name),
            provider=provider,
            wire_model_name=model_name,
            overlay=(
                _build_overlay(overlay_name, provider=provider, model_name=model_name)
                if overlay_name is not None
                else None
            ),
            model_params=model_params,
        )
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
        self.service_tier_supported = service_tier_supported
        if available_service_tiers is None and service_tier_supported:
            available_service_tiers = ("fast", "flex")
        self.available_service_tiers = available_service_tiers or ()
        self._service_tier = service_tier_default
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

    @property
    def service_tier(self) -> str | None:
        return self._service_tier

    @property
    def model_info(self) -> ModelInfo | None:
        if self._model_info_override is not None:
            return self._model_info_override
        return ModelInfo.from_name(self.model_name, self.provider)

    def set_web_search_enabled(self, value: bool | None) -> None:
        if value is not None and not self.web_search_supported:
            raise ValueError("Current model does not support web search configuration.")
        self._web_search_override = value

    def set_web_fetch_enabled(self, value: bool | None) -> None:
        if value is not None and not self.web_fetch_supported:
            raise ValueError("Current model does not support web fetch configuration.")
        self._web_fetch_override = value

    def set_service_tier(self, value: str | None) -> None:
        if value is not None and not self.service_tier_supported:
            raise ValueError("Current model does not support service tier configuration.")
        if value is not None and value not in self.available_service_tiers:
            allowed = ", ".join(self.available_service_tiers) or "standard"
            raise ValueError(
                f"Current model supports only {allowed} or unset (standard) service tier."
            )
        self._service_tier = value


class _StubShellRuntime:
    def __init__(self, output_byte_limit: int) -> None:
        self.output_byte_limit = output_byte_limit


class _StubAgent:
    def __init__(
        self,
        llm: _StubLLM,
        shell_limit: int | None = None,
        *,
        set_model_error: Exception | None = None,
    ) -> None:
        self.llm = llm
        self._llm = llm
        self.shell_runtime = _StubShellRuntime(shell_limit) if shell_limit is not None else None
        self.config = type("Config", (), {"model": llm.model_name})()
        self._set_model_error = set_model_error

    async def set_model(self, model: str | None) -> None:
        if self._set_model_error is not None:
            raise self._set_model_error
        self.config.model = model
        if model is not None:
            self.llm.model_name = model
            self._llm.model_name = model
            self.llm.resolved_model = ResolvedModelSpec(
                raw_input=model,
                selected_model_name=model,
                source="direct",
                model_config=ModelConfig(provider=self.llm.provider, model_name=model),
                provider=self.llm.provider,
                wire_model_name=model,
            )
            self._llm.resolved_model = self.llm.resolved_model

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts


class _StubAgentProvider:
    def __init__(self, agent: _StubAgent) -> None:
        self._instance = agent

    def _agent(self, name: str) -> _StubAgent:  # noqa: ARG002
        return self._instance

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["test"]

    def registered_agent_names(self):
        return ["test"]

    def registered_agents(self):
        return {"test": self._instance}

    def resolve_target_agent_name(self, agent_name: str | None = None):
        return agent_name or "test"

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
    assert "Selected model: claude-opus-4-6." in text_messages
    assert "Display model: claude-opus-4-6." in text_messages
    assert "Wire model: claude-opus-4-6." in text_messages
    assert "Model max output tokens: 128000." in text_messages
    assert "Shell output budget: 21120 bytes (~6.4k tokens, active runtime)." in text_messages


@pytest.mark.asyncio
async def test_model_reasoning_is_overlay_aware_for_context_and_output_limits() -> None:
    llm = _StubLLM(
        "claude-haiku-4-5",
        provider=Provider.ANTHROPIC,
        selected_model_name="haikutiny",
        overlay_name="haikutiny",
        model_info_override=ModelInfo(
            name="claude-haiku-4-5",
            provider=Provider.ANTHROPIC,
            context_window=8192,
            max_output_tokens=1024,
            tokenizes=["text/plain"],
            json_mode=None,
            reasoning=None,
        ),
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

    assert "Provider: anthropic." in text_messages
    assert "Selected model: haikutiny." in text_messages
    assert "Display model: haikutiny." in text_messages
    assert "Wire model: claude-haiku-4-5." in text_messages
    assert "Context window: 8192." in text_messages
    assert "Model max output tokens: 1024." in text_messages


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

    assert "Shell output budget: 9000 bytes (~2.7k tokens, config override)." in text_messages


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
async def test_model_reasoning_includes_runtime_setting_details_when_available() -> None:
    llm = _StubLLM(
        "gpt-5.4",
        web_search_supported=True,
        web_search_default=True,
        web_fetch_supported=True,
        web_fetch_default=False,
        service_tier_supported=True,
        service_tier_default="flex",
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

    assert "Text verbosity: medium." in text_messages
    assert "Service tier: flex." in text_messages
    assert "Web search: enabled." in text_messages
    assert "Web fetch: disabled." in text_messages


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
    assert "Selected model: gpt-4.1." in text_messages
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
async def test_model_reasoning_rounds_noisy_sampling_override_floats() -> None:
    llm = _StubLLM(
        "unsloth/Qwen3.5-9B-GGUF",
        sampling_overrides={
            "temperature": 0.800000011920929,
            "top_p": 0.949999988079071,
            "top_k": 40,
            "min_p": 0.05000000074505806,
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

    assert "Sampling overrides: temperature=0.8, top_p=0.95, top_k=40, min_p=0.05." in text_messages


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
    assert "Selected model: o1." in text_messages
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
async def test_model_fast_reports_when_unsupported() -> None:
    llm = _StubLLM("gpt-4.1", service_tier_supported=False)
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_fast(ctx, agent_name="test", value="status")
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Current model does not support service tier configuration." in text_messages


@pytest.mark.asyncio
async def test_model_fast_toggle_and_status() -> None:
    llm = _StubLLM("gpt-5", service_tier_supported=True)
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    toggle_outcome = await handle_model_fast(ctx, agent_name="test", value=None)
    toggle_text = [str(m.text) for m in toggle_outcome.messages]
    assert "Service tier: set to fast." in toggle_text

    status_outcome = await handle_model_fast(ctx, agent_name="test", value="status")
    status_text = [str(m.text) for m in status_outcome.messages]
    assert "Service tier: fast. Allowed values: on, off, flex, status." in status_text

    off_outcome = await handle_model_fast(ctx, agent_name="test", value="off")
    off_text = [str(m.text) for m in off_outcome.messages]
    assert "Service tier: set to default." in off_text

    flex_outcome = await handle_model_fast(ctx, agent_name="test", value="flex")
    flex_text = [str(m.text) for m in flex_outcome.messages]
    assert "Service tier: set to flex." in flex_text


@pytest.mark.asyncio
async def test_model_fast_codexresponses_omits_flex_value() -> None:
    llm = _StubLLM(
        "gpt-5.4",
        service_tier_supported=True,
        available_service_tiers=("fast",),
    )
    llm.provider = Provider.CODEX_RESPONSES
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    status_outcome = await handle_model_fast(ctx, agent_name="test", value="status")
    status_text = [str(m.text) for m in status_outcome.messages]
    assert "Service tier: default. Allowed values: on, off, status." in status_text

    flex_outcome = await handle_model_fast(ctx, agent_name="test", value="flex")
    flex_text = [str(m.text) for m in flex_outcome.messages]
    assert "Invalid service tier value 'flex'. Allowed values: on, off, status." in flex_text


@pytest.mark.asyncio
async def test_model_fast_rejects_invalid_value() -> None:
    llm = _StubLLM("gpt-5", service_tier_supported=True)
    provider = _StubAgentProvider(_StubAgent(llm, shell_limit=None))
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_fast(ctx, agent_name="test", value="maybe")
    text_messages = [str(m.text) for m in outcome.messages]

    assert "Invalid service tier value 'maybe'. Allowed values: on, off, flex, status." in text_messages


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


@pytest.mark.asyncio
async def test_model_switch_sets_explicit_model_and_requests_session_reset() -> None:
    llm = _StubLLM("claude-haiku-4-5")
    agent = _StubAgent(llm)
    provider = _StubAgentProvider(agent)
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value="gpt-4.1-mini")

    assert agent.config.model == "gpt-4.1-mini"
    assert llm.model_name == "gpt-4.1-mini"
    assert outcome.reset_session is True
    assert any(
        str(message.text) == "Model: switched from claude-haiku-4-5 to gpt-4.1-mini."
        for message in outcome.messages
    )


@pytest.mark.asyncio
async def test_model_switch_uses_selector_when_value_missing() -> None:
    llm = _StubLLM("claude-haiku-4-5")
    agent = _StubAgent(llm)
    provider = _StubAgentProvider(agent)
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(model_selection_response="gpt-5-mini"),
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value=None)

    assert agent.config.model == "gpt-5-mini"
    assert llm.model_name == "gpt-5-mini"
    assert outcome.reset_session is True


@pytest.mark.asyncio
async def test_model_switch_reopens_overlay_selection_on_overlay_provider() -> None:
    llm = _StubLLM(
        "claude-haiku-4-5",
        provider=Provider.ANTHROPIC,
        selected_model_name="haikutiny",
        overlay_name="haikutiny",
    )
    agent = _StubAgent(llm)
    provider = _StubAgentProvider(agent)
    io = _StubIO(model_selection_response="haikutiny")
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=io,
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value=None)

    assert io.last_initial_provider == "overlays"
    assert io.last_default_model == "haikutiny"
    assert outcome.reset_session is False
    assert any("already active" in str(message.text) for message in outcome.messages)


@pytest.mark.asyncio
async def test_model_switch_reopens_vertex_selection_for_anthropic_vertex_model() -> None:
    llm = _StubLLM(
        "claude-sonnet-4-6",
        provider=Provider.ANTHROPIC_VERTEX,
        selected_model_name="anthropic-vertex.claude-sonnet-4-6",
    )
    agent = _StubAgent(llm)
    provider = _StubAgentProvider(agent)
    io = _StubIO(model_selection_response="anthropic-vertex.claude-sonnet-4-6")
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=io,
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value=None)

    assert io.last_initial_provider == "anthropic-vertex"
    assert io.last_default_model == "anthropic-vertex.claude-sonnet-4-6"
    assert outcome.reset_session is False
    assert any("already active" in str(message.text) for message in outcome.messages)


@pytest.mark.asyncio
async def test_model_switch_does_not_reset_session_when_model_is_already_active() -> None:
    llm = _StubLLM("gpt-5-mini")
    agent = _StubAgent(llm)
    provider = _StubAgentProvider(agent)
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value="gpt-5-mini")

    assert outcome.reset_session is False
    assert any("already active" in str(message.text) for message in outcome.messages)


@pytest.mark.asyncio
async def test_model_switch_returns_model_config_errors_without_raising() -> None:
    llm = _StubLLM("claude-haiku-4-5")
    agent = _StubAgent(
        llm,
        set_model_error=ModelConfigError(
            "Model reference '$system.typo' could not be resolved",
            "Available references: $system.default",
        ),
    )
    provider = _StubAgentProvider(agent)
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="test",
        io=_StubIO(),
        settings=Settings(),
    )

    outcome = await handle_model_switch(ctx, agent_name="test", value="$system.typo")
    text_messages = [str(message.text) for message in outcome.messages]

    assert outcome.reset_session is False
    assert text_messages == [
        "Model reference '$system.typo' could not be resolved: Available references: $system.default"
    ]
