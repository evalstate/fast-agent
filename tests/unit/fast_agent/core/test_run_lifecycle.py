from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from fast_agent.core.fastagent import FastAgent, RunRuntime, RunSettings
from fast_agent.core.run_lifecycle import FastAgentRunLifecycle

if TYPE_CHECKING:
    from fast_agent.interfaces import AgentProtocol, FastAgentLLMProtocol, LLMFactoryProtocol


def _unused_llm_factory(agent: "AgentProtocol", **kwargs: object) -> FastAgentLLMProtocol:
    del agent, kwargs
    raise AssertionError("LLM factory should not be called by this test")


def _unused_model_factory(model: str | None = None) -> LLMFactoryProtocol:
    del model
    return _unused_llm_factory


@pytest.mark.asyncio
async def test_run_lifecycle_enter_performs_shared_setup_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("test", parse_cli_args=False)
    calls: list[str] = []

    async def initialize() -> None:
        calls.append("initialize")

    @asynccontextmanager
    async def run_context():
        calls.append("app_enter")
        try:
            yield
        finally:
            calls.append("app_exit")

    def prepare_settings(*, model_override: str | None = None, force_headless: bool = False):
        calls.append(f"settings:{model_override}:{force_headless}")
        return RunSettings(
            quiet_mode=True,
            cli_model_override=model_override,
            noenv_mode=False,
            server_mode=False,
            transport=None,
            is_acp_server_mode=False,
            reload_enabled=False,
        )

    def load_skills() -> list[object]:
        calls.append("load_skills")
        return []

    def before_apply() -> None:
        calls.append("before_apply")

    def apply_skills(default_skills: list[object]) -> None:
        assert default_skills == []
        calls.append("apply_skills")

    def quiet_mode() -> None:
        calls.append("quiet")

    def validate() -> None:
        calls.append("validate")

    def create_runtime(settings: RunSettings) -> RunRuntime:
        calls.append("runtime")
        return RunRuntime(
            model_factory_func=_unused_model_factory,
            global_prompt_context=None,
            is_acp_server_mode=settings.is_acp_server_mode,
            noenv_mode=settings.noenv_mode,
            managed_instances=[],
            instance_lock=asyncio.Lock(),
        )

    monkeypatch.setattr(fast.app, "initialize", initialize)
    monkeypatch.setattr(fast.app, "run", run_context)
    monkeypatch.setattr(fast, "_prepare_run_settings", prepare_settings)
    monkeypatch.setattr(fast, "_load_default_skills_for_run", load_skills)
    monkeypatch.setattr(fast, "_apply_skills_to_agent_configs", apply_skills)
    monkeypatch.setattr(fast, "_configure_quiet_mode_for_run", quiet_mode)
    monkeypatch.setattr(fast, "_validate_run_preconditions", validate)
    monkeypatch.setattr(fast, "_create_run_runtime", create_runtime)

    lifecycle = FastAgentRunLifecycle(fast)
    state = await lifecycle.enter(
        model_override="sonnet",
        force_headless=True,
        before_apply_skills=before_apply,
    )
    await lifecycle.exit(state, None, {}, had_error=False)

    assert calls == [
        "initialize",
        "settings:sonnet:True",
        "app_enter",
        "load_skills",
        "before_apply",
        "apply_skills",
        "quiet",
        "validate",
        "runtime",
        "app_exit",
    ]


@pytest.mark.asyncio
async def test_run_lifecycle_cleans_context_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("test", parse_cli_args=False)
    calls: list[str] = []

    async def initialize() -> None:
        calls.append("initialize")

    @asynccontextmanager
    async def run_context():
        calls.append("app_enter")
        try:
            yield
        finally:
            calls.append("app_exit")

    def prepare_settings(*, model_override: str | None = None, force_headless: bool = False):
        del model_override, force_headless
        return RunSettings(
            quiet_mode=False,
            cli_model_override=None,
            noenv_mode=False,
            server_mode=False,
            transport=None,
            is_acp_server_mode=False,
            reload_enabled=False,
        )

    def validate() -> None:
        calls.append("validate")
        raise RuntimeError("bad setup")

    monkeypatch.setattr(fast.app, "initialize", initialize)
    monkeypatch.setattr(fast.app, "run", run_context)
    monkeypatch.setattr(fast, "_prepare_run_settings", prepare_settings)
    monkeypatch.setattr(fast, "_load_default_skills_for_run", lambda: [])
    monkeypatch.setattr(fast, "_apply_skills_to_agent_configs", lambda _skills: None)
    monkeypatch.setattr(fast, "_validate_run_preconditions", validate)

    lifecycle = FastAgentRunLifecycle(fast)

    with pytest.raises(RuntimeError, match="bad setup"):
        await lifecycle.enter()

    assert calls == ["initialize", "app_enter", "validate", "app_exit"]


@pytest.mark.asyncio
async def test_run_lifecycle_does_not_exit_unentered_app_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("test", parse_cli_args=False)
    calls: list[str] = []

    async def initialize() -> None:
        calls.append("initialize")

    class FailingContext:
        async def __aenter__(self) -> None:
            calls.append("app_enter")
            raise RuntimeError("enter failed")

        async def __aexit__(self, *args: object) -> None:
            del args
            calls.append("app_exit")
            raise RuntimeError("masked")

    def prepare_settings(*, model_override: str | None = None, force_headless: bool = False):
        del model_override, force_headless
        return RunSettings(
            quiet_mode=False,
            cli_model_override=None,
            noenv_mode=False,
            server_mode=False,
            transport=None,
            is_acp_server_mode=False,
            reload_enabled=False,
        )

    monkeypatch.setattr(fast.app, "initialize", initialize)
    monkeypatch.setattr(fast.app, "run", FailingContext)
    monkeypatch.setattr(fast, "_prepare_run_settings", prepare_settings)

    lifecycle = FastAgentRunLifecycle(fast)

    with pytest.raises(RuntimeError, match="enter failed"):
        await lifecycle.enter()

    assert calls == ["initialize", "app_enter"]


@pytest.mark.asyncio
async def test_run_lifecycle_cleans_context_on_cancelled_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = FastAgent("test", parse_cli_args=False)
    calls: list[str] = []

    async def initialize() -> None:
        calls.append("initialize")

    @asynccontextmanager
    async def run_context():
        calls.append("app_enter")
        try:
            yield
        finally:
            calls.append("app_exit")

    def prepare_settings(*, model_override: str | None = None, force_headless: bool = False):
        del model_override, force_headless
        return RunSettings(
            quiet_mode=False,
            cli_model_override=None,
            noenv_mode=False,
            server_mode=False,
            transport=None,
            is_acp_server_mode=False,
            reload_enabled=False,
        )

    def validate() -> None:
        calls.append("validate")
        raise asyncio.CancelledError()

    monkeypatch.setattr(fast.app, "initialize", initialize)
    monkeypatch.setattr(fast.app, "run", run_context)
    monkeypatch.setattr(fast, "_prepare_run_settings", prepare_settings)
    monkeypatch.setattr(fast, "_load_default_skills_for_run", lambda: [])
    monkeypatch.setattr(fast, "_apply_skills_to_agent_configs", lambda _skills: None)
    monkeypatch.setattr(fast, "_validate_run_preconditions", validate)

    lifecycle = FastAgentRunLifecycle(fast)

    with pytest.raises(asyncio.CancelledError):
        await lifecycle.enter()

    assert calls == ["initialize", "app_enter", "validate", "app_exit"]
