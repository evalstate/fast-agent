from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent import AgentResponse

if TYPE_CHECKING:
    from types import ModuleType


def _research_server_module() -> ModuleType:
    path = Path(__file__).parents[3] / "examples" / "a2a" / "research" / "server.py"
    spec = importlib.util.spec_from_file_location("a2a_research_server_example", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_research_decision_round_trips_json() -> None:
    module = _research_server_module()
    decision = module.ResearchDecision(
        kind=module.ResearchDecisionKind.NEEDS_REFINEMENT,
        message="More detail needed",
    )

    parsed = module.ResearchDecision.from_json(decision.to_json())

    assert parsed == decision


@pytest.mark.asyncio
async def test_research_server_loads_agent_card_definitions_from_environment() -> None:
    module = _research_server_module()

    assert module.QUICK_REFINER_AGENT not in module.fast.agents
    assert module.RESEARCH_WORKER_AGENT not in module.fast.agents

    async with module.fast.harness():
        assert module.QUICK_REFINER_AGENT in module.fast.agents
        assert module.RESEARCH_WORKER_AGENT in module.fast.agents


def test_research_refinement_requests_missing_fields_with_context() -> None:
    module = _research_server_module()

    decision = module._decision_for_prompt("research recent A2A work")

    assert decision.kind == module.ResearchDecisionKind.NEEDS_REFINEMENT
    assert "'research recent A2A work'" in decision.message
    assert "the intended audience" in decision.message
    assert "the desired output format" in decision.message
    assert "Goal:" not in decision.message


def test_research_refinement_accepts_natural_language_actionable_task() -> None:
    module = _research_server_module()
    prompt = "goal is to research new models, html report, audience scientists"

    decision = module._decision_for_prompt(prompt)

    assert decision.kind == module.ResearchDecisionKind.BEGIN_RESEARCH
    assert decision.message == "Research task accepted"
    assert decision.goal == prompt


def test_research_refinement_also_accepts_labeled_actionable_task() -> None:
    module = _research_server_module()
    prompt = "Goal: compare A2A. Audience: developers. Output: markdown report."

    decision = module._decision_for_prompt(prompt)

    assert decision.kind == module.ResearchDecisionKind.BEGIN_RESEARCH
    assert decision.message == "Research task accepted"
    assert decision.goal == prompt


def test_huggingface_research_config_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _research_server_module()
    monkeypatch.setenv("FAST_AGENT_RESEARCH_HF_BUCKET", "org/research-bucket")
    monkeypatch.setenv("FAST_AGENT_RESEARCH_HF_IMAGE", "python:3.13")
    monkeypatch.setenv("FAST_AGENT_RESEARCH_HF_FLAVOR", "cpu-upgrade")
    monkeypatch.setenv("FAST_AGENT_RESEARCH_HF_CREATE_BUCKET", "false")
    monkeypatch.setenv("FAST_AGENT_RESEARCH_HF_FORWARD_TOKEN", "true")

    config = module._huggingface_research_environment_config_from_env()

    assert config == module.HuggingFaceResearchEnvironmentConfig(
        bucket="org/research-bucket",
        image="python:3.13",
        flavor="cpu-upgrade",
        token=None,
        namespace=None,
        forward_hf_token=True,
        create_bucket=False,
        private_bucket=None,
    )


@pytest.mark.asyncio
async def test_research_runtime_factory_creates_hf_environment_per_a2a_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _research_server_module()
    ensured: list[str] = []
    environments: list[tuple[str, str]] = []
    opened: list[object] = []

    async def fake_ensure(config: Any) -> None:
        ensured.append(config.bucket)

    class FakeFastAgent:
        def harness(self, *, environment: object) -> object:
            opened.append(environment)
            return FakeHarnessContext()

    class FakeHarnessContext:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            return None

    def fake_environment_factory(config: Any, task_id: str, bucket_path: str) -> object:
        environments.append((task_id, bucket_path))
        return {"bucket": config.bucket, "path": bucket_path}

    monkeypatch.setattr(module, "_ensure_huggingface_bucket", fake_ensure)
    config = module.HuggingFaceResearchEnvironmentConfig(bucket="org/research-bucket")
    factory = module.ResearchRuntimeFactory(
        cast("Any", object()),
        hf_config=config,
        fast_agent_factory=FakeFastAgent,
        environment_factory=fake_environment_factory,
    )
    context = cast("Any", SimpleNamespace(context_id="ctx/one", task_id="task:two"))

    async with factory.open(context) as runtime:
        assert runtime.info.environment_label == "Hugging Face Sandbox"
        assert runtime.info.bucket == "org/research-bucket"
        assert runtime.info.bucket_path == "a2a-research/ctx-one/task-two"

    assert ensured == ["org/research-bucket"]
    assert environments == [("task:two", "a2a-research/ctx-one/task-two")]
    assert opened == [{"bucket": "org/research-bucket", "path": "a2a-research/ctx-one/task-two"}]


@pytest.mark.asyncio
async def test_research_worker_heartbeat_reports_while_runtime_is_busy() -> None:
    module = _research_server_module()
    reports: list[str] = []

    class FakeRuntime:
        async def research(
            self,
            context: object,
            decision: object,
            *,
            progress_handler: object | None = None,
        ) -> object:
            del context, decision, progress_handler
            await asyncio.sleep(0.03)
            return AgentResponse.text("done")

    class FakeProgressHandler:
        async def report(self, message: str) -> None:
            reports.append(message)

    response = await module._run_research_with_progress(
        cast("Any", FakeRuntime()),
        cast("Any", SimpleNamespace()),
        module.ResearchDecision(
            kind=module.ResearchDecisionKind.BEGIN_RESEARCH,
            message="accepted",
            goal="goal",
        ),
        progress_handler=cast("Any", FakeProgressHandler()),
        heartbeat_seconds=0.01,
    )

    assert response.text_content() == "done"
    assert any(message.startswith("Research still running") for message in reports)


def test_research_request_params_enable_loop_progress() -> None:
    module = _research_server_module()
    handler = object()

    params = module._research_request_params(cast("Any", handler))

    assert params is not None
    assert params.emit_loop_progress is True
    assert params.tool_execution_handler is handler
