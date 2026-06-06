from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from acp.helpers import text_block, update_agent_thought_text

from fast_agent.acp.server.live_session_registry import ACPLiveSessionRegistry
from fast_agent.acp.server.prompt_flow import ACPPromptFlow
from fast_agent.acp.server.prompt_flow import PromptFlowHost as ACPPromptFlowHost
from fast_agent.types import LlmStopReason, PromptMessageExtended

if TYPE_CHECKING:
    from fast_agent.acp.server.models import ACPSessionState
    from fast_agent.agents.tool_runner import ToolRunnerHooks
    from fast_agent.core.exceptions import ProviderKeyError
    from fast_agent.core.fastagent import AgentInstance
    from fast_agent.interfaces import AgentProtocol

STRUCTURED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class EmptyStructuredAgent:
    async def generate(
        self,
        _message: PromptMessageExtended,
        *,
        request_params: Any = None,
    ) -> PromptMessageExtended:
        return PromptMessageExtended(role="assistant", content=[])

    async def structured_schema(
        self,
        _message: PromptMessageExtended,
        _schema: dict[str, Any],
        *,
        request_params: Any = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        return (
            None,
            PromptMessageExtended(
                role="assistant",
                content=[],
                stop_reason=LlmStopReason.END_TURN,
            ),
        )


class FakePromptFlowHost(ACPPromptFlowHost):
    def __init__(self, agent: EmptyStructuredAgent) -> None:
        instance = cast("AgentInstance", SimpleNamespace(agents={"main": agent}))
        self._live_sessions = ACPLiveSessionRegistry(sessions={"session-1": instance})
        self._session_lock = asyncio.Lock()
        self._connection = None
        self.primary_agent_name = "main"

    def _resolve_primary_agent_name(self, instance: AgentInstance) -> str | None:
        _ = instance
        return "main"

    async def _build_session_request_params(
        self, agent: Any, session_state: ACPSessionState | None
    ) -> Any:
        _ = agent, session_state
        return None

    def _merge_tool_runner_hooks(
        self, base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        return extra or base

    def _build_status_line_meta(
        self, agent: Any, turn_start_index: int | None
    ) -> dict[str, Any] | None:
        _ = agent, turn_start_index
        return None

    async def _send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None:
        _ = session_id, agent, turn_start_index
        return None

    def _build_auth_required_data(
        self,
        error: ProviderKeyError,
        *,
        agent: AgentProtocol | object | None = None,
    ) -> dict[str, Any]:
        _ = error, agent
        return {}


class CapturingConnection:
    def __init__(self) -> None:
        self.notifications: list[dict[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.notifications.append(
            {
                "session_id": session_id,
                "update": update,
                "kwargs": kwargs,
            }
        )


class DummyStructuredStreamResult:
    stop_reason = LlmStopReason.END_TURN

    def last_text(self) -> str:
        return '{"answer":"ok"}'


@pytest.mark.asyncio
async def test_structured_output_without_assistant_text_returns_refusal() -> None:
    flow = ACPPromptFlow(FakePromptFlowHost(EmptyStructuredAgent()))
    structured_meta: dict[str, Any] = {
        "co.huggingface": {
            "structuredOutput": {
                "schema": STRUCTURED_SCHEMA,
                "mode": "bestEffort",
            }
        }
    }

    response = await flow.prompt_locked(
        prompt=[text_block("return json")],
        session_id="session-1",
        **structured_meta,
    )

    assert response.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_structured_output_streaming_emits_final_status_line_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = FakePromptFlowHost(EmptyStructuredAgent())
    host._connection = CapturingConnection()
    flow = ACPPromptFlow(host)
    status_line_meta = {
        "field_meta": {"openhands.dev/metrics": {"status_line": "tokens: 12 in / 7 out"}}
    }
    structured_meta: dict[str, Any] = {
        "co.huggingface": {
            "structuredOutput": {
                "schema": STRUCTURED_SCHEMA,
                "mode": "bestEffort",
            }
        }
    }

    async def fake_prepare_streaming_context(*, agent: Any, session_id: str) -> dict[str, Any]:
        _ = agent, session_id
        return {
            "stream_listener": None,
            "remove_listener": None,
            "streaming_tasks": [],
            "stream_state": SimpleNamespace(assistant_text_seen=True),
        }

    async def fake_run_with_status_hooks(**_: Any) -> dict[str, Any]:
        return {
            "result": DummyStructuredStreamResult(),
            "structured_parsed": {"answer": "ok"},
        }

    monkeypatch.setattr(host, "_build_status_line_meta", lambda _agent, _index: status_line_meta)
    monkeypatch.setattr(flow, "_prepare_streaming_context", fake_prepare_streaming_context)
    monkeypatch.setattr(flow, "_run_with_status_hooks", fake_run_with_status_hooks)

    response = await flow.prompt_locked(
        prompt=[text_block("return json")],
        session_id="session-1",
        **structured_meta,
    )

    assert response.stop_reason == "end_turn"
    assert response.field_meta == status_line_meta
    assert len(host._connection.notifications) == 1
    assert host._connection.notifications[0]["session_id"] == "session-1"
    assert host._connection.notifications[0]["kwargs"] == status_line_meta


@pytest.mark.asyncio
async def test_reasoning_only_streaming_preserves_final_status_line_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = FakePromptFlowHost(EmptyStructuredAgent())
    host._connection = CapturingConnection()
    flow = ACPPromptFlow(host)
    status_line_meta = {
        "field_meta": {"openhands.dev/metrics": {"status_line": "tokens: 12 in / 7 out"}}
    }
    structured_meta: dict[str, Any] = {
        "co.huggingface": {
            "structuredOutput": {
                "schema": STRUCTURED_SCHEMA,
                "mode": "bestEffort",
            }
        }
    }

    async def fake_prepare_streaming_context(*, agent: Any, session_id: str) -> dict[str, Any]:
        _ = agent

        async def send_reasoning_update() -> None:
            await host._connection.session_update(
                session_id=session_id,
                update=update_agent_thought_text("thinking"),
            )

        return {
            "stream_listener": None,
            "remove_listener": None,
            "streaming_tasks": [asyncio.create_task(send_reasoning_update())],
            "stream_state": SimpleNamespace(assistant_text_seen=False),
        }

    monkeypatch.setattr(host, "_build_status_line_meta", lambda _agent, _index: status_line_meta)
    monkeypatch.setattr(flow, "_prepare_streaming_context", fake_prepare_streaming_context)

    response = await flow.prompt_locked(
        prompt=[text_block("return json")],
        session_id="session-1",
        **structured_meta,
    )

    assert response.stop_reason == "refusal"
    assert response.field_meta == status_line_meta
    assert len(host._connection.notifications) == 2
    assert host._connection.notifications[-1]["session_id"] == "session-1"
    assert host._connection.notifications[-1]["kwargs"] == status_line_meta
    assert host._connection.notifications[-1]["update"].content.text == ""
