"""Exercise live export with real tool turns, archives, and harness sessions."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from mcp_types import (
    CallToolRequest,
    CallToolRequestParams,
    ContentBlock,
    ImageContent,
    TextContent,
    Tool,
)

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner, ToolRunnerHooks
from fast_agent.cli.runtime.agent_setup import _live_atif_session_dir, _run_cli_flow
from fast_agent.config import CompactionSettings, Settings
from fast_agent.constants import OPENAI_COMPACTION_ITEMS
from fast_agent.context import Context
from fast_agent.core.agent_app import AgentApp
from fast_agent.core.fastagent import AgentInstance
from fast_agent.core.harness import HarnessSessions
from fast_agent.history.compaction import compact_conversation
from fast_agent.llm.compaction_support import (
    CompactionStrategy,
    CompactionSupport,
    NativeCompaction,
)
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.model_info import ModelInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.prompt import Prompt
from fast_agent.session import Session, SessionManager
from fast_agent.types import LlmStopReason, PromptMessageExtended, RequestParams
from tests.unit.fast_agent.commands.test_runtime_result_export import _make_request

if TYPE_CHECKING:
    from pathlib import Path


class _CompactingRunSimulator(PassthroughLLM):
    def __init__(
        self,
        strategy: CompactionStrategy,
        terminal_call: int,
        error: BaseException | None,
    ) -> None:
        super().__init__()
        self.retry_count = 0
        self.strategy = strategy
        self.terminal_call = terminal_call
        self.error = error
        self.calls = 0

    @property
    def compaction_support(self) -> CompactionSupport:
        return (
            CompactionSupport.ON_DEMAND
            if self.strategy == CompactionStrategy.OPENAI_COMPACTION
            else CompactionSupport.CLIENT
        )

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="export-simulator",
            provider=Provider.GENERIC,
            context_window=None,
            max_output_tokens=None,
            tokenizes=["text/plain", "image/png"],
            json_mode=None,
            reasoning=None,
        )

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        if tools is None:
            return Prompt.assistant("summary", stop_reason=LlmStopReason.END_TURN)
        self.calls += 1
        if self.calls == self.terminal_call:
            if self.error is not None:
                raise self.error
            return Prompt.assistant("done", stop_reason=LlmStopReason.END_TURN)
        return Prompt.assistant(
            "use tool",
            stop_reason=LlmStopReason.TOOL_USE,
            tool_calls={
                f"call-{self.calls}": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="echo", arguments={"index": self.calls}),
                )
            },
        )

    async def native_compaction(self, messages: list[PromptMessageExtended]) -> NativeCompaction:
        channels: dict[str, list[ContentBlock]] = {
            OPENAI_COMPACTION_ITEMS: [
                TextContent(
                    type="text",
                    text=json.dumps({"type": "compaction", "encrypted_content": "test-checkpoint"}),
                )
            ]
        }
        return NativeCompaction(
            strategy=CompactionStrategy.OPENAI_COMPACTION,
            request="simulated native compaction",
            response=PromptMessageExtended(role="assistant", content=[], channels=channels),
            checkpoint_channels=channels,
            description="opaque checkpoint",
            opaque_bytes=10,
        )


class _MediaToolAgent(ToolAgent):
    def __init__(self, context: Context, *, media_call: int | None) -> None:
        def echo(index: int) -> str:
            self.tool_call = index
            return f"output {index}"

        super().__init__(AgentConfig("agent"), [echo], context=context)
        self.media_call = media_call
        self.tool_call = 0

    def _consume_pending_media_attachments(self) -> list[ContentBlock]:
        if self.tool_call != self.media_call:
            return []
        self.media_call = None
        return [ImageContent(type="image", data="abcd", mime_type="image/png")]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", list(CompactionStrategy))
@pytest.mark.parametrize("termination", ["success", "error", "cancelled"])
@pytest.mark.parametrize("terminal_call", [3, 4])
@pytest.mark.parametrize("media", ["none", "archived", "pending"])
async def test_harness_exports_compacted_tool_turn(
    tmp_path: Path,
    strategy: CompactionStrategy,
    termination: str,
    terminal_call: int,
    media: str,
) -> None:
    manager = SessionManager(
        cwd=tmp_path, home_override=tmp_path / ".fast-agent", respect_env_override=False
    )
    session = manager.create_session()
    manager.set_current_session(session)
    context = Context(
        config=Settings(compaction=CompactionSettings(auto=False)), session_manager=manager
    )
    media_call = 1 if media == "archived" else terminal_call - 1 if media == "pending" else None
    agent = _MediaToolAgent(context, media_call=media_call)
    error = (
        asyncio.CancelledError()
        if termination == "cancelled"
        else RuntimeError("provider unavailable")
        if termination == "error"
        else None
    )
    agent._llm = _CompactingRunSimulator(strategy, terminal_call, error)

    async def compact(runner: ToolRunner, messages: list[PromptMessageExtended]) -> None:
        if runner.iteration == 2:
            result = await compact_conversation(
                agent, settings=CompactionSettings(auto=False, keep_turns=0)
            )
            assert result.archive_file is not None
            assert result.strategy == strategy

    agent.tool_runner_hooks = ToolRunnerHooks(before_llm_call=compact)
    app = AgentApp({"agent": agent})
    instance = AgentInstance(app, {"agent": agent})

    async def create_instance() -> AgentInstance:
        return instance

    async def dispose_instance(owned: AgentInstance) -> None:
        await owned.shutdown()

    async def create_persisted_session(
        session_id: str, owned: AgentInstance, default_agent_name: str | None
    ) -> tuple[SessionManager, Session]:
        return manager, session

    sessions = HarnessSessions(
        create_instance=create_instance,
        dispose_instance=dispose_instance,
        create_persisted_session=create_persisted_session,
    )
    output = tmp_path / "trajectory.json"
    request = _make_request(result_file=None, trajectory_output=output)
    try:
        harness_session = await sessions.create(session.info.name, agent_name="agent")
        if error is None:
            await _run_cli_flow(
                app, request, session_manager=manager, harness_session=harness_session
            )
        else:
            with pytest.raises(type(error)) as caught:
                await _run_cli_flow(
                    app, request, session_manager=manager, harness_session=harness_session
                )
            assert caught.value is error
        assert _live_atif_session_dir(manager, harness_session) == session.directory
        assert list(session.directory.glob("compacted_*.json"))
        # The raw input remains distinct from the timestamped persisted copy.
        assert agent.last_turn_messages[0].timestamp is None
        payload = json.loads(output.read_text())
        assert "test-checkpoint" not in json.dumps(payload)
        steps = [step for step in payload["steps"] if not step.get("is_copied_context")]
        assert [call["tool_call_id"] for step in steps for call in step.get("tool_calls", [])] == [
            f"call-{index}" for index in range(1, terminal_call)
        ]
        assert [
            result["content"]
            for step in steps
            for result in step.get("observation", {}).get("results", [])
            if step["source"] != "system"
        ] == [f"output {index}" for index in range(1, terminal_call)]
        boundaries = [step for step in steps if step.get("extra", {}).get("context_management")]
        assert len(boundaries) == 1
        assert boundaries[0]["extra"]["context_management"]["strategy"] == strategy
        # The second result completed before compaction, even if its next LLM call failed.
        second_result = next(
            step
            for step in steps
            if any(
                result["content"] == "output 2"
                for result in step.get("observation", {}).get("results", [])
            )
        )
        assert second_result["step_id"] < boundaries[0]["step_id"]
        image_parts = [
            part
            for step in steps
            if isinstance(step["message"], list)
            for part in step["message"]
            if part["type"] == "image"
        ]
        assert len(image_parts) == int(media != "none")
        if error is not None:
            assert payload["extra"]["termination"] == {
                "status": termination,
                "error_type": type(error).__name__,
                "message": str(error),
            }
    finally:
        await sessions._close_all()
        manager.close()
