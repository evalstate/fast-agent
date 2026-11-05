"""Deterministic interactive agent used by the opt-in real-herdr input tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import TYPE_CHECKING

from mcp import CallToolRequest
from mcp_types import CallToolRequestParams

from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.cli.runtime.agent_setup import _run_interactive_with_interrupt_recovery
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.core.fastagent import FastAgent
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.mcp.prompt import Prompt
from fast_agent.types.llm_stop_reason import LlmStopReason

if TYPE_CHECKING:
    from mcp_types import Tool

    from fast_agent.llm.request_params import RequestParams
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class Scenario:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.late = False
        self.gated = True
        self.cancelled = False
        self.read_capture = False

    async def emit(self, kind: str, **data: object) -> None:
        self.writer.write((json.dumps({"kind": kind, **data}) + "\n").encode())
        await self.writer.drain()

    async def release(self) -> None:
        assert await self.reader.readline() == b"continue\n"

    async def wait_for_report(self) -> str:
        """Wait for a real hc notification, then hold the boundary for batch tests."""
        if self.read_capture:
            for index in range(180):
                print(f"READ_HISTORY_{index:03d}", flush=True)
            await self.emit("tool_waiting")
            await self.release()
            return "READ_TOOL_COMPLETE"
        await self.emit("tool_waiting")
        process = await asyncio.create_subprocess_exec(
            "hc",
            "wait",
            os.environ["HC_TERMINAL_ID"],
            "--timeout",
            "20",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await process.communicate()
            assert process.returncode == 0, stderr.decode()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        result = stdout.decode()
        await self.emit("wait_returned", result=json.loads(result))
        if self.gated:
            try:
                await self.release()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return result


class ScenarioLLM(PassthroughLLM):
    def __init__(self, scenario: Scenario) -> None:
        super().__init__()
        self.scenario = scenario
        self.followup = False

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: RequestParams | None = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        if self.scenario.cancelled:
            self.followup = False
            self.scenario.cancelled = False
        if self.followup:
            boundary = max(
                i for i, message in enumerate(multipart_messages) if message.tool_results
            )
            current_messages = multipart_messages[boundary:]
        else:
            current_messages = multipart_messages[-1:]
        messages = [
            {
                "role": message.role,
                "text": message.all_text(),
                "tools": list(message.tool_results or {}),
            }
            for message in current_messages
        ]
        if not self.followup and multipart_messages[-1].first_text().startswith("scenario "):
            self.scenario.read_capture = multipart_messages[-1].first_text() == "scenario read"
            self.scenario.late = multipart_messages[-1].first_text() == "scenario late"
            self.scenario.gated = multipart_messages[-1].first_text() != "scenario natural"
            self.followup = True
            await self.scenario.emit("initial", messages=messages)
            return Prompt.assistant(
                "Waiting for the worker report.",
                stop_reason=LlmStopReason.TOOL_USE,
                tool_calls={
                    "wait": CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(name="wait_for_report", arguments={}),
                    )
                },
            )
        if self.followup:
            self.followup = False
            await self.scenario.emit("followup", messages=messages)
            if self.scenario.late:
                # Input delivered here is genuinely late: the next model call
                # has already started. It must become a new turn, exactly once.
                await self.scenario.release()
        else:
            await self.scenario.emit("next_turn", messages=messages)
        text = (
            "READ_FINAL_COMPLETE"
            if self.scenario.read_capture
            else "FINAL: " + " | ".join(message.first_text() for message in current_messages)
        )
        return Prompt.assistant(text, stop_reason=LlmStopReason.END_TURN)


async def main() -> None:
    reader, writer = await asyncio.open_connection("127.0.0.1", int(sys.argv[1]))
    scenario = Scenario(reader, writer)
    fast = FastAgent(
        "hc-input-test",
        parse_cli_args=False,
        isolated=True,
        no_home=True,
        quiet=sys.argv[2] == "quiet",
    )

    fast._settings.ptc.enabled = False

    @fast.agent(name="coordinator", model="passthrough", function_tools=[scenario.wait_for_report])
    async def coordinator() -> None:
        pass

    try:
        async with fast.run() as app:
            agent = app.resolve_agent("coordinator")
            assert isinstance(agent, ToolAgent)
            agent._llm = ScenarioLLM(scenario)
            await scenario.emit("connected")
            # Use the CLI's recovery loop, not a bare asyncio.run cancellation
            # policy, so hc interrupt stops a turn rather than the whole helper.
            request = AgentRunRequest(
                name="hc-input-test",
                instruction=None,
                config_path=None,
                server_list=None,
                agent_cards=None,
                card_tools=None,
                model="passthrough",
                message=None,
                prompt_file=None,
                result_file=None,
                resume=None,
                startup_mcp_servers=None,
                mcp_startup_notices=(),
                agent_name="coordinator",
                target_agent_name="coordinator",
                skills_directory=None,
                home=None,
                no_home=True,
                shell_runtime=False,
                no_shell=True,
                mode="interactive",
                transport="stdio",
                host="127.0.0.1",
                port=0,
                tool_description=None,
                tool_name_template=None,
                instance_scope="shared",
                permissions_enabled=False,
                reload=False,
                watch=False,
            )
            await _run_interactive_with_interrupt_recovery(app, request)
    finally:
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
