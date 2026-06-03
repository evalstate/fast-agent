from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from mcp.types import ListToolsResult, Tool
from rich.text import Text

from fast_agent.agents.agent_types import AgentType
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import tools as tools_handlers
from fast_agent.commands.handlers.tools import handle_list_tools
from fast_agent.tools.tool_sources import SHELL_TOOL_SOURCE, set_tool_source
from fast_agent.types import PromptMessageExtended, RequestParams

if TYPE_CHECKING:
    from pydantic import BaseModel


class _ToolListAgent:
    name = "agent"
    agent_type = AgentType.BASIC
    llm = None
    initialized = True
    instruction = ""
    context = None
    config = None
    message_history: list[Any] = []
    usage_accumulator = None
    card_tool_names: set[str] = set()
    smart_tool_names: set[str] = set()
    agent_backed_tools: dict[str, object] = {}

    async def __call__(self, message: Any) -> str:
        del message
        return ""

    async def send(
        self,
        message: Any,
        request_params: RequestParams | None = None,
    ) -> str:
        del message, request_params
        return ""

    async def generate(
        self,
        messages: Any,
        request_params: RequestParams | None = None,
    ) -> PromptMessageExtended:
        del messages, request_params
        return PromptMessageExtended(role="assistant", content=[])

    async def structured(
        self,
        messages: Any,
        model: type["BaseModel"],
        request_params: RequestParams | None = None,
    ) -> tuple["BaseModel" | None, PromptMessageExtended]:
        del messages, model, request_params
        return None, PromptMessageExtended(role="assistant", content=[])

    async def structured_schema(
        self,
        messages: Any,
        schema: dict[str, Any],
        request_params: RequestParams | None = None,
    ) -> tuple[Any | None, PromptMessageExtended]:
        del messages, schema, request_params
        return None, PromptMessageExtended(role="assistant", content=[])

    async def list_tools(self) -> ListToolsResult:
        tool = Tool(
            name="read_text_file",
            title="Read a file",
            description="Read a UTF-8 text file from disk.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            _meta={
                "ui/appEnabled": True,
                "openai/skybridgeEnabled": True,
                "ui/appTemplate": "ui://file-reader",
            },
        )
        return ListToolsResult(tools=[set_tool_source(tool, SHELL_TOOL_SOURCE)])

    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def set_model(self, model: str | None) -> None:
        del model

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts

    def pop_last_message(self) -> None:
        return None

    async def apply_prompt(
        self,
        prompt: Any,
        arguments: dict[str, str] | None = None,
        as_template: bool = False,
        namespace: str | None = None,
    ) -> str:
        del prompt, arguments, as_template, namespace
        return ""

    async def get_prompt(
        self,
        prompt_name: str,
        arguments: dict[str, str] | None = None,
        namespace: str | None = None,
    ) -> Any:
        del prompt_name, arguments, namespace
        return None

    async def list_prompts(self, namespace: str | None = None) -> Any:
        del namespace
        return None

    async def list_resources(self) -> Any:
        return None

    async def get_resource(self, uri: str) -> Any:
        del uri
        return None

    async def list_mcp_tools(self) -> list[Any]:
        return []

    async def run_tools(self, message: PromptMessageExtended) -> PromptMessageExtended:
        return message

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def attach_llm(self, llm: Any) -> None:
        self.llm = llm

    def with_resource(self, resource: Any) -> "_ToolListAgent":
        del resource
        return self

    async def show_assistant_message(self, message: PromptMessageExtended) -> str:
        del message
        return ""

    async def agent_card(self) -> Any:
        return None


@pytest.mark.asyncio
async def test_handle_list_tools_renders_app_badges_and_details() -> None:
    ctx = CommandContext(
        agent_provider=StaticAgentProvider({"agent": _ToolListAgent()}),
        current_agent_name="agent",
        io=NonInteractiveCommandIOBase(),
    )

    outcome = await handle_list_tools(ctx, agent_name="agent")

    message_text = outcome.messages[0].text
    assert isinstance(message_text, Text)
    rendered = message_text.plain
    assert "read_text_file (Shell) (Apps SDK) (MCP App) Read a file" in rendered
    assert "     args: path, limit\n" in rendered
    assert "     template: ui://file-reader\n" in rendered


def test_format_tool_line_trims_title_and_omits_blank_title() -> None:
    with_title = tools_handlers._format_tool_line("read_text_file", " Read a file ", None)
    blank_title = tools_handlers._format_tool_line("read_text_file", "   ", None)

    assert with_title.plain == "read_text_file Read a file"
    assert blank_title.plain == "read_text_file"
