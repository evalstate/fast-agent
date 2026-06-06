from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.types import ListToolsResult
from rich.text import Text

from fast_agent.commands.context import CommandContext, NonInteractiveCommandIOBase
from fast_agent.commands.handlers.model import handle_model_web_search
from fast_agent.commands.handlers.tools import handle_list_tools
from fast_agent.config import Settings
from fast_agent.llm.provider_types import Provider


class _MutableLlm:
    provider = Provider.RESPONSES
    model_name = "gpt-5"
    model_info = None
    resolved_model = None
    reasoning_effort = None
    reasoning_effort_spec = None
    text_verbosity = None
    text_verbosity_spec = None
    web_search_supported = True
    web_fetch_supported = False
    web_fetch_enabled = False
    x_search_supported = False
    x_search_enabled = False
    task_budget_supported = False
    task_budget_tokens = None
    service_tier_supported = False
    service_tier = None
    available_service_tiers = ()
    default_request_params = None

    def __init__(self) -> None:
        self.web_search_enabled = False

    def set_web_search_enabled(self, value: bool | None) -> None:
        self.web_search_enabled = bool(value)


class _Agent:
    name = "main"
    agent_type = "agent"
    message_history = []
    usage_accumulator = None
    initialized = True
    instruction = ""
    context = None

    def __init__(self, llm: _MutableLlm) -> None:
        self.llm = llm
        self.config = SimpleNamespace(model=llm.model_name)

    async def list_tools(self) -> ListToolsResult:
        return ListToolsResult(tools=[])

    async def initialize(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def set_model(self, model: str | None) -> None:
        self.llm.model_name = model or ""

    def clear(self, *, clear_prompts: bool = False) -> None:
        del clear_prompts

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def pop_last_message(self):
        return None

    async def __call__(self, message):
        return str(message)

    async def send(self, message, request_params=None):
        del request_params
        return str(message)

    async def generate(self, messages, request_params=None):
        del request_params
        return messages

    async def structured(self, messages, model, request_params=None):
        del request_params
        return None, messages

    async def structured_schema(self, messages, schema, request_params=None):
        del schema, request_params
        return None, messages

    async def apply_prompt(self, prompt, arguments=None, as_template=False, namespace=None):
        del arguments, as_template, namespace
        return str(prompt)

    async def get_prompt(self, prompt_name, arguments=None, namespace=None):
        del arguments, namespace
        return SimpleNamespace(name=prompt_name)

    async def list_prompts(self, namespace=None):
        del namespace
        return {}

    async def list_resources(self, namespace=None):
        del namespace
        return {}

    async def list_mcp_tools(self, namespace=None):
        del namespace
        return {}

    async def get_resource(self, resource_uri: str, namespace=None):
        del namespace
        return SimpleNamespace(contents=[], uri=resource_uri)

    async def with_resource(self, prompt_content, resource_uri: str, namespace=None):
        del resource_uri, namespace
        return str(prompt_content)

    async def agent_card(self):
        return SimpleNamespace()

    async def run_tools(self, request, request_params=None):
        del request_params
        return request

    async def show_assistant_message(self, *args, **kwargs) -> None:
        del args, kwargs

    async def attach_llm(self, *args, **kwargs):
        del args, kwargs
        return self.llm


class _Provider:
    def __init__(self, agent: _Agent) -> None:
        self._agent_obj = agent

    def _agent(self, name: str) -> _Agent:
        del name
        return self._agent_obj

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return agent_name or "main"

    def visible_agent_names(self, *, force_include: str | None = None):
        del force_include
        return ["main"]

    def registered_agent_names(self):
        return ["main"]

    def registered_agents(self):
        return {"main": self._agent_obj}

    async def list_prompts(self, namespace: str | None, agent_name: str | None = None):
        del namespace, agent_name
        return {}


def _plain(text: object) -> str:
    if isinstance(text, Text):
        return text.plain
    return str(text)


@pytest.mark.asyncio
async def test_tools_reflects_web_search_toggle_from_model_command() -> None:
    agent = _Agent(_MutableLlm())
    ctx = CommandContext(
        agent_provider=_Provider(agent),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=Settings(),
    )

    await handle_model_web_search(ctx, agent_name="main", value="on")
    outcome = await handle_list_tools(ctx, agent_name="main")

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "web_search (provider-hosted, enabled)" in rendered


@pytest.mark.asyncio
async def test_tools_omits_disabled_provider_hosted_tools() -> None:
    agent = _Agent(_MutableLlm())
    ctx = CommandContext(
        agent_provider=_Provider(agent),
        current_agent_name="main",
        io=NonInteractiveCommandIOBase(),
        settings=Settings(),
    )

    outcome = await handle_list_tools(ctx, agent_name="main")

    rendered = "\n".join(_plain(message.text) for message in outcome.messages)
    assert "web_search" not in rendered
    assert "No tools available for this agent." in rendered
