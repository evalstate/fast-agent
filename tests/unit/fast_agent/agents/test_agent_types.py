"""
Unit tests for agent types and their interactions with the interactive prompt.
"""

import pytest

from fast_agent.agents import McpAgent
from fast_agent.agents.agent_types import AgentConfig, AgentType
from fast_agent.config import MCPServerSettings, MCPSettings, Settings
from fast_agent.context import Context
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.llm.fastagent_llm import FastAgentLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.types import RequestParams


def test_agent_type_default():
    """Test that agent_type defaults to AgentType.BASIC.value"""
    agent = McpAgent(config=AgentConfig(name="test_agent"))
    assert agent.agent_type == AgentType.BASIC


def test_instruction_propagates_to_default_request_params():
    """
    Test that AgentConfig.instruction is propagated to
    default_request_params.systemPrompt when both are provided.

    This reproduces the bug where the instruction is lost when
    a user provides their own default_request_params.
    """
    # Create RequestParams with custom settings but no systemPrompt
    request_params = RequestParams(model="sonnet", temperature=0.7, max_tokens=32768)

    # Verify systemPrompt is not set initially
    assert request_params.system_prompt is None

    # Create AgentConfig with both instruction and default_request_params
    instruction = "You are a helpful assistant specialized in testing."
    config = AgentConfig(
        name="my_agent",
        instruction=instruction,
        default_request_params=request_params,
        model="sonnet",
    )

    # The instruction should be propagated to default_request_params.systemPrompt
    assert config.default_request_params is not None
    assert config.default_request_params.system_prompt == instruction, (
        f"Expected systemPrompt to be '{instruction}', "
        f"but got {config.default_request_params.system_prompt}"
    )


class _StubProviderManagedLLM(FastAgentLLM):
    def __init__(self, provider: Provider = Provider.ANTHROPIC) -> None:
        super().__init__(provider=provider)

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages,
        request_params=None,
        tools=None,
        is_template: bool = False,
    ):
        del request_params, tools, is_template
        return multipart_messages[-1]

    def _convert_extended_messages_to_provider(self, messages):
        del messages
        return []


@pytest.mark.asyncio
async def test_provider_managed_servers_remain_visible_without_local_aggregator_attach() -> None:
    server_settings = {
        "stripe": MCPServerSettings(
            name="stripe",
            management="provider",
            transport="http",
            url="https://mcp.stripe.com",
        ),
        "filesystem": MCPServerSettings(
            name="filesystem",
            command="npx",
            args=["@modelcontextprotocol/server-filesystem"],
        ),
    }
    server_registry = ServerRegistry()
    server_registry.registry = server_settings
    context = Context(
        config=Settings(
            mcp=MCPSettings(
                servers=server_settings,
            )
        ),
        server_registry=server_registry,
    )
    agent = McpAgent(
        config=AgentConfig(name="billing", servers=["stripe", "filesystem"]),
        context=context,
        connection_persistence=False,
    )

    assert agent.aggregator.server_names == ["filesystem"]
    assert agent.list_attached_mcp_servers() == ["stripe"]
    assert await agent.list_servers() == ["filesystem", "stripe"]

    agent.aggregator.initialized = True
    status_map = await agent.get_server_status()
    assert set(status_map) == {"filesystem", "stripe"}
    assert status_map["stripe"].is_connected is True
    assert status_map["stripe"].transport == "http"


@pytest.mark.asyncio
async def test_card_provider_server_lists_visible_name() -> None:
    internal_name = "card-source-revision-docs"
    server = MCPServerSettings(
        name="docs",
        management="provider",
        transport="http",
        url="https://example.com/mcp",
    )
    registry = ServerRegistry()
    registry.register_card(internal_name, server)
    context = Context(
        config=Settings.model_construct(
            mcp=MCPSettings.model_construct(servers={internal_name: server}),
        ),
        server_registry=registry,
    )
    agent = McpAgent(
        config=AgentConfig(name="agent", servers=[internal_name]),
        context=context,
        connection_persistence=False,
    )

    assert agent.list_attached_mcp_servers() == ["docs"]
    assert await agent.list_servers() == ["docs"]


def test_provider_managed_servers_attach_state_to_supported_llm() -> None:
    context = Context(
        config=Settings(
            mcp=MCPSettings(
                servers={
                    "stripe": MCPServerSettings(
                        name="stripe",
                        management="provider",
                        transport="http",
                        url="https://mcp.stripe.com",
                    )
                }
            )
        )
    )
    agent = McpAgent(
        config=AgentConfig(name="billing", servers=["stripe"]),
        context=context,
    )
    llm = _StubProviderManagedLLM(provider=Provider.ANTHROPIC)

    agent._on_llm_attached(llm)

    assert llm.provider_managed_mcp_state.server_names == ("stripe",)


def test_provider_managed_servers_reject_codexresponses_llm() -> None:
    context = Context(
        config=Settings(
            mcp=MCPSettings(
                servers={
                    "stripe": MCPServerSettings(
                        name="stripe",
                        management="provider",
                        transport="http",
                        url="https://mcp.stripe.com",
                    )
                }
            )
        )
    )
    agent = McpAgent(
        config=AgentConfig(name="billing", servers=["stripe"]),
        context=context,
    )
    llm = _StubProviderManagedLLM(provider=Provider.CODEX_RESPONSES)

    with pytest.raises(AgentConfigError, match="OpenAI Responses provider"):
        agent._on_llm_attached(llm)


def test_instruction_takes_precedence_over_systemPrompt():
    """
    Test that AgentConfig.instruction takes precedence over
    default_request_params.systemPrompt when both are provided.

    This ensures that the explicit instruction parameter on AgentConfig
    overrides any systemPrompt already set in the RequestParams.
    """
    # Create RequestParams with a systemPrompt already set
    original_system_prompt = "You are a generic assistant from RequestParams."
    request_params = RequestParams(
        model="sonnet", temperature=0.7, max_tokens=32768, system_prompt=original_system_prompt
    )

    # Verify systemPrompt is set initially
    assert request_params.system_prompt == original_system_prompt

    # Create AgentConfig with BOTH instruction AND default_request_params with systemPrompt
    instruction = "You are a specialized assistant from AgentConfig instruction."
    config = AgentConfig(
        name="my_agent",
        instruction=instruction,
        default_request_params=request_params,
        model="sonnet",
    )

    # The AgentConfig.instruction should take precedence over systemPrompt in RequestParams
    assert config.default_request_params is not None
    assert config.default_request_params.system_prompt == instruction, (
        f"Expected AgentConfig.instruction ('{instruction}') to override "
        f"RequestParams.systemPrompt ('{original_system_prompt}'), "
        f"but got {config.default_request_params.system_prompt}"
    )
