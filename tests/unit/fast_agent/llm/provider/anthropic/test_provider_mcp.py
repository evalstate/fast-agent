from fast_agent.llm.provider.anthropic.llm_anthropic import (
    MCP_CLIENT_BETA,
    AnthropicLLM,
)


def test_anthropic_provider_mcp_enables_mcp_client_beta() -> None:
    llm = AnthropicLLM()

    beta_flags = llm._resolve_anthropic_beta_flags(
        model="sonnet",
        structured_mode=None,
        thinking_enabled=False,
        request_tools=[],
        web_tool_betas=(),
        provider_mcp_enabled=True,
    )

    assert MCP_CLIENT_BETA in beta_flags
