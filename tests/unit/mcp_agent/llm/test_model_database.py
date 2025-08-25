
from fast_agent.agents.llm_agent import LlmAgent
from mcp_agent.core.agent_types import AgentConfig
from mcp_agent.llm.model_database import ModelDatabase
from mcp_agent.llm.model_factory import ModelFactory


def test_model_database_context_windows():
    """Test that ModelDatabase returns expected context windows"""
    # Test known models
    assert ModelDatabase.get_context_window("claude-sonnet-4-0") == 200000
    assert ModelDatabase.get_context_window("gpt-4o") == 128000
    assert ModelDatabase.get_context_window("gemini-2.0-flash") == 1048576

    # Test unknown model
    assert ModelDatabase.get_context_window("unknown-model") is None


def test_model_database_max_tokens():
    """Test that ModelDatabase returns expected max tokens"""
    # Test known models with different max_output_tokens (no cap)
    assert ModelDatabase.get_default_max_tokens("claude-sonnet-4-0") == 64000  # ANTHROPIC_SONNET
    assert ModelDatabase.get_default_max_tokens("gpt-4o") == 16384  # OPENAI_STANDARD
    assert ModelDatabase.get_default_max_tokens("o1") == 100000  # High max_output_tokens

    # Test fallbacks
    assert ModelDatabase.get_default_max_tokens("unknown-model") == 2048
    assert ModelDatabase.get_default_max_tokens(None) == 2048


def test_model_database_tokenizes():
    """Test that ModelDatabase returns expected tokenization types"""
    # Test multimodal model
    claude_tokenizes = ModelDatabase.get_tokenizes("claude-sonnet-4-0")
    assert "text/plain" in claude_tokenizes
    assert "image/jpeg" in claude_tokenizes
    assert "application/pdf" in claude_tokenizes

    # Test unknown model
    assert ModelDatabase.get_tokenizes("unknown-model") is None


def test_llm_uses_model_database_for_max_tokens():
    """Test that LLM instances use ModelDatabase for maxTokens defaults"""

    agent = LlmAgent(AgentConfig(name="Test Agent"))
    # Test with a model that has 8192 max_output_tokens (should get full amount)
    factory = ModelFactory.create_factory("claude-sonnet-4-0")
    llm = factory(agent=agent)
    assert llm.default_request_params.maxTokens == 64000

    # Test with a model that has high max_output_tokens (should get full amount)
    factory2 = ModelFactory.create_factory("o1")
    llm2 = factory2(agent=agent)
    assert llm2.default_request_params.maxTokens == 100000

    # Test with passthrough model (should get its configured max tokens)
    factory3 = ModelFactory.create_factory("passthrough")
    llm3 = factory3(agent=agent)
    expected_max_tokens = ModelDatabase.get_default_max_tokens("passthrough")
    assert llm3.default_request_params.maxTokens == expected_max_tokens


def test_llm_usage_tracking_uses_model_database():
    """Test that usage tracking uses ModelDatabase for context windows"""
    factory = ModelFactory.create_factory("passthrough")
    agent = LlmAgent(AgentConfig(name="Test Agent"))
    llm = factory(agent=agent, model="claude-sonnet-4-0")

    # The usage_accumulator should be able to get context window from ModelDatabase
    # when it has a model set (this happens when turns are added)
    llm.usage_accumulator.model = "claude-sonnet-4-0"
    assert llm.usage_accumulator.context_window_size == 200000
    assert llm.default_request_params.maxTokens == 64000  # Should match ModelDatabase default

    # Test with unknown model
    llm.usage_accumulator.model = "unknown-model"
    assert llm.usage_accumulator.context_window_size is None


def test_openai_provider_preserves_all_settings():
    """Test that OpenAI provider doesn't lose any original settings"""
    factory = ModelFactory.create_factory("gpt-4o")
    agent = LlmAgent(AgentConfig(name="Test Agent"))

    llm = factory(agent=agent, instruction="You are a helpful assistant")

    # Verify all the original OpenAI settings are preserved
    params = llm.default_request_params
    assert params.model == "gpt-4o"
    assert params.parallel_tool_calls  # Should come from base
    assert params.max_iterations == 20  # Should come from base (now 20)
    assert params.use_history  # Should come from base
    assert (
        params.systemPrompt == "You are a helpful assistant"
    )  # Should come from base (self.instruction)
    assert params.maxTokens == 16384  # Model-aware from ModelDatabase (gpt-4o)
