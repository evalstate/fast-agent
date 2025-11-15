"""Tests for the go command with multiple models to ensure parallel is the default."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from fast_agent.cli.commands.go import _run_agent


@pytest.mark.asyncio
async def test_multiple_models_creates_parallel_agent():
    """Test that multiple models automatically create a parallel agent."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class:
        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        # Track decorated agents
        decorated_agents = []
        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                decorated_agents.append(kwargs)
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        def mock_custom(agent_class, **kwargs):
            def decorator(func):
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel
        mock_fast.custom = mock_custom

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.interactive = AsyncMock()
        mock_agent_context.parallel = MagicMock()
        mock_agent_context.parallel.send = AsyncMock()
        mock_agent_context.parallel.generate = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Test with multiple models
        model = "haiku,sonnet,gpt-4"

        await _run_agent(
            model=model,
            instruction="Test instruction",
            mode="interactive"
        )

        # Verify that individual model agents were created
        assert len(decorated_agents) == 3, "Should create 3 individual model agents"

        # Verify agent names match the models
        agent_names = [agent["name"] for agent in decorated_agents]
        assert "haiku" in agent_names
        assert "sonnet" in agent_names
        assert "gpt-4" in agent_names

        # Verify each agent has the correct model
        for agent in decorated_agents:
            assert "model" in agent
            assert agent["model"] in ["haiku", "sonnet", "gpt-4"]

        # Verify parallel agent was created
        assert len(parallel_agents) == 1, "Should create exactly one parallel agent"
        parallel_config = parallel_agents[0]

        # Verify parallel agent configuration
        assert parallel_config["name"] == "parallel", "Parallel agent should be named 'parallel'"
        assert "fan_out" in parallel_config
        assert "fan_in" in parallel_config
        assert parallel_config["fan_in"] == "aggregate"
        assert parallel_config["include_request"] is True

        # Verify fan_out includes all model agents
        fan_out = parallel_config["fan_out"]
        assert len(fan_out) == 3
        assert "haiku" in fan_out
        assert "sonnet" in fan_out
        assert "gpt-4" in fan_out

        # Verify that interactive mode uses the parallel agent
        mock_agent_context.interactive.assert_called_once()
        call_kwargs = mock_agent_context.interactive.call_args[1]
        assert call_kwargs.get("agent_name") == "parallel", "Interactive mode should use 'parallel' as default agent"
        assert call_kwargs.get("pretty_print_parallel") is True


@pytest.mark.asyncio
async def test_multiple_models_message_mode_uses_parallel():
    """Test that message mode with multiple models uses the parallel agent."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class, \
         patch("fast_agent.cli.commands.go.ConsoleDisplay") as mock_display_class:

        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        def mock_custom(agent_class, **kwargs):
            def decorator(func):
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel
        mock_fast.custom = mock_custom

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.parallel = MagicMock()
        mock_agent_context.parallel.send = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Mock display
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display

        # Test with message mode
        await _run_agent(
            model="haiku,sonnet",
            instruction="Test instruction",
            message="Test message",
            mode="interactive"
        )

        # Verify parallel agent was created
        assert len(parallel_agents) == 1
        assert parallel_agents[0]["name"] == "parallel"

        # Verify message was sent to parallel agent
        mock_agent_context.parallel.send.assert_called_once_with("Test message")

        # Verify display shows parallel results
        mock_display.show_parallel_results.assert_called_once_with(mock_agent_context.parallel)


@pytest.mark.asyncio
async def test_multiple_models_prompt_file_mode_uses_parallel():
    """Test that prompt file mode with multiple models uses the parallel agent."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class, \
         patch("fast_agent.cli.commands.go.ConsoleDisplay") as mock_display_class, \
         patch("fast_agent.mcp.prompts.prompt_load.load_prompt") as mock_load_prompt:

        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        def mock_custom(agent_class, **kwargs):
            def decorator(func):
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel
        mock_fast.custom = mock_custom

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.parallel = MagicMock()
        mock_agent_context.parallel.generate = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Mock display and prompt loading
        mock_display = MagicMock()
        mock_display_class.return_value = mock_display
        mock_load_prompt.return_value = [{"role": "user", "content": "Test prompt"}]

        # Test with prompt file mode
        await _run_agent(
            model="haiku,sonnet,gpt-4",
            instruction="Test instruction",
            prompt_file="test_prompt.txt",
            mode="interactive"
        )

        # Verify parallel agent was created
        assert len(parallel_agents) == 1
        assert parallel_agents[0]["name"] == "parallel"

        # Verify prompt was loaded
        mock_load_prompt.assert_called_once()

        # Verify prompt was passed to parallel agent
        mock_agent_context.parallel.generate.assert_called_once()

        # Verify display shows parallel results
        mock_display.show_parallel_results.assert_called_once_with(mock_agent_context.parallel)


@pytest.mark.asyncio
async def test_single_model_does_not_create_parallel():
    """Test that a single model does not create a parallel agent."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class:
        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        # Track decorated agents
        decorated_agents = []
        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                decorated_agents.append(kwargs)
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.interactive = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Test with single model
        await _run_agent(
            model="haiku",
            instruction="Test instruction",
            mode="interactive"
        )

        # Verify only one agent was created
        assert len(decorated_agents) == 1
        assert decorated_agents[0]["model"] == "haiku"

        # Verify NO parallel agent was created
        assert len(parallel_agents) == 0, "Single model should not create parallel agent"


@pytest.mark.asyncio
async def test_no_model_does_not_create_parallel():
    """Test that no model specification does not create a parallel agent."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class:
        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        # Track decorated agents
        decorated_agents = []
        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                decorated_agents.append(kwargs)
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.interactive = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Test with no model
        await _run_agent(
            model=None,
            instruction="Test instruction",
            mode="interactive"
        )

        # Verify only one agent was created (using default model)
        assert len(decorated_agents) == 1

        # Verify NO parallel agent was created
        assert len(parallel_agents) == 0, "No model should not create parallel agent"


@pytest.mark.asyncio
async def test_parallel_agent_has_correct_fan_in():
    """Test that the parallel agent uses a silent fan-in aggregator."""
    with patch("fast_agent.cli.commands.go.FastAgent") as mock_fast_agent_class:
        # Setup mocks
        mock_fast = MagicMock()
        mock_fast_agent_class.return_value = mock_fast
        mock_fast.app.initialize = AsyncMock()

        # Track custom agents (for the aggregator)
        custom_agents = []
        parallel_agents = []

        def mock_agent(**kwargs):
            def decorator(func):
                return func
            return decorator

        def mock_parallel(**kwargs):
            def decorator(func):
                parallel_agents.append(kwargs)
                return func
            return decorator

        def mock_custom(agent_class, **kwargs):
            def decorator(func):
                custom_agents.append({"class": agent_class, **kwargs})
                return func
            return decorator

        mock_fast.agent = mock_agent
        mock_fast.parallel = mock_parallel
        mock_fast.custom = mock_custom

        # Mock the run context
        mock_agent_context = MagicMock()
        mock_agent_context.interactive = AsyncMock()
        mock_fast.run.return_value.__aenter__ = AsyncMock(return_value=mock_agent_context)
        mock_fast.run.return_value.__aexit__ = AsyncMock()

        # Test with multiple models
        await _run_agent(
            model="haiku,sonnet",
            instruction="Test instruction",
            mode="interactive"
        )

        # Verify aggregator was created
        assert len(custom_agents) == 1
        aggregator = custom_agents[0]

        # Verify aggregator configuration
        assert aggregator["name"] == "aggregate"
        assert aggregator["model"] == "passthrough"
        assert "SilentFanInAgent" in str(aggregator["class"])

        # Verify parallel agent uses this aggregator
        assert len(parallel_agents) == 1
        assert parallel_agents[0]["fan_in"] == "aggregate"
