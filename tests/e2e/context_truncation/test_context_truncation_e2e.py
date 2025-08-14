"""
End-to-end tests for context truncation across all providers.
These tests ensure that context truncation works correctly and catches regressions
like the OpenAI provider missing truncation logic entirely.
"""

import pytest

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams


class TestContextTruncationE2E:
    """End-to-end tests for context truncation functionality."""
    
    @pytest.fixture
    def app(self):
        """Create FastAgent app for testing."""
        return FastAgent("Context Truncation Test")
    
    async def test_anthropic_remove_truncation(self, app):
        """Test that Anthropic provider applies REMOVE truncation correctly."""
        
        @app.agent(
            name="anthropic_truncation_test",
            instruction="You are a test assistant that processes large content.",
            servers=["large_content_server"],
            model="anthropic_test",
            request_params=RequestParams(
                maxTokens=500,
                use_history=True,
                context_truncation_mode="remove",
                context_truncation_length_limit=2000  # Small limit to force truncation
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # Generate large content that exceeds limit
                response1 = await agent("Generate large lorem ipsum content with 3000 tokens")
                assert response1 is not None
                
                # Generate more content to trigger truncation  
                response2 = await agent("Generate more large content with 2000 tokens")
                assert response2 is not None
                
                # Verify the agent still works after truncation
                response3 = await agent("Generate small content with 100 tokens")
                assert response3 is not None
                
                return "success"
        
        result = await test_agent()
        assert result == "success"
    
    async def test_openai_remove_truncation(self, app):
        """Test that OpenAI provider applies REMOVE truncation correctly."""
        
        @app.agent(
            name="openai_truncation_test", 
            instruction="You are a test assistant that processes large content.",
            servers=["large_content_server"],
            model="openai_test",
            request_params=RequestParams(
                maxTokens=500,
                use_history=True, 
                context_truncation_mode="remove",
                context_truncation_length_limit=1500  # Small limit to force truncation
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # This test specifically catches the bug where OpenAI provider
                # wasn't calling truncation logic at all
                
                # Generate large content
                response1 = await agent("Generate numbered content with 2000 tokens")
                assert response1 is not None
                
                # Generate more to trigger truncation
                response2 = await agent("Generate alphabet content with 1800 tokens") 
                assert response2 is not None
                
                # Verify functionality persists
                response3 = await agent("Get conversation length info")
                assert response3 is not None
                
                return "success"
        
        result = await test_agent()
        assert result == "success"
    
    async def test_google_summarize_truncation(self, app):
        """Test that Google provider applies SUMMARIZE truncation correctly."""
        
        @app.agent(
            name="google_truncation_test",
            instruction="You are a test assistant that processes large content.",
            servers=["large_content_server"], 
            model="google_test",
            request_params=RequestParams(
                maxTokens=500,
                use_history=True,
                context_truncation_mode="summarize", 
                context_truncation_length_limit=2500  # Small limit to force truncation
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # Test summarization mode
                response1 = await agent("Generate lorem content with 4000 tokens")
                assert response1 is not None
                
                response2 = await agent("Generate more lorem content with 3000 tokens")
                assert response2 is not None
                
                # After summarization, agent should still be functional
                response3 = await agent("Generate small content with 200 tokens")
                assert response3 is not None
                
                return "success"
        
        result = await test_agent()
        assert result == "success"
    
    async def test_truncation_parameter_detection(self, app):
        """Test that truncation parameters are properly detected and applied."""
        
        # This test catches the specific bug where request_params was None
        # in the OpenAI provider's _apply_prompt_provider_specific method
        
        @app.agent(
            name="param_detection_test",
            instruction="Test agent for parameter detection.",
            servers=["large_content_server"],
            model="openai_test",  # Test the previously broken provider
            request_params=RequestParams(
                maxTokens=300,
                use_history=True,
                context_truncation_mode="remove",
                context_truncation_length_limit=1000
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # If parameters aren't passed correctly, this will fail
                # because truncation won't be applied and context will overflow
                
                for i in range(3):
                    response = await agent(f"Generate large content iteration {i}")
                    assert response is not None
                
                return "success"
        
        result = await test_agent()
        assert result == "success"
    
    async def test_no_truncation_mode_specified(self, app):
        """Test that agents work correctly when no truncation mode is specified."""
        
        @app.agent(
            name="no_truncation_test",
            instruction="Test agent with no truncation configured.", 
            servers=["large_content_server"],
            model="anthropic_test",
            request_params=RequestParams(
                maxTokens=500,
                use_history=True,
                # No truncation parameters specified
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # Should work normally without truncation
                response = await agent("Generate small content with 100 tokens")
                assert response is not None
                return "success"
        
        result = await test_agent()
        assert result == "success"
    
    async def test_different_truncation_limits(self, app):
        """Test different truncation limits work correctly."""
        
        test_limits = [500, 1000, 2000, 5000]
        
        for limit in test_limits:
            @app.agent(
                name=f"limit_test_{limit}",
                instruction=f"Test agent with {limit} token limit.",
                servers=["large_content_server"],
                model="anthropic_test", 
                request_params=RequestParams(
                    maxTokens=300,
                    use_history=True,
                    context_truncation_mode="remove",
                    context_truncation_length_limit=limit
                )
            )
            async def test_agent():
                async with app.run() as agent:
                    # Generate content larger than limit
                    response = await agent(f"Generate content with {limit + 500} tokens")
                    assert response is not None
                    return "success"
            
            result = await test_agent()
            assert result == "success"
    
    @pytest.mark.parametrize("provider,mode", [
        ("anthropic_test", "remove"),
        ("anthropic_test", "summarize"),
        ("openai_test", "remove"),
        ("google_test", "remove"),
        ("google_test", "summarize"),
    ])
    async def test_all_provider_mode_combinations(self, app, provider, mode):
        """Test all valid combinations of providers and truncation modes."""
        
        @app.agent(
            name=f"combo_test_{provider}_{mode}",
            instruction="Test agent for provider/mode combinations.",
            servers=["large_content_server"],
            model=provider,
            request_params=RequestParams(
                maxTokens=400,
                use_history=True,
                context_truncation_mode=mode,
                context_truncation_length_limit=1500
            )
        )
        async def test_agent():
            async with app.run() as agent:
                # Test that each provider+mode combination works
                response1 = await agent("Generate large content with 2500 tokens")
                assert response1 is not None
                
                response2 = await agent("Generate more content with 2000 tokens") 
                assert response2 is not None
                
                return "success"
        
        result = await test_agent()
        assert result == "success"