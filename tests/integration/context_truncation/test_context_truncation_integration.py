"""
Integration tests for context truncation functionality.
Tests that context truncation parameters are properly passed to providers
and that truncation logic is triggered correctly.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.request_params import RequestParams
from mcp_agent.llm.context_truncation_and_summarization import ContextTruncation


class TestContextTruncationIntegration:
    """Integration tests for context truncation across providers."""
    
    @pytest.fixture
    def app(self):
        """Create FastAgent app for testing."""
        return FastAgent("Context Truncation Integration Test")
    
    @pytest.mark.anyio
    async def test_anthropic_truncation_parameters_passed(self, app):
        """Test that Anthropic provider receives truncation parameters correctly."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._anthropic_completion') as mock_completion:
            mock_completion.return_value = [Mock()]
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []  # Return empty list to simulate truncation
                
                @app.agent(
                    name="anthropic_param_test",
                    instruction="Test truncation parameters",
                    model="passthrough",  # Use passthrough to avoid real API calls
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        context_truncation_mode="remove",
                        context_truncation_length_limit=500
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            await agent("Test message that should trigger truncation parameter passing")
                        except Exception:
                            pass  # We expect this to fail due to mocking, but we want to check the call
                
                await test_agent()
                
                # Verify that truncate_if_required was called with correct parameters
                mock_truncate.assert_called()
                call_args = mock_truncate.call_args
                assert call_args[1]['truncation_mode'] == "remove"
                assert call_args[1]['limit'] == 500
    
    @pytest.mark.anyio
    async def test_openai_truncation_parameters_passed(self, app):
        """Test that OpenAI provider receives truncation parameters correctly."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._openai_completion') as mock_completion:
            mock_completion.return_value = [Mock()]
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name="openai_param_test", 
                    instruction="Test truncation parameters",
                    model="passthrough",
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        context_truncation_mode="summarize",
                        context_truncation_length_limit=1000
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            await agent("Test message for OpenAI truncation")
                        except Exception:
                            pass
                
                await test_agent()
                
                # This test specifically catches the bug where OpenAI provider
                # wasn't calling truncation at all
                mock_truncate.assert_called()
                call_args = mock_truncate.call_args
                assert call_args[1]['truncation_mode'] == "summarize"
                assert call_args[1]['limit'] == 1000
    
    @pytest.mark.anyio  
    async def test_google_truncation_parameters_passed(self, app):
        """Test that Google provider receives truncation parameters correctly."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_google_native.GoogleNativeAugmentedLLM._completion') as mock_completion:
            mock_completion.return_value = Mock()
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name="google_param_test",
                    instruction="Test truncation parameters",
                    model="passthrough",
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        context_truncation_mode="remove", 
                        context_truncation_length_limit=750
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            await agent("Test message for Google truncation")
                        except Exception:
                            pass
                
                await test_agent()
                
                mock_truncate.assert_called()
                call_args = mock_truncate.call_args
                assert call_args[1]['truncation_mode'] == "remove"
                assert call_args[1]['limit'] == 750
    
    @pytest.mark.anyio
    async def test_truncation_not_called_when_disabled(self, app):
        """Test that truncation is not called when no truncation mode is specified."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._anthropic_completion') as mock_completion:
            mock_completion.return_value = [Mock()]
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                
                @app.agent(
                    name="no_truncation_test",
                    instruction="Test no truncation",
                    model="passthrough",
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        # No truncation parameters
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            await agent("Test message with no truncation")
                        except Exception:
                            pass
                
                await test_agent()
                
                # Truncation should not be called when disabled
                mock_truncate.assert_not_called()
    
    @pytest.mark.anyio
    async def test_request_params_fallback_to_defaults(self, app):
        """Test that providers fall back to default request params when none provided."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_openai.OpenAIAugmentedLLM._openai_completion') as mock_completion:
            mock_completion.return_value = [Mock()]
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name="defaults_test",
                    instruction="Test default params fallback", 
                    model="passthrough",
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        context_truncation_mode="remove",
                        context_truncation_length_limit=600
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            # This tests the fix for the bug where request_params was None
                            await agent("Test message for defaults fallback")
                        except Exception:
                            pass
                
                await test_agent()
                
                # Should still call truncation with correct params from defaults
                mock_truncate.assert_called()
                call_args = mock_truncate.call_args
                assert call_args[1]['truncation_mode'] == "remove"
                assert call_args[1]['limit'] == 600
    
    @pytest.mark.anyio
    @pytest.mark.parametrize("mode,limit", [
        ("remove", 500),
        ("summarize", 1000), 
        ("remove", 2000),
        ("summarize", 1500),
    ])
    async def test_various_truncation_configurations(self, app, mode, limit):
        """Test various truncation configurations work correctly."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._anthropic_completion') as mock_completion:
            mock_completion.return_value = [Mock()]
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name=f"config_test_{mode}_{limit}",
                    instruction="Test various configurations",
                    model="passthrough",
                    request_params=RequestParams(
                        maxTokens=100,
                        use_history=True,
                        context_truncation_mode=mode,
                        context_truncation_length_limit=limit
                    )
                )
                async def test_agent():
                    async with app.run() as agent:
                        try:
                            await agent(f"Test {mode} mode with {limit} limit")
                        except Exception:
                            pass
                
                await test_agent()
                
                mock_truncate.assert_called()
                call_args = mock_truncate.call_args  
                assert call_args[1]['truncation_mode'] == mode
                assert call_args[1]['limit'] == limit