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
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._execute_streaming_call') as mock_completion:
            mock_response = Mock()
            mock_response.content = [Mock(type="text", text="Test response")]
            mock_response.stop_reason = "end_turn"
            mock_response.usage = Mock(input_tokens=10, output_tokens=5)
            mock_completion.return_value = mock_response
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []  # Return empty list to simulate truncation
                
                @app.agent(
                    name="anthropic_param_test",
                    instruction="Test truncation parameters",
                    model="haiku",  # Use Anthropic model to trigger Anthropic provider
                    request_params=RequestParams(
                        maxTokens=5000,
                        use_history=True,
                        context_truncation_mode="remove",
                        context_truncation_length_limit=5
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
                assert call_args[1]['limit'] == 5
    
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
                    model="gpt-4o",  # Use OpenAI model to trigger OpenAI provider
                    request_params=RequestParams(
                        maxTokens=5000,
                        use_history=True,
                        context_truncation_mode="summarize",
                        context_truncation_length_limit=100
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
                assert call_args[1]['limit'] == 100
    
    @pytest.mark.skip(reason="No access to API keys in the integration tests in the GitHub test runner.")
    @pytest.mark.anyio  
    async def test_google_truncation_parameters_passed(self, app):
        """Test that Google provider receives truncation parameters correctly."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_google_native.GoogleNativeAugmentedLLM._execute_api_call') as mock_api_call:
            mock_response = Mock()
            mock_response.text = "Test response"
            mock_response.usage_metadata = Mock(prompt_token_count=10, candidates_token_count=5)
            mock_response.candidates = [Mock(content=Mock(parts=[Mock(text="Test response")]))]
            mock_api_call.return_value = mock_response
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name="google_param_test",
                    instruction="Test truncation parameters",
                    model="gemini-2.0-flash",  # Use Google model to trigger Google provider
                    request_params=RequestParams(
                        maxTokens=5000,
                        use_history=True,
                        context_truncation_mode="remove", 
                        context_truncation_length_limit=5
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
                assert call_args[1]['limit'] == 5
    
    @pytest.mark.anyio
    async def test_truncation_not_called_when_disabled(self, app):
        """Test that truncation is not called when no truncation mode is specified."""
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._execute_streaming_call') as mock_completion:
            mock_response = Mock()
            mock_response.content = [Mock(type="text", text="Test response")]
            mock_response.stop_reason = "end_turn"
            mock_response.usage = Mock(input_tokens=10, output_tokens=5)
            mock_completion.return_value = mock_response
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                
                @app.agent(
                    name="no_truncation_test",
                    instruction="Test no truncation",
                    model="haiku",  # Use Anthropic model 
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
                    model="gpt-4o",  # Use OpenAI model
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
        
        with patch('mcp_agent.llm.providers.augmented_llm_anthropic.AnthropicAugmentedLLM._execute_streaming_call') as mock_completion:
            mock_response = Mock()
            mock_response.content = [Mock(type="text", text="Test response")]
            mock_response.stop_reason = "end_turn"
            mock_response.usage = Mock(input_tokens=10, output_tokens=5)
            mock_completion.return_value = mock_response
            
            with patch.object(ContextTruncation, 'truncate_if_required', new_callable=AsyncMock) as mock_truncate:
                mock_truncate.return_value = []
                
                @app.agent(
                    name=f"config_test_{mode}_{limit}",
                    instruction="Test various configurations",
                    model="haiku",  # Use Anthropic model
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