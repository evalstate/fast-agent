"""
Integration test for AWS Bedrock connection.
This test verifies that the fast-agent can connect to Bedrock and use it properly.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.llm.providers.augmented_llm_bedrock import BedrockAugmentedLLM


@pytest.fixture
def mock_boto3():
    """Create a mock boto3 client for testing."""
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3') as mock:
        # Mock successful response for invoke_model
        mock_client = MagicMock()
        mock_response = {
            'body': MagicMock()
        }
        # Mock the read method to return a JSON string
        mock_response['body'].read.return_value = (
            '{"content": [{"type": "text", "text": "I am Claude on Bedrock."}]}'
        )
        mock_client.invoke_model.return_value = [mock_response], {}
        
        # Configure the boto3 mock to return our mock client
        mock.client.return_value = mock_client
        
        # Configure boto3.Session to return a mock session
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock.Session.return_value = mock_session
        
        yield mock


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_bedrock_authenticated_connection(mock_boto3):
    """Test that the agent can connect to Bedrock with valid authentication."""
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Connection Test")
    
    @fast.agent(
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent's LLM is a BedrockAugmentedLLM instance
            assert isinstance(agent.agent.llm, BedrockAugmentedLLM)
            
            # Send a test message
            response = await agent.send("Identify yourself")
            assert "Claude on Bedrock" in response
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_real_connection(fast_agent):
    """
    Test bedrock connection with potential real credentials.
    
    Note: This test requires real AWS credentials to be available.
    It will be skipped if the REAL_BEDROCK_TEST environment variable is not set to "true".
    """
    if os.environ.get("REAL_BEDROCK_TEST") != "true":
        pytest.skip("Skipping real Bedrock connection test (set REAL_BEDROCK_TEST=true to run)")
    
    # Use the FastAgent instance from the test directory fixture
    fast = fast_agent
    
    @fast.agent(
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant. Keep your answers brief."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send a simple query
            response = await agent.send("What is 2+2?")
            assert "4" in response
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_configuration_loading(mock_boto3):
    """Test that Bedrock configuration is properly loaded from config file."""
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Config Test")
    
    @fast.agent(
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    async def agent_function():
        async with fast.run() as agent:
            # Verify the agent's LLM is configured correctly
            llm = agent.agent.llm
            assert isinstance(llm, BedrockAugmentedLLM)
            
            # Check that the configuration was loaded from the config file
            assert llm.region == "us-east-1"
            assert llm.use_default_credentials is True
            
            # Check the model parameters
            assert isinstance(llm.default_params, dict)
            assert isinstance(llm.model_params, dict)
            
            # Verify a specific parameter from the config
            if "anthropic.claude-3-5-sonnet-20241022-v2:0" in llm.model_params:
                assert llm.model_params["anthropic.claude-3-5-sonnet-20241022-v2:0"]["temperature"] == 0.5
    
    await agent_function()