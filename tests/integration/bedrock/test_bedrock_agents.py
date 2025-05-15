"""
Integration tests for Bedrock-based agents in fast-agent.
These tests verify that agent workflows function correctly with Bedrock models.
"""

import pytest
from unittest.mock import patch, MagicMock

from mcp_agent.core.fastagent import FastAgent
from mcp_agent.core.prompt import Prompt
from mcp_agent.mcp.interfaces import AgentParallelExecutionStrategy


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
            '{"content": [{"type": "text", "text": "This is a response from bedrock."}]}'
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
async def test_bedrock_agent_workflow(mock_boto3):
    """Test a simple workflow with a Bedrock agent."""
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Agent Workflow Test")
    
    @fast.agent(
        "bedrock_agent",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Send a sequence of messages
            first_response = await agent.bedrock_agent.send("Hello")
            assert "response from bedrock" in first_response
            
            # Send a follow-up
            second_response = await agent.bedrock_agent.send("Tell me more")
            assert "response from bedrock" in second_response
            
            # Check that the message history is maintained
            assert len(agent.bedrock_agent.message_history) >= 4  # 2 user messages + 2 responses
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_agent_structured_output(mock_boto3):
    """Test structured output with a Bedrock agent."""
    
    # Mock the invoke_model to return JSON
    mock_boto3.client.return_value.invoke_model.side_effect = lambda **kwargs: (
        [{
            'body': MagicMock(
                read=MagicMock(
                    return_value='{"content": [{"type": "text", "text": "{\\"name\\": \\"Alice\\", \\"age\\": 30}"}]}'
                )
            )
        }],
        {}
    )
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Structured Output Test")
    
    # Define a simple Person model
    from pydantic import BaseModel
    
    class Person(BaseModel):
        name: str
        age: int
    
    @fast.agent(
        "bedrock_agent",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a helpful assistant. Return structured JSON."
    )
    async def agent_function():
        async with fast.run() as agent:
            # Request structured output
            system_message = Prompt.system(
                "Always respond with JSON in the format: {\"name\": \"string\", \"age\": number}"
            )
            
            result = await agent.bedrock_agent.structured(
                Person,
                "Give me a person named Alice who is 30 years old.",
                system_message
            )
            
            # Verify the structured output
            assert isinstance(result, Person)
            assert result.name == "Alice"
            assert result.age == 30
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_with_chain_workflow(mock_boto3):
    """Test Bedrock with a chain workflow."""
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Chain Workflow Test")
    
    @fast.agent(
        "summarizer",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a text summarizer. Summarize the input text briefly."
    )
    async def summarizer(text: str) -> str:
        """Summarize the input text."""
        return f"Summary of: {text}"
    
    @fast.agent(
        "enhancer",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are a text enhancer. Improve the input text."
    )
    async def enhancer(text: str) -> str:
        """Enhance the input text."""
        return f"Enhanced: {text}"
    
    @fast.chain()
    async def workflow(text: str) -> str:
        """A simple chain workflow."""
        summary = await summarizer(text)
        enhanced = await enhancer(summary)
        return enhanced
    
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Run the chain workflow
            result = await workflow("This is a test text for the workflow.")
            
            # Verify the result (since we're mocking, we know exactly what to expect)
            assert "Enhanced: Summary of: This is a test text for the workflow." == result
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bedrock_with_parallel_workflow(mock_boto3):
    """Test Bedrock with a parallel workflow."""
    
    # Create a FastAgent
    fast = FastAgent("Bedrock Parallel Workflow Test")
    
    @fast.agent(
        "analyzer1",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are analyzer 1."
    )
    async def analyzer1(text: str) -> str:
        """First analyzer."""
        return f"Analysis 1: {text}"
    
    @fast.agent(
        "analyzer2",
        model="bedrock.anthropic.claude-3-5-sonnet-20241022-v2:0",
        instruction="You are analyzer 2."
    )
    async def analyzer2(text: str) -> str:
        """Second analyzer."""
        return f"Analysis 2: {text}"
    
    @fast.parallel(
        execution_strategy=AgentParallelExecutionStrategy.PARALLEL
    )
    async def parallel_workflow(text: str) -> list:
        """A parallel workflow."""
        result1 = await analyzer1(text)
        result2 = await analyzer2(text)
        return [result1, result2]
    
    @fast.agent()
    async def agent_function():
        async with fast.run() as agent:
            # Run the parallel workflow
            results = await parallel_workflow("Test text for parallel analysis.")
            
            # Verify the results
            assert len(results) == 2
            assert "Analysis 1: Test text for parallel analysis." == results[0]
            assert "Analysis 2: Test text for parallel analysis." == results[1]
    
    await agent_function()