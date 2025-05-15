"""
Pytest configuration fixtures for Bedrock integration tests.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from mcp_agent.core.fastagent import FastAgent


@pytest.fixture
def fast_agent():
    """
    Create a FastAgent instance for testing.
    Uses the configuration in the current directory.
    """
    return FastAgent("Bedrock Integration Test", config_path="fastagent.config.yaml")


@pytest.fixture
def mock_boto3():
    """
    Provide a mock boto3 module for testing.
    This ensures tests don't actually make AWS API calls.
    """
    with patch('mcp_agent.llm.providers.augmented_llm_bedrock.boto3') as mock:
        # Mock successful response for invoke_model
        mock_client = MagicMock()
        mock_response = {
            'body': MagicMock()
        }
        # Mock the read method to return a JSON string
        mock_response['body'].read.return_value = (
            '{"content": [{"type": "text", "text": "This is a mocked response from Bedrock."}]}'
        )
        mock_client.invoke_model.return_value = [mock_response], {}
        
        # Configure the boto3 mock to return our mock client
        mock.client.return_value = mock_client
        
        # Configure boto3.Session to return a mock session
        mock_session = MagicMock()
        mock_session.client.return_value = mock_client
        mock.Session.return_value = mock_session
        
        yield mock


@pytest.fixture
def mock_image_resource():
    """
    Create a mock image resource for multimodal testing.
    """
    from pydantic import AnyUrl
    from mcp.types import ImageResourceContents, EmbeddedResource
    
    # Create a mock image resource
    image_resource = ImageResourceContents(
        uri=AnyUrl("file:///test/image.png"),
        mimeType="image/png",
        byteValues=[0, 1, 2, 3, 4]  # Dummy byte values
    )
    
    # Create an embedded resource
    image_embedded = EmbeddedResource(
        type="resource",
        resource=image_resource
    )
    
    return image_embedded