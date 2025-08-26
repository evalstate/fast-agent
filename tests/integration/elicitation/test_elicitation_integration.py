"""
Integration tests for elicitation handler functionality.

These tests verify that:
1. Custom elicitation handlers work correctly (decorator precedence)
2. Config-based elicitation modes work (auto_cancel, forms, none)
3. Elicitation capabilities are properly advertised to servers
"""

from typing import TYPE_CHECKING, Any, Dict

import pytest
from mcp.shared.context import RequestContext
from mcp.types import ElicitRequestParams, ElicitResult

from mcp_agent.logging.logger import get_logger

if TYPE_CHECKING:
    from mcp import ClientSession

logger = get_logger(__name__)


async def custom_test_elicitation_handler(
    context: RequestContext["ClientSession", Any],
    params: ElicitRequestParams,
) -> ElicitResult:
    """Test handler that returns predictable responses for integration testing."""
    logger.info(f"Test elicitation handler called with: {params.message}")
    
    if params.requestedSchema:
        # Generate test data based on the schema for round-trip verification
        properties = params.requestedSchema.get("properties", {})
        content: Dict[str, Any] = {}
        
        # Provide test values for each field
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string")
            
            if field_type == "string":
                if field_name == "name":
                    content[field_name] = "Test User"
                elif field_name == "email":
                    content[field_name] = "test@example.com"
                elif field_name == "role":
                    # Check for enum values
                    enum_values = field_def.get("enum", [])
                    content[field_name] = enum_values[0] if enum_values else "developer"
                else:
                    content[field_name] = f"test-{field_name}"
            elif field_type == "integer":
                if field_name == "age":
                    content[field_name] = 30
                else:
                    content[field_name] = 42
            elif field_type == "number":
                content[field_name] = 3.14
            elif field_type == "boolean":
                if field_name == "subscribe_newsletter":
                    content[field_name] = True
                else:
                    content[field_name] = False
            elif field_type == "array":
                content[field_name] = ["test-item"]
            elif field_type == "object":
                content[field_name] = {"test": "value"}
        
        logger.info(f"Test handler returning: {content}")
        return ElicitResult(action="accept", content=content)
    else:
        # No schema, return simple response
        content = {"response": "test-response-no-schema"}
        logger.info(f"Test handler returning: {content}")
        return ElicitResult(action="accept", content=content)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_custom_elicitation_handler(fast_agent):
    """Test that custom elicitation handler works (highest precedence)."""
    fast = fast_agent
    
    @fast.agent(
        "custom-handler-agent",
        servers=["resource_forms"],
        elicitation_handler=custom_test_elicitation_handler,  # Custom handler
    )
    async def agent_function():
        async with fast.run() as agent:
            # First check that elicitation capability is advertised
            capabilities_result = await agent.get_resource("elicitation://client-capabilities")
            capabilities_text = str(capabilities_result)
            
            # Should have elicitation capability
            assert "✓ Elicitation" in capabilities_text, f"Elicitation capability not advertised: {capabilities_text}"
            
            # Now test the actual elicitation with our custom handler
            result = await agent.get_resource("elicitation://user-profile")
            result_str = str(result)
            
            # Verify we got expected test data from our custom handler
            assert "Test User" in result_str, f"Custom handler not used, got: {result_str}"
            assert "test@example.com" in result_str, f"Custom handler not used, got: {result_str}"
    
    await agent_function()


@pytest.mark.integration  
@pytest.mark.asyncio
async def test_forms_mode_capability_advertisement(fast_agent):
    """Test that forms mode advertises elicitation capability when no custom handler provided."""
    fast = fast_agent
    
    @fast.agent(
        "forms-agent",
        servers=["resource_forms"],
        # No elicitation_handler provided - should use config mode (forms is default)
    )
    async def agent_function():
        async with fast.run() as agent:
            # Check capabilities - should have elicitation capability
            capabilities_result = await agent.get_resource("elicitation://client-capabilities")
            capabilities_text = str(capabilities_result)
            
            # Should advertise elicitation capability in forms mode
            assert "✓ Elicitation" in capabilities_text, f"Forms mode should advertise elicitation: {capabilities_text}"
    
    await agent_function()


@pytest.mark.integration
@pytest.mark.asyncio  
async def test_elicitation_precedence_decorator_over_config(fast_agent):
    """Test that decorator-provided handler takes precedence over config."""
    fast = fast_agent
    
    @fast.agent(
        "precedence-test-agent", 
        servers=["resource_forms"],
        elicitation_handler=custom_test_elicitation_handler,  # Should override config
    )
    async def agent_function():
        async with fast.run() as agent:
            # Test actual elicitation behavior
            result = await agent.get_resource("elicitation://user-profile")
            result_str = str(result)
            
            # Should get test data from our custom handler, not config behavior
            assert "Test User" in result_str, f"Decorator precedence failed: {result_str}"
    
    await agent_function()