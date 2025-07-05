"""
Test demonstrating elicitation handler configuration modes using centralized config.

This test shows how elicitation mode can be configured at the application level
using the main fastagent.config.yaml file with well-named server configurations.
"""

import pytest

from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.asyncio
async def test_forms_mode(fast_agent):
    """Test that 'forms' mode (default) advertises elicitation capability."""
    
    @fast_agent.agent(
        "forms-elicitation-agent",
        servers=["elicitation_forms_mode"],
        # No elicitation_handler provided - should use config mode
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            # Check capabilities reported by server
            result = await agent.get_resource("elicitation://client-capabilities")
            capabilities_text = str(result)
            print(f"Server reports capabilities: {capabilities_text}")
            
            # Should HAVE elicitation capability
            assert "✓ Elicitation" in capabilities_text, (
                f"Forms mode test failed - elicitation capability NOT advertised. "
                f"Got: {capabilities_text}"
            )
            print("✓ Forms mode working - elicitation capability advertised")
    
    await test_agent()


@pytest.mark.asyncio
async def test_custom_handler_mode(fast_agent):
    """Test that custom handlers work (highest precedence)."""
    from test_elicitation_handler import custom_elicitation_handler
    
    @fast_agent.agent(
        "custom-handler-agent",
        servers=["elicitation_custom_handler"],
        elicitation_handler=custom_elicitation_handler,  # Custom handler (highest precedence)
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            # Check capabilities - should have elicitation
            capabilities_result = await agent.get_resource("elicitation://client-capabilities")
            capabilities_text = str(capabilities_result)
            
            assert "✓ Elicitation" in capabilities_text, (
                f"Custom handler mode failed - elicitation capability not advertised. "
                f"Got: {capabilities_text}"
            )
            
            # Test the actual elicitation - should use our custom handler
            result = await agent.get_resource("elicitation://user-profile")
            result_str = str(result)
            
            assert "Test User" in result_str, (
                f"Custom handler mode failed - custom handler not used. "
                f"Expected 'Test User' in result, got: {result_str}"
            )
            print("✓ Custom handler mode working - custom handler used")
    
    await test_agent()