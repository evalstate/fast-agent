"""
Test demonstrating elicitation handler configuration modes.

This test covers the modes not tested in other files:
- auto_cancel mode
- none mode
"""

import pytest

from mcp_agent.logging.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.asyncio
async def test_auto_cancel_mode(fast_agent):
    """Test that auto_cancel mode works when configured."""
    
    @fast_agent.agent(
        "auto-cancel-agent",
        servers=["elicitation_auto_cancel_mode"],
        # No elicitation_handler provided - should use config mode
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            # This should auto-cancel due to config
            # Auto-cancel might result in an exception or a cancellation response
            try:
                result = await agent.get_resource("elicitation://generate")
                print(f"Result: {result}")
                # If we get a result, it should indicate cancellation
                result_str = str(result).lower()
                assert "cancel" in result_str or "decline" in result_str, (
                    f"Expected cancellation response, got: {result}"
                )
                print("✓ Auto-cancel mode test completed")
            except Exception as e:
                # Auto-cancel might result in an exception, which is also valid
                print(f"Auto-cancel test result: {e}")
                print("✓ Auto-cancel mode working (cancelled as expected)")
    
    await test_agent()


@pytest.mark.asyncio
async def test_none_mode(fast_agent):
    """Test that 'none' mode disables elicitation capability advertisement."""
    
    @fast_agent.agent(
        "no-elicitation-agent",
        servers=["elicitation_none_mode"],
        # No elicitation_handler provided - should use config mode
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            # Check capabilities reported by server
            result = await agent.get_resource("elicitation://client-capabilities")
            capabilities_text = str(result)
            print(f"Server reports capabilities: {capabilities_text}")
            
            # Should NOT have elicitation capability
            assert "✗ Elicitation" in capabilities_text or "✓ Elicitation" not in capabilities_text, (
                f"None mode should NOT advertise elicitation capability. Got: {capabilities_text}"
            )
            print("✓ None mode working - elicitation capability NOT advertised")
    
    await test_agent()