"""
Test demonstrating elicitation handler configuration modes.

This test covers the modes not tested in other files:
- auto_cancel mode
- none mode
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_auto_cancel_mode(fast_agent):
    """Test that auto_cancel mode works when configured."""

    @fast_agent.agent(
        "auto-cancel-agent",
        servers=["elicitation_auto_cancel_mode"],
        # No elicitation_handler provided - should use config mode
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            app = agent["auto-cancel-agent"]
            result = await app.call_tool("simple_rating", {})
            assert not result.is_error
            assert "cancel" in str(result).lower()

    await test_agent()


async def test_none_mode(fast_agent):
    """Test that 'none' mode disables elicitation capability advertisement."""

    @fast_agent.agent(
        "no-elicitation-agent",
        servers=["elicitation_none_mode"],
        # No elicitation_handler provided - should use config mode
    )
    async def test_agent():
        async with fast_agent.run() as agent:
            app = agent["no-elicitation-agent"]
            # Check capabilities reported by server
            result = await app.call_tool("client_capabilities", {})
            capabilities_text = str(result)
            print(f"Server reports capabilities: {capabilities_text}")

            # Should NOT have elicitation capability
            assert not result.is_error
            assert "✗ Elicitation" in capabilities_text
            print("✓ None mode working - elicitation capability NOT advertised")

    await test_agent()
