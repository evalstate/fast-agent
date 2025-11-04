"""
Test the prompt command processing functionality.
"""

import pytest

from fast_agent.ui.enhanced_prompt import handle_special_commands


@pytest.mark.asyncio
async def test_command_handling_for_prompts():
    """Test the command handling functions for /prompts and /prompt commands."""
    # Test /prompts command after it's been pre-processed
    # The pre-processed form of "/prompts" is {"select_prompt": True, "prompt_name": None}
    result = await handle_special_commands({"select_prompt": True, "prompt_name": None}, True)
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "select_prompt" in result, "Result should have select_prompt key"
    assert result["select_prompt"] is True
    assert "prompt_name" in result
    assert result["prompt_name"] is None

    # Test /prompt <number> command after pre-processing
    # The pre-processed form is {"select_prompt": True, "prompt_index": 3}
    result = await handle_special_commands({"select_prompt": True, "prompt_index": 3}, True)
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "select_prompt" in result
    assert "prompt_index" in result
    assert result["prompt_index"] == 3

    # Test /prompt <name> command after pre-processing
    # The pre-processed form is "SELECT_PROMPT:my-prompt"
    result = await handle_special_commands("SELECT_PROMPT:my-prompt", True)
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "select_prompt" in result
    assert "prompt_name" in result
    assert result["prompt_name"] == "my-prompt"
