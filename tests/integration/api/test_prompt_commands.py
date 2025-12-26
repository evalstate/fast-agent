"""
Test the prompt command processing functionality.
"""

import pytest

from fast_agent.ui.command_payloads import SelectPromptCommand, is_command_payload
from fast_agent.ui.enhanced_prompt import handle_special_commands


@pytest.mark.asyncio
async def test_command_handling_for_prompts():
    """Test the command handling functions for /prompts and /prompt commands."""
    # Test /prompts command after it's been pre-processed
    # The pre-processed form of "/prompts" is a SelectPromptCommand dataclass
    input_cmd = SelectPromptCommand(prompt_name=None, prompt_index=None)
    result = await handle_special_commands(input_cmd, True)
    assert is_command_payload(result), "Result should be a command payload"
    assert isinstance(result, SelectPromptCommand)
    assert result.prompt_name is None
    assert result.prompt_index is None

    # Test /prompt <number> command after pre-processing
    # The pre-processed form is a SelectPromptCommand with prompt_index
    input_cmd = SelectPromptCommand(prompt_name=None, prompt_index=3)
    result = await handle_special_commands(input_cmd, True)
    assert is_command_payload(result), "Result should be a command payload"
    assert isinstance(result, SelectPromptCommand)
    assert result.prompt_index == 3

    # Test /prompt <name> command after pre-processing
    # The pre-processed form is "SELECT_PROMPT:my-prompt"
    result = await handle_special_commands("SELECT_PROMPT:my-prompt", True)
    assert is_command_payload(result), "Result should be a command payload"
    assert isinstance(result, SelectPromptCommand)
    assert result.prompt_name == "my-prompt"
