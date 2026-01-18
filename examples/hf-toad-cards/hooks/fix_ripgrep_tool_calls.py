"""
Combined hook to fix ripgrep tool call issues.

Fixes two common problems:
1. Invalid -R flag in ripgrep commands
2. Hallucinated tool name variations (exec → execute)
"""

from fast_agent.hooks.hook_context import HookContext
from fast_agent.core.logging.logger import get_logger

logger = get_logger(__name__)


# Map of incorrect tool names to correct ones
TOOL_NAME_CORRECTIONS = {
    "exec": "execute",
    "executescript": "execute",
    "execscript": "execute",
    "executor": "execute",
    "exec_command": "execute",
}


async def fix_ripgrep_tool_calls(ctx: HookContext) -> None:
    """
    Fix common ripgrep agent tool call issues before execution.
    
    1. Strips invalid -R flag from ripgrep commands
    2. Corrects hallucinated tool name variations (exec* → execute)
    
    Args:
        ctx: Hook context with access to the message and tool calls
    """
    # Only process before_tool_call hooks
    if ctx.hook_type != "before_tool_call":
        return
    
    message = ctx.message
    
    # Check if message has tool calls
    if not message.tool_calls:
        return
    
    # Iterate through all tool calls
    for tool_id, tool_call in message.tool_calls.items():
        # Fix 1: Correct hallucinated tool names
        original_name = tool_call.params.name
        
        # Check for exact match in corrections map
        if original_name in TOOL_NAME_CORRECTIONS:
            corrected_name = TOOL_NAME_CORRECTIONS[original_name]
            tool_call.params.name = corrected_name
            
            logger.warning(
                "Corrected hallucinated tool name",
                data={
                    "tool_id": tool_id,
                    "original": original_name,
                    "corrected": corrected_name,
                }
            )
        # Also check for any other "exec*" variants not in map
        elif original_name.startswith("exec") and original_name != "execute":
            corrected_name = "execute"
            tool_call.params.name = corrected_name
            
            logger.warning(
                "Corrected unknown exec* variant to execute",
                data={
                    "tool_id": tool_id,
                    "original": original_name,
                    "corrected": corrected_name,
                }
            )
        
        # Fix 2: Strip -R flag from ripgrep commands
        # Only process execute tool calls
        if tool_call.params.name != "execute":
            continue
        
        # Get the command arguments
        args = tool_call.params.arguments
        if not isinstance(args, dict):
            continue
        
        command = args.get("command")
        if not command or not isinstance(command, str):
            continue
        
        # Check if command contains rg and -R flag
        if "rg" not in command:
            continue
        
        # Look for -R flag (with space before and after, or at end)
        modified = False
        original_command = command
        
        # Replace " -R " with " "
        if " -R " in command:
            command = command.replace(" -R ", " ")
            modified = True
        
        # Replace " -R" at end of line or command
        if command.endswith(" -R"):
            command = command[:-3]
            modified = True
        
        # Replace " -R\n" in multiline commands
        if " -R\n" in command:
            command = command.replace(" -R\n", "\n")
            modified = True
        
        if modified:
            logger.warning(
                "Stripped invalid -R flag from ripgrep command",
                data={
                    "tool_id": tool_id,
                    "original": original_command,
                    "modified": command,
                }
            )
            
            # Update the command in place
            args["command"] = command
