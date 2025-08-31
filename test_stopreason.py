#!/usr/bin/env python3
"""Examples of using StopReason from mcp.types"""

from typing import Literal, get_args

from mcp.types import CreateMessageResult, StopReason

# StopReason is a Union type of Literal['endTurn', 'stopSequence', 'maxTokens'] | str
# To get the literal values, we can extract them:
stop_reason_literals = get_args(get_args(StopReason)[0])
print(f"StopReason literal values: {stop_reason_literals}")
# Output: ('endTurn', 'stopSequence', 'maxTokens')

# Common usage patterns:

# 1. Direct string comparison (simplest approach)
def check_stop_reason_v1(result: CreateMessageResult) -> None:
    if result.stopReason == "endTurn":
        print("Stop reason is endTurn")
    elif result.stopReason == "stopSequence":
        print("Stop reason is stopSequence")
    elif result.stopReason == "maxTokens":
        print("Stop reason is maxTokens")
    else:
        print(f"Stop reason is custom: {result.stopReason}")

# 2. Using constants for better maintainability
class StopReasonConstants:
    END_TURN = "endTurn"
    STOP_SEQUENCE = "stopSequence"
    MAX_TOKENS = "maxTokens"

def check_stop_reason_v2(result: CreateMessageResult) -> None:
    if result.stopReason == StopReasonConstants.END_TURN:
        print("Stop reason is END_TURN")
    elif result.stopReason == StopReasonConstants.STOP_SEQUENCE:
        print("Stop reason is STOP_SEQUENCE")
    elif result.stopReason == StopReasonConstants.MAX_TOKENS:
        print("Stop reason is MAX_TOKENS")

# 3. Using a mapping for actions
STOP_REASON_HANDLERS = {
    "endTurn": lambda: print("Normal conversation end"),
    "stopSequence": lambda: print("Hit a stop sequence"),
    "maxTokens": lambda: print("Reached token limit"),
}

def check_stop_reason_v3(result: CreateMessageResult) -> None:
    handler = STOP_REASON_HANDLERS.get(result.stopReason)
    if handler:
        handler()
    else:
        print(f"Unknown stop reason: {result.stopReason}")

# 4. Type-safe enum-like approach using Literal
StopReasonLiteral = Literal["endTurn", "stopSequence", "maxTokens"]

def is_known_stop_reason(reason: str | None) -> bool:
    """Check if a stop reason is one of the known literal values"""
    return reason in ("endTurn", "stopSequence", "maxTokens")

# 5. Extract valid stop reasons programmatically
def get_valid_stop_reasons() -> tuple[str, ...]:
    """Get the valid stop reason literals from the type definition"""
    from typing import get_args
    # Get the Literal type from the Union
    literal_type = get_args(StopReason)[0]
    # Get the literal values
    return get_args(literal_type)

# Example usage:
if __name__ == "__main__":
    from mcp.types import TextContent
    
    # Create a sample result
    sample_result = CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text="Hello!"),
        model="test-model",
        stopReason="endTurn"
    )
    
    print("\n--- Method 1: Direct comparison ---")
    check_stop_reason_v1(sample_result)
    
    print("\n--- Method 2: Using constants ---")
    check_stop_reason_v2(sample_result)
    
    print("\n--- Method 3: Using handlers ---")
    check_stop_reason_v3(sample_result)
    
    print("\n--- Method 4: Type checking ---")
    print(f"Is '{sample_result.stopReason}' a known stop reason? {is_known_stop_reason(sample_result.stopReason)}")
    
    print("\n--- Method 5: Get valid reasons ---")
    print(f"Valid stop reasons: {get_valid_stop_reasons()}")