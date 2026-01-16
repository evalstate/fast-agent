"""Custom hooks for testing."""

from fast_agent.hooks import HookContext

# Track hook invocations for testing
_hook_calls: list[dict] = []


def get_hook_calls() -> list[dict]:
    """Get recorded hook calls."""
    return _hook_calls


def clear_hook_calls() -> None:
    """Clear recorded hook calls."""
    _hook_calls.clear()


async def track_after_turn_complete(ctx: HookContext) -> None:
    """Track after_turn_complete hook calls."""
    _hook_calls.append(
        {
            "hook_type": ctx.hook_type,
            "iteration": ctx.iteration,
            "is_turn_complete": ctx.is_turn_complete,
            "history_len": len(ctx.message_history),
        }
    )
