"""Built-in auto-compaction hooks.

After a turn completes, compacts the agent's history when server-observed
context usage crosses the configured threshold (``compaction.threshold`` in
settings). Failures never break the turn; history is left untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fast_agent.context import get_current_context
from fast_agent.core.logging.logger import get_logger
from fast_agent.history.compaction import (
    CompactionSkipped,
    compact_conversation,
    should_auto_compact,
)
from fast_agent.hooks.hook_messages import show_hook_failure, show_hook_message

if TYPE_CHECKING:
    from fast_agent.config import CompactionSettings
    from fast_agent.history.compaction import CompactableAgent
    from fast_agent.hooks.hook_context import HookContext

logger = get_logger(__name__)

_HOOK_NAME = "auto-compact"


def _resolve_compaction_settings(ctx: "HookContext") -> "CompactionSettings | None":
    config = ctx.context.config if ctx.context else None
    if config is None:
        current = get_current_context()
        config = current.config if current else None
    return config.compaction if config else None


def _effective_use_history(ctx: "HookContext") -> bool:
    request_params = ctx.request_params
    if request_params is not None:
        return request_params.use_history
    return ctx.agent.config.use_history


async def auto_compact_history(ctx: "HookContext") -> None:
    """Compact history after the turn when context usage crossed the threshold."""
    if not ctx.is_turn_complete:
        return
    await _auto_compact_history(ctx, min_keep_turns=0)

async def auto_compact_history_mid_turn(ctx: "HookContext") -> None:
    """Compact history during a tool loop while preserving the active turn."""
    if ctx.is_turn_complete:
        return
    await _auto_compact_history(ctx, min_keep_turns=1)


async def _auto_compact_history(ctx: "HookContext", *, min_keep_turns: int) -> None:
    settings = _resolve_compaction_settings(ctx)
    if settings is None or not settings.auto:
        return

    if ctx.agent.config.tool_only:
        return

    if not _effective_use_history(ctx):
        return

    if not should_auto_compact(ctx.usage, settings):
        return

    usage = ctx.usage
    percent = usage.context_usage_percentage if usage else None
    suffix = " mid-turn" if min_keep_turns else ""
    show_hook_message(
        ctx,
        f"context at {percent:.0f}% — compacting history{suffix}"
        if percent
        else f"compacting history{suffix}",
        hook_name=_HOOK_NAME,
        style="cyan",
    )

    try:
        result = await compact_conversation(
            cast("CompactableAgent", ctx.agent),
            settings=settings,
            min_keep_turns=min_keep_turns,
        )
    except CompactionSkipped as exc:
        logger.info("Auto-compaction skipped", data={"reason": str(exc)})
        return
    except Exception as exc:
        show_hook_failure(ctx, hook_name=_HOOK_NAME, error=exc)
        logger.exception("Auto-compaction failed; history unchanged")
        return

    from rich.text import Text

    from fast_agent.ui.compaction_display import compaction_summary_lines

    combined = Text("\n").join(compaction_summary_lines(result))
    show_hook_message(ctx, combined, hook_name=_HOOK_NAME, style="cyan")
