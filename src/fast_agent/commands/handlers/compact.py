"""Shared /compact command handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.history.compaction import (
    DEFAULT_COMPACTION_PROMPT,
    CompactableAgent,
    CompactionError,
    CompactionSkipped,
    compact_conversation,
    estimate_tokens,
    persist_compacted_session,
    plan_compaction,
    resolve_compaction_prompt,
)
from fast_agent.ui.compaction_display import compaction_summary_lines, context_bar

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext
    from fast_agent.config import CompactionSettings

# Rough allowance for the summary the model has not written yet (preview only).
_PREVIEW_SUMMARY_TOKEN_ALLOWANCE = 500


def _compaction_settings(ctx: "CommandContext") -> "CompactionSettings":
    return ctx.resolve_settings().compaction


def _compactable_agent(ctx: "CommandContext", agent_name: str) -> CompactableAgent:
    return cast("CompactableAgent", ctx.agent_provider._agent(agent_name))


async def handle_compact(
    ctx: "CommandContext",
    *,
    agent_name: str,
    instructions: str | None = None,
) -> CommandOutcome:
    """Compact the agent's history into a checkpoint summary plus recent turns."""
    outcome = CommandOutcome()
    settings = _compaction_settings(ctx)
    agent = _compactable_agent(ctx, agent_name)

    try:
        result = await compact_conversation(
            agent,
            settings=settings,
            instructions=instructions,
        )
    except (CompactionSkipped, CompactionError) as exc:
        outcome.add_message(str(exc), channel="warning", agent_name=agent_name)
        return outcome
    except Exception as exc:
        outcome.add_message(
            f"Compaction failed; history unchanged: {exc}",
            channel="error",
            agent_name=agent_name,
        )
        return outcome

    await persist_compacted_session(agent, no_home=ctx.no_home)

    body = Text("\n").join(compaction_summary_lines(result))
    outcome.add_message(
        body,
        channel="info",
        title="History compacted",
        agent_name=agent_name,
    )
    outcome.add_message(
        "Summary kept in history (see /history). Restore with /history load <archive> if needed.",
        channel="system",
        agent_name=agent_name,
    )
    return outcome


async def handle_compact_preview(
    ctx: "CommandContext",
    *,
    agent_name: str,
) -> CommandOutcome:
    """Show what compaction would do, without calling the model."""
    outcome = CommandOutcome()
    settings = _compaction_settings(ctx)
    agent = _compactable_agent(ctx, agent_name)
    history = list(agent.message_history)

    try:
        plan = plan_compaction(history, keep_turns=settings.keep_turns)
    except CompactionSkipped as exc:
        outcome.add_message(str(exc), channel="warning", agent_name=agent_name)
        return outcome

    usage = agent.usage_accumulator
    tokens_now = usage.current_context_tokens if usage else 0
    window = usage.context_window_size if usage else None
    estimated_after = (
        estimate_tokens(plan.templates + plan.retained_tail) + _PREVIEW_SUMMARY_TOKEN_ALLOWANCE
    )

    lines: list[Text] = []
    context_line = Text("context ", style="dim")
    context_line.append_text(context_bar(tokens_now if tokens_now > 0 else None, window))
    context_line.append("  →  ", style="dim")
    context_line.append_text(context_bar(estimated_after, window))
    context_line.append(" est", style="dim")
    lines.append(context_line)

    detail = Text()
    detail.append(f"{len(plan.compact_region)} messages would be summarized")
    if plan.templates:
        detail.append(f"  •  {len(plan.templates)} template messages kept", style="dim")
    detail.append(
        f"  •  last {settings.keep_turns} turns kept verbatim ({len(plan.retained_tail)} messages)",
        style="dim",
    )
    lines.append(detail)

    auto_state = (
        f"auto-compaction at {settings.threshold * 100:.0f}% of context window"
        if settings.auto
        else "auto-compaction off"
    )
    lines.append(Text(auto_state + "  •  tune with compaction settings in config", style="dim"))

    outcome.add_message(
        Text("\n").join(lines),
        channel="info",
        title="Compaction preview (no model call)",
        agent_name=agent_name,
    )
    return outcome


async def handle_compact_prompt(
    ctx: "CommandContext",
    *,
    agent_name: str,
) -> CommandOutcome:
    """Show the active compaction prompt and where it comes from."""
    outcome = CommandOutcome()
    settings = _compaction_settings(ctx)
    prompt_text = resolve_compaction_prompt(settings)
    source = (
        "built-in" if prompt_text == DEFAULT_COMPACTION_PROMPT else "config (compaction.prompt)"
    )

    outcome.add_message(
        prompt_text,
        channel="info",
        title=f"Compaction prompt ({source})",
        agent_name=agent_name,
        render_markdown=True,
    )
    outcome.add_message(
        "> Override with `compaction.prompt` in `fastagent.config.yaml` "
        "(inline text or file path). Relative prompt paths resolve from the loaded config "
        "file directory, including `FAST_AGENT_HOME` configs. Add one-off focus with "
        "`/compact <instructions>`.",
        channel="system",
        agent_name=agent_name,
        render_markdown=True,
    )
    return outcome
