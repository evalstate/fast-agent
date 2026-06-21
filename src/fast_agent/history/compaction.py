"""
Conversation history compaction.

Replaces older history with a model-generated checkpoint summary while keeping
leading template messages and the most recent turns verbatim. The summary is a
user message carrying a ``fast-agent-compaction`` channel with the prompt and
counts used, so the operation stays inspectable after the fact.

Used by the ``/compact`` command and the automatic post-turn trigger
(``fast_agent.hooks.compaction``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from mcp.types import TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.event_progress import ProgressAction

if TYPE_CHECKING:
    from fast_agent.config import CompactionSettings
    from fast_agent.context import Context
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)

DEFAULT_COMPACTION_PROMPT = """\
You are performing a CONTEXT CHECKPOINT COMPACTION. Write a handoff summary of this \
conversation for another LLM that will resume the work with no other context.

Include:
- The user's goals and key decisions made so far
- Current progress and important results (including key tool outputs)
- Important context, constraints, or user preferences
- What remains to be done (clear next steps)
- Any critical data, file paths, identifiers, or references needed to continue

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. \
Respond with the summary only."""

SUMMARY_NOTICE = (
    "Earlier conversation history was compacted into the checkpoint summary below. "
    "Treat it as authoritative context for the work completed so far."
)

_CHARS_PER_TOKEN = 4
_MIN_COMPACTABLE_MESSAGES = 2


class CompactionSkipped(Exception):
    """Raised when there is nothing useful to compact."""


class CompactionError(Exception):
    """Raised when a compaction attempt fails; history is left untouched."""


class CompactableAgent(Protocol):
    """Agent surface required for compaction."""

    @property
    def name(self) -> str: ...

    @property
    def message_history(self) -> list[PromptMessageExtended]: ...

    def load_message_history(self, messages: list[PromptMessageExtended] | None) -> None: ...

    @property
    def usage_accumulator(self) -> "UsageAccumulator | None": ...

    @property
    def llm(self) -> "FastAgentLLMProtocol | None": ...

    @property
    def context(self) -> "Context | None": ...


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """Retention plan for a history, computed without any model call."""

    templates: list[PromptMessageExtended]
    compact_region: list[PromptMessageExtended]
    retained_tail: list[PromptMessageExtended]

    @property
    def messages_before(self) -> int:
        return len(self.templates) + len(self.compact_region) + len(self.retained_tail)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    """Outcome of a completed compaction."""

    agent_name: str
    summary_text: str
    messages_before: int
    messages_after: int
    tokens_before: int | None
    tokens_after_estimate: int
    context_window: int | None
    archive_file: str | None


def is_compaction_message(message: PromptMessageExtended) -> bool:
    """True when the message is a compaction checkpoint summary."""
    return bool(message.channels and FAST_AGENT_COMPACTION_CHANNEL in message.channels)


def compaction_metadata(message: PromptMessageExtended) -> dict[str, object] | None:
    """Return the recorded compaction metadata (prompt, counts) when present."""
    if not message.channels:
        return None
    blocks = message.channels.get(FAST_AGENT_COMPACTION_CHANNEL)
    if not blocks:
        return None
    block = blocks[0]
    if not isinstance(block, TextContent):
        return None
    try:
        data = json.loads(block.text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def resolve_compaction_prompt(settings: "CompactionSettings | None") -> str:
    """Resolve the active summarization prompt (config override or built-in)."""
    configured = settings.prompt if settings else None
    if not configured or not configured.strip():
        return DEFAULT_COMPACTION_PROMPT
    candidate = configured.strip()
    if len(candidate) < 1024 and "\n" not in candidate:
        try:
            path = Path(candidate).expanduser()
            if not path.is_absolute() and settings is not None and settings._config_file:
                path = Path(settings._config_file).expanduser().resolve().parent / path
            if path.is_file():
                return path.read_text(encoding="utf-8").strip() or DEFAULT_COMPACTION_PROMPT
        except OSError:
            pass
    return configured


def should_auto_compact(
    usage: "UsageAccumulator | None",
    settings: "CompactionSettings",
) -> bool:
    """True when server-observed context usage has crossed the configured threshold."""
    if not settings.auto or usage is None:
        return False
    window = usage.context_window_size
    if not window or window <= 0:
        return False
    current = usage.current_context_tokens
    if current <= 0:
        return False
    return (current / window) >= settings.threshold


def _turn_start_indices(messages: list[PromptMessageExtended]) -> list[int]:
    """Indices where real user turns start.

    A turn start is a user message that is neither a tool result nor a prior
    compaction summary. Summaries from earlier compactions look like user
    messages but must not be counted as turns, otherwise ``keep_turns`` protects
    them and the recent real turns, leaving nothing to compact.
    """
    return [
        index
        for index, message in enumerate(messages)
        if message.role == "user"
        and not message.tool_results
        and not is_compaction_message(message)
    ]


def estimate_tokens(messages: list[PromptMessageExtended]) -> int:
    """Rough token estimate for a message list (text plus serialized tool traffic)."""
    chars = 0
    for message in messages:
        chars += len(message.all_text())
        if message.tool_calls:
            for call in message.tool_calls.values():
                try:
                    chars += len(call.model_dump_json())
                except Exception:
                    chars += 64
        if message.tool_results:
            for result in message.tool_results.values():
                try:
                    chars += len(result.model_dump_json())
                except Exception:
                    chars += 64
    return max(1, chars // _CHARS_PER_TOKEN)


def plan_compaction(
    history: list[PromptMessageExtended],
    *,
    keep_turns: int,
) -> CompactionPlan:
    """
    Split history into (templates, compact region, retained tail).

    Leading template messages are always preserved. The retained tail starts at a
    user-turn boundary so assistant tool calls stay paired with their results.
    At least the first turn is always eligible for compaction.
    """
    template_count = 0
    for message in history:
        if message.is_template:
            template_count += 1
        else:
            break

    templates = list(history[:template_count])
    body = list(history[template_count:])

    if len(body) < _MIN_COMPACTABLE_MESSAGES:
        raise CompactionSkipped("Not enough conversation history to compact.")

    turn_starts = _turn_start_indices(body)
    if not turn_starts:
        raise CompactionSkipped("No completed user turns to compact.")

    effective_keep = min(max(keep_turns, 0), len(turn_starts) - 1)
    tail_start = turn_starts[-effective_keep] if effective_keep > 0 else len(body)

    compact_region = body[:tail_start]
    if len(compact_region) < _MIN_COMPACTABLE_MESSAGES:
        raise CompactionSkipped(
            "Recent turns already cover the whole conversation; nothing older to compact."
        )

    return CompactionPlan(
        templates=templates,
        compact_region=compact_region,
        retained_tail=body[tail_start:],
    )


def build_summary_message(
    summary_text: str,
    *,
    prompt_text: str,
    instructions: str | None,
    messages_compacted: int,
    tokens_before: int | None,
    context_window: int | None,
    model: str | None,
) -> PromptMessageExtended:
    """Build the checkpoint summary as a user message with a typed compaction channel."""
    visible = f"[COMPACTED HISTORY]\n{SUMMARY_NOTICE}\n\n{summary_text}"
    metadata = {
        "compacted_at": datetime.now(timezone.utc).isoformat(),
        "messages_compacted": messages_compacted,
        "tokens_before": tokens_before,
        "context_window": context_window,
        "model": model,
        "prompt": prompt_text,
        "instructions": instructions,
    }
    message = Prompt.user(visible)
    message.channels = {
        FAST_AGENT_COMPACTION_CHANNEL: [
            TextContent(type="text", text=json.dumps(metadata, ensure_ascii=False))
        ]
    }
    return message


def _session_persistence_enabled(agent: CompactableAgent) -> bool:
    from fast_agent.context import get_current_context

    context = agent.context or get_current_context()
    config = context.config if context else None
    if config is None:
        return True
    return not config._fast_agent_noenv and config.session_history


def _archive_history(
    agent: CompactableAgent,
    history: list[PromptMessageExtended],
) -> str | None:
    """Write the pre-compaction history into the active session directory, if any.

    Uses a ``compacted_*`` filename so session history rotation and resume never
    pick it up; it exists purely as a recoverable archive.
    """
    try:
        from fast_agent.context import get_current_context
        from fast_agent.mcp.prompt_serialization import save_messages
        from fast_agent.session import get_session_manager
        from fast_agent.session.identity import (
            SessionSaveContext,
            normalize_session_store_scope,
            resolve_session_for_save,
        )

        context = agent.context or get_current_context()
        if not _session_persistence_enabled(agent):
            return None

        acp_context = context.acp if context else None
        session_context = SessionSaveContext(
            acp_session_id=acp_context.session_id if acp_context else None,
            session_cwd=_resolved_path(acp_context.session_cwd) if acp_context else None,
            session_store_scope=normalize_session_store_scope(
                acp_context.session_store_scope if acp_context else "workspace"
            ),
            session_store_cwd=_resolved_path(acp_context.session_store_cwd)
            if acp_context
            else None,
        )
        identity = resolve_session_for_save(
            current_session=None,
            get_manager=lambda cwd: get_session_manager(cwd=cwd),
            context=session_context,
            seed_metadata={"agent_name": agent.name},
        )
        manager = identity.manager
        session = identity.session

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_agent = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent.name)
        filename = f"compacted_{stamp}_{safe_agent}.json"
        filepath = session.directory / filename
        save_messages(history, str(filepath))
        manager.set_current_session(session)
        return str(filepath)
    except Exception as exc:
        logger.warning(
            "Failed to archive pre-compaction history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
        return None


def _resolved_path(raw_path: object | None) -> Path | None:
    if not raw_path:
        return None
    return Path(str(raw_path)).expanduser().resolve()


async def compact_conversation(
    agent: CompactableAgent,
    *,
    settings: "CompactionSettings",
    instructions: str | None = None,
) -> CompactionResult:
    """
    Compact the agent's history into a checkpoint summary plus recent turns.

    The summarization request is a side-channel model call (no tools, history
    untouched on failure). On success the agent history is replaced and the
    usage accumulator's context size is overridden with an estimate until the
    next turn reports server-observed usage.
    """
    llm = agent.llm
    if llm is None:
        raise CompactionError(f"Agent '{agent.name}' has no attached LLM.")

    history = list(agent.message_history)
    plan = plan_compaction(history, keep_turns=settings.keep_turns)

    usage = agent.usage_accumulator
    tokens_before = usage.current_context_tokens if usage else None
    context_window = usage.context_window_size if usage else None
    if tokens_before is not None and tokens_before <= 0:
        tokens_before = None

    prompt_text = resolve_compaction_prompt(settings)
    request_text = prompt_text
    if instructions and instructions.strip():
        request_text = (
            f"{prompt_text}\n\nAdditional focus for this summary:\n{instructions.strip()}"
        )

    summary_source = plan.templates + plan.compact_region
    previous_verb = llm.verb
    llm.verb = ProgressAction.COMPACTING
    try:
        response = await llm.generate(
            summary_source + [Prompt.user(request_text)], None, tools=None
        )
    finally:
        llm.verb = previous_verb
    summary_text = (response.last_text() or "").strip()
    if not summary_text:
        raise CompactionError("Compaction model returned an empty summary; history unchanged.")

    archive_file = _archive_history(agent, history)

    summary_message = build_summary_message(
        summary_text,
        prompt_text=prompt_text,
        instructions=instructions,
        messages_compacted=len(plan.compact_region),
        tokens_before=tokens_before,
        context_window=context_window,
        model=usage.model if usage else None,
    )

    new_history = plan.templates + [summary_message] + plan.retained_tail
    agent.load_message_history(new_history)

    tokens_after_estimate = estimate_tokens(new_history)
    if usage is not None:
        usage.set_context_estimate(tokens_after_estimate)

    logger.info(
        "Compacted conversation history",
        data={
            "agent": agent.name,
            "messages_before": plan.messages_before,
            "messages_after": len(new_history),
            "tokens_before": tokens_before,
            "tokens_after_estimate": tokens_after_estimate,
            "archive_file": archive_file,
        },
    )

    return CompactionResult(
        agent_name=agent.name,
        summary_text=summary_text,
        messages_before=plan.messages_before,
        messages_after=len(new_history),
        tokens_before=tokens_before,
        tokens_after_estimate=tokens_after_estimate,
        context_window=context_window,
        archive_file=archive_file,
    )


async def persist_compacted_session(agent: CompactableAgent, *, noenv: bool = False) -> None:
    """Persist the agent's (now compacted) history into the active session."""

    class _StubRunner:
        iteration = 0
        request_params = None

    try:
        if noenv or not _session_persistence_enabled(agent):
            return

        from fast_agent.hooks.hook_context import HookAgentProtocol, HookContext
        from fast_agent.hooks.session_history import save_session_history

        history = agent.message_history
        if not history:
            return
        await save_session_history(
            HookContext(
                runner=_StubRunner(),
                agent=cast("HookAgentProtocol", agent),
                message=history[-1],
                hook_type="after_compaction",
            )
        )
    except Exception as exc:
        logger.warning(
            "Failed to persist compacted session history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
