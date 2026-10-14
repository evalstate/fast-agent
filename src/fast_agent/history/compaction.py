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

import hashlib
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

from mcp_types import ContentBlock, EmbeddedResource, ImageContent, ResourceLink, TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL
from fast_agent.core.logging.logger import get_logger
from fast_agent.event_progress import ProgressAction
from fast_agent.llm.compaction_support import (
    CompactionStrategy,
    CompactionSupport,
    compaction_trigger_tokens,
    resolve_compaction_support,
)
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.tool_result_metadata import tool_result_media_preview_exclusion
from fast_agent.types.llm_stop_reason import LlmStopReason
from fast_agent.utils.count_display import format_bytes

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from fast_agent.config import CompactionSettings
    from fast_agent.context import Context
    from fast_agent.interfaces import FastAgentLLMProtocol
    from fast_agent.llm.usage_tracking import UsageAccumulator
    from fast_agent.session.session_manager import SessionManager
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
- Distinguish durable state from ephemeral runtime state. Do not claim that a tool or \
capability is currently available or unavailable; tell the resumed model to check its \
current declarations.

Be concise, structured, and focused on helping the next LLM seamlessly continue the work. \
Respond with the summary only."""


def resolve_compaction_prompt(settings: "CompactionSettings | None") -> str:
    """Resolve the active summarization prompt (config override or built-in)."""
    configured = settings.configured_prompt() if settings else None
    return configured or DEFAULT_COMPACTION_PROMPT


LEGACY_SUMMARY_NOTICE = (
    "Earlier conversation history was compacted into the checkpoint summary below. "
    "Treat it as authoritative context for the work completed so far."
)
SUMMARY_NOTICE = (
    "Earlier conversation history was compacted into the checkpoint summary below. "
    "Treat completed work and durable state as authoritative. Current system and developer "
    "instructions, configuration, and declared tools override conflicting runtime-state "
    "claims in the summary."
)

_CHARS_PER_TOKEN = 4
_IMAGE_TOKENS: Final = 2000
_MIN_COMPACTABLE_MESSAGES = 2
_SUMMARY_TOKEN_ALLOWANCE = 2048
MID_TURN_RECENT_TOOL_EXCHANGES = 3


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
    strategy: CompactionStrategy = CompactionStrategy.SUMMARY
    """``summary`` for fast-agent's own summary, or the provider-native strategy."""
    checkpoint_bytes: int | None = None
    """Encrypted size of a provider-native checkpoint; unset for summaries."""

    @property
    def is_native(self) -> bool:
        return self.strategy != CompactionStrategy.SUMMARY


def describe_provider_checkpoint(
    description: str, *, opaque_bytes: int, archive_file: str | None
) -> str:
    """Visible text of a provider-native checkpoint: what it is, its size, its archive."""
    archive = archive_file or "none (session history disabled)"
    return f"{description}\nEncrypted size: {format_bytes(opaque_bytes)}. Archive: {archive}."


def is_compaction_message(message: PromptMessageExtended) -> bool:
    """True when the message is a compaction checkpoint summary."""
    return bool(message.channels and FAST_AGENT_COMPACTION_CHANNEL in message.channels)


def normalize_compaction_notice(message: PromptMessageExtended) -> None:
    """Update legacy checkpoint guidance without altering its summary or metadata."""
    if not is_compaction_message(message):
        return
    for content in message.content:
        if isinstance(content, TextContent) and LEGACY_SUMMARY_NOTICE in content.text:
            content.text = content.text.replace(LEGACY_SUMMARY_NOTICE, SUMMARY_NOTICE, 1)
            return


def should_auto_compact(
    usage: "UsageAccumulator | None",
    settings: "CompactionSettings",
    *,
    llm: "FastAgentLLMProtocol | None" = None,
    projected_context_tokens: int | None = None,
) -> bool:
    """True when observed or projected context usage crossed the compaction trigger."""
    if not settings.auto or usage is None:
        return False
    trigger = compaction_trigger_tokens(llm, settings, usage.context_window_size)
    current = (
        projected_context_tokens
        if projected_context_tokens is not None
        else usage.current_context_tokens
    )
    if trigger is None or current is None or current <= 0:
        return False
    return current >= trigger


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


def _is_image(content: ContentBlock) -> bool:
    if isinstance(content, ImageContent):
        return True
    if isinstance(content, ResourceLink):
        return bool(content.mime_type and content.mime_type.startswith("image/"))
    if isinstance(content, EmbeddedResource):
        mime_type = content.resource.mime_type
        return bool(mime_type and mime_type.startswith("image/"))
    return False


def estimate_tokens(messages: list[PromptMessageExtended]) -> int:
    """Estimate text and serialized non-image payloads, plus 2000 tokens per image.

    Image cost is a provider-independent placeholder, not a tokenizer estimate.
    Never count image base64 or image resource URLs as text, including in tool
    results and channels. Display-only tool result media previews are excluded.
    """
    chars = 0
    images = 0
    for message in messages:
        texts: list[str] = []
        for content in message.content:
            if _is_image(content):
                images += 1
                continue
            text = get_text(content)
            if text is not None:
                texts.append(text)
                continue
            try:
                chars += len(content.model_dump_json())
            except Exception:
                chars += 64
        chars += len("\n".join(texts))
        if message.tool_calls:
            for call in message.tool_calls.values():
                try:
                    chars += len(call.model_dump_json())
                except Exception:
                    chars += 64
        if message.tool_results:
            for result in message.tool_results.values():
                image_indices = {
                    index for index, content in enumerate(result.content) if _is_image(content)
                }
                images += len(image_indices)
                try:
                    chars += len(
                        result.model_dump_json(
                            exclude={
                                "content": image_indices,
                                **tool_result_media_preview_exclusion(),
                            }
                        )
                    )
                except Exception:
                    chars += 64
        if message.channels:
            for blocks in message.channels.values():
                for block in blocks:
                    if _is_image(block):
                        images += 1
                        continue
                    try:
                        chars += len(block.model_dump_json())
                    except Exception:
                        chars += 64
    return max(1, chars // _CHARS_PER_TOKEN + images * _IMAGE_TOKENS)


def _plan_compaction_with_budget(
    history: list[PromptMessageExtended],
    *,
    keep_turns: int,
    max_tokens_after: int | None,
    recent_tool_exchanges: int | None = None,
) -> CompactionPlan:
    """Return a compaction plan, reducing kept turns when the tail is too large."""
    if recent_tool_exchanges is not None:
        if max_tokens_after is None or max_tokens_after <= 0:
            return _plan_mid_turn_compaction(
                history,
                recent_tool_exchanges=max(recent_tool_exchanges, 0),
            )

        plan: CompactionPlan | None = None
        for exchanges_to_keep in range(max(recent_tool_exchanges, 0), -1, -1):
            try:
                candidate = _plan_mid_turn_compaction(
                    history,
                    recent_tool_exchanges=exchanges_to_keep,
                )
            except CompactionSkipped:
                continue
            projected = (
                estimate_tokens(candidate.templates + candidate.retained_tail)
                + _SUMMARY_TOKEN_ALLOWANCE
            )
            if projected <= max_tokens_after:
                return candidate
            plan = candidate
        if plan is not None:
            return plan
        raise CompactionSkipped("No older tool-loop history to compact.")

    plan = plan_compaction(history, keep_turns=keep_turns)
    if max_tokens_after is None or max_tokens_after <= 0:
        return plan

    for candidate_keep in range(max(keep_turns, 0), -1, -1):
        candidate = plan_compaction(history, keep_turns=candidate_keep)
        projected = (
            estimate_tokens(candidate.templates + candidate.retained_tail)
            + _SUMMARY_TOKEN_ALLOWANCE
        )
        if projected <= max_tokens_after:
            return candidate
        plan = candidate
    return plan


def _plan_mid_turn_compaction(
    history: list[PromptMessageExtended],
    *,
    recent_tool_exchanges: int = MID_TURN_RECENT_TOOL_EXCHANGES,
) -> CompactionPlan:
    """Compact a tool loop while retaining recent complete exchanges and the pending call."""
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

    pending_call = body[-1]
    if (
        pending_call.role != "assistant"
        or not pending_call.tool_calls
        or pending_call.stop_reason != LlmStopReason.TOOL_USE
    ):
        raise CompactionSkipped("No trailing pending tool call to preserve.")

    tail_start = len(body) - 1
    cursor = tail_start
    exchanges_kept = 0
    while cursor >= 2 and exchanges_kept < recent_tool_exchanges:
        request = body[cursor - 2]
        result = body[cursor - 1]
        request_ids = set(request.tool_calls or {})
        result_ids = set(result.tool_results or {})
        if (
            request.role != "assistant"
            or request.stop_reason != LlmStopReason.TOOL_USE
            or not request_ids
            or request_ids != result_ids
            or result.role != "user"
        ):
            break
        tail_start = cursor - 2
        cursor -= 2
        exchanges_kept += 1

    compact_region = body[:tail_start]
    if len(compact_region) < _MIN_COMPACTABLE_MESSAGES:
        raise CompactionSkipped("No older tool-loop history to compact.")

    return CompactionPlan(
        templates=templates,
        compact_region=compact_region,
        retained_tail=body[tail_start:],
    )


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
    strategy: CompactionStrategy = CompactionStrategy.SUMMARY,
    provider_channels: Mapping[str, list[ContentBlock]] | None = None,
    prompt_text: str | None,
    instructions: str | None,
    messages_compacted: int,
    tokens_before: int | None,
    context_window: int | None,
    model: str | None,
    archive_metadata: dict[str, object] | None = None,
) -> PromptMessageExtended:
    """Build the checkpoint as a user message with a typed compaction channel.

    Provider-native checkpoints also carry ``provider_channels``; their provider
    replays those instead of the visible text, which other providers receive.
    """
    visible = f"[COMPACTED HISTORY]\n{SUMMARY_NOTICE}\n\n{summary_text}"
    metadata = {
        "compacted_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "messages_compacted": messages_compacted,
        "tokens_before": tokens_before,
        "context_window": context_window,
        "model": model,
        "prompt": prompt_text,
        "instructions": instructions,
    }
    if archive_metadata is not None:
        metadata.update(archive_metadata)
    message = Prompt.user(visible)
    message.channels = {
        **(provider_channels or {}),
        FAST_AGENT_COMPACTION_CHANNEL: [
            TextContent(type="text", text=json.dumps(metadata, ensure_ascii=False))
        ],
    }
    return message


def _session_persistence_enabled(agent: CompactableAgent) -> bool:
    from fast_agent.context import get_current_context

    context = agent.context or get_current_context()
    config = context.config if context else None
    if config is None:
        return True
    return not config._fast_agent_no_home and config.session_history


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
        manager = context.session_manager if context is not None else None
        if manager is None:
            manager = get_session_manager()

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
            get_manager=lambda cwd: _resolve_active_session_manager(manager, cwd),
            context=session_context,
            seed_metadata={"agent_name": agent.name},
        )
        session = identity.session

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        safe_agent = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent.name)
        filename = f"compacted_{stamp}_{uuid.uuid4().hex}_{safe_agent}.json"
        filepath = session.directory / filename
        with tempfile.NamedTemporaryFile(
            dir=session.directory, prefix=".compaction-", suffix=".json", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            save_messages(history, str(temporary))
            # Publish only complete bytes, without overwriting an existing archive.
            os.link(temporary, filepath)
        finally:
            temporary.unlink(missing_ok=True)
        manager.set_current_session(session)
        return str(filepath)
    except RuntimeError as exc:
        if "session manager" in str(exc).lower():
            raise
        logger.warning(
            "Failed to archive pre-compaction history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
        return None
    except Exception as exc:
        logger.warning(
            "Failed to archive pre-compaction history",
            data={"error": str(exc), "error_type": type(exc).__name__},
        )
        return None


def _resolve_active_session_manager(
    manager: "SessionManager", cwd: Path | None
) -> "SessionManager":
    if cwd is not None and cwd.resolve() != manager.workspace_dir:
        raise RuntimeError(
            "Compaction archive requested a different cwd than the active session manager."
        )
    return manager


def _resolved_path(raw_path: object | None) -> Path | None:
    if not raw_path:
        return None
    return Path(str(raw_path)).expanduser().resolve()


@contextmanager
def _compacting(
    llm: FastAgentLLMProtocol, action: ProgressAction = ProgressAction.COMPACTING
) -> Iterator[None]:
    """Temporarily label side-channel model progress as compaction."""
    previous_verb = llm.verb
    llm.verb = action
    try:
        yield
    finally:
        llm.verb = previous_verb


@dataclass(frozen=True, slots=True)
class _CheckpointContent:
    """Uniform content for publishing a summary or provider checkpoint."""

    text: str
    strategy: CompactionStrategy
    request_text: str
    prompt_text: str | None
    response: PromptMessageExtended
    provider_channels: Mapping[str, list[ContentBlock]]
    checkpoint_bytes: int | None

    def visible_text(self, archive_name: str | None) -> str:
        if self.checkpoint_bytes is None:
            return self.text
        return describe_provider_checkpoint(
            self.text, opaque_bytes=self.checkpoint_bytes, archive_file=archive_name
        )


async def _client_summary(
    llm: "FastAgentLLMProtocol",
    summary_source: list[PromptMessageExtended],
    request_text: str,
    prompt_text: str,
) -> _CheckpointContent:
    """Summarize with a side-channel call to the agent's own model (no tools)."""
    with _compacting(llm):
        response = await llm.generate(
            summary_source + [Prompt.user(request_text)], None, tools=None
        )
    summary_text = (response.last_text() or "").strip()
    if not summary_text:
        raise CompactionError("Compaction model returned an empty summary; history unchanged.")
    return _CheckpointContent(
        text=summary_text,
        strategy=CompactionStrategy.SUMMARY,
        request_text=request_text,
        prompt_text=prompt_text,
        response=response,
        provider_channels={},
        checkpoint_bytes=None,
    )


def uses_native_compaction(
    llm: "FastAgentLLMProtocol | None",
    settings: "CompactionSettings",
    *,
    instructions: str | None = None,
) -> bool:
    """Whether compaction produces a provider-native checkpoint.

    Native checkpoints take no prompt: a custom ``compaction.prompt`` or focus
    instructions ask for a written summary, so either selects fast-agent's summary.
    """
    if resolve_compaction_support(llm, settings) is not CompactionSupport.ON_DEMAND:
        return False
    return not settings.has_custom_prompt and not (instructions and instructions.strip())


async def _native_checkpoint(
    llm: "FastAgentLLMProtocol",
    summary_source: list[PromptMessageExtended],
) -> _CheckpointContent:
    """Ask the provider for an opaque checkpoint of the compact region."""
    with _compacting(llm, ProgressAction.COMPACTING_CHECKPOINT):
        try:
            native = await llm.native_compaction(summary_source)
        except Exception as exc:
            raise CompactionError(f"Provider compaction failed; history unchanged: {exc}") from exc
    return _CheckpointContent(
        text=native.description,
        strategy=native.strategy,
        request_text=native.request,
        prompt_text=None,
        response=_without_channels(native.response, native.checkpoint_channels),
        provider_channels=native.checkpoint_channels,
        checkpoint_bytes=native.opaque_bytes,
    )


def _without_channels(
    message: PromptMessageExtended, names: Mapping[str, object]
) -> PromptMessageExtended:
    channels = {
        name: blocks for name, blocks in (message.channels or {}).items() if name not in names
    }
    return message.model_copy(update={"channels": channels or None})


def _audit_response(response: PromptMessageExtended) -> dict[str, object]:
    """Serialize the call for exports; native content already excludes checkpoint payloads."""
    return response.model_dump(by_alias=True, mode="json", exclude_none=True)


async def compact_conversation(
    agent: CompactableAgent,
    *,
    settings: "CompactionSettings",
    instructions: str | None = None,
    recent_tool_exchanges: int | None = None,
    pending_context_tokens: int = 0,
    on_start: Callable[[], None] | None = None,
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

    pending_context_tokens = max(pending_context_tokens, 0)
    usage = agent.usage_accumulator
    tokens_before = usage.current_context_tokens if usage else None
    if tokens_before is not None:
        tokens_before += pending_context_tokens
    context_window = usage.context_window_size if usage else None
    if tokens_before is not None and tokens_before <= 0:
        tokens_before = None

    history = list(agent.message_history)
    trigger = compaction_trigger_tokens(llm, settings, context_window)
    max_tokens_after = max(trigger - pending_context_tokens, 1) if trigger is not None else None
    plan = _plan_compaction_with_budget(
        history,
        keep_turns=settings.keep_turns,
        max_tokens_after=max_tokens_after,
        recent_tool_exchanges=recent_tool_exchanges,
    )

    prompt_text = resolve_compaction_prompt(settings)
    request_text = prompt_text
    if instructions and instructions.strip():
        request_text = (
            f"{prompt_text}\n\nAdditional focus for this summary:\n{instructions.strip()}"
        )

    if on_start is not None:
        # Planning succeeded: compaction will now call the model.
        on_start()
    summary_source = plan.templates + plan.compact_region
    if uses_native_compaction(llm, settings, instructions=instructions):
        checkpoint = await _native_checkpoint(llm, summary_source)
    else:
        checkpoint = await _client_summary(llm, summary_source, request_text, prompt_text)

    archive_file = _archive_history(agent, history)
    if archive_file is None and _session_persistence_enabled(agent):
        raise CompactionError("Compaction archive failed; history unchanged.")
    archive_name = Path(archive_file).name if archive_file else None
    summary_text = checkpoint.visible_text(archive_name)

    summary_message = build_summary_message(
        summary_text,
        strategy=checkpoint.strategy,
        provider_channels=checkpoint.provider_channels,
        prompt_text=checkpoint.prompt_text,
        instructions=instructions,
        messages_compacted=len(plan.compact_region),
        tokens_before=tokens_before,
        context_window=context_window,
        model=usage.model if usage else None,
        archive_metadata={
            "archive_file": archive_name,
            "archive_sha256": (
                hashlib.sha256(Path(archive_file).read_bytes()).hexdigest()
                if archive_file
                else None
            ),
            "template_messages": len(plan.templates),
            "retained_messages": len(plan.retained_tail),
            "checkpoint_bytes": checkpoint.checkpoint_bytes,
            # The summarization call is a real model call: keep its request and
            # response (with usage/timing channels) so exports can account for it.
            "summary_request": checkpoint.request_text,
            "summary_response": _audit_response(checkpoint.response),
        },
    )

    new_history = plan.templates + [summary_message] + plan.retained_tail
    agent.load_message_history(new_history)

    # Opaque checkpoint payloads are far smaller in tokens than their encrypted text.
    estimate_history = (
        plan.templates
        + [_without_channels(summary_message, checkpoint.provider_channels)]
        + plan.retained_tail
    )
    tokens_after_estimate = estimate_tokens(estimate_history) + pending_context_tokens
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
        strategy=checkpoint.strategy,
        checkpoint_bytes=checkpoint.checkpoint_bytes,
    )


async def persist_compacted_session(agent: CompactableAgent, *, no_home: bool = False) -> None:
    """Persist the agent's (now compacted) history into the active session."""

    class _StubRunner:
        iteration = 0
        request_params = None

    try:
        if no_home or not _session_persistence_enabled(agent):
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
