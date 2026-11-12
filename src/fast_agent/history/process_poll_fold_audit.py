"""Typed audit archive for managed-process poll history folding, and its expansion."""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from mcp_types import TextContent
from pydantic import BaseModel, model_validator

from fast_agent.constants import FAST_AGENT_PROCESS_POLL_FOLD
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class ArchivedPollExchange(BaseModel):
    """One exact poll request/result pair preserved outside effective history."""

    call_id: str
    request: PromptMessageExtended
    result: PromptMessageExtended

    @model_validator(mode="after")
    def validate_exchange(self) -> "ArchivedPollExchange":
        if set(self.request.tool_calls or {}) != {self.call_id}:
            raise ValueError("Archived poll request does not match its call ID")
        if set(self.result.tool_results or {}) != {self.call_id}:
            raise ValueError("Archived poll result does not match its call ID")
        return self


class ArchivedContextRewrite(BaseModel):
    """A historical rewrite of the model-visible prompt context.

    `summary` is the exact text prepended to the retained observation. `fold`
    records this rewrite's own contribution (the usage and assistant updates of
    the polls it removed) alongside cumulative counters; per-poll data appears
    in exactly one rewrite so the audit stays linear in poll count.
    """

    after_call_id: str
    summary: str
    fold: dict[str, object]
    removed_call_ids: list[str]
    retained_call_ids: list[str]


class ProcessPollFoldAudit(BaseModel):
    """Lossless exchanges and context rewrites for one cumulative fold."""

    removed_exchanges: list[ArchivedPollExchange]
    retained_exchanges: list[ArchivedPollExchange]
    context_rewrites: list[ArchivedContextRewrite]

    @model_validator(mode="after")
    def validate_archive(self) -> "ProcessPollFoldAudit":
        if not self.retained_exchanges:
            raise ValueError("Poll fold audit must retain at least one exchange")
        if not self.context_rewrites:
            raise ValueError("Poll fold audit must contain a context rewrite")
        exchanges = [*self.removed_exchanges, *self.retained_exchanges]
        call_ids = [exchange.call_id for exchange in exchanges]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Poll fold audit contains duplicate call IDs")
        known_call_ids = set(call_ids)
        for rewrite in self.context_rewrites:
            referenced_call_ids = {
                rewrite.after_call_id,
                *rewrite.removed_call_ids,
                *rewrite.retained_call_ids,
            }
            if not referenced_call_ids <= known_call_ids:
                raise ValueError("Context rewrite references unknown call IDs")
        retained_call_ids = [exchange.call_id for exchange in self.retained_exchanges]
        latest_rewrite = self.context_rewrites[-1]
        if (
            latest_rewrite.after_call_id != retained_call_ids[-1]
            or latest_rewrite.retained_call_ids != retained_call_ids
        ):
            raise ValueError("Latest context rewrite is not anchored to a retained call")
        return self

    @property
    def exchanges(self) -> list[ArchivedPollExchange]:
        """Every archived exchange in original poll order."""
        return [*self.removed_exchanges, *self.retained_exchanges]


@dataclass(frozen=True, slots=True)
class HistoryMessage:
    """A history message kept as-is by fold expansion."""

    index: int
    message: PromptMessageExtended


@dataclass(frozen=True, slots=True)
class FoldExpansion:
    """Archived poll exchanges replacing the folded pair whose result is at ``index``."""

    index: int
    audit: ProcessPollFoldAudit


type PollHistorySegment = HistoryMessage | FoldExpansion


def process_poll_fold_audit(message: PromptMessageExtended) -> ProcessPollFoldAudit | None:
    """The fold audit carried by a folded tool-result message, if any.

    Raises ``ValueError`` when the message is folded but its audit is missing or invalid.
    """
    fold: dict[str, object] | None = None
    for block in reversed((message.channels or {}).get(FAST_AGENT_PROCESS_POLL_FOLD, ())):
        if not isinstance(block, TextContent):
            continue
        try:
            value = json.loads(block.text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            fold = value
            break
    if fold is None:
        return None
    if "audit" not in fold:
        raise ValueError("Managed-process poll fold is missing its audit archive")
    try:
        return ProcessPollFoldAudit.model_validate(fold["audit"])
    except ValueError as exc:
        raise ValueError("Managed-process poll fold audit archive is invalid") from exc


def expand_process_poll_folds(
    history: Sequence[PromptMessageExtended],
) -> list[PollHistorySegment]:
    """Restore folded managed-process polling to the exact archived exchanges.

    Each folded request/result pair becomes a ``FoldExpansion``; earlier retained
    exchanges that precede the pair are dropped because the audit archives them too.
    """
    segments: list[PollHistorySegment] = []
    index = 0
    while index < len(history):
        message = history[index]
        audit = process_poll_fold_audit(history[index + 1]) if index + 1 < len(history) else None
        if audit is None:
            segments.append(HistoryMessage(index, message))
            index += 1
            continue
        result_message = history[index + 1]
        *earlier_retained, retained_call_id = [
            exchange.call_id for exchange in audit.retained_exchanges
        ]
        if retained_call_id not in (message.tool_calls or {}) or retained_call_id not in (
            result_message.tool_results or {}
        ):
            raise ValueError("Managed-process poll fold retained exchange is invalid")
        retained_suffix = 2 * len(earlier_retained)
        if retained_suffix > len(segments):
            raise ValueError("Managed-process poll fold retained-step archive is inconsistent")
        if earlier_retained:
            suffix_call_ids = [
                call_id
                for segment in segments[-retained_suffix:]
                if isinstance(segment, HistoryMessage)
                for call_id in (segment.message.tool_calls or {})
            ]
            if suffix_call_ids != earlier_retained:
                raise ValueError("Managed-process poll fold retained call IDs are inconsistent")
            del segments[-retained_suffix:]
        segments.append(FoldExpansion(index + 1, audit))
        index += 2
    return segments


def restore_process_poll_history(
    history: Sequence[PromptMessageExtended],
) -> list[PromptMessageExtended]:
    """History with every folded poll exchange restored, in original order."""
    restored: list[PromptMessageExtended] = []
    for segment in expand_process_poll_folds(history):
        if isinstance(segment, HistoryMessage):
            restored.append(segment.message)
        else:
            for exchange in segment.audit.exchanges:
                restored.extend((exchange.request, exchange.result))
    return restored
