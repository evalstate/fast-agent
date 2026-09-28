"""Verified, sequence-based recovery of compacted ATIF history.

Archives are snapshots, not additive event logs. Only a checkpoint's linked
snapshot (or one uniquely verified legacy snapshot) is expanded. Current and
previous session snapshots must never be unioned to recover a trajectory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mcp_types import TextContent

from fast_agent.constants import FAST_AGENT_COMPACTION_CHANNEL, FAST_AGENT_PROCESS_POLL_FOLD
from fast_agent.types import PromptMessageExtended

COMPACTION_BOUNDARY = "fast-agent-atif-compaction-boundary"
COPIED_CONTEXT = "fast-agent-atif-copied-context"
_EXPORT_MARKERS = (COMPACTION_BOUNDARY, COPIED_CONTEXT)


class HistoryReconstructionError(ValueError):
    """A complete audit history cannot be established from the available evidence."""


def _metadata(message: PromptMessageExtended, channel: str) -> dict[str, object] | None:
    blocks = (message.channels or {}).get(channel)
    if blocks is None:
        return None
    if len(blocks) != 1 or not isinstance(blocks[0], TextContent):
        raise HistoryReconstructionError("Malformed compaction metadata")
    try:
        value = json.loads(blocks[0].text)
    except ValueError:
        raise HistoryReconstructionError("Malformed compaction metadata") from None
    if not isinstance(value, dict):
        raise HistoryReconstructionError("Malformed compaction metadata")
    return value


def _count(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if type(value) is not int or value < 0:
        raise HistoryReconstructionError("Invalid compaction boundary count")
    return value


def _load_archive(path: Path, expected_digest: str | None = None) -> list[PromptMessageExtended]:
    # The permissive general history loader can silently skip malformed messages.
    # Audit recovery must validate every message instead.
    try:
        data = path.read_bytes()
        if expected_digest is not None and hashlib.sha256(data).hexdigest() != expected_digest:
            raise HistoryReconstructionError("Compaction archive digest mismatch")
        payload = json.loads(data)
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            raise ValueError
        return [PromptMessageExtended.model_validate(item) for item in payload["messages"]]
    except HistoryReconstructionError:
        raise
    except (OSError, ValueError):
        raise HistoryReconstructionError("Unreadable or malformed compaction archive") from None


def reconstruct_history(
    history: list[PromptMessageExtended],
    session_dir: Path | None,
    agent_name: str,
    *,
    _lineage: frozenset[Path] = frozenset(),
) -> list[PromptMessageExtended]:
    """Restore originals once, followed by an explicit model-context boundary.

    A retained tail has already executed when compaction occurs: it stays in its
    original position, before the boundary, not replayed after the summary.
    """
    checkpoints = [
        (index, metadata)
        for index, message in enumerate(history)
        if (metadata := _metadata(message, FAST_AGENT_COMPACTION_CHANNEL)) is not None
    ]
    if not checkpoints:
        return list(history)
    if len(checkpoints) != 1:
        raise HistoryReconstructionError("Ambiguous compaction checkpoints")
    index, metadata = checkpoints[0]
    if not all(message.is_template for message in history[:index]):
        raise HistoryReconstructionError("Compaction checkpoint has a non-template prefix")
    if session_dir is None:
        raise HistoryReconstructionError("Full export requires compaction archives")
    compacted = _count(metadata, "messages_compacted")
    if compacted == 0:
        raise HistoryReconstructionError("Empty compaction boundary")
    archive_name = metadata.get("archive_file")
    linked = "archive_file" in metadata
    if linked:
        if not isinstance(archive_name, str) or Path(archive_name).name != archive_name:
            raise HistoryReconstructionError("Compaction archive unavailable or invalid")
        candidates = [session_dir / archive_name]
    else:
        safe_agent = "".join(c if c.isalnum() or c in "-_" else "_" for c in agent_name)
        candidates = sorted(session_dir.glob(f"compacted_*_{safe_agent}.json"))

    matches: list[tuple[Path, list[PromptMessageExtended], int]] = []
    for path in candidates:
        if path.is_symlink() or path.resolve().parent != session_dir.resolve():
            raise HistoryReconstructionError("Compaction archive escapes session directory")
        digest = metadata.get("archive_sha256") if linked else None
        if linked and (not isinstance(digest, str) or not digest):
            raise HistoryReconstructionError("Compaction archive digest unavailable")
        archived = _load_archive(path, digest if isinstance(digest, str) else None)
        if linked:
            if _count(metadata, "template_messages") != index:
                raise HistoryReconstructionError("Compaction template boundary mismatch")
        tail_count = len(archived) - index - compacted
        if tail_count < 0:
            continue
        if linked and _count(metadata, "retained_messages") != tail_count:
            raise HistoryReconstructionError("Compaction retained boundary mismatch")
        if archived[:index] != history[:index]:
            continue
        tail = archived[len(archived) - tail_count :] if tail_count else []
        if tail != history[index + 1 : index + 1 + tail_count]:
            continue
        if not linked:
            # Legacy archives have no identity. Require positive sequence evidence,
            # timestamped originals, and proximity to the checkpoint's archive time.
            # Summary-only legacy histories cannot establish this contract.
            try:
                stamp = datetime.strptime(path.name[10:25], "%Y%m%d-%H%M%S")
                checkpoint_time = datetime.fromisoformat(str(metadata["compacted_at"]))
                delta = (_utc(checkpoint_time) - stamp.replace(tzinfo=timezone.utc)).total_seconds()
            except (KeyError, ValueError):
                continue
            if not tail_count or not 0 <= delta < 5:
                continue
            if any(
                message.timestamp is None
                and FAST_AGENT_COMPACTION_CHANNEL not in (message.channels or {})
                for message in archived
            ):
                continue
            if any(
                message.timestamp is not None and _utc(message.timestamp) > _utc(checkpoint_time)
                for message in archived
            ):
                continue
        matches.append((path, archived, tail_count))
    if len(matches) != 1:
        raise HistoryReconstructionError("Missing, conflicting, or ambiguous compaction archive")
    path, archived, tail_count = matches[0]
    resolved_path = path.resolve()
    if resolved_path in _lineage:
        raise HistoryReconstructionError("Cyclic compaction archive lineage")
    original = reconstruct_history(
        archived, session_dir, agent_name, _lineage=_lineage | {resolved_path}
    )
    boundary = history[index].model_copy(deep=True)
    try:
        compacted_at = datetime.fromisoformat(str(metadata["compacted_at"]))
    except (KeyError, ValueError):
        raise HistoryReconstructionError("Invalid compaction timestamp") from None
    boundary.timestamp = compacted_at
    summary_call = {
        key: metadata[key] for key in ("summary_request", "summary_response") if key in metadata
    }
    boundary.channels = {
        COMPACTION_BOUNDARY: [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "type": "compaction",
                        "strategy": "summary",
                        "boundary": "replace",
                        "scope": "previous_model_context",
                        "template_messages": index,
                        "compacted_messages": compacted,
                        "retained_messages": tail_count,
                        "replacement_order": [
                            "system_prompt",
                            "templates",
                            "summary",
                            "retained_tail",
                        ],
                        "archive_verified": True,
                        "summary_usage": "recorded" if summary_call else "unavailable",
                        **({"summary_call": summary_call} if summary_call else {}),
                    }
                ),
            )
        ]
    }
    tail = history[index + 1 : index + 1 + tail_count]
    # Under an ATIF "replace" boundary only later steps are model-visible, so the
    # context that survives compaction is re-emitted as copied context.
    copies = [
        *(_copied_context(message, "template", compacted_at) for message in history[:index]),
        *(_copied_context(message, "retained_tail", compacted_at) for message in tail),
    ]
    return original + [boundary, *copies] + history[index + 1 + tail_count :]


def _copied_context(
    message: PromptMessageExtended, role: str, timestamp: datetime
) -> PromptMessageExtended:
    copy = message.model_copy(deep=True)
    channels = {
        name: blocks
        for name, blocks in (copy.channels or {}).items()
        # The copy is the model-visible form: a nested summary is plain user text,
        # and a poll fold is its folded prompt rather than another audit expansion.
        if name not in (FAST_AGENT_COMPACTION_CHANNEL, FAST_AGENT_PROCESS_POLL_FOLD)
    }
    channels[COPIED_CONTEXT] = [TextContent(type="text", text=json.dumps({"role": role}))]
    copy.channels = channels
    copy.timestamp = timestamp
    return copy


def is_export_marker(message: PromptMessageExtended) -> bool:
    """Whether a message is a reconstruction artifact rather than an original."""
    channels = message.channels or {}
    return any(marker in channels for marker in _EXPORT_MARKERS)


def reconcile_transient(
    history: list[PromptMessageExtended],
    transient: list[PromptMessageExtended],
    *,
    overlap_history: list[PromptMessageExtended] | None = None,
) -> list[PromptMessageExtended]:
    """Join a final turn using a unique exact sequence alignment, never a set union."""
    if not history:
        return list(transient)
    if not transient:
        return list(history)
    originals = [
        message
        for message in (history if overlap_history is None else overlap_history)
        if not is_export_marker(message)
    ]
    # Include containment: a persisted final turn needs no second copy.
    matches = [
        start
        for start in range(len(originals))
        if originals[start : start + min(len(transient), len(originals) - start)]
        == transient[: min(len(transient), len(originals) - start)]
    ]
    if len(matches) != 1:
        raise HistoryReconstructionError("Transient turn has no unique history overlap")
    overlap = min(len(transient), len(originals) - matches[0])
    additions = transient[overlap:]
    if not additions:
        return list(history)
    # A transient turn can retain evidence completed before a post-turn summary
    # that the persisted archive did not capture. Keep trailing context boundaries
    # (and their copied context) at their observed times instead of moving that
    # evidence after compaction.
    split = len(history)
    while split and is_export_marker(history[split - 1]):
        split -= 1
    trailing = history[split:]
    result = list(history[:split])
    for message in additions:
        if trailing:
            if message.timestamp is None:
                raise HistoryReconstructionError("Transient boundary placement lacks a timestamp")
            while trailing:
                boundary = trailing[0]
                if boundary.timestamp is None:
                    raise HistoryReconstructionError("Compaction boundary lacks a timestamp")
                if _utc(message.timestamp) < _utc(boundary.timestamp):
                    break
                result.append(trailing.pop(0))
        result.append(message)
    return result + trailing


def _utc(timestamp: datetime) -> datetime:
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
