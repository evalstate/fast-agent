"""Constrained BIOES Viterbi decoding for token privacy labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

IMPOSSIBLE = -1_000_000_000.0


@dataclass(frozen=True, slots=True)
class DecodedTokenSpan:
    """A decoded token-label span."""

    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ViterbiTables:
    """Precomputed BIOES transition / start / end masks as numpy arrays.

    `transitions[i, j]` is `0` for valid `prev=i -> cur=j` transitions and
    `IMPOSSIBLE` otherwise. `start_mask` / `end_mask` likewise gate the
    first / last timestep.
    """

    transitions: Any  # ndarray (L, L)
    start_mask: Any  # ndarray (L,)
    end_mask: Any  # ndarray (L,)


def build_viterbi_tables(labels: list[str], np_module: Any) -> ViterbiTables:
    """Build numpy transition tables for a label set. Call once per session."""

    label_count = len(labels)
    transitions = np_module.full((label_count, label_count), IMPOSSIBLE, dtype=np_module.float32)
    for previous_index, previous_label in enumerate(labels):
        for current_index, current_label in enumerate(labels):
            if _valid_transition(previous_label, current_label):
                transitions[previous_index, current_index] = 0.0
    start_mask = np_module.array(
        [0.0 if _valid_start(label) else IMPOSSIBLE for label in labels],
        dtype=np_module.float32,
    )
    end_mask = np_module.array(
        [0.0 if _valid_end(label) else IMPOSSIBLE for label in labels],
        dtype=np_module.float32,
    )
    return ViterbiTables(transitions=transitions, start_mask=start_mask, end_mask=end_mask)


def constrained_viterbi_np(
    logits: Any,
    tables: ViterbiTables,
    np_module: Any,
) -> list[int]:
    """Vectorized Viterbi over a `(T, L)` logit array. Returns a path of length T."""

    timesteps, label_count = logits.shape
    if timesteps == 0 or label_count == 0:
        return []

    transitions = tables.transitions
    scores = logits[0] + tables.start_mask
    backpointers = np_module.empty((timesteps, label_count), dtype=np_module.int64)
    backpointers[0] = 0
    label_arange = np_module.arange(label_count)
    for token_index in range(1, timesteps):
        # candidates[i, j] = scores[i] + transitions[i, j]; pick best i per j.
        candidates = scores[:, None] + transitions
        best_previous = np_module.argmax(candidates, axis=0)
        scores = candidates[best_previous, label_arange] + logits[token_index]
        backpointers[token_index] = best_previous

    final_scores = scores + tables.end_mask
    last = int(np_module.argmax(final_scores))
    path = [0] * timesteps
    path[-1] = last
    for token_index in range(timesteps - 1, 0, -1):
        last = int(backpointers[token_index, last])
        path[token_index - 1] = last
    return path


def constrained_viterbi(token_scores: list[list[float]], labels: list[str]) -> list[int]:
    """Reference pure-Python decoder. Used by tests; runtime path uses numpy."""

    if not token_scores:
        return []
    label_count = len(labels)
    if label_count == 0:
        return []

    valid_start = [_valid_start(label) for label in labels]
    valid_end = [_valid_end(label) for label in labels]
    predecessors = [
        [
            previous_index
            for previous_index, previous_label in enumerate(labels)
            if _valid_transition(previous_label, label)
        ]
        for label in labels
    ]

    scores: list[list[float]] = [
        [
            token_scores[0][label_index] if valid_start[label_index] else IMPOSSIBLE
            for label_index in range(label_count)
        ]
    ]
    backpointers: list[list[int]] = [[0] * label_count]

    for token_index in range(1, len(token_scores)):
        previous = scores[-1]
        current: list[float] = []
        current_backpointers: list[int] = []
        for label_index in range(label_count):
            best_score = IMPOSSIBLE
            best_previous = 0
            for previous_index in predecessors[label_index]:
                score = previous[previous_index] + token_scores[token_index][label_index]
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
            current.append(best_score)
            current_backpointers.append(best_previous)
        scores.append(current)
        backpointers.append(current_backpointers)

    final_scores = [
        score if valid_end[index] else IMPOSSIBLE
        for index, score in enumerate(scores[-1])
    ]
    last = max(range(label_count), key=lambda index: final_scores[index])
    path = [last]
    for token_index in range(len(token_scores) - 1, 0, -1):
        last = backpointers[token_index][last]
        path.append(last)
    path.reverse()
    return path


def token_spans_from_path(path: list[int], labels: list[str]) -> list[DecodedTokenSpan]:
    """Convert decoded label indices to token spans."""

    spans: list[DecodedTokenSpan] = []
    index = 0
    while index < len(path):
        label = labels[path[index]]
        prefix, kind = _split_label(label)
        if prefix == "S" and kind is not None:
            spans.append(DecodedTokenSpan(label=_normalize_kind(kind), start=index, end=index + 1))
            index += 1
            continue
        if prefix == "B" and kind is not None:
            end = index + 1
            while end < len(path):
                next_prefix, next_kind = _split_label(labels[path[end]])
                if next_kind != kind:
                    break
                if next_prefix == "E":
                    spans.append(
                        DecodedTokenSpan(label=_normalize_kind(kind), start=index, end=end + 1)
                    )
                    end += 1
                    break
                if next_prefix != "I":
                    break
                end += 1
            index = end
            continue
        index += 1
    return spans


def _valid_start(label: str) -> bool:
    prefix, _ = _split_label(label)
    return prefix in {"O", "B", "S"}


def _valid_end(label: str) -> bool:
    prefix, _ = _split_label(label)
    return prefix in {"O", "E", "S"}


def _valid_transition(previous: str, current: str) -> bool:
    previous_prefix, previous_kind = _split_label(previous)
    current_prefix, current_kind = _split_label(current)
    if previous_prefix in {"O", "E", "S"}:
        return current_prefix in {"O", "B", "S"}
    if previous_prefix in {"B", "I"}:
        return current_prefix in {"I", "E"} and previous_kind == current_kind
    return False


def _split_label(label: str) -> tuple[str, str | None]:
    if label == "O":
        return "O", None
    prefix, separator, kind = label.partition("-")
    if separator:
        return prefix.upper(), kind
    prefix, separator, kind = label.partition("_")
    if separator and prefix.upper() in {"B", "I", "E", "S"}:
        return prefix.upper(), kind
    return label.upper(), None


def _normalize_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized.startswith("private_") or normalized == "secret":
        return normalized
    return normalized
