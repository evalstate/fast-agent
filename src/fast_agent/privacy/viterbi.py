"""Constrained BIOES Viterbi decoding for token privacy labels."""

from __future__ import annotations

from dataclasses import dataclass

IMPOSSIBLE = -1_000_000_000.0


@dataclass(frozen=True, slots=True)
class DecodedTokenSpan:
    """A decoded token-label span."""

    label: str
    start: int
    end: int


def constrained_viterbi(log_probs: list[list[float]], labels: list[str]) -> list[int]:
    """Decode a valid BIOES path from per-token log probabilities."""

    if not log_probs:
        return []
    label_count = len(labels)
    if label_count == 0:
        return []

    scores: list[list[float]] = [
        [
            log_probs[0][label_index] if _valid_start(labels[label_index]) else IMPOSSIBLE
            for label_index in range(label_count)
        ]
    ]
    backpointers: list[list[int]] = [[0] * label_count]

    for token_index in range(1, len(log_probs)):
        previous = scores[-1]
        current: list[float] = []
        current_backpointers: list[int] = []
        for label_index, label in enumerate(labels):
            best_score = IMPOSSIBLE
            best_previous = 0
            for previous_index, previous_label in enumerate(labels):
                if not _valid_transition(previous_label, label):
                    continue
                score = previous[previous_index] + log_probs[token_index][label_index]
                if score > best_score:
                    best_score = score
                    best_previous = previous_index
            current.append(best_score)
            current_backpointers.append(best_previous)
        scores.append(current)
        backpointers.append(current_backpointers)

    final_scores = [
        score if _valid_end(labels[index]) else IMPOSSIBLE
        for index, score in enumerate(scores[-1])
    ]
    last = max(range(label_count), key=lambda index: final_scores[index])
    path = [last]
    for token_index in range(len(log_probs) - 1, 0, -1):
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
