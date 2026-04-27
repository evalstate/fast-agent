"""Parity / invariant tests for the BIOES Viterbi decoders."""

from __future__ import annotations

import numpy as np
import pytest

from fast_agent.privacy.viterbi import (
    build_viterbi_tables,
    constrained_viterbi,
    constrained_viterbi_np,
    token_spans_from_path,
)

# Mirrors the privacy-filter label schema (subset is fine — the constraint
# logic is the same shape).
_LABELS = [
    "O",
    "B-private_email",
    "I-private_email",
    "E-private_email",
    "S-private_email",
    "B-private_person",
    "I-private_person",
    "E-private_person",
    "S-private_person",
]


def _random_logits(timesteps: int, label_count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((timesteps, label_count)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_numpy_viterbi_matches_pure_python_reference(seed: int) -> None:
    logits = _random_logits(timesteps=24, label_count=len(_LABELS), seed=seed)
    tables = build_viterbi_tables(_LABELS, np)

    reference = constrained_viterbi(logits.tolist(), _LABELS)
    actual = constrained_viterbi_np(logits, tables, np)

    assert actual == reference


def test_numpy_viterbi_respects_bioes_constraints() -> None:
    # Heavily bias every timestep toward I-private_email; constraints must
    # still produce a valid B/I*/E or O sequence rather than a bare I run.
    timesteps = 6
    label_count = len(_LABELS)
    logits = np.full((timesteps, label_count), -5.0, dtype=np.float32)
    logits[:, _LABELS.index("I-private_email")] = 10.0

    tables = build_viterbi_tables(_LABELS, np)
    path = constrained_viterbi_np(logits, tables, np)

    decoded = [_LABELS[i] for i in path]
    # First label must be a valid start (O / B-* / S-*); last must be O / E-* / S-*.
    assert decoded[0].split("-", 1)[0] in {"O", "B", "S"}
    assert decoded[-1].split("-", 1)[0] in {"O", "E", "S"}
    # No bare-I runs: every I-X must be preceded by B-X or another I-X of the same kind.
    for index in range(1, len(decoded)):
        prefix, _, kind = decoded[index].partition("-")
        if prefix == "I":
            previous_prefix, _, previous_kind = decoded[index - 1].partition("-")
            assert previous_prefix in {"B", "I"}
            assert previous_kind == kind


def test_token_spans_from_path_yields_expected_spans() -> None:
    path = [
        _LABELS.index("O"),
        _LABELS.index("B-private_email"),
        _LABELS.index("I-private_email"),
        _LABELS.index("E-private_email"),
        _LABELS.index("O"),
        _LABELS.index("S-private_person"),
        _LABELS.index("O"),
    ]

    spans = token_spans_from_path(path, _LABELS)

    assert [(span.label, span.start, span.end) for span in spans] == [
        ("private_email", 1, 4),
        ("private_person", 5, 6),
    ]
