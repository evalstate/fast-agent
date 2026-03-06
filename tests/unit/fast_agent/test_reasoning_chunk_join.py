from fast_agent.utils.reasoning_chunk_join import normalize_reasoning_delta


def test_normalize_reasoning_delta_inserts_space_after_sentence_break() -> None:
    last_char = None
    emitted = ""
    parts = [
        "approach.",
        "Specifying session retrieval format",
        "Selecting session retrieval method",
    ]

    for part in parts:
        delta = normalize_reasoning_delta(last_char, part)
        emitted += delta
        last_char = emitted[-1] if emitted else None

    assert emitted == "approach. Specifying session retrieval format Selecting session retrieval method"


def test_normalize_reasoning_delta_preserves_contractions() -> None:
    last_char = None
    emitted = ""
    for part in ["don", "'t do that"]:
        delta = normalize_reasoning_delta(last_char, part)
        emitted += delta
        last_char = emitted[-1] if emitted else None

    assert emitted == "don't do that"


def test_normalize_reasoning_delta_preserves_identifier_fragments() -> None:
    last_char = None
    emitted = ""
    for part in ["session", "_id is required"]:
        delta = normalize_reasoning_delta(last_char, part)
        emitted += delta
        last_char = emitted[-1] if emitted else None

    assert emitted == "session_id is required"
