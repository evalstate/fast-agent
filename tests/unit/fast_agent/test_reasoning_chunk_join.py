from fast_agent.utils.reasoning_chunk_join import normalize_reasoning_delta


def test_normalize_reasoning_delta_inserts_space_after_sentence_break() -> None:
    emitted = ""
    parts = [
        "approach.",
        "Specifying session retrieval format",
        "Selecting session retrieval method",
    ]

    for part in parts:
        delta = normalize_reasoning_delta(emitted, part)
        emitted += delta

    assert emitted == "approach. Specifying session retrieval format Selecting session retrieval method"


def test_normalize_reasoning_delta_preserves_tool_style_names() -> None:
    emitted = ""
    for part in ["voice.", "fetch", "(id=session_id, mode='transcript')"]:
        delta = normalize_reasoning_delta(emitted, part)
        emitted += delta

    assert emitted == "voice.fetch(id=session_id, mode='transcript')"
