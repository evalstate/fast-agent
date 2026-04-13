from fast_agent.utils.reasoning_chunk_join import (
    ReasoningTextAccumulator,
    join_reasoning_segments,
    normalize_reasoning_delta,
)


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


def test_normalize_reasoning_delta_promotes_markdown_heading_to_new_paragraph() -> None:
    last_char = None
    emitted = ""
    parts = [
        "avoid extending that syntax.",
        "**Structuring config for clarity**\n\nI'm focusing on using structured config.",
    ]

    for part in parts:
        delta = normalize_reasoning_delta(last_char, part)
        emitted += delta
        last_char = emitted[-1] if emitted else None

    assert emitted == (
        "avoid extending that syntax.\n\n"
        "**Structuring config for clarity**\n\n"
        "I'm focusing on using structured config."
    )


def test_normalize_reasoning_delta_keeps_inline_bold_sentence_spacing() -> None:
    last_char = None
    emitted = ""
    parts = [
        "The answer is",
        "**probably yes** for now",
    ]

    for part in parts:
        delta = normalize_reasoning_delta(last_char, part)
        emitted += delta
        last_char = emitted[-1] if emitted else None

    assert emitted == "The answer is **probably yes** for now"


def test_reasoning_text_accumulator_normalizes_streamed_reasoning() -> None:
    accumulator = ReasoningTextAccumulator(normalizer=normalize_reasoning_delta)

    for part in [
        "avoid extending that syntax.",
        "**Structuring config for clarity**\n\nI'm focusing on using structured config.",
    ]:
        accumulator.append(part)

    assert accumulator.text() == (
        "avoid extending that syntax.\n\n"
        "**Structuring config for clarity**\n\n"
        "I'm focusing on using structured config."
    )
    assert accumulator.parts() == [
        "avoid extending that syntax.",
        "\n\n**Structuring config for clarity**\n\nI'm focusing on using structured config.",
    ]


def test_reasoning_text_accumulator_defaults_to_identity_join() -> None:
    accumulator = ReasoningTextAccumulator()
    accumulator.append("thinking")
    accumulator.append(" harder")

    assert accumulator.text() == "thinking harder"
    assert accumulator.parts() == ["thinking", " harder"]


def test_join_reasoning_segments_preserves_heading_paragraph_breaks() -> None:
    assert join_reasoning_segments(
        [
            (
                "**Deciding on naming and implementation**\n\n"
                "I think I should prepare an implementation checklist, "
                "needing just one or two from them."
            ),
            (
                "**Identifying key decisions**\n\n"
                "I think I should emphasize that there are really only "
                "three decisions to make."
            ),
        ]
    ) == (
        "**Deciding on naming and implementation**\n\n"
        "I think I should prepare an implementation checklist, "
        "needing just one or two from them.\n\n"
        "**Identifying key decisions**\n\n"
        "I think I should emphasize that there are really only "
        "three decisions to make."
    )
