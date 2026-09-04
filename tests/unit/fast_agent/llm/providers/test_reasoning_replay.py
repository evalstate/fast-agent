from types import SimpleNamespace

from fast_agent.llm.provider.openai.reasoning_replay import (
    REASONING_REPLAY_SCHEMA,
    REASONING_REPLAY_VERSION,
    capture_reasoning_replay,
    parse_reasoning_replay,
)


def test_capture_reasoning_replay_allowlists_provider_input_fields() -> None:
    envelope = capture_reasoning_replay(
        SimpleNamespace(
            type="reasoning",
            id="rs_1",
            summary=[
                SimpleNamespace(type="summary_text", text="First."),
                SimpleNamespace(type="summary_text", text="Second."),
            ],
            content=[],
            encrypted_content="cipher",
            status="completed",
            future_sdk_field="must-not-cross-boundary",
        )
    )

    assert envelope == {
        "schema": REASONING_REPLAY_SCHEMA,
        "version": REASONING_REPLAY_VERSION,
        "item": {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [
                {"type": "summary_text", "text": "First."},
                {"type": "summary_text", "text": "Second."},
            ],
            "encrypted_content": "cipher",
        },
    }


def test_capture_reasoning_replay_preserves_valid_nonempty_content() -> None:
    envelope = capture_reasoning_replay(
        SimpleNamespace(
            type="reasoning",
            id="rs_1",
            summary=[],
            content=[SimpleNamespace(type="reasoning_text", text="Visible reasoning.")],
            encrypted_content="cipher",
            status="completed",
        )
    )

    assert envelope is not None
    assert envelope["item"]["content"] == [{"type": "reasoning_text", "text": "Visible reasoning."}]


def test_parse_reasoning_replay_accepts_only_canonical_envelope() -> None:
    parsed = parse_reasoning_replay(
        {
            "schema": REASONING_REPLAY_SCHEMA,
            "version": REASONING_REPLAY_VERSION,
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Summary."}],
                "encrypted_content": "cipher",
            },
        }
    )

    assert parsed == {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "Summary."}],
        "encrypted_content": "cipher",
    }


def test_parse_reasoning_replay_rejects_noncanonical_records() -> None:
    for payload in (
        {
            "type": "reasoning",
            "id": "rs_legacy",
            "encrypted_content": "legacy-cipher",
        },
        {
            "type": "reasoning",
            "id": "rs_snapshot",
            "summary": [{"type": "summary_text", "text": "Item summary."}],
            "content": None,
            "encrypted_content": "snapshot-cipher",
            "status": "completed",
            "future_sdk_field": "must-not-cross-boundary",
        },
    ):
        assert parse_reasoning_replay(payload) is None


def test_parse_reasoning_replay_rejects_unknown_canonical_version() -> None:
    assert (
        parse_reasoning_replay(
            {
                "schema": REASONING_REPLAY_SCHEMA,
                "version": 2,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "cipher",
                },
            }
        )
        is None
    )


def test_parse_reasoning_replay_rejects_unknown_fields() -> None:
    assert (
        parse_reasoning_replay(
            {
                "schema": REASONING_REPLAY_SCHEMA,
                "version": REASONING_REPLAY_VERSION,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "cipher",
                    "future_sdk_field": "unsupported",
                },
            }
        )
        is None
    )


def test_reasoning_replay_requires_id_summary_and_ciphertext() -> None:
    assert (
        capture_reasoning_replay(
            SimpleNamespace(
                type="reasoning",
                id="",
                summary=[],
                content=[],
                encrypted_content="cipher",
                status="completed",
            )
        )
        is None
    )
    assert (
        capture_reasoning_replay(
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                summary=None,
                content=[],
                encrypted_content="cipher",
                status="completed",
            )
        )
        is None
    )
    assert (
        capture_reasoning_replay(
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                summary=[],
                content=[],
                encrypted_content="",
                status="completed",
            )
        )
        is None
    )


def test_capture_reasoning_replay_requires_completed_output() -> None:
    assert (
        capture_reasoning_replay(
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                summary=[],
                content=[],
                encrypted_content="cipher",
                status="in_progress",
            )
        )
        is None
    )


def test_reasoning_replay_rejects_malformed_nonempty_content() -> None:
    assert (
        capture_reasoning_replay(
            SimpleNamespace(
                type="reasoning",
                id="rs_1",
                summary=[],
                content=[SimpleNamespace(type="future_reasoning_type", text="unsupported")],
                encrypted_content="cipher",
                status="completed",
            )
        )
        is None
    )
