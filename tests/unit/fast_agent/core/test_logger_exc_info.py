from fast_agent.core.logging.logger import Logger


def test_coerce_exc_info_removes_falsey_exc_info() -> None:
    assert Logger._coerce_exc_info({"exc_info": None, "message": "kept"}) == {"message": "kept"}


def test_coerce_exc_info_formats_exception_object() -> None:
    error = RuntimeError("boom")

    payload = Logger._coerce_exc_info({"exc_info": error})

    assert payload["error"] == "boom"
    assert payload["error_type"] == "RuntimeError"
    assert "RuntimeError: boom" in payload["exception"]


def test_coerce_exc_info_preserves_existing_error_fields() -> None:
    error = ValueError("actual")

    payload = Logger._coerce_exc_info(
        {
            "exc_info": (ValueError, error, error.__traceback__),
            "error": "custom",
            "error_type": "CustomError",
        }
    )

    assert payload["error"] == "custom"
    assert payload["error_type"] == "CustomError"
    assert "ValueError: actual" in payload["exception"]


def test_coerce_exc_info_falls_back_for_invalid_tuple() -> None:
    payload = Logger._coerce_exc_info({"exc_info": ("not-an-exception", "boom", None)})

    assert payload == {"exception": "('not-an-exception', 'boom', None)"}
