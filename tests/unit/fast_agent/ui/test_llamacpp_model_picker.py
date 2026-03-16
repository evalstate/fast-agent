from __future__ import annotations

from typing import Any, cast

from fast_agent.llm.llamacpp_discovery import LlamaCppModelListing
from fast_agent.ui.llamacpp_model_picker import _LlamaCppModelPicker


def test_llamacpp_picker_enter_returns_generate_overlay_action() -> None:
    class _FakeApp:
        def __init__(self) -> None:
            self.result = None

        def exit(self, *, result) -> None:
            self.result = result

    class _FakeEvent:
        def __init__(self, app: _FakeApp) -> None:
            self.app = app

    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
            LlamaCppModelListing(
                model_id="meta-llama/Llama-3.2-3B-Instruct",
                owned_by="llamacpp",
                training_context_window=131072,
            ),
        )
    )
    picker.state.index = 1

    enter_binding = next(
        binding
        for binding in picker._create_key_bindings().bindings
        if getattr(binding.handler, "__name__", "") == "_accept"
    )

    app = _FakeApp()
    cast("Any", enter_binding.handler)(_FakeEvent(app))

    assert app.result is not None
    assert app.result.action == "generate_overlay"
    assert app.result.model_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_llamacpp_picker_details_include_generate_overlay_hint() -> None:
    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
        )
    )

    rendered = "".join(fragment for _, fragment in picker._render_details())

    assert "Enter = generate overlay" in rendered
    assert "training context: 262144" in rendered
