from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from fast_agent.llm.llamacpp_discovery import LlamaCppModelListing
from fast_agent.ui.llamacpp_model_picker import LlamaCppModelPickerContext, _LlamaCppModelPicker


def test_llamacpp_picker_enter_on_actions_returns_selected_action_and_model() -> None:
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
    picker.state.model_index = 1
    picker.state.action_index = 1
    picker.state.focus = "actions"

    enter_binding = next(
        binding
        for binding in picker._create_key_bindings().bindings
        if getattr(binding.handler, "__name__", "") == "_accept"
    )

    app = _FakeApp()
    cast("Any", enter_binding.handler)(_FakeEvent(app))

    assert app.result is not None
    assert app.result.action == "start_now_with_shell"
    assert app.result.model_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_llamacpp_picker_enter_on_models_switches_focus_to_actions() -> None:
    class _FakeApp:
        def exit(self, *, result) -> None:
            del result
            raise AssertionError("Enter on the model list should not exit immediately")

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
        )
    )
    picker.state.focus = "models"

    enter_binding = next(
        binding
        for binding in picker._create_key_bindings().bindings
        if getattr(binding.handler, "__name__", "") == "_accept"
    )

    cast("Any", enter_binding.handler)(_FakeEvent(_FakeApp()))

    assert picker.state.focus == "actions"


def test_llamacpp_picker_details_include_start_now_and_generate_overlay_hints() -> None:
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

    assert "selected action: Start now" in rendered
    assert "context: training: 262144 / runtime: not loaded" in rendered
    assert "Import writes a reusable overlay for this model." in rendered
    assert "Enter on models = choose action" in rendered


def test_llamacpp_picker_includes_start_now_smart_action() -> None:
    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
        )
    )

    rendered_actions = "".join(fragment for _, fragment in picker._render_actions())

    assert "Start now (Smart)" in rendered_actions


def test_llamacpp_picker_hides_model_cursor_when_actions_are_focused() -> None:
    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
        )
    )
    picker.state.focus = "actions"

    rendered_models = "".join(fragment for _, fragment in picker._render_models())
    rendered_actions = "".join(fragment for _, fragment in picker._render_actions())

    assert "❯" not in rendered_models
    assert "❯ Start now" in rendered_actions


def test_llamacpp_picker_formats_model_rows_as_name_only() -> None:
    model = LlamaCppModelListing(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        owned_by="llamacpp",
        training_context_window=131072,
    )

    row = _LlamaCppModelPicker._model_row_label(model, width=60)

    assert len(row) <= 60
    assert "meta-llama/Llama-3.2-3B-Instruct" in row
    assert "llamacpp" not in row


@pytest.mark.asyncio
async def test_llamacpp_picker_lazy_loads_runtime_context() -> None:
    async def _load_runtime_context(model_id: str) -> int | None:
        assert model_id == "unsloth/Qwen3.5-9B-GGUF"
        return 75264

    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=262144,
            ),
        ),
        runtime_context_loader=_load_runtime_context,
    )

    picker._ensure_runtime_context_loading()
    await asyncio.sleep(0)

    rendered = "".join(fragment for _, fragment in picker._render_details())

    assert "context: training: 262144 / runtime: 75264" in rendered


@pytest.mark.asyncio
async def test_llamacpp_picker_lazy_load_can_upgrade_training_context() -> None:
    async def _load_runtime_context(model_id: str) -> LlamaCppModelPickerContext:
        assert model_id == "unsloth/Qwen3.5-9B-GGUF"
        return LlamaCppModelPickerContext(
            runtime_context_window=75264,
            training_context_window=262144,
        )

    picker = _LlamaCppModelPicker(
        (
            LlamaCppModelListing(
                model_id="unsloth/Qwen3.5-9B-GGUF",
                owned_by="llamacpp",
                training_context_window=None,
            ),
        ),
        runtime_context_loader=_load_runtime_context,
    )

    picker._ensure_runtime_context_loading()
    await asyncio.sleep(0)

    rendered = "".join(fragment for _, fragment in picker._render_details())

    assert "context: training: 262144 / runtime: 75264" in rendered
