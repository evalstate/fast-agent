from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _ensure_hf_inference_acp_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "publish" / "hf-inference-acp" / "src"
    sys.path.insert(0, str(package_root))


@pytest.mark.asyncio
async def test_wizard_can_restart_after_complete_with_go(monkeypatch: pytest.MonkeyPatch) -> None:
    _ensure_hf_inference_acp_on_path()

    # Avoid hitting real auth state, importing huggingface_hub, or making network calls.
    import hf_inference_acp.wizard.wizard_llm as wizard_llm
    from hf_inference_acp.wizard.stages import WizardStage
    from hf_inference_acp.wizard.wizard_llm import WizardSetupLLM

    monkeypatch.setattr(wizard_llm, "has_hf_token", lambda: False)

    llm = WizardSetupLLM()
    llm._state.first_message = False
    llm._state.stage = WizardStage.COMPLETE

    response = await llm._process_stage("go")

    assert llm._state.stage == WizardStage.TOKEN_GUIDE
    assert "Step 1" in response
    assert "Setup Complete" not in response


@pytest.mark.asyncio
async def test_wizard_completion_callback_fires_only_once() -> None:
    _ensure_hf_inference_acp_on_path()

    from hf_inference_acp.wizard.stages import WizardStage
    from hf_inference_acp.wizard.wizard_llm import WizardSetupLLM

    llm = WizardSetupLLM()
    llm._state.first_message = False
    llm._state.stage = WizardStage.COMPLETE

    callback_calls: list[WizardStage] = []

    async def _callback(state) -> None:  # type: ignore[no-untyped-def]
        callback_calls.append(state.stage)

    llm.set_completion_callback(_callback)

    first = await llm._handle_complete("")
    second = await llm._handle_complete("")

    assert "Setup Complete" in first
    assert "already complete" in second
    assert len(callback_calls) == 1
