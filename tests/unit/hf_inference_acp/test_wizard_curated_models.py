from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _ensure_hf_inference_acp_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "publish" / "hf-inference-acp" / "src"
    sys.path.insert(0, str(package_root))


@pytest.mark.asyncio
async def test_wizard_model_selection_uses_curated_ids() -> None:
    pytest.importorskip("ruamel.yaml")
    _ensure_hf_inference_acp_on_path()

    from hf_inference_acp.wizard.model_catalog import CURATED_MODELS
    from hf_inference_acp.wizard.stages import WizardStage
    from hf_inference_acp.wizard.wizard_llm import WizardSetupLLM

    llm = WizardSetupLLM()
    llm._state.first_message = False  # skip welcome
    llm._state.stage = WizardStage.MODEL_SELECT

    # Pick the first curated model by number.
    response = await llm._handle_model_select("1")
    assert llm._state.selected_model == CURATED_MODELS[0].id
    assert llm._state.stage == WizardStage.MCP_CONNECT
    assert "Step 3" in response

