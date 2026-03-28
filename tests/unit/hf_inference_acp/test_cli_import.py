from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_types import Provider


def _ensure_hf_inference_acp_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "publish" / "hf-inference-acp" / "src"
    sys.path.insert(0, str(package_root))


def test_cli_import_registers_wizard_setup_model() -> None:
    pytest.importorskip("ruamel.yaml")
    _ensure_hf_inference_acp_on_path()

    ModelDatabase.unregister_runtime_model_params("wizard-setup")
    ModelFactory.MODEL_SPECIFIC_CLASSES.pop("wizard-setup", None)
    sys.modules.pop("hf_inference_acp.cli", None)

    cli = importlib.import_module("hf_inference_acp.cli")

    assert cli is not None
    assert ModelDatabase.get_default_provider("wizard-setup") == Provider.FAST_AGENT
    assert ModelFactory.MODEL_SPECIFIC_CLASSES["wizard-setup"].__name__ == "WizardSetupLLM"
