"""Unit tests for the internal resource validation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_check_internal_resources_module() -> "ModuleType":
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "check_internal_resources.py"
    spec = importlib.util.spec_from_file_location("check_internal_resources_script", script_path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_check_module = _load_check_internal_resources_module()
validate_internal_resources = _check_module.validate_internal_resources


def test_validate_internal_resources_returns_known_uri() -> None:
    uris = validate_internal_resources()

    assert "internal://fast-agent/smart-agent-cards" in uris
