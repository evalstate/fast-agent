from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


class _PluginInterfaceModule(types.ModuleType):
    BuildHookInterface: type[object]


def _load_hatch_build_module() -> "ModuleType":
    module_path = Path(__file__).resolve().parents[2] / "hatch_build.py"
    spec = importlib.util.spec_from_file_location("hatch_build_module", module_path)
    assert spec is not None
    loader = spec.loader
    assert loader is not None

    plugin_interface = _PluginInterfaceModule("hatchling.builders.hooks.plugin.interface")

    class BuildHookInterface:
        pass

    plugin_interface.BuildHookInterface = BuildHookInterface
    sys.modules.setdefault("hatchling", types.ModuleType("hatchling"))
    sys.modules.setdefault("hatchling.builders", types.ModuleType("hatchling.builders"))
    sys.modules.setdefault("hatchling.builders.hooks", types.ModuleType("hatchling.builders.hooks"))
    sys.modules.setdefault(
        "hatchling.builders.hooks.plugin",
        types.ModuleType("hatchling.builders.hooks.plugin"),
    )
    sys.modules["hatchling.builders.hooks.plugin.interface"] = plugin_interface

    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


_hatch_build = _load_hatch_build_module()
_example_mappings = _hatch_build._example_mappings
_generated_artifacts_ignore = _hatch_build.GENERATED_ARTIFACTS_IGNORE


def test_example_mappings_include_markdown_assets() -> None:
    mappings = _example_mappings()

    assert mappings["examples/markdown"] == (
        Path("src") / "fast_agent" / "resources" / "examples" / "markdown"
    )


def test_generated_artifacts_ignore_excludes_runtime_homes() -> None:
    ignored = _generated_artifacts_ignore(
        "examples/workflows",
        [".fast-agent", "agent.py", "fastagent.jsonl", "stream-debug"],
    )

    assert ignored == {".fast-agent", "fastagent.jsonl", "stream-debug"}
