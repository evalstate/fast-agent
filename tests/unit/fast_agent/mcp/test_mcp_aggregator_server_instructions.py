import asyncio
import enum
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp.types import Tool

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = PROJECT_ROOT / "src" / "fast_agent" / "mcp" / "mcp_aggregator.py"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Backport StrEnum for Python < 3.11 environments
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass

    enum.StrEnum = StrEnum  # type: ignore[attr-defined]

# Provide minimal stubs for optional dependencies referenced during import
if "a2a" not in sys.modules:
    a2a_module = types.ModuleType("a2a")
    a2a_types_module = types.ModuleType("a2a.types")
    setattr(a2a_types_module, "AgentCard", object)
    setattr(a2a_types_module, "AgentSkill", object)
    a2a_module.types = a2a_types_module  # type: ignore[attr-defined]
    sys.modules["a2a"] = a2a_module
    sys.modules["a2a.types"] = a2a_types_module

spec = importlib.util.spec_from_file_location("fast_agent.mcp.mcp_aggregator", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Failed to load mcp_aggregator module for testing")
_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_module)

MCPAggregator = _module.MCPAggregator
NamespacedTool = _module.NamespacedTool


def test_get_server_instructions_does_not_implicitly_connect() -> None:
    aggregator = MCPAggregator(
        server_names=["huggingface", "optional"],
        connection_persistence=True,
        context=None,
        name="test-agent",
    )
    aggregator.display = None  # type: ignore[attr-defined]

    aggregator._namespaced_tool_map = {  # type: ignore[attr-defined]
        "huggingface.tool_a": NamespacedTool(
            tool=Tool(name="tool_a", inputSchema={"type": "object"}),
            server_name="huggingface",
            namespaced_tool_name="huggingface.tool_a",
        )
    }

    huggingface_conn = SimpleNamespace(
        server_instructions="hf instructions",
        is_healthy=lambda: True,
    )

    fake_manager = SimpleNamespace(
        running_servers={"huggingface": huggingface_conn},
        # If get_server() is called, it means we're implicitly connecting, which this test forbids.
        get_server=AsyncMock(side_effect=AssertionError("get_server() should not be called")),
    )
    aggregator._persistent_connection_manager = fake_manager  # type: ignore[attr-defined]

    result = asyncio.run(aggregator.get_server_instructions())
    assert result == {"huggingface": ("hf instructions", ["tool_a"])}
