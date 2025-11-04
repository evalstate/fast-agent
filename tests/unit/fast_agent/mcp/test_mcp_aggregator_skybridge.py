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
SkybridgeServerConfig = _module.SkybridgeServerConfig
SKYBRIDGE_MIME_TYPE = _module.SKYBRIDGE_MIME_TYPE
NamespacedTool = _module.NamespacedTool


def _create_aggregator() -> MCPAggregator:
    """Create an aggregator instance suitable for unit testing."""
    aggregator = MCPAggregator(
        server_names=["test"],
        connection_persistence=False,
        context=None,
        name="test-agent",
    )
    # Bypass the normal display setup for unit tests
    aggregator.display = None  # type: ignore[attr-defined]
    return aggregator


def test_skybridge_detection_marks_valid_resources() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=Tool(
                name="tool_a",
                inputSchema={"type": "object"},
                _meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=SimpleNamespace(contents=[SimpleNamespace(mimeType=SKYBRIDGE_MIME_TYPE)])
    )

    server_name, config = asyncio.run(aggregator._evaluate_skybridge_for_server("test"))

    assert server_name == "test"
    assert isinstance(config, SkybridgeServerConfig)
    assert config.enabled is True
    assert len(config.ui_resources) == 1
    assert config.ui_resources[0].is_skybridge is True
    assert config.ui_resources[0].warning is None
    assert not config.warnings
    assert len(config.tools) == 1
    tool_cfg = config.tools[0]
    assert tool_cfg.is_valid is True
    assert tool_cfg.template_uri is not None
    assert tool_cfg.resource_uri == config.ui_resources[0].uri
    aggregator._list_resources_from_server.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test", check_support=False
    )
    aggregator._get_resource_from_server.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test", "ui://component/app"
    )


def test_skybridge_detection_warns_on_invalid_mime() -> None:
    aggregator = _create_aggregator()
    aggregator.server_supports_feature = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=Tool(
                name="tool_a",
                inputSchema={"type": "object"},
                _meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=SimpleNamespace(contents=[SimpleNamespace(mimeType="text/html")])
    )

    _, config = asyncio.run(aggregator._evaluate_skybridge_for_server("test"))

    assert config.enabled is False
    assert len(config.ui_resources) == 1
    assert (
        config.ui_resources[0].warning
        == "served as 'text/html' instead of 'text/html+skybridge'"
    )
    assert config.warnings
    assert config.warnings[0] == (
        "ui://component/app: served as 'text/html' instead of 'text/html+skybridge'"
    )
    assert len(config.tools) == 1
    tool_cfg = config.tools[0]
    assert tool_cfg.is_valid is False
    assert (
        tool_cfg.warning
        == "Tool 'test.tool_a' references resource 'ui://component/app' served as 'text/html' "
        "instead of 'text/html+skybridge'"
    )
    aggregator._list_resources_from_server.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test", check_support=False
    )
    aggregator._get_resource_from_server.assert_awaited_once_with(  # type: ignore[attr-defined]
        "test", "ui://component/app"
    )


def test_skybridge_detection_handles_missing_resources_capability() -> None:
    aggregator = _create_aggregator()
    aggregator.server_supports_feature = AsyncMock(return_value=False)  # type: ignore[attr-defined]
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=Tool(
                name="tool_a",
                inputSchema={"type": "object"},
                _meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock()  # type: ignore[attr-defined]
    aggregator._get_resource_from_server = AsyncMock()  # type: ignore[attr-defined]

    _, config = asyncio.run(aggregator._evaluate_skybridge_for_server("test"))

    assert config.supports_resources is False
    assert config.enabled is False
    aggregator._list_resources_from_server.assert_not_called()  # type: ignore[attr-defined]
    aggregator._get_resource_from_server.assert_not_called()  # type: ignore[attr-defined]
    assert len(config.tools) == 1


def test_list_tools_marks_skybridge_meta() -> None:
    aggregator = _create_aggregator()
    aggregator.initialized = True

    tool = Tool(
        name="tool_a",
        inputSchema={"type": "object"},
        _meta={"openai/outputTemplate": "ui://component/app"},
    )

    namespaced = NamespacedTool(
        tool=tool,
        server_name="test",
        namespaced_tool_name="test.tool_a",
    )

    aggregator._namespaced_tool_map = {"test.tool_a": namespaced}  # type: ignore[attr-defined]
    aggregator._server_to_tool_map["test"] = [namespaced]

    aggregator._skybridge_configs["test"] = SkybridgeServerConfig(  # type: ignore[attr-defined]
        server_name="test",
        supports_resources=True,
        ui_resources=[
            _module.SkybridgeResourceConfig(
                uri=_module.AnyUrl("ui://component/app"),
                mime_type=SKYBRIDGE_MIME_TYPE,
                is_skybridge=True,
            )
        ],
        tools=[
            _module.SkybridgeToolConfig(
                tool_name="tool_a",
                namespaced_tool_name="test.tool_a",
                template_uri=_module.AnyUrl("ui://component/app"),
                resource_uri=_module.AnyUrl("ui://component/app"),
                is_valid=True,
            )
        ],
    )

    tools_result = asyncio.run(aggregator.list_tools())
    assert len(tools_result.tools) == 1
    meta = tools_result.tools[0].meta or {}
    assert meta.get("openai/skybridgeEnabled") is True
    assert meta.get("openai/skybridgeTemplate") == "ui://component/app"


def test_skybridge_resource_without_tool_warns() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)  # type: ignore[attr-defined]
    aggregator._server_to_tool_map["test"] = []  # type: ignore[attr-defined]
    aggregator._list_resources_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(  # type: ignore[attr-defined]
        return_value=SimpleNamespace(contents=[SimpleNamespace(mimeType=SKYBRIDGE_MIME_TYPE)])
    )

    _, config = asyncio.run(aggregator._evaluate_skybridge_for_server("test"))

    assert config.enabled is True
    assert not config.tools
    assert any(
        "no tools expose them" in warning.lower() for warning in config.warnings
    )
