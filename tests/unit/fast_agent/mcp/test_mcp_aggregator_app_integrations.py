import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from mcp_types import Tool

from fast_agent.mcp.app_integrations import (
    AppIntegrationKind,
    AppResourceConfig,
    AppServerConfig,
    AppToolConfig,
    extract_app_tool_metadata,
)
from fast_agent.mcp.app_integrations.mcp_apps import MCP_APPS_MIME_TYPE
from fast_agent.mcp.app_integrations.openai_apps_sdk import OPENAI_APPS_SDK_MIME_TYPE
from fast_agent.mcp.mcp_aggregator import MCPAggregator, NamespacedTool
from fast_agent.ui.console_display import ConsoleDisplay


def _tool_with_meta(name: str, input_schema: dict[str, Any], meta: dict[str, Any]) -> Tool:
    return Tool.model_validate(
        {
            "name": name,
            "inputSchema": input_schema,
            "_meta": meta,
        }
    )


def _create_aggregator() -> MCPAggregator:
    """Create an aggregator instance suitable for unit testing."""
    aggregator = MCPAggregator(
        server_names=["test"],
        connection_persistence=False,
        context=None,
        name="test-agent",
    )
    return aggregator


def test_mcp_apps_nested_metadata_takes_precedence_over_flat_and_openai_metadata() -> None:
    metadata = extract_app_tool_metadata(
        {
            "ui": {"resourceUri": "ui://component/nested"},
            "ui/resourceUri": "ui://component/flat",
            "openai/outputTemplate": "ui://component/openai",
        },
        namespaced_tool_name="test.tool_a",
    )

    assert metadata is not None
    assert metadata.kind is AppIntegrationKind.MCP_APPS
    assert str(metadata.resource_uri) == "ui://component/nested"


def test_mcp_apps_invalid_visibility_falls_back_with_a_warning() -> None:
    metadata = extract_app_tool_metadata(
        {
            "ui": {
                "resourceUri": "ui://component/app",
                "visibility": ["app", "unsupported"],
            }
        },
        namespaced_tool_name="test.tool_a",
    )

    assert metadata is not None
    assert metadata.visibility == ["app"]
    assert metadata.warnings == ["invalid _meta.ui.visibility values ignored: unsupported"]


def test_openai_apps_sdk_detection_marks_valid_resources() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(
            contents=[SimpleNamespace(mime_type=OPENAI_APPS_SDK_MIME_TYPE)]
        )
    )

    server_name, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert server_name == "test"
    assert isinstance(config, AppServerConfig)
    assert config.enabled is True
    assert len(config.resources) == 1
    assert config.resources[0].kind is AppIntegrationKind.OPENAI_APPS_SDK
    assert config.resources[0].warning is None
    assert not config.warnings
    assert len(config.tools) == 1
    tool_cfg = config.tools[0]
    assert tool_cfg.is_valid is True
    assert tool_cfg.resource_uri is not None
    assert tool_cfg.linked_resource_uri == config.resources[0].uri
    aggregator._list_resources_from_server.assert_awaited_once_with("test", check_support=False)
    aggregator._get_resource_from_server.assert_awaited_once_with("test", "ui://component/app")


def test_mcp_apps_detection_marks_valid_resources() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"ui": {"resourceUri": "ui://component/app"}},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(contents=[SimpleNamespace(mime_type=MCP_APPS_MIME_TYPE)])
    )

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.enabled is True
    assert len(config.resources) == 1
    assert config.resources[0].kind is AppIntegrationKind.MCP_APPS
    assert len(config.tools) == 1
    tool_cfg = config.tools[0]
    assert tool_cfg.is_valid is True
    assert tool_cfg.kind is AppIntegrationKind.MCP_APPS
    assert tool_cfg.visibility == ["model", "app"]


def test_mcp_apps_detection_supports_flat_resource_uri() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"ui/resourceUri": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(contents=[SimpleNamespace(mime_type=MCP_APPS_MIME_TYPE)])
    )

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.tools[0].is_valid is True
    assert config.tools[0].kind is AppIntegrationKind.MCP_APPS


def test_mcp_apps_detection_warns_on_openai_apps_sdk_mime() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"ui": {"resourceUri": "ui://component/app"}},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(
            contents=[SimpleNamespace(mime_type=OPENAI_APPS_SDK_MIME_TYPE)]
        )
    )

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.enabled is True
    assert config.resources[0].kind is AppIntegrationKind.OPENAI_APPS_SDK
    assert config.tools[0].is_valid is False
    assert "instead of 'text/html;profile=mcp-app'" in (config.tools[0].warning or "")


def test_openai_apps_sdk_detection_warns_on_invalid_mime() -> None:
    aggregator = _create_aggregator()
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(contents=[SimpleNamespace(mime_type="text/html")])
    )

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.enabled is False
    assert len(config.resources) == 1
    assert config.resources[0].warning == "served as 'text/html' instead of 'text/html+skybridge'"
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
    aggregator._list_resources_from_server.assert_awaited_once_with("test", check_support=False)
    aggregator._get_resource_from_server.assert_awaited_once_with("test", "ui://component/app")


def test_app_integration_detection_handles_missing_resources_capability() -> None:
    aggregator = _create_aggregator()
    aggregator.server_supports_feature = AsyncMock(return_value=False)
    aggregator._server_to_tool_map["test"] = [
        NamespacedTool(
            tool=_tool_with_meta(
                name="tool_a",
                input_schema={"type": "object"},
                meta={"openai/outputTemplate": "ui://component/app"},
            ),
            server_name="test",
            namespaced_tool_name="test.tool_a",
        )
    ]
    aggregator._list_resources_from_server = AsyncMock()
    aggregator._get_resource_from_server = AsyncMock()

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.supports_resources is False
    assert config.enabled is False
    aggregator._list_resources_from_server.assert_not_called()
    aggregator._get_resource_from_server.assert_not_called()
    assert len(config.tools) == 1


def test_list_tools_marks_openai_apps_sdk_meta() -> None:
    aggregator = _create_aggregator()
    aggregator.initialized = True

    tool = _tool_with_meta(
        name="tool_a",
        input_schema={"type": "object"},
        meta={"openai/outputTemplate": "ui://component/app"},
    )

    namespaced = NamespacedTool(
        tool=tool,
        server_name="test",
        namespaced_tool_name="test.tool_a",
    )

    aggregator._namespaced_tool_map = {"test.tool_a": namespaced}
    aggregator._server_to_tool_map["test"] = [namespaced]

    aggregator._app_integration_configs["test"] = AppServerConfig(
        server_name="test",
        supports_resources=True,
        resources=[
            AppResourceConfig(
                uri="ui://component/app",
                mime_type=OPENAI_APPS_SDK_MIME_TYPE,
                kind=AppIntegrationKind.OPENAI_APPS_SDK,
            )
        ],
        tools=[
            AppToolConfig(
                tool_name="tool_a",
                namespaced_tool_name="test.tool_a",
                resource_uri="ui://component/app",
                linked_resource_uri="ui://component/app",
                kind=AppIntegrationKind.OPENAI_APPS_SDK,
            )
        ],
    )

    tools_result = asyncio.run(aggregator.list_tools())
    assert len(tools_result.tools) == 1
    meta = tools_result.tools[0].meta or {}
    assert meta.get("fast-agent/appIntegrationKind") == "openai_apps_sdk"
    assert meta.get("fast-agent/appResourceUri") == "ui://component/app"


def test_list_tools_marks_mcp_apps_meta_and_hides_app_only_tools() -> None:
    aggregator = _create_aggregator()
    aggregator.initialized = True

    model_tool = _tool_with_meta(
        name="model_tool",
        input_schema={"type": "object"},
        meta={"ui": {"resourceUri": "ui://component/model", "visibility": ["model"]}},
    )
    app_tool = _tool_with_meta(
        name="app_tool",
        input_schema={"type": "object"},
        meta={"ui": {"resourceUri": "ui://component/app", "visibility": ["app"]}},
    )

    model_namespaced = NamespacedTool(
        tool=model_tool,
        server_name="test",
        namespaced_tool_name="test.model_tool",
    )
    app_namespaced = NamespacedTool(
        tool=app_tool,
        server_name="test",
        namespaced_tool_name="test.app_tool",
    )

    aggregator._namespaced_tool_map = {
        "test.model_tool": model_namespaced,
        "test.app_tool": app_namespaced,
    }
    aggregator._server_to_tool_map["test"] = [model_namespaced, app_namespaced]
    aggregator._app_integration_configs["test"] = AppServerConfig(
        server_name="test",
        supports_resources=True,
        tools=[
            AppToolConfig(
                tool_name="model_tool",
                namespaced_tool_name="test.model_tool",
                resource_uri="ui://component/model",
                linked_resource_uri="ui://component/model",
                kind=AppIntegrationKind.MCP_APPS,
                visibility=["model"],
            ),
            AppToolConfig(
                tool_name="app_tool",
                namespaced_tool_name="test.app_tool",
                resource_uri="ui://component/app",
                linked_resource_uri="ui://component/app",
                kind=AppIntegrationKind.MCP_APPS,
                visibility=["app"],
            ),
        ],
    )

    tools_result = asyncio.run(aggregator.list_tools())

    assert [tool.name for tool in tools_result.tools] == ["test.model_tool"]
    meta = tools_result.tools[0].meta or {}
    assert meta.get("fast-agent/appIntegrationKind") == "mcp_apps"
    assert meta.get("fast-agent/appResourceUri") == "ui://component/model"
    assert meta.get("ui") == {
        "resourceUri": "ui://component/model",
        "visibility": ["model"],
    }


def test_tool_list_refresh_rebuilds_app_visibility_before_commit() -> None:
    aggregator = _create_aggregator()
    aggregator.initialized = True
    aggregator.validate_server = AsyncMock(return_value=True)
    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._execute_on_server = AsyncMock(
        return_value=SimpleNamespace(
            tools=[
                _tool_with_meta(
                    name="app_only",
                    input_schema={"type": "object"},
                    meta={
                        "ui": {
                            "resourceUri": "ui://component/app-only",
                            "visibility": ["app"],
                        }
                    },
                )
            ]
        )
    )
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app-only")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(contents=[SimpleNamespace(mime_type=MCP_APPS_MIME_TYPE)])
    )

    class _SilentDisplay(ConsoleDisplay):
        async def show_tool_update(
            self,
            updated_server: str,
            agent_name: str | None = None,
        ) -> None:
            del updated_server, agent_name

    aggregator.display = _SilentDisplay(config=None)

    asyncio.run(aggregator._refresh_server_tools("test"))

    config = aggregator._app_integration_configs["test"]
    assert len(config.tools) == 1
    assert config.tools[0].is_app_only
    assert asyncio.run(aggregator.list_tools()).tools == []


def test_app_resource_without_tool_warns() -> None:
    aggregator = _create_aggregator()

    aggregator.server_supports_feature = AsyncMock(return_value=True)
    aggregator._server_to_tool_map["test"] = []
    aggregator._list_resources_from_server = AsyncMock(
        return_value=[SimpleNamespace(uri="ui://component/app")]
    )
    aggregator._get_resource_from_server = AsyncMock(
        return_value=SimpleNamespace(
            contents=[SimpleNamespace(mime_type=OPENAI_APPS_SDK_MIME_TYPE)]
        )
    )

    _, config = asyncio.run(aggregator._evaluate_app_integrations_for_server("test"))

    assert config.enabled is True
    assert not config.tools
    assert any("no tools expose them" in warning.lower() for warning in config.warnings)
