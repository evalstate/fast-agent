from __future__ import annotations

from fast_agent.agents.smart_agent import _enable_smart_tooling


class _SmartToolHarness:
    def __init__(self) -> None:
        self.tools: list[object] = []

    def add_tool(self, tool: object) -> None:
        self.tools.append(tool)

    async def smart(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def validate(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def mcp_connect(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_list_resources(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_get_resource(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_with_resource(self, *args, **kwargs):
        del args, kwargs
        return ""

    async def smart_complete_resource_argument(self, *args, **kwargs):
        del args, kwargs
        return ""


def test_enable_smart_tooling_registers_resource_tools() -> None:
    harness = _SmartToolHarness()

    _enable_smart_tooling(harness)

    names = {getattr(tool, "name", "") for tool in harness.tools}
    assert "smart" in names
    assert "validate" in names
    assert "mcp_connect" in names
    assert "smart_list_resources" in names
    assert "smart_get_resource" in names
    assert "smart_with_resource" in names
    assert "smart_complete_resource_argument" in names
