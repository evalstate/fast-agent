from mcp_types import Tool

from fast_agent.mcp.mcp_aggregator import MCPToolCatalog, NamespacedTool


def _tool(server_name: str, local_name: str) -> NamespacedTool:
    return NamespacedTool(
        tool=Tool(name=local_name, input_schema={"type": "object"}),
        server_name=server_name,
        namespaced_tool_name=f"{server_name}__{local_name}",
    )


def test_tool_catalog_is_an_ordered_structural_snapshot() -> None:
    first = _tool("first-server", "search")
    second = _tool("second", "search")
    unique = _tool("second", "render")
    by_name = {
        second.namespaced_tool_name: second,
        first.namespaced_tool_name: first,
        unique.namespaced_tool_name: unique,
    }
    by_server = {
        "first-server": [first],
        "second": [second, unique],
    }

    catalog = MCPToolCatalog.snapshot(
        by_namespaced_name=by_name,
        by_server=by_server,
        server_names=["first-server", "second"],
    )
    by_name.clear()
    by_server["second"].clear()

    assert catalog.namespaced_tool("first-server__search") is first
    assert catalog.first_tool_named("search") is second
    assert catalog.routable_tool_names() == {
        "first-server__search",
        "second__search",
        "second__render",
        "search",
        "render",
    }
    assert catalog.server_tool_names("second") == ("search", "render")
    assert catalog.resolve_tool_name("first-server__search").local_name == "search"
    assert catalog.resolve_tool_name("first-server__other").server_name == "first-server"
    assert catalog.resolve_tool_name("render").server_name == "second"
    assert catalog.resolve_tool_name("missing").server_name == "first-server"


def test_filtered_routable_names_match_execution_destinations() -> None:
    first_tools = [_tool("first", name) for name in ("shared", "only_first", "second__shared")]
    second_tools = [
        _tool("second", name)
        for name in ("shared", "only_second", "first__missing", "unknown__alias")
    ]
    # Display order deliberately differs from the router's first-server precedence.
    catalog = MCPToolCatalog.snapshot(
        by_namespaced_name={
            tool.namespaced_tool_name: tool for tool in [*second_tools, *first_tools]
        },
        by_server={"first": first_tools, "second": second_tools},
        server_names=["first", "second"],
    )
    for allowed_server in ("first", "second"):

        def permitted(name: str) -> bool:
            return name.startswith(f"{allowed_server}__")

        allowed_names = catalog.routable_tool_names(permitted)
        for name in catalog.routable_tool_names():
            destination = catalog.resolve_tool(name)
            assert (name in allowed_names) == (
                destination is not None and permitted(destination.namespaced_tool_name)
            )
