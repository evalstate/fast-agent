from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio, SseMcpServer

from fast_agent.acp.server.mcp_server_conversion import convert_acp_mcp_server


def test_convert_stdio_session_mcp_server() -> None:
    settings = convert_acp_mcp_server(
        McpServerStdio(
            name="filesystem",
            command="/usr/bin/python",
            args=["server.py", "--stdio"],
            env=[
                EnvVariable(name="API_KEY", value="secret"),
                EnvVariable(name="MODE", value="test"),
            ],
        )
    )

    assert settings.name == "filesystem"
    assert settings.transport == "stdio"
    assert settings.command == "/usr/bin/python"
    assert settings.args == ["server.py", "--stdio"]
    assert settings.env == {"API_KEY": "secret", "MODE": "test"}


def test_convert_http_session_mcp_server() -> None:
    settings = convert_acp_mcp_server(
        HttpMcpServer(
            name="docs",
            type="http",
            url="https://example.com/mcp",
            headers=[
                HttpHeader(name="Authorization", value="Bearer token"),
                HttpHeader(name="X-Client", value="pytest"),
            ],
        )
    )

    assert settings.name == "docs"
    assert settings.transport == "http"
    assert settings.url == "https://example.com/mcp"
    assert settings.headers == {
        "Authorization": "Bearer token",
        "X-Client": "pytest",
    }


def test_convert_sse_session_mcp_server() -> None:
    settings = convert_acp_mcp_server(
        SseMcpServer(
            name="events",
            type="sse",
            url="https://example.com/sse",
            headers=[HttpHeader(name="X-API-Key", value="secret")],
        )
    )

    assert settings.name == "events"
    assert settings.transport == "sse"
    assert settings.url == "https://example.com/sse"
    assert settings.headers == {"X-API-Key": "secret"}
