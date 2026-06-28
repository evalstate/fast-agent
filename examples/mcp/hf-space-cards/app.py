from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP
from fastmcp.server.auth import RemoteAuthProvider
from pydantic import AnyHttpUrl
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse

from fast_agent import FastAgent
from fast_agent.mcp.auth.middleware import HFAuthHeaderMiddleware
from fast_agent.mcp.auth.providers.huggingface import HuggingFaceTokenVerifier
from fast_agent.mcp.server import HarnessMCPAdapter, HarnessMCPAdapterOptions
from fast_agent.mcp.server.common import get_fast_agent_version, get_oauth_config

os.environ.setdefault("FAST_AGENT_SERVE_OAUTH", "huggingface")

ROOT = Path(__file__).parent
SERVER_NAME = "fast-agent card MCP server"


def auth_provider() -> RemoteAuthProvider | None:
    oauth_provider, oauth_scopes, resource_url = get_oauth_config()
    if oauth_provider != "huggingface":
        return None
    return RemoteAuthProvider(
        token_verifier=HuggingFaceTokenVerifier(),
        authorization_servers=[AnyHttpUrl("https://huggingface.co")],
        base_url=AnyHttpUrl(resource_url),
        scopes_supported=oauth_scopes,
        resource_name=SERVER_NAME,
    )


fast = FastAgent(SERVER_NAME, config_path=ROOT / "fast-agent.yaml")
fast.load_agents(ROOT / "agents")

mcp = FastMCP(
    name=SERVER_NAME,
    instructions=(
        "This MCP server exposes AgentCard-defined fast-agent tools. The research "
        "tool uses Hugging Face Inference Providers with the caller's OAuth token."
    ),
    version=get_fast_agent_version(),
    auth=auth_provider(),
)


@mcp.custom_route("/", methods=["GET"])
async def root_info(request: Any) -> PlainTextResponse:
    del request
    return PlainTextResponse(
        "fast-agent AgentCard MCP server. Connect with an MCP client and Hugging Face OAuth."
    )


async def main() -> None:
    async with fast.harness() as harness:
        adapter = HarnessMCPAdapter(
            harness.app(),
            HarnessMCPAdapterOptions(
                default_agent="researcher",
                session_scope="request",
                metadata={"deployment_shape": "agent_cards"},
                cleanup_session=harness.sessions.delete,
            ),
        )
        adapter.register_agent_tool(
            mcp,
            name="research",
            agent="researcher",
            description="Research a topic and return a concise answer.",
            input_schema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to research."},
                    "depth": {
                        "type": "string",
                        "enum": ["quick", "deep"],
                        "default": "quick",
                    },
                },
                "required": ["topic"],
            },
            render_arguments="Research {{topic}}.\nDepth: {{depth}}",
        )
        adapter.register_agent_tool(
            mcp,
            name="chat",
            agent="researcher",
            description="Send a plain message to the researcher AgentCard.",
        )
        await mcp.run_http_async(
            transport="http",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "7860")),
            middleware=[Middleware(cast("Any", HFAuthHeaderMiddleware))],
        )


if __name__ == "__main__":
    asyncio.run(main())
