"""Non-Codex tool discovery must not load the OpenAI provider stack."""

import subprocess
import sys


def test_non_codex_tool_listing_and_stale_run_do_not_import_openai() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import sys

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.tools.codex_web_search import CodexWebSearchAdapter

async def main():
    agent = McpAgent(AgentConfig(name="import-test", servers=[]), context=Context())
    for llm in (None, PassthroughLLM()):
        agent._llm = llm
        assert "web_run" not in {tool.name for tool in (await agent.list_tools()).tools}
        result = await CodexWebSearchAdapter(agent).run()
        assert result.is_error
        assert result.content[0].text == "Web search is disabled."
    loaded = [name for name in sys.modules if (
        name == "openai" or name.startswith("openai.")
        or name.startswith("fast_agent.llm.provider.openai")
    )]
    assert not loaded, f"Non-Codex tool discovery loaded OpenAI: {loaded}"

asyncio.run(main())
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
