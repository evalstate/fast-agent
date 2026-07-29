from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.agents.subagent_tool import SUBAGENT_TOOL_NAME, install_subagent_tool
from fast_agent.constants import FAST_AGENT_SUBAGENT_RESULT_METADATA
from fast_agent.core import Core
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.mcp.helpers.content_helpers import get_text
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.local_shell_executor import LocalEnvironment

MODEL = os.environ.get(
    "TEST_SUBAGENT_TRANSCRIPT_MODEL",
    "codexresponses.gpt-5.6-terra?reasoning=high",
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_codexresponses_subagent_produces_readable_temporary_transcript(tmp_path) -> None:
    if not MODEL.startswith("codexresponses."):
        pytest.fail("TEST_SUBAGENT_TRANSCRIPT_MODEL must select a codexresponses provider model")

    config_path = Path(__file__).parent / "fastagent.config.yaml"
    core = Core(settings=config_path)
    environment = LocalEnvironment(
        logger=get_logger(__name__),
        working_directory=tmp_path,
    )
    await core.initialize()
    parent = McpAgent(
        AgentConfig(
            "transcript-parent",
            model=MODEL,
            subagents=True,
        ),
        context=core.context,
        shell_environment=environment,
    )
    await parent.attach_llm(ModelFactory.create_factory(MODEL))
    assert install_subagent_tool(parent, label_generator=lambda: "e2e")

    transcript_path: str | None = None
    try:
        result = await parent.call_tool(
            SUBAGENT_TOOL_NAME,
            {
                "message": (
                    "Reply with the exact token ARTIFACT_E2E_OK and no other text. "
                    "Do not call tools."
                )
            },
        )
        assert not result.is_error
        response = get_text(result.content[0])
        assert response is not None
        assert response.startswith("ARTIFACT_E2E_OK\n\n")

        assert result.meta is not None
        details = result.meta[FAST_AGENT_SUBAGENT_RESULT_METADATA]
        raw_transcript_path = details["transcript_path"]
        assert isinstance(raw_transcript_path, str)
        transcript_path = raw_transcript_path
        assert transcript_path in response
        transcript = await environment.read_text(transcript_path)
        assert "FAST_AGENT_SUBAGENT_TRANSCRIPT 1" in transcript
        assert "ARTIFACT_E2E_OK" in transcript
        assert "=== STATUS completed ===" in transcript
        search = await environment.execute(
            ShellExecutionRequest(
                command=f"rg -n ARTIFACT_E2E_OK -- {shlex.quote(transcript_path)}"
            )
        )
        assert search.result.exit_code == 0
        assert "ARTIFACT_E2E_OK" in search.result.stdout
    finally:
        await parent.shutdown()
        if transcript_path is not None:
            assert not Path(transcript_path).exists()
        await environment.close()
        await core.cleanup()
