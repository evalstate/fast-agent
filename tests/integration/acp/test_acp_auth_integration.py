from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from acp.exceptions import RequestError
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapabilities, Implementation
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from conftest import _initialize_agent  # noqa: E402
from test_client import TestClient  # noqa: E402


@pytest.mark.asyncio
@pytest.mark.integration
async def test_acp_initialize_survives_missing_provider_keys_and_prompts_fail_lazily(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            default_model: gpt-4.1
            logger:
              progress_display: false
              show_chat: false
              show_tools: false
            """
        )
    )

    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(config_path),
        "--transport",
        "acp",
        "--name",
        "fast-agent-acp-auth-test",
    ]
    client = TestClient()
    environment_dir = tmp_path / ".fast-agent"
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        cwd=tmp_path,
        env={
            "ENVIRONMENT_DIR": str(environment_dir),
            "HOME": str(tmp_path),
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "AZURE_API_KEY": "",
            "CODEX_API_KEY": "",
            "HF_TOKEN": "",
        },
    ) as (connection, process):
        init_response = await _initialize_agent(
            connection,
            process,
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapabilities(read_text_file=True, write_text_file=True),
                terminal=False,
            ),
            client_info=Implementation(name="pytest-auth-client", version="0.0.1"),
        )

        auth_methods = getattr(init_response, "auth_methods", None) or getattr(
            init_response, "authMethods", None
        )
        assert auth_methods is not None
        assert any(
            getattr(method, "id", None) == "fast-agent-ai-secrets" for method in auth_methods
        )

        session_response = await connection.new_session(mcp_servers=[], cwd=str(tmp_path))
        assert session_response.session_id

        with pytest.raises(RequestError) as exc_info:
            await connection.prompt(
                session_id=session_response.session_id,
                prompt=[text_block("hello")],
            )

    assert exc_info.value.code == -32000
    data = exc_info.value.data
    assert isinstance(data, dict)
    assert data["methodId"] == "fast-agent-ai-secrets"
    assert data["message"] == "OpenAI API key not configured"
    assert data["configFile"] == "fastagent.secrets.yaml"
    assert data["docsUrl"] == "https://fast-agent.ai/ref/config_file/"
    assert data["envVars"] == ["OPENAI_API_KEY"]
    assert "fast-agent model setup" in data["recommendedCommands"]
