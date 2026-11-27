from __future__ import annotations

import subprocess
import sys


def test_provider_neutral_runtime_import_defers_optional_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fast_agent.llm.fastagent_llm; "
                "import fast_agent.agents.mcp_agent; "
                "assert 'anthropic' not in sys.modules; "
                "assert 'openai' not in sys.modules; "
                "assert 'fast_agent.agents.shell_runtime' not in sys.modules; "
                "assert 'fast_agent.agents.environment_filesystem_runtime' "
                "not in sys.modules; "
                "assert 'fast_agent.agents.local_filesystem_runtime' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_provider_neutral_llm_import_defers_huggingface_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fast_agent.llm.fastagent_llm; "
                "assert 'fast_agent.llm.hf_inference_lookup' not in sys.modules; "
                "assert 'huggingface_hub' not in sys.modules; "
                "from fast_agent.llm import lookup_inference_providers; "
                "assert callable(lookup_inference_providers); "
                "assert 'fast_agent.llm.hf_inference_lookup' in sys.modules; "
                "assert 'huggingface_hub' in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_anthropic_provider_import_loads_only_its_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import fast_agent.llm.provider.anthropic.llm_anthropic; "
                "import sys; "
                "assert 'anthropic' in sys.modules; "
                "assert 'openai' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_constructing_copilot_chat_model_defers_openai_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.model_factory import ModelFactory

context = Context(config=Settings())
agent = ToolAgent(AgentConfig("test"), context=context)
llm = ModelFactory.create_factory("copilot.gemini-4-neon")(agent, context=context)
assert llm.default_request_params.model == "gemini-4-neon"
assert "openai" not in sys.modules
assert "anthropic" not in sys.modules
""",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
