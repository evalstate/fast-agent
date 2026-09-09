"""Live HF route smoke tests using model defaults, including output token limits."""

import asyncio

import pytest
from huggingface_hub import get_token

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.types.llm_stop_reason import LlmStopReason


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend", ("baseten", "deepinfra", "together", "novita", "fireworks-ai", "scaleway")
)
async def test_deepseek_flash_route_accepts_default_request(backend: str) -> None:
    token = get_token()
    if not token:
        pytest.skip("Set HF_TOKEN or log in with hf auth login to run live route tests")
    settings = Settings(hf=HuggingFaceSettings(api_key=token))
    settings.logger.show_chat = False
    settings.logger.show_tools = False
    settings.logger.progress_display = False
    core = Core(settings=settings)
    await core.initialize()
    try:
        agent = LlmAgent(AgentConfig("hf-route-smoke"), core.context)
        await agent.attach_llm(
            ModelFactory.create_factory(f"hf.deepseek-ai/DeepSeek-V4-Flash-0731:{backend}")
        )
        # Keep the full default output budget to catch provider validation errors,
        # but ask for a tiny answer and bound the runtime of each paid request.
        async with asyncio.timeout(45):
            result = await agent.generate(
                "What is 2 + 2? Reply with only the single digit answer. No explanation."
            )
        assert result.stop_reason is LlmStopReason.END_TURN
        text = result.last_text()
        assert text is not None and text.strip() == "4"
    finally:
        await core.cleanup()
