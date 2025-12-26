"""
Regression tests for max_tokens being incorrectly set to 2048 in ACP mode.

The issue: When ACP mode rebuilds the system prompt via _build_session_request_params,
it creates RequestParams(systemPrompt=resolved) which has maxTokens=2048 as a class default.
The merge logic should preserve the model-aware maxTokens from default_request_params.
"""

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.types import RequestParams


class TestModelDatabaseLookup:
    """Test that ModelDatabase correctly looks up the kimi model."""

    def test_kimi_k2_instruct_0905_max_tokens(self):
        """ModelDatabase should return 16384 for moonshotai/kimi-k2-instruct-0905."""
        max_tokens = ModelDatabase.get_default_max_tokens("moonshotai/kimi-k2-instruct-0905")
        assert max_tokens == 16384, f"Expected 16384, got {max_tokens}"

    def test_kimi_k2_instruct_0905_case_insensitive(self):
        """ModelDatabase lookup should be case-insensitive."""
        # The DEFAULT_HUGGINGFACE_MODEL uses title case
        max_tokens = ModelDatabase.get_default_max_tokens("moonshotai/Kimi-K2-Instruct-0905")
        assert max_tokens == 16384, f"Expected 16384, got {max_tokens}"

    def test_kimi_model_params_exist(self):
        """Verify the model params are correctly configured."""
        params = ModelDatabase.get_model_params("moonshotai/kimi-k2-instruct-0905")
        assert params is not None, "Model params should exist"
        assert params.max_output_tokens == 16384
        assert params.context_window == 262144


class TestRequestParamsExcludeUnset:
    """Test Pydantic exclude_unset behavior with RequestParams."""

    def test_exclude_unset_excludes_default_max_tokens(self):
        """When only systemPrompt is set, maxTokens should be excluded with exclude_unset=True."""
        params = RequestParams(systemPrompt="test prompt")
        dumped = params.model_dump(exclude_unset=True)

        assert "systemPrompt" in dumped, "systemPrompt should be in dumped dict"
        assert "maxTokens" not in dumped, (
            f"maxTokens should NOT be in exclude_unset dump, but got: {dumped}"
        )

    def test_explicitly_set_max_tokens_included(self):
        """When maxTokens is explicitly set, it should be included."""
        params = RequestParams(systemPrompt="test", maxTokens=8192)
        dumped = params.model_dump(exclude_unset=True)

        assert dumped.get("maxTokens") == 8192

    def test_recreated_from_dump_loses_unset_tracking(self):
        """
        BUG TEST: When RequestParams is recreated from model_dump(), all fields
        become "set" and override base params during merge.

        This is what happens in apply_model() when it does:
            params_without_model = self.config.default_request_params.model_dump(exclude={"model"})
            self.config.default_request_params = RequestParams(**params_without_model)

        The recreated RequestParams has ALL fields including maxTokens=2048 as "set".
        """
        # Simulate: original params with maxTokens=2048 (class default, never explicitly set)
        original = RequestParams(systemPrompt="test")
        assert original.maxTokens == 2048  # Class default

        # This is what apply_model does - dump and recreate
        dumped = original.model_dump(exclude={"model"})
        recreated = RequestParams(**dumped)

        # The recreated params now has maxTokens as "set" (not "unset")
        recreated_dump = recreated.model_dump(exclude_unset=True)

        # BUG: maxTokens is now considered "set" because it was passed to __init__
        assert "maxTokens" in recreated_dump, (
            "After dump/recreate, maxTokens should be 'set' (this documents the bug)"
        )
        assert recreated_dump["maxTokens"] == 2048


class TestHuggingFaceLLMMaxTokens:
    """Test that HuggingFaceLLM gets correct maxTokens from ModelDatabase."""

    def _make_hf_llm(self, model: str) -> HuggingFaceLLM:
        settings = Settings(hf=HuggingFaceSettings())
        context = Context(config=settings)
        return HuggingFaceLLM(context=context, model=model, name="test-agent")

    def test_kimi_k2_instruct_0905_gets_correct_max_tokens(self):
        """HuggingFaceLLM should initialize with model-aware maxTokens."""
        llm = self._make_hf_llm("moonshotai/kimi-k2-instruct-0905")

        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384, got {llm.default_request_params.maxTokens}"
        )

    def test_kimi_default_model_gets_correct_max_tokens(self):
        """Default HuggingFace model should get correct maxTokens."""
        # Uses DEFAULT_HUGGINGFACE_MODEL = "moonshotai/Kimi-K2-Instruct-0905"
        llm = self._make_hf_llm("moonshotai/Kimi-K2-Instruct-0905")

        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384, got {llm.default_request_params.maxTokens}"
        )


class TestACPModeMaxTokensMerge:
    """Test the merge behavior that occurs in ACP mode."""

    def _make_hf_llm(self, model: str) -> HuggingFaceLLM:
        settings = Settings(hf=HuggingFaceSettings())
        context = Context(config=settings)
        return HuggingFaceLLM(context=context, model=model, name="test-agent")

    def test_acp_system_prompt_merge_preserves_max_tokens(self):
        """
        Simulates ACP mode: _build_session_request_params returns RequestParams(systemPrompt=...)
        which gets merged with the LLM's default_request_params.

        The model-aware maxTokens from default_request_params should be preserved.
        """
        llm = self._make_hf_llm("moonshotai/kimi-k2-instruct-0905")

        # This is what ACP's _build_session_request_params does
        acp_request_params = RequestParams(systemPrompt="Updated system prompt for ACP session")

        # This is what get_request_params does
        merged = llm.get_request_params(acp_request_params)

        # The merged params should have:
        # - systemPrompt from the ACP request params
        # - maxTokens from default_request_params (model-aware: 16384)
        assert merged.systemPrompt == "Updated system prompt for ACP session"
        assert merged.maxTokens == 16384, (
            f"maxTokens should be 16384 (from default_request_params), got {merged.maxTokens}"
        )

    def test_acp_merge_with_explicit_max_tokens_override(self):
        """If ACP explicitly sets maxTokens, it should override the default."""
        llm = self._make_hf_llm("moonshotai/kimi-k2-instruct-0905")

        # If ACP explicitly wanted different maxTokens
        acp_request_params = RequestParams(systemPrompt="test", maxTokens=4096)

        merged = llm.get_request_params(acp_request_params)

        # Explicit value should win
        assert merged.maxTokens == 4096


class TestMergeRequestParamsLogic:
    """Direct tests of the _merge_request_params logic."""

    def _make_hf_llm(self, model: str) -> HuggingFaceLLM:
        settings = Settings(hf=HuggingFaceSettings())
        context = Context(config=settings)
        return HuggingFaceLLM(context=context, model=model, name="test-agent")

    def test_merge_with_only_system_prompt_set(self):
        """Merge should not override maxTokens when only systemPrompt is provided."""
        llm = self._make_hf_llm("moonshotai/kimi-k2-instruct-0905")

        default = llm.default_request_params
        provided = RequestParams(systemPrompt="new prompt")

        # Verify the setup
        assert default.maxTokens == 16384, "Default should have model-aware maxTokens"

        # Check what exclude_unset gives us
        provided_dump = provided.model_dump(exclude_unset=True)
        assert "maxTokens" not in provided_dump, "maxTokens should be excluded"

        # Perform merge
        merged = llm._merge_request_params(default, provided)

        assert merged.maxTokens == 16384, (
            f"Merged maxTokens should be 16384, got {merged.maxTokens}"
        )
        assert merged.systemPrompt == "new prompt"


class TestAttachLLMFlow:
    """Test the full attach_llm flow through ModelFactory."""

    @pytest.mark.asyncio
    async def test_attach_llm_via_factory_gets_correct_max_tokens(self):
        """Test that creating an LLM via attach_llm preserves model-aware maxTokens."""
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))
        # Note: correct syntax is hf.model (dot), not hf:model (colon)
        factory = ModelFactory.create_factory("hf.moonshotai/kimi-k2-instruct-0905")

        llm = await agent.attach_llm(factory)

        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384, got {llm.default_request_params.maxTokens}"
        )
        assert llm.default_request_params.model == "moonshotai/kimi-k2-instruct-0905"

    @pytest.mark.asyncio
    async def test_kimi_alias_via_factory_gets_correct_max_tokens(self):
        """Test that the 'kimi' alias resolves correctly and gets proper maxTokens."""
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))
        # Use the kimi alias - this is what /set-model kimi would resolve to
        factory = ModelFactory.create_factory("kimi")

        llm = await agent.attach_llm(factory)

        # kimi alias resolves to hf.moonshotai/Kimi-K2-Instruct-0905:groq
        # which should get maxTokens=16384 from KIMI_MOONSHOT
        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384, got {llm.default_request_params.maxTokens}"
        )

    @pytest.mark.asyncio
    async def test_kimithink_alias_via_factory_gets_correct_max_tokens(self):
        """Test that the 'kimithink' alias resolves correctly and gets proper maxTokens."""
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))
        # Use the kimithink alias
        factory = ModelFactory.create_factory("kimithink")

        llm = await agent.attach_llm(factory)

        # kimithink alias resolves to hf.moonshotai/Kimi-K2-Thinking:together
        # which should get maxTokens=16384 from KIMI_MOONSHOT_THINKING
        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384, got {llm.default_request_params.maxTokens}"
        )

    @pytest.mark.asyncio
    async def test_apply_model_flow_preserves_model_max_tokens(self):
        """
        Test that model switching correctly uses the new model's maxTokens.

        When switching models via /set-model, the new model should determine
        its own maxTokens from ModelDatabase, not inherit from the old model.

        The fix excludes maxTokens (and model) from the dump so the new model
        can determine its own maxTokens from ModelDatabase.
        """
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))

        # Simulate model switching - exclude both model and maxTokens
        # (this is what apply_model now does after the fix)
        original_params = RequestParams(systemPrompt="original prompt")
        params_without_model_specific = original_params.model_dump(
            exclude={"model", "maxTokens"}
        )
        recreated_params = RequestParams(**params_without_model_specific)

        # Create the factory and attach LLM with the params
        factory = ModelFactory.create_factory("kimi")
        llm = await agent.attach_llm(factory, request_params=recreated_params)

        # maxTokens should be 16384 from the new model's ModelDatabase entry
        assert llm.default_request_params.maxTokens == 16384, (
            f"Expected 16384 (from KIMI_MOONSHOT), got {llm.default_request_params.maxTokens}"
        )
        # systemPrompt should be preserved from the original params
        assert llm.default_request_params.systemPrompt == "original prompt"

    @pytest.mark.asyncio
    async def test_attach_llm_then_acp_merge_preserves_max_tokens(self):
        """
        Full flow test: create agent via factory, then simulate ACP merge.

        This tests the entire path that the user is experiencing issues with.
        """
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))
        # Note: correct syntax is hf.model (dot), not hf:model (colon)
        factory = ModelFactory.create_factory("hf.moonshotai/kimi-k2-instruct-0905")

        llm = await agent.attach_llm(factory)

        # Verify LLM was created with correct maxTokens
        assert llm.default_request_params.maxTokens == 16384, (
            f"Initial maxTokens should be 16384, got {llm.default_request_params.maxTokens}"
        )

        # Now simulate what ACP does: create RequestParams with only systemPrompt
        acp_request_params = RequestParams(systemPrompt="Updated for ACP session")

        # This is what happens in generate() -> get_request_params()
        merged = llm.get_request_params(acp_request_params)

        assert merged.maxTokens == 16384, (
            f"After ACP merge, maxTokens should still be 16384, got {merged.maxTokens}"
        )
        assert merged.systemPrompt == "Updated for ACP session"

    @pytest.mark.asyncio
    async def test_colon_syntax_bug_model_name_corrupted(self):
        """
        BUG TEST: Using hf:model syntax (colon) corrupts the model name.

        The model name becomes ':moonshotai/kimi-k2-instruct-0905' (with leading colon)
        which causes ModelDatabase lookup to fail, resulting in maxTokens=2048.
        """
        from fast_agent.llm.model_factory import ModelFactory

        agent = LlmAgent(AgentConfig(name="Test Agent"))
        # This is the WRONG syntax but users may try it - exposes the bug
        factory = ModelFactory.create_factory("hf:moonshotai/kimi-k2-instruct-0905")

        llm = await agent.attach_llm(factory)

        # This documents the current buggy behavior
        # The model name gets corrupted with a leading colon
        assert llm.default_request_params.model == ":moonshotai/kimi-k2-instruct-0905", (
            f"BUG: Expected corrupted model name, got {llm.default_request_params.model}"
        )
        # And maxTokens falls back to 2048 because the model lookup fails
        assert llm.default_request_params.maxTokens == 2048, (
            f"BUG: Expected 2048 (fallback), got {llm.default_request_params.maxTokens}"
        )
