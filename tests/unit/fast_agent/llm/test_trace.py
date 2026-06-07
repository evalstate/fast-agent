import os

from fast_agent.llm.trace import llm_trace_enabled, set_llm_trace_enabled, toggle_llm_trace


def test_llm_trace_toggle_updates_runtime_state() -> None:
    set_llm_trace_enabled(False)

    assert toggle_llm_trace() is True
    assert llm_trace_enabled() is True
    assert os.environ["FAST_AGENT_LLM_TRACE"] == "1"

    assert toggle_llm_trace() is False
    assert llm_trace_enabled() is False
    assert "FAST_AGENT_LLM_TRACE" not in os.environ
