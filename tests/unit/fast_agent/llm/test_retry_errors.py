from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.fastagent_llm import FastAgentLLM


def test_provider_key_errors_with_retryable_terms_are_not_fatal() -> None:
    error = ProviderKeyError("Provider unavailable", "QUOTA exhausted; retry after timeout")

    assert FastAgentLLM._is_fatal_retry_error(error) is False


def test_provider_key_errors_without_retryable_terms_are_fatal() -> None:
    error = ProviderKeyError("Missing API key")

    assert FastAgentLLM._is_fatal_retry_error(error) is True
