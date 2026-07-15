import httpx
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import APIError as OpenAIAPIError
from openai import BadRequestError as OpenAIBadRequestError

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.fastagent_llm import FastAgentLLM


def test_provider_key_errors_with_retryable_terms_are_not_fatal() -> None:
    error = ProviderKeyError("Provider unavailable", "QUOTA exhausted; retry after timeout")

    assert FastAgentLLM._is_fatal_retry_error(error) is False


def test_provider_key_errors_without_retryable_terms_are_fatal() -> None:
    error = ProviderKeyError("Missing API key")

    assert FastAgentLLM._is_fatal_retry_error(error) is True


def test_context_length_code_is_fatal() -> None:
    error = RuntimeError("request failed (code: context_length_exceeded)")

    assert FastAgentLLM._is_fatal_retry_error(error) is True


def test_openai_api_error_context_length_code_is_fatal() -> None:
    error = OpenAIAPIError(
        "Your input exceeds the context window.",
        httpx.Request("POST", "https://api.openai.com/v1/responses"),
        body={"code": "context_length_exceeded"},
    )

    assert "context_length_exceeded" not in str(error)
    assert FastAgentLLM._is_fatal_retry_error(error) is True


def test_openai_bad_request_is_fatal() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(400, request=request)
    error = OpenAIBadRequestError(
        "input exceeds the context window",
        response=response,
        body={"code": "context_length_exceeded"},
    )

    assert FastAgentLLM._is_fatal_retry_error(error) is True


def test_anthropic_bad_request_is_fatal() -> None:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    error = AnthropicBadRequestError(
        "prompt is too long",
        response=response,
        body={"error": {"type": "invalid_request_error"}},
    )

    assert FastAgentLLM._is_fatal_retry_error(error) is True
