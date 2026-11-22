"""OpenAI SDK client construction for Copilot gateway routes."""

from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from fast_agent.llm.provider.copilot.broker import CopilotEndpoint


def create_openai_client(endpoint: CopilotEndpoint) -> AsyncOpenAI:
    # Match the SDK's spelling so its default User-Agent is replaced, not
    # sent alongside the native fast-agent identity under another casing.
    headers = {
        "User-Agent" if name.lower() == "user-agent" else name: value
        for name, value in endpoint.headers.items()
    }
    client = AsyncOpenAI(
        api_key="",
        base_url=endpoint.base_url,
        default_headers=headers,
        max_retries=0,
        organization="",
        project="",
        admin_api_key="",
        # The broker validates credentials, including header-only bearer auth.
        _enforce_credentials=False,
        http_client=DefaultAsyncHttpxClient(trust_env=False, follow_redirects=False),
    )
    # The SDK merges OPENAI_CUSTOM_HEADERS during construction; only broker
    # headers belong on this client, without changing the process environment.
    client._custom_headers = headers
    return client
