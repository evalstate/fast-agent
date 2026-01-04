"""Context variables for request-scoped authentication."""

from contextvars import ContextVar

# Stores the bearer token for the current request.
# Used to pass through to LLM providers (e.g., HuggingFace).
# Each async task has its own isolated copy of this variable.
request_bearer_token: ContextVar[str | None] = ContextVar("request_bearer_token", default=None)
