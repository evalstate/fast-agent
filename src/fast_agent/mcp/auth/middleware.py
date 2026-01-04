"""Middleware for handling HuggingFace-specific authentication headers."""

from starlette.types import ASGIApp, Receive, Scope, Send


class HFAuthHeaderMiddleware:
    """
    Middleware that copies X-HF-Authorization to Authorization header.

    HuggingFace Spaces use X-HF-Authorization for authentication, but
    FastMCP's BearerAuthBackend only checks the standard Authorization header.
    This middleware normalizes the headers so both work.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])

        # Check if Authorization header already exists
        has_auth = any(k.lower() == b"authorization" for k, _ in headers)

        # If no Authorization but X-HF-Authorization exists, copy it
        if not has_auth:
            for key, value in headers:
                if key.lower() == b"x-hf-authorization":
                    # Add as Authorization header
                    new_headers = list(headers) + [(b"authorization", value)]
                    scope = dict(scope, headers=new_headers)
                    break

        await self.app(scope, receive, send)
