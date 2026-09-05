"""Search without an agent; supply Codex credentials explicitly through the environment.

From the repository root:
    uv run examples/web-search/standalone.py 'OpenAI news'

Required: CODEX_API_KEY, CODEX_ACCOUNT_ID (the caller's ChatGPT account ID).
Optional: CODEX_BASE_URL, WEB_SEARCH_MODEL, WEB_SEARCH_SESSION_ID.
"""

import argparse
import asyncio
import os
from uuid import uuid4

from fast_agent.tools.web_search import (
    SearchCommands,
    SearchQuery,
    SearchRequest,
    WebSearchClient,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Web search query")
    args = parser.parse_args()

    # This example reads caller-supplied credentials; the library does not discover them.
    headers = {
        "Authorization": f"Bearer {os.environ['CODEX_API_KEY']}",
        "chatgpt-account-id": os.environ["CODEX_ACCOUNT_ID"],
    }
    session_id = os.environ.get("WEB_SEARCH_SESSION_ID") or str(uuid4())
    async with WebSearchClient(
        base_url=os.environ.get("CODEX_BASE_URL", "https://chatgpt.com/backend-api/codex"),
        headers=headers,
    ) as client:
        result = await client.search(
            SearchRequest(
                id=session_id,
                model=os.environ.get("WEB_SEARCH_MODEL", "gpt-6-astra"),
                commands=SearchCommands(search_query=[SearchQuery(q=args.query)]),
            )
        )
        # Authoritative text; result.results separately preserves opaque structured metadata.
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
