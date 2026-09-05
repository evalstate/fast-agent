# Standalone web search

An auth-independent client for Codex's `POST alpha/search` API. No agent, provider,
history, environment, or MCP registration is performed by this package.

Protocol reference: OpenAI Codex source at `ddf04ad267`, particularly
`codex-rs/codex-api/src/search.rs`, `codex-api/src/endpoint/search.rs`, and
`ext/web-search/src/`. This is an internal Codex endpoint, not a public API
contract; availability and supported operations may change.

```python
from fast_agent.tools.web_search import (
    SearchCommands,
    SearchQuery,
    SearchRequest,
    WebSearchClient,
)

async with WebSearchClient(
    base_url="https://your-api.example/backend-api/codex",
    headers={"Authorization": "Bearer caller-supplied-token"},
) as client:
    result = await client.search(
        SearchRequest(
            id="caller-owned-stable-session-id",
            model="caller-selected-model",
            commands=SearchCommands(search_query=[SearchQuery(q="OpenAI news")]),
            input="Find recent announcements",
            max_output_tokens=2500,
        )
    )
    print(result.output)
```

Reuse the same `id` across calls that refer to earlier search/page references.
`SearchRequest` also accepts `settings: SearchSettings` and
`reasoning: SearchReasoning`. All command groups are available: `search_query`,
`image_query`, `open`, `click`, `find`, `screenshot`, `finance`, `weather`, `sports`,
and `time`, with optional `response_length`.

`input` accepts text or a list of JSON-typed Responses item objects. This keeps
conversation item variants extensible without duplicating Codex's protocol model
hierarchy. Request models reject unknown fields and invalid integer/enum values;
absent optional fields are omitted on the wire. Nested JSON nulls are retained.

`SearchResponse` preserves `output`, `encrypted_output`, opaque JSON `results`,
and future response fields. Missing results (`None`) remain distinct from `[]`.
Output is not truncated, rewritten, or converted into citations by the client.

For an MCP tool, use `commands_schema()` as its input schema and
`WEB_SEARCH_DESCRIPTION` as its description. The schema exposes only commands;
the integrating caller supplies session ID, model, input and settings. The
original description covers operations and Markdown source/image citations.

Pass `http_client=existing_async_client` to share connections. `aclose()` and the
async context manager close only clients created by this package. Closing the
wrapper prevents further searches, even when the underlying client is borrowed.
Timeout defaults to 60 seconds per HTTP operation and can be a float or
`httpx.Timeout`. There is no overall request deadline or success-body size limit.

The client makes one POST and disables redirects. It does not retry, refresh auth,
or discover credentials. Injected clients retain their own authentication hooks,
default headers and transport policy: callers must disable transport retries if
single-attempt semantics are required. `WebSearchError.kind` is `http`,
`transport`, or `response`; HTTP errors also expose `status_code`. Diagnostics are
bounded and exclude response bodies, URLs and credentials. Error bodies are not
read; successful response text and structured results are kept in full.
