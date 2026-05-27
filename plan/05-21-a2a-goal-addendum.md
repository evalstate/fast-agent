# A2A Goal Addendum

This addendum extends the active A2A completion goal with the latest review
requirements that are not represented in the immutable goal tracker text.

## Documentation recordings

- Include at least one asciinema recording that shows a real fast-agent A2A
  client streaming from a real fast-agent A2A server backed by an LLM.
- The preferred provider smoke path is:
  - server model: `codexresponses.gpt-5.4-mini`;
  - server tools: Hugging Face MCP server;
  - client prompt: ask for a Markdown-formatted answer about trending Hugging
    Face models;
  - expected user-visible behavior: the client receives streaming updates before
    the final task completion.
- Keep this recording separate from deterministic fake-server recordings. The
  deterministic recordings and tests remain the required regression signal; the
  real-LLM recording is a provider/network smoke demonstration.

Current implementation notes:

- `docs/docs/assets/a2a/a2a-real-llm-hf-streaming.cast` is the expected checked-in
  cast file.
- `uv run scripts/a2a_docs_pipeline.py record-real-llm` is the expected
  regeneration command.
- The checked-in cast must not contain provider tokens, bearer headers, or other
  secrets.

## Structured JSON output

A2A protocol support for structured JSON is through `Part.data`, not through an
LLM-output-schema negotiation feature. The fast-agent integration should treat
structured JSON as protocol data only when it is represented as structured
content, and should keep ordinary model text as text.

Expected fast-agent mapping:

- inbound A2A `Part.data` maps into fast-agent prompt content as formatted JSON
  text unless a richer internal structured-content representation is added later;
- outbound fast-agent `TextResourceContents` with
  `mimeType="application/json"` maps to A2A `Part.data`;
- ordinary model text that happens to contain JSON remains a text artifact;
- docs must make this distinction explicit so users do not assume A2A provides
  model-level JSON schema enforcement.

Open follow-up:

- If fast-agent adds a first-class structured-output content object later, the
  A2A bridge should map that object directly to `Part.data` instead of requiring
  JSON `TextResourceContents`.
