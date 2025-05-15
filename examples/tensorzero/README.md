# About the tensorzero / fast-agent integration

[TensorZero](https://www.tensorzero.com/) is an open source project designed to help LLM application developers rapidly improve their inference calls. Its core features include:

- A uniform inference interface to all leading LLM platforms.
- The ability to dynamic route to different platforms and program failovers.
- Automated parameter tuning and training
- Advance templating features for your system prompts
- Organization of LLM inference data into a Clickhouse DB allowing for sophisticated downstream analytics
- A bunch of other good stuff is always in development

`tensorzero` is powerful heavy, so we provide here a quickstart example that combines the basic components of `fast-agent`, an MCP server, `tensorzero`, and other supporting services into a cohesive whole.

## Quickstart guide

- Build and activate the `uv` `fast-agent` environment
- Ensure that ports `3000`, `4000`, `8000`, `9000`, and `9001` are unallocated before running this demo.
- Run `cp .env.sample .env` and then drop in at least one of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Make sure the accounts are funded.
- `make up`
- `make agent`

The demo test's our implementation's ability to:

- Implement the T0 model gateway as an inference backend
- Implement T0's dynamic templating feature
- Have in-conversation memory
- Describe and execute tool calls
- Remember previous tool calls

A version of a conversation to test all of this could be:

```
Hi.

Tell me a poem.

Do you have any tools that you can use?

Please demonstrate the use of that tool on your last response.

Please summarize the conversation so far.

What tool calls have you executed in this session, and what were their results?
```

## Multimodal support

Run `make image-test` to test the gateway's ability to handle base64-encoded image data

## Development notes:

- `make stop` will stop the MCP server and the tensorzero server
- `make tenzorzero-logs` will tail the tensorzero server logs
- `make mcp-logs` will tail the MCP server logs
- Generic `make logs` dumps all log output from all services to terminal
