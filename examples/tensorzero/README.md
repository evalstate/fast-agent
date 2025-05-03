# About the tensorzero / fast-agent integration

[TensorZero](https://www.tensorzero.com/) is an open source project designed to help LLM application developers rapidly improve their inference calls. Its core features include:

- A uniform inference interface to all leading LLM platforms.
- The ability to dynamic route to different platforms and program failovers.
- Automated parameter tuning and training
- Advance templating features for your system prompts
- Organization of LLM inference data into a Clickhouse DB allowing for sophisticated downstream analytics
- A bunch of other good stuff is always in development

`tensorzero` is not the lightest framework, so we provide here a quickstart example that combines the basic components of `fast-agent`, an MCP server, and `tensorzero` into a cohesive whole.

## Quickstart guide

Ensure that ports `8000` and `3000` are unallocated before running this demo.

`cp .env.sample .env`, followed by population with values. note: you can run this without either your ANTHROPIC or OPENAI keys
