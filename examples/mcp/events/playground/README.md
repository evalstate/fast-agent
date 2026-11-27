---
title: MCP Events Playground
emoji: 🔔
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive SEP-3415 MCP Events (poll) test server
---

# MCP Events Playground (SEP-3415)

A standalone Streamable HTTP [MCP Events (SEP-3415)](https://github.com/modelcontextprotocol/modelcontextprotocol) test server with a browser control plane for triggering events, gaps (`truncated`), revocations (`Forbidden`), and session `--resume` replay.

- **MCP Streamable HTTP endpoint:** `/mcp`
- **Browser dashboard:** `/`

## Event Types

| Event | Delivery | Arguments | Description |
| --- | --- | --- | --- |
| `job.status_changed` | `poll` | `{"job_id": "..."}` | Fires when a job moves between `queued`, `running`, `done`, `failed`. |
| `system.heartbeat` | `poll` | `{}` | Server heartbeat with no required arguments. |
| `webhook.only_alert` | `webhook` | `{}` | Non-pollable event for testing client UI handling. |

## Tools

- `start_job(job_id, description)` — emits `job.status_changed` (`queued`).
- `advance_job(job_id, stage, detail)` — emits `job.status_changed` for any stage.
- `get_job_details(job_id)` — returns recorded stages for a job.

## Run Locally

From the `fast-agent` repo:

```bash
uv run examples/mcp/events/playground/app.py --port 8765
```

Then connect with `fast-agent`:

```bash
uv run fast-agent go --url http://127.0.0.1:8765/mcp
```

Or attach from an existing session:

```text
/mcp connect http://127.0.0.1:8765/mcp --name playground
/events select
```

## Deploy to a Hugging Face Space

Create a **Docker** Space on Hugging Face and upload the 4 files in this directory (`app.py`, `Dockerfile`, `requirements.txt`, `README.md`):

```bash
hf repo create mcp-events-playground --repo-type space --space-sdk docker
hf upload <your-username>/mcp-events-playground examples/mcp/events/playground . --repo-type space
```

Once built, connect to `https://<your-username>-mcp-events-playground.hf.space/mcp`.
