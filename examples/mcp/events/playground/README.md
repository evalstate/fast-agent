---
title: MCP Events Test Lab
emoji: 🔔
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Interactive SEP-3415 MCP Events (poll) test server
---

# MCP Events Test Lab (SEP-3415)

A standalone Streamable HTTP [MCP Events (SEP-3415)](https://github.com/modelcontextprotocol/modelcontextprotocol) test server with isolated per-room sandboxes and a browser control plane for triggering events, gaps (`truncated`), revocations (`Forbidden`), and session `--resume` replay.

- **Browser dashboard:** `/` redirects to an isolated room at `/r/<room_id>`
- **Room-scoped MCP Streamable HTTP endpoint:** `/r/<room_id>/mcp`
- **Public fallback MCP endpoint:** `/mcp`

Each room keeps its own streams, poll logs, revocations, and `nextPollMs` setting, and expires after 4 hours of inactivity.

## Event Types

| Event | Delivery | Arguments | Description |
| --- | --- | --- | --- |
| `job.status_changed` | `poll` | `{"job_id": "..."}` | Fires when a job moves between `queued`, `running`, `done`, `failed`. |
| `system.heartbeat` | `poll` | `{}` | Server heartbeat with no required arguments. |
| `webhook.only_alert` | `webhook` | `{}` | Non-pollable event for testing client UI handling. |

## Tools

- `start_job(job_id, description)` emits `job.status_changed` (`queued`).
- `advance_job(job_id, stage, detail)` emits `job.status_changed` for any stage.
- `get_job_details(job_id)` returns recorded stages for a job.

## Run Locally

From the `fast-agent` repo:

```bash
uv run examples/mcp/events/playground/app.py --port 8765
```

Open `http://127.0.0.1:8765` to get an isolated room URL, then copy the generated `fast-agent go --url ...` or `/mcp connect ...` command.

## Deploy to a Hugging Face Space

Create a **Docker** Space on Hugging Face and upload the contents of this directory (`app.py`, `Dockerfile`, `requirements.txt`, `README.md`, and `static/`):

```bash
hf repo create mcp-events-playground --repo-type space --space-sdk docker
hf upload <your-username>/mcp-events-playground examples/mcp/events/playground . --repo-type space
```
