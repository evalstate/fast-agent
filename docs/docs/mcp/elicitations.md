---
title: Quick Start - MCP Elicitations
description: Collect request-scoped user input from modern MCP servers
social:
  title: MCP Elicitations
  tagline: Collect structured user input while an MCP request is active.
  description: Collect structured user input while an MCP request is active.
  alt: fast-agent social card — MCP Elicitations
---


# Quick Start: MCP Elicitations

An MCP server can use
[elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
to request additional user input **while it is processing another client
request**. Elicitation is not an unsolicited or background prompt.

The quickstart demonstrates the complete modern flow:

1. The client calls `start_t4_small_sandbox`.
2. The server needs a region, duration, and maximum budget.
3. The active tool request pauses and presents an elicitation form.
4. The user accepts, declines, or cancels the question.
5. The client retries that same request with the user's response.
6. The server returns a simulated start result.

No infrastructure is created and no charges are incurred.

<div
  class="fa-terminal-demo"
  data-fa-asciinema-cast="../../assets/mcp/elicitation-sandbox.cast"
  data-fa-asciinema-cols="96"
  data-fa-asciinema-rows="24"
  data-fa-asciinema-poster="npt:0:03"
  data-fa-asciinema-speed="1"
  data-fa-asciinema-idle-time-limit="1.3"
  data-fa-asciinema-fit="width"
  data-fa-asciinema-autoplay="true"
>
  <div class="fa-terminal-theme-switch" aria-label="Terminal theme">
    <button type="button" data-fa-terminal-theme="auto">Auto</button>
    <button type="button" data-fa-terminal-theme="light">Light</button>
    <button type="button" data-fa-terminal-theme="dark">Dark</button>
  </div>
  <div data-fa-asciinema-target></div>
</div>

<!--
Cast asset:
- Source: docs/docs/assets/mcp/elicitation-sandbox.cast
- Regenerate: uv run scripts/docs.py cast-build elicitation-sandbox
-->

## Setup

Make sure you have the `uv` [package manager](https://docs.astral.sh/uv/)
installed, then:

=== "Linux/macOS"

    ```bash
    mkdir fast-agent && cd fast-agent
    uv venv
    source .venv/bin/activate
    uv pip install fast-agent-mcp
    fast-agent quickstart elicitations
    cd elicitations
    ```

=== "Windows"

    ```pwsh
    mkdir fast-agent; cd fast-agent
    uv venv
    .venv\Scripts\activate
    uv pip install fast-agent-mcp
    fast-agent quickstart elicitations
    cd elicitations
    ```

Run the demo:

```bash
uv run sandbox_demo.py
```

Choose a region, duration, and maximum simulated budget, then select **Accept**
to approve starting the sandbox at `$0.40 per hour`. Use **Decline** or
**Cancel** to finish without creating anything. The budget is an in-band
numeric limit, not payment information, and the example never creates a
charge.

## Approval actions

Approval is part of the elicitation result, not a field in the form schema.
Every elicitation resolves with one of these protocol actions:

- `accept`: approve the question and, for form mode, return the form content.
- `decline`: explicitly refuse the request.
- `cancel`: dismiss the interaction without making a decision.

The example therefore does not add an `approve`/`disapprove` property to
`SandboxRequest`. The question itself carries the price, and the tool handles
`AcceptedElicitation`, `DeclinedElicitation`, and `CancelledElicitation`
separately.

## Server

The resolver describes input needed to continue the active tool request:

```python title="sandbox_server.py"
def request_sandbox_details() -> Elicit[SandboxRequest]:
    return Elicit(
        "Start t4-small sandbox at $0.40 per hour?",
        SandboxRequest,
    )
```

The resolved response is injected into the tool only when the same request
continues:

```python title="sandbox_server.py"
@server.tool()
def start_t4_small_sandbox(
    request: Annotated[
        ElicitationResult[SandboxRequest],
        Resolve(request_sandbox_details),
    ],
) -> str: ...
```

The resolver must be deterministic and side-effect free because it may run
again as the request continues. Perform the operation only in the tool body
after validating the accepted response.

Form elicitation is in-band. Never use it for passwords, API keys, payment
details, or other sensitive data.

## Client

The demo calls the tool directly, so it does not need an LLM:

```python title="sandbox_demo.py"
sandbox = cast("McpAgent", agent.sandbox)
result = await sandbox.call_tool("start_t4_small_sandbox", {})
```

An LLM-backed agent can select the same tool from a natural-language request.
The elicitation remains associated with that active tool call either way.

## Custom handler

Pass `elicitation_handler` to an agent to replace the built-in form UI. The
custom-handler demo supplies deterministic content for the simulated sandbox:

```bash
uv run custom_handler_demo.py
```

```python title="custom_handler_demo.py"
async def development_sandbox_handler(
    context: ClientRequestContext,
    params: ElicitRequestParams,
) -> ElicitResult:
    assert isinstance(params, ElicitRequestFormParams)
    return ElicitResult(
        action="accept",
        content={
            "region": "eu-west-1",
            "duration_hours": 2,
            "max_budget_usd": 1.00,
        },
    )


@fast.agent(
    "sandbox_custom",
    servers=["sandbox_server"],
    elicitation_handler=development_sandbox_handler,
)
```

This pattern is useful for deterministic development and test flows. Do not
automatically approve consequential production operations without an
equivalent user-control boundary.

## Modern URL elicitation

Run the URL example with:

```bash
uv run url_demo.py
```

The URL server returns an `InputRequiredResult` while processing
`request_console_access`. The embedded URL request is associated with that
tool call:

```python title="url_server.py"
return InputRequiredResult(
    input_requests={
        "authorize_console": ElicitRequest(
            params=ElicitRequestURLParams(
                message=f"Authorize browser access to sandbox {sandbox_id}.",
                url=authorization_url,
            )
        )
    },
    request_state=request_state,
)
```

`url_demo.py` registers a per-agent handler that displays the URL and waits
for an explicit `accept`, `decline`, or `cancel` choice. The MCP client then
retries the original tool with its original arguments, the response keyed by
`authorize_console`, and the opaque `requestState`.

The custom handler is intentional: the current built-in URL handler displays
or queues the URL and immediately returns `accept`. Override it when the
application requires an explicit consent choice.

In URL mode, `accept` means consent to navigate—it does **not** mean OAuth,
payment, credential entry, or another external operation completed. The
example therefore returns:

```text
external completion was not verified and no console access was granted
```

A production server must independently verify its browser callback before
granting access or performing a protected operation. The example uses
`example.com`, omits the legacy `elicitationId`, and does not exchange
credentials.

## Configuration

The example forces modern protocol negotiation and uses the built-in forms
handler:

```yaml title="fast-agent.yaml"
mcp:
  servers:
    sandbox_server:
      target: "uv run sandbox_server.py"
      protocol_mode: modern
      elicitation:
        mode: forms
```

Elicitation handling can be configured as `forms` (the default),
`auto-cancel`, or `none`. In `auto-cancel` mode, **fast-agent** advertises the
capability but cancels requests automatically. With `none`, it does not
advertise the capability.
