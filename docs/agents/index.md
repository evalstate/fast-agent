# Files and Resources

## Attaching Files

You can include files in a conversation using Paths:

```python
from fast_agent import Prompt
from pathlib import Path

plans = await agent.send(
    Prompt.user(
        "Summarise this PDF",
        Path("secret-plans.pdf")
    )
)
```

This works for any mime type that can be tokenized by the model.

## MCP Resources

MCP Server resources can be conveniently included in a message with:

```python
description = await agent.with_resource(
    "What is in this image?",
    "resource://images/cat.png",
    "mcp_image_server",
)
```

## Prompt Files

Prompt Files can include Resources:

```md title="agent_script.txt"
---USER
Please extract the major colours from this CSS file:
---RESOURCE
index.css
```

They can either be loaded with `fast_agent.load_prompt`, or delivered via the built-in `prompt-server`.
