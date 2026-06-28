---
social:
  title: Internal Models
  tagline: Use built-in model aliases and internal model definitions effectively.
  description: Use built-in model aliases and internal model definitions effectively.
  alt: fast-agent social card — Internal Models
---

**fast-agent** comes with two internal models to aid development and testing: `passthrough` and `playback`.

## Passthrough

By default, the `passthrough` model echos messages sent to it.

### Fixed Responses

By sending a `***FIXED_RESPONSE <message>` message, the model will return `<message>` to any request.

### Tool Calling

By sending a `***CALL_TOOL <tool_name> [<json>]` message, the model will call the specified MCP Tool, and return a string containing the results.

## Playback

The `playback` model replays the first conversation sent to it. A typical usage may look like this:

```markdown title="playback.txt"
---USER
Good morning!
---ASSISTANT
Hello
---USER
Generate some JSON
---ASSISTANT
{
   "city": "London",
   "temperature": 72
}
```

This can then be used with the `prompt-server`; you can apply the MCP Prompt to the agent either programmatically with `apply_prompt` or with the `/prompts` command in the interactive shell.

Alternatively, you can load the file with `load_prompt`.

JSON contents can be converted to structured outputs:

```python
from pathlib import Path

from fast_agent import Prompt, PromptMessageExtended, load_prompt
from pydantic import BaseModel


class Weather(BaseModel):
    city: str
    temperature: int


@fast.agent(name="playback", model="playback")

...

playback_messages: list[PromptMessageExtended] = load_prompt(Path("playback.txt"))
# Set up the Conversation
loaded = await agent.playback.generate(playback_messages)
assert loaded.first_text().startswith("HISTORY LOADED")

response: str = await agent.playback.send("Good morning!") # Returns Hello
temperature, _ = await agent.playback.structured([Prompt.user("Generate some JSON")], Weather)

```

When the `playback` runs out of messages, it returns `MESSAGES EXHAUSTED (list size [a]) ([b] overage)`.

List size is the total number of messages originally loaded, overage is the number of requests made after exhaustion.
