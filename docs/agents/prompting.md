# Prompting Agents

**fast-agent** provides a flexible MCP based API for sending messages to agents, with convenience methods for handling Files, Prompts and Resources.

Read more about the use of MCP types in **fast-agent** [here](../mcp/types.md).

## Sending Messages

The simplest way of sending a message to an agent is the `send` method:

```python
response: str = await agent.send("how are you?")
```

This returns the text of the agent's response as a string, making it ideal for simple interactions.

You can attach files by using `Prompt.user()` method to construct your message:

```python
from fast_agent import Prompt
from pathlib import Path

plans: str = await agent.send(
    Prompt.user(
        "Summarise this PDF",
        Path("secret-plans.pdf")
    )
)
```

`Prompt.user()` automatically converts content to the appropriate MCP Type. For example, `image/png` becomes `ImageContent` and `application/pdf` becomes an EmbeddedResource.

You can also use MCP Types directly - for example:

```python
from mcp.types import ImageContent, TextContent

mcp_text: TextContent = TextContent(type="text", text="Analyse this image.")
mcp_image: ImageContent = ImageContent(type="image", 
                          mimeType="image/png",
                          data=base_64_encoded)

response: str  = await agent.send(
    Prompt.user(
        mcp_text,
        mcp_image
    )
)
```

> Note: use `Prompt.assistant()` to produce messages for the `assistant` role.

### Using `generate()` and multipart content

The `generate()` method allows you to access multimodal content from an agent, or its Tool Calls as well as send conversational pairs.

```python
from fast_agent import FastAgent, Prompt, PromptMessageExtended

message = Prompt.user("Describe an image of a sunset")

response: PromptMessageExtended = await agent.generate([message])

print(response.last_text())  # Main text response
```

The key difference between `send()` and `generate()` is that `generate()` returns a `PromptMessageExtended` object, giving you access to the complete response structure:

- `last_text()`: Gets the last text response - usually the Assistant message without Tool Call/Response information.
- `first_text()`: Gets the first text content if multiple text blocks exist
- `all_text()`: Combines all text content in the response - including Tall Call/Response information.
- `content`: Direct access to the full list of content parts, including Images and EmbeddedResources

This is particularly useful when working with multimodal responses or tool outputs:

```python
# Generate a response that might include multiple content types
response = await agent.generate([
    Prompt.user("Analyze this image", Path("chart.png"))
])

for content in response.content:
    if content.type == "text":
        print("Text response:", content.text[:100], "...")
    elif content.type == "image":
        print("Image content:", content.mimeType)
    elif content.type == "resource":
        print("Resource:", content.resource.uri)
```

You can also use `generate()` for multi-turn conversations by passing multiple messages:

```python
messages = [
    Prompt.user("What is the capital of France?"),
    Prompt.assistant("The capital of France is Paris."),
    Prompt.user("And what is its population?")
]

response = await agent.generate(messages)
```

The `generate()` method provides the foundation for working with content returned by the LLM, and MCP Tool, Prompt and Resource calls.

### Using `structured()` for typed responses

When you need the agent to return data in a specific format, use the `structured()` method. This parses the agent's response into a Pydantic model:

```python
from pydantic import BaseModel
from typing import List

# Define your expected response structure
class CityInfo(BaseModel):
    name: str
    country: str
    population: int
    landmarks: List[str]

# Request structured information
result, message = await agent.structured(
    [Prompt.user("Tell me about Paris")], 
    CityInfo
)

# Now you have strongly typed data
if result:
    print(f"City: {result.name}, Population: {result.population:,}")
    for landmark in result.landmarks:
        print(f"- {landmark}")
```

The `structured()` method returns a tuple containing:
1. The parsed Pydantic model instance (or `None` if parsing failed)
2. The full `PromptMessageExtended` response

This approach is ideal for:
- Extracting specific data points in a consistent format
- Building workflows where agents need structured inputs/outputs
- Integrating agent responses with typed systems

Always check if the first value is `None` to handle cases where the response couldn't be parsed into your model:

```python
result, message = await agent.structured([Prompt.user("Describe Paris")], CityInfo)

if result is None:
    # Fall back to the text response
    print("Could not parse structured data, raw response:")
    print(message.last_text())
```

The `structured()` method provides the same request parameter options as `generate()`.

!!! note

    LLMs produce JSON when producing Structured responses, which can conflict with Tool Calls. Use a `chain` to combine Tool Calls with Structured Outputs. 



## MCP Prompts

Apply a Prompt from an MCP Server to the agent with:

```python
response: str = await agent.apply_prompt(
    "setup_sizing",
    arguments={"units": "metric"}
)
```

You can list and get Prompts from attached MCP Servers:

```python
from mcp.types import GetPromptResult, PromptMessage

prompt: GetPromptResult = await agent.get_prompt("setup_sizing")
first_message: PromptMessage = prompt.messages[0]
```

and send the native MCP `PromptMessage` to the agent with:

```python
response: str = await agent.send(first_message)
```

> If the last message in the conversation is from the `assistant`, it is returned as the response.

## MCP Resources

`Prompt.user` also works with MCP Resources:

```python
from mcp.types import ReadResourceResult

resource: ReadResourceResult = await agent.get_resource(
    "resource://images/cat.png", "mcp_server_name" 
)
response: str = await agent.send(
    Prompt.user("What is in this image?", resource)
)
```

Alternatively, use the _with_resource_ convenience method:

```python
response: str = await agent.with_resource(
    "What is in this image?",
    "resource://images/cat.png",
    "mcp_server_name",
)

```

## Prompt Files

Long prompts can be stored in text files, and loaded with the `load_prompt` utility:

```python
from fast_agent import PromptMessageExtended, load_prompt
from pathlib import Path

prompt: list[PromptMessageExtended] = load_prompt(Path("two_cities.txt"))
result: str = await agent.send(prompt[0])
```

```markdown title="two_cities.txt"
### The Period

It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was
the epoch of incredulity, ...
```

Prompts files can contain conversations to aid in-context learning or allow you to replay conversations with the Playback LLM:

```markdown title="sizing_conversation.txt"
---USER
the moon
---ASSISTANT
object: MOON
size: 3,474.8
units: KM
---USER
the earth
---ASSISTANT
object: EARTH
size: 12,742
units: KM
---USER
how big is a tiger?
---ASSISTANT
object: TIGER
size: 1.2
units: M
```

Multiple messages (conversations) can be applied with the `generate()` method:

```python
from fast_agent import PromptMessageExtended, load_prompt
from pathlib import Path

prompt: list[PromptMessageExtended] = load_prompt(Path("sizing_conversation.txt"))
result: PromptMessageExtended = await agent.generate(prompt)
```

Conversation files can also be used to include resources:

```markdown title="prompt_secret_plans.txt"
---USER
Please review the following documents:
---RESOURCE
secret_plan.pdf
---RESOURCE
repomix.xml
---ASSISTANT
Thank you for those documents, the PDF contains secret plans, and some
source code was attached to achieve those plans. Can I help further?
```

```python
from fast_agent import PromptMessageExtended, load_prompt
from pathlib import Path

prompt: list[PromptMessageExtended] = load_prompt(Path("prompt_secret_plans.txt"))
result: PromptMessageExtended = await agent.generate(prompt)
```

!!! Note "File Format / MCP Serialization"

    If the filetype is `json`, fast-agent saves a `{"messages": [...]}` JSON container. It can contain either MCP `PromptMessage` objects (legacy) or `PromptMessageExtended` objects (preserves tool calls, channels, etc). `fast_agent.load_prompt` and `prompt-server` will load either the text or JSON format directly.
    See [History Saving](../models/index.md#history-saving) to learn how to save a conversation to a file for editing or playback.


### Using the `prompt-server`

Prompt files can also be served using the inbuilt `prompt-server`. The `prompt-server` command is installed with `fast-agent` making it convenient to set up and use:

```yaml title="fastagent.config.yaml"
mcp:
  servers:
    prompts:
      command: "prompt-server"
      args: ["prompt_secret_plans.txt"]
```

This configures an MCP Server that will serve a `prompt_secret_plans` MCP Prompt, and `secret_plan.pdf` and `repomix.xml` as MCP Resources.

If arguments are supplied in the template file, these are also handled by the `prompt-server`

```markdown title="prompt_with_args.txt"
---USER
Hello {{assistant_name}}, how are you?
---ASSISTANT
Great to meet you {{user_name}} how can I be of assistance?
```
