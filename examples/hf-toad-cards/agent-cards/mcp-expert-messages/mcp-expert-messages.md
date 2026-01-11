---USER
What are prompts in MCP?

---ASSISTANT
Let me search for the prompts documentation.

```bash
rg -n --heading -C 3 'prompt' specification/docs/specification/draft/server/ -g '*.mdx' | head -60
```

---USER
specification/docs/specification/draft/server/prompts.mdx
6-
7-<Info>**Protocol Revision**: draft</Info>
8-
9:The Model Context Protocol (MCP) provides a standardized way for servers to expose prompt
10-templates to clients. Prompts allow servers to provide structured messages and
11-instructions for interacting with language models. Clients can discover available
12-prompts, retrieve their contents, and provide arguments to customize them.
13-
14-## User Interaction Model
15-
16:Prompts are designed to be **user-controlled**, meaning they are exposed from servers to
17-clients with the intention of the user being able to explicitly select them for use.

---ASSISTANT
From `specification/docs/specification/draft/server/prompts.mdx:9-17`:

> "The Model Context Protocol (MCP) provides a standardized way for servers to expose prompt templates to clients. Prompts allow servers to provide structured messages and instructions for interacting with language models."

Key point: Prompts are **user-controlled** — they're intended for users to explicitly select, not for automatic invocation.

Let me find the SDK implementation:

```bash
rg -n 'class Prompt|class PromptArgument' python-sdk/src/mcp/types.py -A 8
```

---USER
964:class PromptArgument(BaseModel):
965-    """An argument for a prompt template."""
966-
967-    name: str
968-    """The name of the argument."""
969-    description: str | None = None
970-    """A human-readable description of the argument."""
971-    required: bool | None = None
972-    """Whether this argument must be provided."""
--
976:class Prompt(BaseMetadata):
977-    """A prompt or prompt template that the server offers."""
978-
979-    description: str | None = None
980-    """An optional description of what this prompt provides."""
981-    arguments: list[PromptArgument] | None = None
982-    """A list of arguments to use for templating the prompt."""
983-    icons: list[Icon] | None = None
984-    """An optional list of icons for this prompt."""

---ASSISTANT
The SDK defines prompts in `python-sdk/src/mcp/types.py:964-984`:

```python
class PromptArgument(BaseModel):
    name: str
    description: str | None = None
    required: bool | None = None

class Prompt(BaseMetadata):
    description: str | None = None
    arguments: list[PromptArgument] | None = None
    icons: list[Icon] | None = None
```

**Summary:** Prompts are server-defined templates for LLM interactions. They're explicitly user-triggered (e.g., via slash commands), have optional arguments for customization, and can include icons for UI display. Use them to expose reusable workflows like "review this code" or "summarize this document".

---USER
What is the Icon class in MCP?

---ASSISTANT
Let me search for the Icon class in the SDK types:

```bash
rg -n 'class Icon' python-sdk/src/mcp/types.py -A 10
```

---USER
150:class Icon(BaseModel):
151-    """An icon for display to users."""
152-
153-    url: AnyUrl
154-    """A data URI or HTTP URL for an icon."""
155-    size: int | None = None
156-    """The size of the icon in pixels. If unspecified, 512 is assumed."""
157-    mimeType: str | None = None
158-    """The MIME type of the icon. If unspecified, `image/webp` is assumed."""
159-

---ASSISTANT
From `python-sdk/src/mcp/types.py:150-159`:

```python
class Icon(BaseModel):
    """An icon for display to users."""
    url: AnyUrl        # Data URI or HTTP URL
    size: int | None   # Pixel size (default: 512)
    mimeType: str | None  # MIME type (default: image/webp)
```

Icons are used in `Prompt`, `Tool`, and `Resource` classes to provide visual representation in client UIs.
