# Quick Start: MCP Elicitations

This quickstart demonstrates **fast-agent**'s elicitation feature - a powerful way to collect structured data from users through forms and interactive prompts.

## What are Elicitations?

Elicitations allow MCP servers to request structured input from users through type-safe forms. This enables:
- User preference collection
- Account registration flows  
- Configuration wizards
- Interactive feedback forms
- Any scenario requiring structured user input

## Examples Included

### 1. Forms Demo (`forms_demo.py`)
A showcase of different form types with beautiful rich console output. Uses the passthrough model to display forms directly to users.

```bash
uv run forms_demo.py
```

This example demonstrates:
- User profile collection
- Preference settings
- Simple yes/no ratings
- Detailed feedback forms

### 2. Account Creation Assistant (`account_creation.py`)
An AI-powered account creation workflow where the LLM initiates the account signup process and the user fills out the form.

```bash
# Configure your LLM first (edit fastagent.config.yaml)
uv run account_creation.py --model gpt-4o
```

This example shows:
- LLM initiating elicitation via tool calls
- User filling out the form (not the LLM)
- Tool call errors when user cancels/declines
- Success handling when form is completed

### 3. Game Character Creator (`game_character.py` + `game_character_handler.py`)
A whimsical example with custom elicitation handling, featuring animated dice rolls and visual effects.

```bash
uv run game_character.py
```

Features:
- Custom elicitation handler (in separate module for clarity)
- Animated progress bars and typewriter effects
- Epic dice roll mechanics with cosmic bonuses
- Interactive character creation with theatrical flair
- Demonstrates proper handler file organization

## Getting Started

1. **Setup your environment:**
   ```bash
   # Activate your Python environment
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   
   # Install dependencies if needed
   uv pip install fast-agent-mcp
   ```

2. **For the account creation example**, rename `fastagent.secrets.yaml.example` to `fastagent.secrets.yaml` and add your API keys.

3. **Run an example:**
   ```bash
   # Try the forms demo first (quiet mode enabled programmatically)
   uv run forms_demo.py
   ```

## How Elicitations Work

### Elicitation Modes

**fast-agent** supports these elicitation modes:

1. **`forms`** - Shows forms to users (great with passthrough model)
2. **`auto_cancel`** - Automatically cancels all elicitations
3. **`none`** - No elicitation handling
4. **Custom handler** - Use your own handler function (overrides mode setting)

Configure modes in `fastagent.config.yaml`:

```yaml
mcp:
  servers:
    my_server:
      command: "uv"
      args: ["run", "server.py"]
      elicitation:
        mode: "forms"  # or "auto" or "custom"
```

### Creating Your Own Elicitations

#### Basic Server-Side Elicitation

```python
from pydantic import BaseModel, Field

class UserPrefs(BaseModel):
    theme: str = Field(
        description="Color theme",
        json_schema_extra={
            "enum": ["light", "dark"],
            "enumNames": ["Light Mode", "Dark Mode"]
        }
    )
    notifications: bool = Field(True, description="Enable notifications?")

# In your MCP server:
result = await mcp.get_context().elicit(
    "Configure your preferences",
    schema=UserPrefs
)
```

#### Custom Elicitation Handler

For advanced interactive experiences, create a custom handler:

```python
async def my_custom_handler(context, params) -> ElicitResult:
    # Your custom logic here - animations, special effects, etc.
    content = {"field": "value"}
    return ElicitResult(action="accept", content=content)

# Register with your agent:
@fast.agent(
    "my-agent",
    servers=["my_server"],
    elicitation_handler=my_custom_handler
)
```

See `game_character_handler.py` for a complete example with animations and effects.

## Next Steps

- Explore the example code to understand different patterns
- Try modifying the forms in `elicitation_server.py`
- Create your own custom elicitation handlers
- Check the [documentation](https://fast-agent.ai) for advanced features

## Tips

- Use `rich` for beautiful console output
- Test with passthrough model first, then try real LLMs
- Custom handlers enable creative interactions
- Validate user input in your schemas using Pydantic

Happy form building! 🚀