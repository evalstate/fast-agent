"""Console entrypoint for hf-inference-acp."""

from __future__ import annotations

import asyncio
import sys
from importlib.resources import files

from fast_agent import FastAgent

from hf_inference_acp.agents import HuggingFaceAgent, SetupAgent
from hf_inference_acp.config import (
    CONFIG_FILE,
    ensure_config_exists,
    get_default_model,
    has_hf_token,
)


def get_setup_instruction() -> str:
    """Generate the instruction for the Setup agent."""
    token_status = "set" if has_hf_token() else "NOT SET"
    default_model = get_default_model()

    return f"""You are the HuggingFace Inference Setup assistant.

# Available Commands

Use these slash commands to configure the agent:

- `/set-model <model>` - Set the default model for inference
- `/login` - Get instructions for logging in to HuggingFace
- `/check` - Verify huggingface_hub installation and configuration

# Current Status

- **Config file**: `{CONFIG_FILE}`
- **HF_TOKEN**: {token_status}
- **Default model**: `{default_model}`

To start using the AI assistant, ensure HF_TOKEN is set and switch to the "Hugging Face" mode."""


def get_hf_instruction() -> str:
    """Generate the instruction for the HuggingFace agent.

    Uses file_silent templates to include optional markdown files.
    """
    # Load the system prompt template from resources
    try:
        resource_path = (
            files("hf_inference_acp").joinpath("resources").joinpath("hf.system_prompt.md")
        )
        if resource_path.is_file():
            return resource_path.read_text()
    except Exception:
        pass

    # Fallback to basic instruction
    return """You are a helpful AI assistant powered by HuggingFace Inference API.

{{file_silent:AGENTS.md}}
{{file_silent:huggingface.md}}
"""


async def run_agents() -> None:
    """Main async function to set up and run the agents."""
    # Ensure config exists
    config_path = ensure_config_exists()

    # Determine which agent should be default based on HF_TOKEN presence
    hf_token_present = has_hf_token()

    # Get the default model from config
    default_model = get_default_model()

    # Create FastAgent instance
    fast = FastAgent(
        name="hf-inference-acp",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )

    # Register the Setup agent (passthrough LLM)
    # This is always available for configuration
    @fast.custom(
        SetupAgent,
        name="setup",
        instruction=get_setup_instruction(),
        model="passthrough",
        default=not hf_token_present,
    )
    async def setup_agent():
        pass

    # Only register the HuggingFace agent if HF_TOKEN is present
    # This prevents model initialization errors when token is missing
    if hf_token_present:
        # Register the HuggingFace agent (uses HF LLM)
        # Note: HuggingFace MCP server is connected lazily via /connect command
        @fast.custom(
            HuggingFaceAgent,
            name="huggingface",
            instruction=get_hf_instruction(),
            model=default_model,
            servers=[],  # Empty - use /connect to add HuggingFace MCP server
            default=True,
        )
        async def hf_agent():
            pass

    # Start the ACP server
    await fast.start_server(transport="acp")


def main() -> None:
    """Console script entrypoint for `hf-inference-acp`."""
    try:
        asyncio.run(run_agents())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
