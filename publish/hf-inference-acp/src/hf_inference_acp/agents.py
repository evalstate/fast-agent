"""Custom agents for hf-inference-acp."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.acp import ACPAwareMixin, ACPCommand
from fast_agent.acp.acp_aware_mixin import ACPModeInfo
from fast_agent.agents import McpAgent
from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.context import Context
    from hf_inference_acp.wizard.stages import WizardState

from hf_inference_acp.hf_config import (
    CONFIG_FILE,
    get_default_model,
    has_hf_token,
    update_model_in_config,
)

logger = get_logger(__name__)


class SetupAgent(ACPAwareMixin, McpAgent):
    """
    Setup agent for configuring HuggingFace inference.

    Provides slash commands for:
    - Setting the default model
    - Logging in to HuggingFace
    - Checking the configuration
    """

    def __init__(
        self,
        config: "AgentConfig",
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        """Initialize the Setup agent."""
        McpAgent.__init__(self, config=config, context=context, **kwargs)
        self._context = context

    async def attach_llm(self, llm_factory, model=None, request_params=None, **kwargs):
        """Override to set up wizard callback after LLM is attached."""
        llm = await super().attach_llm(llm_factory, model, request_params, **kwargs)

        # Set up wizard callback if LLM supports it
        if hasattr(llm, "set_completion_callback"):
            llm.set_completion_callback(self._on_wizard_complete)

        return llm

    async def _on_wizard_complete(self, state: "WizardState") -> None:
        """
        Called when the setup wizard completes successfully.

        Attempts to auto-switch to HuggingFace mode if available.
        """
        logger.info(
            "Wizard completed",
            name="wizard_complete",
            model=state.selected_model,
            username=state.hf_username,
        )

        # Try to switch to HuggingFace mode
        if self._context and self._context.acp:
            try:
                # Check if huggingface mode is available
                available_modes = self._context.acp.available_modes
                if "huggingface" in available_modes:
                    await self._context.acp.switch_mode("huggingface")
                    logger.info("Auto-switched to HuggingFace mode")
                else:
                    logger.info(
                        "HuggingFace mode not available for auto-switch. "
                        "User may need to restart the agent."
                    )
            except Exception as e:
                logger.warning(f"Failed to auto-switch mode: {e}")

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        """Declare slash commands for the Setup agent."""
        return {
            "set-model": ACPCommand(
                description="Set the default model for HuggingFace inference",
                input_hint="<model-name>",
                handler=self._handle_set_model,
            ),
            "login": ACPCommand(
                description="Log in to HuggingFace (runs huggingface-cli login)",
                handler=self._handle_login,
            ),
            "check": ACPCommand(
                description="Verify huggingface_hub installation and configuration",
                handler=self._handle_check,
            ),
        }

    def acp_mode_info(self) -> ACPModeInfo | None:
        """Provide mode info for ACP clients."""
        return ACPModeInfo(name="Setup", description="Configure Hugging Face settings")

    async def _handle_set_model(self, arguments: str) -> str:
        """Handler for /set-model command."""
        model = arguments.strip()
        if not model:
            return (
                "Error: Please provide a model name.\n\n"
                "Example: `/set-model hf.moonshotai/Kimi-K2-Instruct-0905`"
            )

        try:
            update_model_in_config(model)
            return f"Default model set to: `{model}`\n\nConfig file updated: `{CONFIG_FILE}`"
        except Exception as e:
            return f"Error updating config: {e}"

    async def _handle_login(self, arguments: str) -> str:
        """Handler for /login command."""
        return (
            "To log in to Hugging Face, please run the following command in your terminal:\n\n"
            "```bash\n"
            "huggingface-cli login\n"
            "```\n\n"
            "Or set the `HF_TOKEN` environment variable with your token:\n\n"
            "```bash\n"
            "export HF_TOKEN=your_token_here\n"
            "```\n\n"
            "You can get your token from https://huggingface.co/settings/tokens"
        )

    async def _handle_check(self, arguments: str) -> str:
        """Handler for /check command."""
        lines = ["# HuggingFace Configuration Check\n"]

        # Check huggingface_hub installation
        try:
            import huggingface_hub

            lines.append(
                f"- **huggingface_hub**: installed (version {huggingface_hub.__version__})"
            )
        except ImportError:
            lines.append("- **huggingface_hub**: NOT INSTALLED")
            lines.append("  Run: `pip install huggingface_hub`")

        # Check HF_TOKEN
        if has_hf_token():
            lines.append("- **HF_TOKEN**: set")
        else:
            lines.append("- **HF_TOKEN**: NOT SET")
            lines.append("  Use `/login` or set `HF_TOKEN` environment variable")

        # Check config file
        lines.append(f"- **Config file**: `{CONFIG_FILE}`")
        if CONFIG_FILE.exists():
            lines.append("  Status: exists")
            lines.append(f"  Default model: `{get_default_model()}`")
        else:
            lines.append("  Status: will be created on first use")

        return "\n".join(lines)


class HuggingFaceAgent(ACPAwareMixin, McpAgent):
    """
    Main HuggingFace inference agent.

    This is a standard agent that uses the HuggingFace LLM provider.
    Supports lazy connection to HuggingFace MCP server via /connect command.
    """

    def __init__(
        self,
        config: "AgentConfig",
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        """Initialize the HuggingFace agent."""
        McpAgent.__init__(self, config=config, context=context, **kwargs)
        self._context = context
        self._hf_mcp_connected = False

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        """Declare slash commands for the HuggingFace agent."""
        return {
            "connect": ACPCommand(
                description="Connect to HuggingFace MCP server",
                handler=self._handle_connect,
            ),
        }

    def acp_mode_info(self) -> ACPModeInfo | None:
        """Provide mode info for ACP clients."""
        return ACPModeInfo(
            name="Hugging Face",
            description="AI assistant powered by Hugging Face Inference API",
        )

    async def _handle_connect(self, arguments: str) -> str:
        """Handler for /connect command - lazily connect to HuggingFace MCP server."""
        if self._hf_mcp_connected:
            return "Already connected to HuggingFace MCP server."

        if not has_hf_token():
            return (
                "**Error**: HF_TOKEN not set.\n\n"
                "Please set your HuggingFace token first:\n"
                "```bash\n"
                "export HF_TOKEN=your_token_here\n"
                "```\n\n"
                "Or switch to Setup mode and use `/login` for instructions."
            )

        try:
            # Add huggingface server to aggregator if not present
            if "huggingface" not in self._aggregator.server_names:
                self._aggregator.server_names.append("huggingface")

            # Reset initialized flag to force reconnection
            self._aggregator.initialized = False

            # Load/connect to the server
            await self._aggregator.load_servers()

            self._hf_mcp_connected = True

            # Get available tools
            tools_result = await self._aggregator.list_tools()
            tool_names = [t.name for t in tools_result.tools] if tools_result.tools else []

            if tool_names:
                tool_list = "\n".join(f"- `{name}`" for name in tool_names[:10])
                more = f"\n- ... and {len(tool_names) - 10} more" if len(tool_names) > 10 else ""
                return (
                    "Connected to HuggingFace MCP server.\n\n"
                    f"**Available tools ({len(tool_names)}):**\n{tool_list}{more}"
                )
            else:
                return "Connected to Hugging Face MCP server.\n\nNo tools available."

        except Exception as e:
            return f"**Error connecting to HuggingFace MCP server:**\n\n`{e}`"
