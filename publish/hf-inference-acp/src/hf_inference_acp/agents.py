"""Custom agents for hf-inference-acp."""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from acp.helpers import text_block, tool_content
from acp.schema import ToolCallProgress, ToolCallStart

from fast_agent.acp import ACPAwareMixin, ACPCommand
from fast_agent.acp.acp_aware_mixin import ACPModeInfo
from fast_agent.agents import McpAgent
from fast_agent.core.direct_factory import get_model_factory
from fast_agent.core.logging.logger import get_logger
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.hf_auth import add_hf_auth_header

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.context import Context
    from hf_inference_acp.wizard.stages import WizardState

from hf_inference_acp.hf_config import (
    CONFIG_FILE,
    get_default_model,
    get_hf_token_source,
    has_hf_token,
    update_model_in_config,
)
from hf_inference_acp.wizard.model_catalog import format_model_list_help

logger = get_logger(__name__)


def _normalize_hf_model(model: str) -> str:
    """Normalize a HuggingFace model string by adding hf. prefix if needed.

    If the model looks like a HuggingFace model (org/model format) but doesn't
    have the hf. prefix, add it automatically.

    Examples:
        moonshotai/Kimi-K2-Thinking:together -> hf.moonshotai/Kimi-K2-Thinking:together
        hf.moonshotai/Kimi-K2-Thinking:together -> hf.moonshotai/Kimi-K2-Thinking:together
        kimi -> kimi (alias, unchanged)
        gpt-4o -> gpt-4o (no /, unchanged)
    """
    from fast_agent.llm.model_factory import ModelFactory

    # Already has hf. prefix
    if model.startswith("hf."):
        return model

    # Check if it's a known alias
    if model in ModelFactory.MODEL_ALIASES:
        return model

    # If it has org/model format, add hf. prefix
    if "/" in model:
        return f"hf.{model}"

    return model


async def _lookup_and_format_providers(model: str) -> str | None:
    """Look up inference providers for a model and return a formatted message.

    Returns None if the model is not a HuggingFace model (no '/').
    """
    import random

    # Extract the HF model ID from various formats
    model_id = model

    # Strip hf. prefix if present
    if model_id.startswith("hf."):
        model_id = model_id[3:]

    # Strip :provider suffix if present
    if ":" in model_id:
        model_id = model_id.rsplit(":", 1)[0]

    # Must have org/model format
    if "/" not in model_id:
        return None

    from fast_agent.llm.hf_inference_lookup import lookup_inference_providers

    try:
        result = await lookup_inference_providers(model_id)
        if result.has_providers:
            providers = result.format_provider_list()
            model_strings = result.format_model_strings()
            example = random.choice(model_strings)
            return (
                f"**Available providers:** {providers}\n\n"
                f"**Autoroutes if no provider specified. Example use:** `/set-model {example}`"
            )
        elif result.exists:
            return "No inference providers currently available for this model."
        else:
            return None
    except Exception:
        return None


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
                if state.selected_model:
                    await self._apply_model_to_running_hf_agent(state.selected_model)

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

    async def _apply_model_to_running_hf_agent(self, model: str) -> bool:
        """
        If the HuggingFace agent exists in this ACP session, update it in-place.

        Returns True if we found the agent and attempted an update.
        """
        acp = self.acp
        if not acp or not acp.slash_handler:
            return False
        instance = getattr(acp.slash_handler, "instance", None)
        agents = getattr(instance, "agents", None) if instance else None
        if not isinstance(agents, dict):
            return False
        hf_agent = agents.get("huggingface")
        if not hf_agent or not hasattr(hf_agent, "apply_model"):
            return False
        try:
            await hf_agent.apply_model(model)
            return True
        except Exception:
            return True

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
                description="Log in to HuggingFace (runs `hf auth login`)",
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

    @property
    def acp_session_commands_allowlist(self) -> set[str]:
        """Restrict built-in session commands for setup flows."""
        return {"status", "skills"}

    async def _handle_set_model(self, arguments: str) -> str:
        """Handler for /set-model command."""
        from fast_agent.llm.model_factory import ModelFactory

        model = arguments.strip()
        if not model:
            return format_model_list_help()

        # Normalize the model string (auto-add hf. prefix if needed)
        model = _normalize_hf_model(model)

        # Validate the model string before saving to config
        try:
            ModelFactory.parse_model_string(model)
        except Exception as e:
            return f"Error: Invalid model `{model}` - {e}"

        # Look up inference providers for this model
        provider_info = await _lookup_and_format_providers(model)

        try:
            update_model_in_config(model)
            applied = await self._apply_model_to_running_hf_agent(model)
            applied_note = "\n\nApplied to the running Hugging Face agent." if applied else ""
            provider_prefix = f"{provider_info}\n\n" if provider_info else ""
            return (
                f"{provider_prefix}"
                f"Default model set to: `{model}`\n\nConfig file updated: `{CONFIG_FILE}`"
                f"{applied_note}"
            )
        except Exception as e:
            return f"Error setting model: {e}"

    async def _handle_login(self, arguments: str) -> str:
        """Handler for /login command."""
        return (
            "To log in to Hugging Face, please run the following command in your terminal:\n\n"
            "```bash\n"
            "hf auth login\n"
            "```\n\n"
            "Or set the `HF_TOKEN` environment variable with your token:\n\n"
            "```bash\n"
            "export HF_TOKEN=your_token_here\n"
            "```\n\n"
            "You can get your token from https://huggingface.co/settings/tokens"
        )

    async def _handle_check(self, arguments: str) -> str:
        """Handler for /check command."""
        lines = ["# Hugging Face Configuration Check\n"]

        # Check huggingface_hub installation
        try:
            import huggingface_hub

            lines.append(
                f"- **huggingface_hub**: installed (version {huggingface_hub.__version__})"
            )
        except ImportError:
            lines.append("- **huggingface_hub**: NOT INSTALLED")
            lines.append("  Run: `uv tool install -U huggingface_hub`")

        # Check HF_TOKEN
        if has_hf_token():
            # Prefer the original discovery source recorded at startup (if present),
            # otherwise re-run discovery (may report "env" if auto-populated).
            source = os.environ.get("FAST_AGENT_HF_TOKEN_SOURCE") or get_hf_token_source()
            suffix = f" (source: {source})" if source else ""
            lines.append(f"- **HF_TOKEN**: set{suffix}")
        else:
            lines.append("- **HF_TOKEN**: NOT SET")
            lines.append("  Use `/login` or set `HF_TOKEN` environment variable")

        # Check config file and show model with provider info
        lines.append(f"- **Config file**: `{CONFIG_FILE}`")
        if CONFIG_FILE.exists():
            lines.append("  - Status: exists")
            lines.append(f"  - Default model: `{get_default_model()}`")

            # provider_info = await self._get_model_provider_info(default_model)
            # if provider_info:
            #     lines.append(f"  - {provider_info}")

        else:
            lines.append("  - Status: will be created on first use")

        return "\n".join(lines)

    async def _get_model_provider_info(self, model: str) -> str | None:
        """Get a brief provider info string for a model.

        Returns None if providers cannot be looked up or model is not a HF model.
        """
        from fast_agent.llm.hf_inference_lookup import lookup_inference_providers

        # Extract the HF model ID from various formats
        model_id = model

        # Strip hf. prefix if present
        if model_id.startswith("hf."):
            model_id = model_id[3:]

        # Strip :provider suffix if present
        if ":" in model_id:
            model_id = model_id.rsplit(":", 1)[0]

        # Must have org/model format
        if "/" not in model_id:
            return None

        try:
            result = await lookup_inference_providers(model_id)
            if result.has_providers:
                providers = result.format_provider_list()
                return f"Available providers: {providers}"
            elif result.exists:
                return "No inference providers available"
            else:
                return None
        except Exception:
            return None


class HuggingFaceAgent(ACPAwareMixin, McpAgent):
    """
    Main Hugging Face inference agent.

    This is a standard agent that uses the Hugging Face LLM provider.
    Supports lazy connection to Hugging Face MCP server via /connect command.
    """

    def __init__(
        self,
        config: "AgentConfig",
        context: "Context | None" = None,
        **kwargs,
    ) -> None:
        """Initialize the Hugging  Face agent."""
        McpAgent.__init__(self, config=config, context=context, **kwargs)
        self._context = context
        self._hf_mcp_connected = False

    @property
    def acp_commands(self) -> dict[str, ACPCommand]:
        """Declare slash commands for the Hugging Face agent."""
        return {
            "connect": ACPCommand(
                description="Connect to Hugging Face MCP server",
                handler=self._handle_connect,
            ),
            "set-model": ACPCommand(
                description="Set the active Hugging Face model for this session",
                input_hint="<model-name>",
                handler=self._handle_set_model,
            ),
        }

    def acp_mode_info(self) -> ACPModeInfo | None:
        """Provide mode info for ACP clients."""
        return ACPModeInfo(
            name="Hugging Face",
            description="AI assistant powered by Hugging Face Inference API",
        )

    def _ensure_hf_token_and_header(self) -> None:
        """
        Best-effort sync of HF token from provider config into environment,
        then attach Authorization/X-HF-Authorization to the MCP server config.
        """
        # Prefer provider config token if env isn't set yet
        try:
            if (
                not os.environ.get("HF_TOKEN")
                and self._context
                and getattr(self._context, "config", None)
            ):
                provider_token = ProviderKeyManager.get_config_file_key("hf", self._context.config)
                if provider_token:
                    os.environ["HF_TOKEN"] = provider_token
        except Exception:
            pass

        # Inject auth header onto the huggingface server if missing
        try:
            registry = getattr(self._context, "server_registry", None)
            if not registry:
                return
            server_config = registry.get_server_config("huggingface")
            if not server_config or not getattr(server_config, "url", None):
                return

            existing_headers = dict(server_config.headers or {})
            existing_keys = {k.lower() for k in existing_headers}
            if {"authorization", "x-hf-authorization"} & existing_keys:
                return

            updated_headers = add_hf_auth_header(server_config.url, existing_headers)
            if updated_headers is None or updated_headers == existing_headers:
                return

            server_config.headers = updated_headers
            registry.registry["huggingface"] = server_config
        except Exception:
            # Non-fatal; connection attempts will still proceed with existing headers
            return

    async def _handle_connect(self, arguments: str) -> str:
        """Handler for /connect command - lazily connect to Hugging Face MCP server."""
        # Refresh HF token/header (important when setup flow captured token after start)
        self._ensure_hf_token_and_header()

        if self._hf_mcp_connected:
            return "Already connected to Hugging Face MCP server."

        if not has_hf_token():
            return (
                "**Error**: HF_TOKEN not set.\n\n"
                "Please set your Hugging Face token first:\n"
                "```bash\n"
                "export HF_TOKEN=your_token_here\n"
                "```\n\n"
                "Or switch to Setup mode and use `/login` for instructions."
            )

        tool_call_id = str(uuid.uuid4())

        async def _send_connect_update(
            *,
            title: str | None = None,
            status: str | None = None,
            message: str | None = None,
        ) -> None:
            if not self.acp:
                return
            try:
                content = [tool_content(text_block(message))] if message else None
                await self.acp.send_session_update(
                    ToolCallProgress(
                        tool_call_id=tool_call_id,
                        title=title,
                        status=status,  # type: ignore[arg-type]
                        content=content,
                        session_update="tool_call_update",
                    )
                )
            except Exception:
                return

        try:
            if self.acp:
                await self.acp.send_session_update(
                    ToolCallStart(
                        tool_call_id=tool_call_id,
                        title="Connect HuggingFace MCP server",
                        kind="fetch",
                        status="in_progress",
                        session_update="tool_call",
                    )
                )
                await _send_connect_update(
                    title="Starting connection…",
                    status="in_progress",
                    message=f"Starting connection (session {self.acp.session_id})",
                )

            # Add huggingface server to aggregator if not present
            if "huggingface" not in self._aggregator.server_names:
                self._aggregator.server_names.append("huggingface")

            # Reset initialized flag to force reconnection
            self._aggregator.initialized = False

            # Load/connect to the server
            await _send_connect_update(title="Connecting and initializing…", status="in_progress")
            await self._aggregator.load_servers(force_connect=True)

            self._hf_mcp_connected = True
            await _send_connect_update(title="Connected", status="in_progress")

            # Rebuild system prompt to include fresh server instructions
            await _send_connect_update(
                title="Rebuilding system prompt…", status="in_progress"
            )
            await self.rebuild_instruction_templates()

            # Get available tools
            await _send_connect_update(title="Fetching available tools…", status="in_progress")
            tools_result = await self._aggregator.list_tools()
            tool_names = [t.name for t in tools_result.tools] if tools_result.tools else []

            if tool_names:
                preview = ", ".join(tool_names[:10])
                suffix = f" (+{len(tool_names) - 10} more)" if len(tool_names) > 10 else ""
                await _send_connect_update(
                    title="Connected (tools available)",
                    status="completed",
                    message=f"Available tools: {preview}{suffix}",
                )
            else:
                await _send_connect_update(
                    title="Connected (no tools found)",
                    status="completed",
                    message="No tools available from the server.",
                )

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
            await _send_connect_update(
                title="Connection failed",
                status="failed",
                message=str(e),
            )
            return f"**Error connecting to HuggingFace MCP server:**\n\n`{e}`"

    async def _handle_set_model(self, arguments: str) -> str:
        """Handler for /set-model in Hugging Face mode."""
        from fast_agent.llm.model_factory import ModelFactory

        model = arguments.strip()
        if not model:
            return format_model_list_help()

        # Normalize the model string (auto-add hf. prefix if needed)
        model = _normalize_hf_model(model)

        # Validate the model string before applying
        try:
            ModelFactory.parse_model_string(model)
        except Exception as e:
            return f"Error: Invalid model `{model}` - {e}"

        # Look up inference providers for this model
        provider_info = await _lookup_and_format_providers(model)

        try:
            # Apply model first - if this fails, don't update config
            await self.apply_model(model)
            update_model_in_config(model)
            provider_prefix = f"{provider_info}\n\n" if provider_info else ""
            return f"{provider_prefix}Active model set to: `{model}`\n\nConfig file updated: `{CONFIG_FILE}`"
        except Exception as e:
            return f"Error setting model: {e}"

    async def apply_model(self, model: str) -> None:
        """
        Switch the active LLM model for this running agent.

        This updates the agent config and re-attaches a fresh LLM instance.
        """
        new_model = (model or "").strip()
        if not new_model:
            return
        if not self.context:
            return

        # Update agent config so future attachments use the new model spec.
        # Do not set `RequestParams.model`: that field is a provider-level override and can
        # accidentally force the full spec (or even None) into the outgoing request body.
        self.config.model = new_model
        if self.config.default_request_params is not None:
            params_without_model = self.config.default_request_params.model_dump(exclude={"model"})
            self.config.default_request_params = RequestParams(**params_without_model)

        llm_factory = get_model_factory(
            self.context,
            model=new_model,
        )
        await self.attach_llm(
            llm_factory,
            request_params=self.config.default_request_params,
            api_key=self.config.api_key,
        )
