"""Custom agents for hf-inference-acp."""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
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
    copy_toad_cards_from_resources,
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


def _resolve_alias_display(model: str) -> tuple[str, str] | None:
    """Resolve alias to full model string for display, preserving suffix overrides."""
    from fast_agent.llm.model_factory import ModelFactory

    if not model:
        return None

    alias_key = model
    alias_suffix: str | None = None
    if ":" in model:
        alias_key, alias_suffix = model.rsplit(":", 1)

    alias_target = ModelFactory.MODEL_ALIASES.get(alias_key)
    if not alias_target:
        return None

    resolved = alias_target
    if alias_suffix:
        if ":" in resolved:
            resolved = resolved.rsplit(":", 1)[0]
        resolved = f"{resolved}:{alias_suffix}"

    return model, resolved


def _collect_agent_card_warnings(context: "Context | None") -> list[str]:
    if not context:
        return []
    warnings = getattr(context, "agent_card_errors", None)
    if not warnings:
        return []
    cleaned: list[str] = []
    for message in warnings:
        message_text = str(message).strip()
        if message_text:
            cleaned.append(message_text)
    return cleaned


async def _lookup_and_format_providers(model: str) -> str | None:
    """Look up inference providers for a model and return a formatted message.

    Returns None if the model is not a HuggingFace model (no '/').
    """
    from fast_agent.llm.hf_inference_lookup import (
        format_provider_help_message,
        lookup_inference_providers,
        normalize_hf_model_id,
    )

    model_id = normalize_hf_model_id(model)
    if model_id is None:
        return None

    try:
        result = await lookup_inference_providers(model_id)
        return format_provider_help_message(result)
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
        self._record_agent_card_warnings()

    def _record_agent_card_warnings(self) -> None:
        for warning in _collect_agent_card_warnings(self._context):
            self._record_warning(warning)

    async def attach_llm(self, llm_factory, model=None, request_params=None, **kwargs):
        """Override to set up wizard callback after LLM is attached."""
        llm = await super().attach_llm(llm_factory, model, request_params, **kwargs)

        # Set up wizard callback if LLM supports it
        callback_setter = getattr(llm, "set_completion_callback", None)
        if callback_setter is not None:
            callback_setter(self._on_wizard_complete)

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
            "reset": ACPCommand(
                description="Reset the local .fast-agent directory",
                input_hint="confirm",
                handler=self._handle_reset,
            ),
            "quickstart": ACPCommand(
                description="Install example agent and tool cards to .fast-agent directory",
                handler=self._handle_quickstart,
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
        from fast_agent.llm.hf_inference_lookup import validate_hf_model
        from fast_agent.llm.model_factory import ModelFactory

        raw_model = arguments.strip()
        model = raw_model
        if not model:
            return format_model_list_help()

        alias_info = _resolve_alias_display(raw_model)

        # Normalize the model string (auto-add hf. prefix if needed)
        model = _normalize_hf_model(model)

        # Validate the model string format
        try:
            ModelFactory.parse_model_string(model)
        except Exception as e:
            return f"Error: Invalid model `{model}` - {e}"

        # Validate model exists on HuggingFace and has providers
        validation = await validate_hf_model(model, aliases=ModelFactory.MODEL_ALIASES)
        if not validation.valid:
            return validation.error or "Error: Model validation failed"

        try:
            update_model_in_config(model)
            applied = await self._apply_model_to_running_hf_agent(model)
            applied_note = "\n\nApplied to the running Hugging Face agent." if applied else ""
            provider_prefix = (
                f"{validation.display_message}\n\n" if validation.display_message else ""
            )
            if alias_info:
                alias_display, resolved_alias = alias_info
                model_status = f"Active model set to: `{alias_display}` (`{resolved_alias}`)"
            else:
                model_status = f"Default model set to: `{model}`"
            return (
                f"{provider_prefix}"
                f"{model_status}\n\nConfig file updated: `{CONFIG_FILE}`"
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

    async def _handle_reset(self, arguments: str) -> str:
        """Handler for /reset command."""
        confirmation = arguments.strip().lower()
        if confirmation not in {"confirm", "yes", "y"}:
            return (
                "This will delete the local `.fast-agent` directory and recreate it empty.\n\n"
                "To proceed, run:\n"
                "`/reset confirm`"
            )

        base_dir = Path(self.config.cwd) if self.config.cwd else Path.cwd()
        target_dir = (base_dir / ".fast-agent").resolve()

        try:
            if target_dir.exists():
                shutil.rmtree(target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "agent-cards").mkdir(parents=True, exist_ok=True)
            (target_dir / "tool-cards").mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            return f"Error resetting `{target_dir}`: {exc}"

        return (
            "Reset complete.\n\n"
            f"Recreated `{target_dir}` with empty `agent-cards` and `tool-cards` folders."
        )

    async def _handle_quickstart(self, arguments: str) -> str:
        """Handler for /quickstart command - copy toad-cards to .fast-agent directory."""
        base_dir = Path(self.config.cwd) if self.config.cwd else Path.cwd()
        target_dir = (base_dir / ".fast-agent").resolve()

        # Parse arguments for force flag
        args = arguments.strip().lower()
        force = args in ("force", "--force", "-f")

        # Check if directory exists with content
        if target_dir.exists() and not force:
            existing_items = []
            if (target_dir / "agent-cards").exists():
                agent_cards = list((target_dir / "agent-cards").glob("*.md"))
                if agent_cards:
                    existing_items.append("agent-cards")
            if (target_dir / "tool-cards").exists():
                tool_cards = list((target_dir / "tool-cards").glob("*.md"))
                if tool_cards:
                    existing_items.append("tool-cards")

            if existing_items:
                return (
                    f"`.fast-agent` directory already exists with: {', '.join(existing_items)}\n\n"
                    "Use `/quickstart force` to overwrite existing files."
                )

        # Attempt to copy from fast-agent-mcp package resources
        try:
            created = self._copy_toad_cards_from_resources(target_dir, force)
        except Exception as exc:
            return f"Error installing examples: {exc}"

        if not created:
            return "No files were copied. The example resources may not be available."

        # Generate success message
        lines = [
            "## Quickstart Examples Installed",
            "",
            f"Installed {len(created)} files to `.fast-agent/`",
            "",
            "**Directory structure:**",
            "```",
            ".fast-agent/",
            "├── agent-cards/          # Agent definitions (loaded automatically)",
            "├── tool-cards/           # Tool definitions (loaded automatically)",
            "├── shared/               # Shared context snippets",
            "├── skills/               # Tool definitions (loaded on-demand)",
            "```",
            "",
            "The cards are now installed. Restart Toad and use `ctrl+o` to switch Agents.",
            "",
            "**Available agent cards:**",
        ]

        # List agent cards
        agent_cards_dir = target_dir / "agent-cards"
        if agent_cards_dir.exists():
            for card_file in sorted(agent_cards_dir.glob("*.md")):
                lines.append(f"- `{card_file.stem}`")

        return "\n".join(lines)

    def _copy_toad_cards_from_resources(self, target_dir: Path, force: bool) -> list[str]:
        """Copy toad-cards from fast-agent-mcp package resources."""
        created = copy_toad_cards_from_resources(target_dir, force)
        if not created:
            raise RuntimeError(
                "Example files not found. Ensure fast-agent-mcp is installed correctly."
            )
        return created

    async def _get_model_provider_info(self, model: str) -> str | None:
        """Get a brief provider info string for a model.

        Returns None if providers cannot be looked up or model is not a HF model.
        """
        from fast_agent.llm.hf_inference_lookup import (
            format_provider_summary,
            lookup_inference_providers,
            normalize_hf_model_id,
        )

        model_id = normalize_hf_model_id(model)
        if model_id is None:
            return None

        try:
            result = await lookup_inference_providers(model_id)
            return format_provider_summary(result)
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
        self._record_agent_card_warnings()

    def _record_agent_card_warnings(self) -> None:
        for warning in _collect_agent_card_warnings(self._context):
            self._record_warning(warning)

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
            await _send_connect_update(title="Rebuilding system prompt…", status="in_progress")
            await self._apply_instruction_templates()

            # Get available tools
            await _send_connect_update(title="Fetching available tools…", status="in_progress")
            tools_result = await self._aggregator.list_tools()
            tool_names = [t.name for t in tools_result.tools] if tools_result.tools else []

            # Send final progress update (but don't mark as completed yet -
            # the return value serves as the completion signal)
            if tool_names:
                await _send_connect_update(
                    title=f"Connected ({len(tool_names)} tools)",
                    status="completed",
                )
            else:
                await _send_connect_update(
                    title="Connected (no tools)",
                    status="completed",
                )

            if tool_names:
                tool_list = "\n".join(f"- `{name}`" for name in tool_names[:10])
                more = f"\n- ... and {len(tool_names) - 10} more" if len(tool_names) > 10 else ""
                return (
                    "Connected to Hugging Face MCP server.\n\n"
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
        from fast_agent.llm.hf_inference_lookup import validate_hf_model
        from fast_agent.llm.model_factory import ModelFactory

        raw_model = arguments.strip()
        model = raw_model
        if not model:
            return format_model_list_help()

        alias_info = _resolve_alias_display(raw_model)

        # Normalize the model string (auto-add hf. prefix if needed)
        model = _normalize_hf_model(model)

        # Validate the model string format
        try:
            ModelFactory.parse_model_string(model)
        except Exception as e:
            return f"Error: Invalid model `{model}` - {e}"

        # Validate model exists on HuggingFace and has providers
        validation = await validate_hf_model(model, aliases=ModelFactory.MODEL_ALIASES)
        if not validation.valid:
            return validation.error or "Error: Model validation failed"

        try:
            # Apply model first - if this fails, don't update config
            await self.apply_model(model)
            update_model_in_config(model)
            provider_prefix = (
                f"{validation.display_message}\n\n" if validation.display_message else ""
            )
            if alias_info:
                alias_display, resolved_alias = alias_info
                model_status = f"Active model set to: `{alias_display}` (`{resolved_alias}`)"
            else:
                model_status = f"Active model set to: `{model}`"
            return f"{provider_prefix}{model_status}\n\nConfig file updated: `{CONFIG_FILE}`"
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
        # Also exclude `maxTokens` so the new model can determine its own value from ModelDatabase.
        self.config.model = new_model
        if self.config.default_request_params is not None:
            params_without_model = self.config.default_request_params.model_dump(
                exclude={"model", "maxTokens"}
            )
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
