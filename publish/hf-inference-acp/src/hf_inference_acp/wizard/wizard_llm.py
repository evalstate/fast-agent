"""Wizard-style setup LLM for HuggingFace inference configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import create_turn_usage_from_messages
from hf_inference_acp.hf_config import (
    CONFIG_FILE,
    has_hf_token,
    update_mcp_server_load_on_start,
    update_model_in_config,
)
from hf_inference_acp.wizard.model_catalog import CURATED_MODELS
from hf_inference_acp.wizard.stages import WizardStage, WizardState

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.llm.fastagent_llm import RequestParams
    from fast_agent.types import PromptMessageExtended

logger = get_logger(__name__)


class WizardSetupLLM(PassthroughLLM):
    """
    A wizard-style LLM that guides users through HF setup.

    Unlike PassthroughLLM which echoes input, this drives a
    structured setup flow with state management.
    """

    def __init__(
        self,
        provider: Provider = Provider.FAST_AGENT,
        name: str = "HFSetupWizard",
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(name=name, provider=provider, **kwargs)
        self._state = WizardState()
        self._on_complete_callback: Callable[["WizardState"], Any] | None = None
        self._completion_callback_fired = False
        self.logger = get_logger(__name__)

    def set_completion_callback(self, callback: Callable[["WizardState"], Any]) -> None:
        """Set callback to be called when wizard completes."""
        self._on_complete_callback = callback

    async def _apply_prompt_provider_specific(
        self,
        multipart_messages: list[PromptMessageExtended],
        request_params: "RequestParams | None" = None,
        tools: list[Tool] | None = None,
        is_template: bool = False,
    ) -> PromptMessageExtended:
        """Process user input through the wizard state machine."""
        # Add messages to history
        self.history.extend(multipart_messages, is_prompt=is_template)

        last_message = multipart_messages[-1]

        # If already an assistant response, return as-is
        if last_message.role == "assistant":
            return last_message

        # Get user input
        user_input = last_message.first_text().strip()

        # Check for slash commands - passthrough for handler
        if user_input.startswith("/"):
            result = Prompt.assistant(user_input)
            self._track_usage(multipart_messages, result)
            return result

        # Process through wizard state machine
        response = await self._process_stage(user_input)

        result = Prompt.assistant(response)
        self._track_usage(multipart_messages, result)
        return result

    def _reset_wizard(self) -> None:
        """Reset wizard state so the flow can be re-run."""
        self._state = WizardState()
        self._completion_callback_fired = False

    def _is_restart_command(self, cmd: str) -> bool:
        """Return True if the user intends to restart the wizard."""
        if cmd in ("restart", "reset"):
            return True
        # Convenience: after completion, users often try "go"/"setup" to re-run.
        if self._state.stage == WizardStage.COMPLETE and cmd in ("go", "setup", "start", "begin"):
            return True
        return False

    def _track_usage(
        self,
        input_messages: list[PromptMessageExtended],
        result: PromptMessageExtended,
    ) -> None:
        """Track usage for billing/analytics."""
        tool_call_count = len(result.tool_calls) if result.tool_calls else 0
        turn_usage = create_turn_usage_from_messages(
            input_content=input_messages[-1].all_text(),
            output_content=result.all_text(),
            model="wizard-setup",
            model_type="wizard-setup",
            tool_calls=tool_call_count,
            delay_seconds=0.0,
        )
        self.usage_accumulator.add_turn(turn_usage)

    async def _process_stage(self, user_input: str) -> str:
        """Process current stage and return response."""
        cmd = user_input.lower().strip()

        if self._is_restart_command(cmd):
            self._reset_wizard()
            # Treat the restart request as the first message of a fresh wizard.
        return await self._process_stage_inner(user_input)

    async def _process_stage_inner(self, user_input: str) -> str:
        """Internal stage processor."""
        # Handle first message - show welcome, but if user already typed a
        # recognized command, process it immediately without making them repeat
        if self._state.first_message:
            self._state.first_message = False
            cmd = user_input.lower().strip()
            if cmd in ("go", "start", "begin", "y", "yes", "ok"):
                # User already said go, skip welcome and proceed
                self._state.stage = WizardStage.TOKEN_CHECK
                return await self._handle_token_check(user_input)
            elif cmd == "skip":
                # User wants to skip wizard
                self._state.stage = WizardStage.WELCOME
                return await self._handle_welcome(user_input)
            else:
                # Show welcome for any other input
                return self._render_welcome()

        # Route to appropriate handler based on current stage
        handlers = {
            WizardStage.WELCOME: self._handle_welcome,
            WizardStage.TOKEN_CHECK: self._handle_token_check,
            WizardStage.TOKEN_GUIDE: self._handle_token_guide,
            WizardStage.TOKEN_VERIFY: self._handle_token_verify,
            WizardStage.MODEL_SELECT: self._handle_model_select,
            WizardStage.MCP_CONNECT: self._handle_mcp_connect,
            WizardStage.CONFIRM: self._handle_confirm,
            WizardStage.COMPLETE: self._handle_complete,
        }

        handler = handlers.get(self._state.stage)
        if handler:
            return await handler(user_input)
        return "Unknown wizard state. Type 'restart' to begin again."

    def _render_welcome(self) -> str:
        """Render the welcome message."""
        self._state.stage = WizardStage.WELCOME
        return """# Hugging Face Inference Providers Setup Wizard

---

Welcome! This wizard will help you configure:

1. Your Hugging Face token (required for API access)
2. Your default inference model
3. Whether to connect to the Hugging Face MCP server on startup

Type `go` to begin, or `skip` to use slash commands instead.
"""

    async def _handle_welcome(self, user_input: str) -> str:
        """Handle welcome stage input."""
        cmd = user_input.lower().strip()
        if cmd == "skip":
            return """
Wizard mode skipped. You can use these commands:

  /login     - Get instructions for setting up your token
  /set-model - Set the default model
  /check     - Verify your configuration
  /skills add - Install skills from the marketplace

Type any command to continue.
"""
        elif cmd in ("go", "start", "begin", "y", "yes", "ok"):
            # Proceed to token check
            self._state.stage = WizardStage.TOKEN_CHECK
            return await self._handle_token_check(user_input)
        else:
            return self._render_welcome()

    async def _handle_token_check(self, user_input: str) -> str:
        """Check if HF_TOKEN is present and route accordingly."""
        if has_hf_token():
            # Token present, verify it
            self._state.stage = WizardStage.TOKEN_VERIFY
            return await self._handle_token_verify(user_input)
        else:
            # No token, show guide
            self._state.stage = WizardStage.TOKEN_GUIDE
            return self._render_token_guide()

    def _render_token_guide(self) -> str:
        """Render token setup instructions."""
        return f"""## Step 1 - Hugging Face Token Setup

Your Hugging Face token is not configured.

Set it using one of these methods:

**Option A** - Run `hf auth login`. In Toad, type `$hf auth login`

**Option B** - Set environment variable:
    `export HF_TOKEN=hf_YOUR_TOKEN` and restart Toad

**Option C** - Add to config file `{CONFIG_FILE}`:

```yaml
hf:
    api_key: hf_your_token_here
```

Create a `READ` or `WRITE` token at: https://huggingface.co/settings/tokens

---

Type `check` after setting your token to continue.
"""

    async def _handle_token_guide(self, user_input: str) -> str:
        """Handle token guide stage input."""
        cmd = user_input.lower().strip()

        if cmd in ("check", "verify"):
            # Re-check token
            self._state.stage = WizardStage.TOKEN_CHECK
            return await self._handle_token_check(user_input)
        elif cmd in ("quit", "exit", "q"):
            return "Setup cancelled. Run the agent again when you're ready to continue."
        else:
            return self._render_token_guide()

    async def _handle_token_verify(self, user_input: str) -> str:
        """Verify the HF token by calling the API."""
        try:
            from huggingface_hub import whoami

            user_info = whoami()
            username = user_info.get("name", "unknown")
            self._state.token_verified = True
            self._state.hf_username = username

            # Token is already available via huggingface_hub (from hf auth login or HF_TOKEN env)
            # ProviderKeyManager discovers it automatically, no need to copy to config file

            # Move to model selection
            self._state.stage = WizardStage.MODEL_SELECT
            return f"""## Step 1 - Hugging Face Token Setup

Token verified - connected as: `{username}`

{self._render_model_selection()}"""
        except Exception as e:
            self._state.token_verified = False
            self._state.error_message = str(e)
            self._state.stage = WizardStage.TOKEN_GUIDE
            return f"""
Token verification failed: {e}

{self._render_token_guide()}"""

    def _render_model_selection(self) -> str:
        """Render model selection prompt."""
        lines = [
            "## Step 2 : Select Default Model",
            "",
            "Choose your default inference model by entering a number:",
            "",
        ]

        for i, model in enumerate(CURATED_MODELS, start=1):
            lines.append(f"  {i}. {model.display_name} (`{model.id}`)")
            lines.append(f"     {model.description}")
            lines.append("")

        custom_index = len(CURATED_MODELS) + 1
        lines.extend(
            [
                f"  {custom_index}. Custom model (enter model ID manually)",
                "",
                "---",
                f"Enter a number (1-{custom_index}), a curated ID (e.g. `kimi`), or type a model ID directly:",
                "",
            ]
        )
        return "\n".join(lines)

    async def _handle_model_select(self, user_input: str) -> str:
        """Handle model selection input."""
        user_input = user_input.strip()

        custom_index = len(CURATED_MODELS) + 1
        if user_input == str(custom_index):
            # Custom model entry
            return """
Enter the full model ID (e.g., hf.organization/model-name):
"""

        if user_input.isdigit():
            selection = int(user_input)
            if 1 <= selection <= len(CURATED_MODELS):
                chosen = CURATED_MODELS[selection - 1]
                # Store the curated ID (shortform alias) in config.
                self._state.selected_model = chosen.id
                self._state.selected_model_display = chosen.display_name
            else:
                return f"Invalid selection: '{user_input}'\n\n{self._render_model_selection()}"
        elif any(m.id.lower() == user_input.lower() for m in CURATED_MODELS):
            chosen = next(m for m in CURATED_MODELS if m.id.lower() == user_input.lower())
            self._state.selected_model = chosen.id
            self._state.selected_model_display = chosen.display_name
        elif user_input.startswith("hf.") or "/" in user_input:
            # Direct model ID entry
            if not user_input.startswith("hf."):
                user_input = f"hf.{user_input}"
            self._state.selected_model = user_input
            self._state.selected_model_display = user_input
        else:
            return f"Invalid selection: '{user_input}'\n\n{self._render_model_selection()}"

        # Skip skills selection step and move to MCP connection
        self._state.stage = WizardStage.MCP_CONNECT
        return self._render_mcp_connect()

    def _render_mcp_connect(self) -> str:
        """Render MCP server connection prompt."""
        return """## Step 3 - Hugging Face MCP Server

The Hugging Face MCP server provides additional tools for working with
models, datasets, and spaces on Hugging Face.

Would you like to connect to the Hugging Face MCP server on startup?

- [y] Yes - connect automatically on startup
- [n] No - I'll connect manually when needed (use /connect)

Enter y or n:
"""

    async def _handle_mcp_connect(self, user_input: str) -> str:
        """Handle MCP connect step input."""
        cmd = user_input.lower().strip()

        if cmd in ("y", "yes"):
            self._state.mcp_load_on_start = True
            self._state.stage = WizardStage.CONFIRM
            return "\n".join(
                [
                    "## Skills (Optional)",
                    "",
                    "Skills are available. Use `/skills add` to install.",
                    "",
                    self._render_confirmation(),
                ]
            )
        elif cmd in ("n", "no"):
            self._state.mcp_load_on_start = False
            self._state.stage = WizardStage.CONFIRM
            return "\n".join(
                [
                    "## Skills (Optional)",
                    "",
                    "Skills are available. Use `/skills add` to install.",
                    "",
                    self._render_confirmation(),
                ]
            )
        elif cmd in ("quit", "exit", "q"):
            return "Setup cancelled. Your configuration was not changed."
        else:
            return self._render_mcp_connect()

    def _render_confirmation(self) -> str:
        """Render confirmation prompt."""
        mcp_status = "Yes" if self._state.mcp_load_on_start else "No"
        return f"""## Confirm Settings

- **Model**: {self._state.selected_model_display}
  `{self._state.selected_model}`
- **MCP server on startup**: {mcp_status}

- [y] Confirm and save
- [c] Change model selection
- [q] Quit without saving
"""

    async def _handle_confirm(self, user_input: str) -> str:
        """Handle confirmation input."""
        cmd = user_input.lower().strip()

        if cmd in ("c", "change", "back"):
            self._state.stage = WizardStage.MODEL_SELECT
            return self._render_model_selection()
        elif cmd in ("q", "quit", "exit"):
            return "Setup cancelled. Your configuration was not changed."
        elif cmd in ("y", "yes", "confirm", "ok", "save"):
            # Save configuration
            try:
                update_model_in_config(self._state.selected_model)
                update_mcp_server_load_on_start("huggingface", self._state.mcp_load_on_start)
                self._state.stage = WizardStage.COMPLETE
                return await self._handle_complete(user_input)
            except Exception as e:
                return f"Error saving configuration: {e}\n\nTry again or type 'q' to quit."
        else:
            return self._render_confirmation()

    async def _handle_complete(self, user_input: str) -> str:
        """Handle completion - show success and trigger callback."""
        if self._state.stage == WizardStage.COMPLETE and self._completion_callback_fired:
            return "Setup is already complete. Type `restart` to run the wizard again."

        # Call completion callback (once per completion) if set
        if self._on_complete_callback and not self._completion_callback_fired:
            try:
                await self._on_complete_callback(self._state)
                self._completion_callback_fired = True
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")
                self._completion_callback_fired = True

        mcp_status = "Yes" if self._state.mcp_load_on_start else "No"
        return f"""## Setup Complete!

Your configuration has been saved:
  - Token: verified (connected as `{self._state.hf_username or "unknown"}`)
  - Model: `{self._state.selected_model}`
  - MCP server on startup: {mcp_status}

You're now ready to use the Hugging Face assistant!

Some tips:
 - `AGENTS.md` and `huggingface.md` are automatically loaded in the System Prompt
 - You can include content from URLs with `{{{{url:https://gist.github.com/...}}}}` syntax
 - Tool Permissions are set on a per-project basis. use `/status auth` and `/status authreset` to manage. 
 - Customise the Hugging Face MCP Server at `https://huggingface.co/settings/mcp`
 - Join https://huggingface.co/toad-hf-inference-explorers to claim **$10** in free inference credits!

Transferring to chat mode...

(If you need to re-run this wizard later, return to `setup` mode and type `go` or `setup`.)
"""
