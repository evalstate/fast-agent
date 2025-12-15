"""Wizard-style setup LLM for HuggingFace inference configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mcp import CallToolRequest
from mcp.types import CallToolRequestParams

from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt import Prompt
from fast_agent.llm.internal.passthrough import PassthroughLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import create_turn_usage_from_messages
from fast_agent.types import PromptMessageExtended
from fast_agent.types.llm_stop_reason import LlmStopReason
from hf_inference_acp.hf_config import has_hf_token, update_model_in_config
from hf_inference_acp.wizard.stages import WizardStage, WizardState

if TYPE_CHECKING:
    from mcp import Tool

    from fast_agent.llm.fastagent_llm import RequestParams

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

        # Handlers can return either a string or a PromptMessageExtended (for tool calls)
        if isinstance(response, PromptMessageExtended):
            result = response
        else:
            result = Prompt.assistant(response)

        self._track_usage(multipart_messages, result)
        return result

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

    async def _process_stage(self, user_input: str) -> str | PromptMessageExtended:
        """Process current stage and return response (string or PromptMessageExtended for tool calls)."""
        # Handle first message - show welcome
        if self._state.first_message:
            self._state.first_message = False
            return self._render_welcome()

        # Route to appropriate handler based on current stage
        handlers = {
            WizardStage.WELCOME: self._handle_welcome,
            WizardStage.TOKEN_CHECK: self._handle_token_check,
            WizardStage.TOKEN_GUIDE: self._handle_token_guide,
            WizardStage.TOKEN_VERIFY: self._handle_token_verify,
            WizardStage.MODEL_SELECT: self._handle_model_select,
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
1. Your default inference model

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
        return """## Step 1 -  Hugging Face Token Setup

Your Hugging Face token is not configured.

Options:
  [1] Run interactive login (hf auth login)
  [2] I'll set HF_TOKEN manually

Enter 1 or 2, or type `check` after setting your token:
"""

    def _render_manual_token_instructions(self) -> str:
        """Render manual token setup instructions."""
        return """
To set up your token manually:

  1. Visit: https://huggingface.co/settings/tokens
  2. Create a new token with "Read" access
  3. Set it using one of these methods:

Option A - Environment variable:
    export HF_TOKEN=hf_your_token_here

Option B - CLI login (in a separate terminal):
    hf auth login

--------------------------------------------------------------------------------
Type `check` to verify your token, or `1` to run interactive login.
"""

    async def _handle_token_guide(self, user_input: str) -> str | PromptMessageExtended:
        """Handle token guide stage input."""
        cmd = user_input.lower().strip()

        if cmd == "1" or cmd == "login":
            # Return a proper tool call request for interactive login
            # After the tool runs, user will need to type 'check' to continue
            self._state.awaiting_login_result = True
            tool_call_id = f"hf_login_{self._correlation_id}"
            self._correlation_id += 1
            return Prompt.assistant(
                "Running `hf auth login`... Follow the prompts in the terminal.\n\n"
                "Type `check` when done to verify your token.",
                tool_calls={
                    tool_call_id: CallToolRequest(
                        method="tools/call",
                        params=CallToolRequestParams(
                            name="execute",
                            arguments={"command": "hf auth login"},
                        ),
                    )
                },
                stop_reason=LlmStopReason.TOOL_USE,
            )
        elif cmd == "2" or cmd == "manual":
            return self._render_manual_token_instructions()
        elif cmd in ("check", "verify"):
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
            return f"""Token verified - connected as: `{username}`
            
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
        return """## Step 2 : Select Default Model
================================================================================

Choose your default inference model by entering a number:

  1. Kimi K2 Instruct (moonshotai/Kimi-K2-Instruct-0905)
     Fast, capable instruct model - good general purpose choice

  2. DeepSeek R1 (deepseek-ai/DeepSeek-R1)
     Advanced reasoning model with strong capabilities

  3. Qwen3 235B (Qwen/Qwen3-235B-A22B)
     Large parameter model, high capability

  4. Llama 4 Maverick 17B (meta-llama/Llama-4-Maverick-17B-128E-Instruct)
     Meta's latest Llama model with strong performance

  5. Custom model (enter model ID manually)

--------------------------------------------------------------------------------
Enter a number (1-5) or type a model ID directly:
"""

    async def _handle_model_select(self, user_input: str) -> str:
        """Handle model selection input."""
        user_input = user_input.strip()

        # Map numbers to models
        model_map = {
            "1": "hf.moonshotai/Kimi-K2-Instruct-0905",
            "2": "hf.deepseek-ai/DeepSeek-R1",
            "3": "hf.Qwen/Qwen3-235B-A22B",
            "4": "hf.meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        }

        display_map = {
            "1": "Kimi K2 Instruct",
            "2": "DeepSeek R1",
            "3": "Qwen3 235B",
            "4": "Llama 4 Maverick 17B",
        }

        if user_input == "5":
            # Custom model entry
            return """
Enter the full model ID (e.g., hf.organization/model-name):
"""

        if user_input in model_map:
            self._state.selected_model = model_map[user_input]
            self._state.selected_model_display = display_map[user_input]
        elif user_input.startswith("hf.") or "/" in user_input:
            # Direct model ID entry
            if not user_input.startswith("hf."):
                user_input = f"hf.{user_input}"
            self._state.selected_model = user_input
            self._state.selected_model_display = user_input
        else:
            return f"Invalid selection: '{user_input}'\n\n{self._render_model_selection()}"

        # Move to confirmation
        self._state.stage = WizardStage.CONFIRM
        return self._render_confirmation()

    def _render_confirmation(self) -> str:
        """Render confirmation prompt."""
        return f"""
================================================================================
        Confirm Selection
================================================================================

You selected: {self._state.selected_model_display}
Model ID: {self._state.selected_model}

[y] Confirm and save
[c] Change selection
[q] Quit without saving
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
                self._state.stage = WizardStage.COMPLETE
                return await self._handle_complete(user_input)
            except Exception as e:
                return f"Error saving configuration: {e}\n\nTry again or type 'q' to quit."
        else:
            return self._render_confirmation()

    async def _handle_complete(self, user_input: str) -> str:
        """Handle completion - show success and trigger callback."""
        # Call completion callback if set
        if self._on_complete_callback:
            try:
                await self._on_complete_callback(self._state)
            except Exception as e:
                self.logger.warning(f"Completion callback failed: {e}")

        return f"""
================================================================================
        Setup Complete!
================================================================================

Your configuration has been saved:
  - Token: verified (connected as {self._state.hf_username or "unknown"})
  - Model: {self._state.selected_model}

You're now ready to use the Hugging Face assistant!

Switching to chat mode...
"""
