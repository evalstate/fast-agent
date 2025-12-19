"""
Skills Commands Mixin for ACP agents.

Provides slash commands for managing skills:
- /skills - List installed skills
- /skills add - Show marketplace and install skills
- /skills remove - Remove installed skills
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fast_agent.acp.acp_aware_mixin import ACPCommand
from fast_agent.constants import DEFAULT_SKILLS_INSTALL_DIR, DEFAULT_SKILLS_MARKETPLACE_URL
from fast_agent.core.logging.logger import get_logger
from fast_agent.skills.manager import (
    MarketplaceInfo,
    fetch_marketplace,
    format_installed_skills,
    format_marketplace_skills,
    install_skill,
    list_installed_skills,
    remove_skill,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class SkillsCommandState:
    """
    State for multi-step skills commands (like add/remove with selection).

    This tracks pending operations when a user needs to select a skill by number.
    """

    def __init__(self) -> None:
        self.pending_add_marketplace: MarketplaceInfo | None = None
        self.pending_remove_skills: list | None = None


class SkillsCommandsMixin:
    """
    Mixin providing skills management slash commands.

    Agents using this mixin can include skills commands in their acp_commands.
    The mixin requires access to:
    - self.context (for configuration)
    - self.rebuild_instruction_templates() (for updating system prompt after changes)
    """

    # State for pending operations
    _skills_state: SkillsCommandState | None = None

    def _get_skills_state(self) -> SkillsCommandState:
        """Get or create the skills command state."""
        if self._skills_state is None:
            self._skills_state = SkillsCommandState()
        return self._skills_state

    def _get_marketplace_url(self) -> str:
        """Get the configured marketplace URL."""
        try:
            ctx = getattr(self, "context", None)
            if ctx and hasattr(ctx, "config") and ctx.config:
                skills_settings = getattr(ctx.config, "skills", None)
                if skills_settings and skills_settings.marketplace_url:
                    return skills_settings.marketplace_url
        except Exception:
            pass
        return DEFAULT_SKILLS_MARKETPLACE_URL

    def _get_install_dir(self) -> str:
        """Get the skills installation directory."""
        return DEFAULT_SKILLS_INSTALL_DIR

    def _get_base_dir(self) -> Path:
        """Get the base directory for skill operations."""
        return Path.cwd()

    def get_skills_commands(self) -> dict[str, ACPCommand]:
        """
        Return the skills-related ACP commands.

        Call this from your agent's acp_commands property to include skills commands.
        """
        return {
            "skills": ACPCommand(
                description="Manage skills (list, add, remove)",
                input_hint="[add|remove] [name|number]",
                handler=self._handle_skills,
            ),
        }

    async def _handle_skills(self, arguments: str) -> str:
        """
        Handle the /skills command and its subcommands.

        Usage:
            /skills           - List installed skills
            /skills add       - Show marketplace skills
            /skills add <n>   - Install skill by number or name
            /skills remove    - Show installed skills for removal
            /skills remove <n> - Remove skill by number or name
        """
        args = arguments.strip().split(maxsplit=1)

        if not args:
            # /skills - List installed skills
            return await self._handle_skills_list()

        subcommand = args[0].lower()
        subargs = args[1] if len(args) > 1 else ""

        if subcommand == "add":
            return await self._handle_skills_add(subargs)
        elif subcommand == "remove":
            return await self._handle_skills_remove(subargs)
        elif subcommand == "q":
            # Clear any pending state
            state = self._get_skills_state()
            state.pending_add_marketplace = None
            state.pending_remove_skills = None
            return "Skills operation cancelled."
        else:
            # Try to interpret as a number for pending operations
            return await self._handle_skills_selection(subcommand)

    async def _handle_skills_list(self) -> str:
        """List installed skills."""
        manifests = list_installed_skills(
            base_dir=self._get_base_dir(),
            install_dir=self._get_install_dir(),
        )
        return format_installed_skills(manifests)

    async def _handle_skills_add(self, arguments: str) -> str:
        """Handle /skills add command."""
        state = self._get_skills_state()

        # If no arguments, show marketplace
        if not arguments.strip():
            return await self._show_marketplace()

        # Check if it's "q" to quit
        if arguments.strip().lower() == "q":
            state.pending_add_marketplace = None
            return "Skills installation cancelled."

        # Try to interpret as number or name
        return await self._install_skill_by_ref(arguments.strip())

    async def _show_marketplace(self) -> str:
        """Fetch and display the marketplace skills."""
        state = self._get_skills_state()

        try:
            marketplace_url = self._get_marketplace_url()
            marketplace = await fetch_marketplace(marketplace_url)
            state.pending_add_marketplace = marketplace

            # Get installed skill names for comparison
            installed = list_installed_skills(
                base_dir=self._get_base_dir(),
                install_dir=self._get_install_dir(),
            )
            installed_names = {m.name for m in installed}

            return format_marketplace_skills(marketplace, installed_names)

        except Exception as e:
            logger.error("Failed to fetch marketplace", data={"error": str(e)})
            return f"Failed to fetch skills marketplace: {e}"

    async def _install_skill_by_ref(self, ref: str) -> str:
        """Install a skill by number or name."""
        state = self._get_skills_state()
        marketplace = state.pending_add_marketplace

        # If no marketplace loaded, try to fetch it
        if not marketplace:
            try:
                marketplace_url = self._get_marketplace_url()
                marketplace = await fetch_marketplace(marketplace_url)
                state.pending_add_marketplace = marketplace
            except Exception as e:
                return f"Failed to fetch marketplace: {e}"

        if not marketplace.skills:
            return "No skills available in the marketplace."

        # Find the skill by number or name
        skill = None
        try:
            idx = int(ref) - 1  # Convert to 0-indexed
            if 0 <= idx < len(marketplace.skills):
                skill = marketplace.skills[idx]
        except ValueError:
            # Not a number, try by name
            ref_lower = ref.lower()
            for s in marketplace.skills:
                if s.name.lower() == ref_lower:
                    skill = s
                    break

        if not skill:
            return f"Skill '{ref}' not found. Use `/skills add` to see available skills."

        # Install the skill
        success, message = await install_skill(
            skill,
            marketplace,
            base_dir=self._get_base_dir(),
            install_dir=self._get_install_dir(),
            marketplace_url=self._get_marketplace_url(),
        )

        if success:
            # Clear pending state
            state.pending_add_marketplace = None

            # Rebuild system prompt
            await self._rebuild_after_skill_change()

            return f"✓ Installed skill: **{skill.name}**\n\n{message}\n\nSystem prompt updated."
        else:
            return f"✗ Failed to install skill: **{skill.name}**\n\n{message}"

    async def _handle_skills_remove(self, arguments: str) -> str:
        """Handle /skills remove command."""
        state = self._get_skills_state()

        # If no arguments, show installed skills
        if not arguments.strip():
            return await self._show_installed_for_removal()

        # Check if it's "q" to quit
        if arguments.strip().lower() == "q":
            state.pending_remove_skills = None
            return "Skills removal cancelled."

        # Try to interpret as number or name
        return await self._remove_skill_by_ref(arguments.strip())

    async def _show_installed_for_removal(self) -> str:
        """Show installed skills for removal selection."""
        state = self._get_skills_state()

        manifests = list_installed_skills(
            base_dir=self._get_base_dir(),
            install_dir=self._get_install_dir(),
        )

        if not manifests:
            return "No skills installed."

        state.pending_remove_skills = manifests

        lines = ["# Remove a Skill", ""]
        for idx, manifest in enumerate(manifests, 1):
            lines.append(f"**{idx}. {manifest.name}**")
            if manifest.description:
                lines.append(f"   {manifest.description}")
            lines.append("")

        lines.append("---")
        lines.append("Use `/skills remove <number>` or `/skills remove <name>` to remove a skill.")
        lines.append("Use `q` to cancel.")

        return "\n".join(lines)

    async def _remove_skill_by_ref(self, ref: str) -> str:
        """Remove a skill by number or name."""
        state = self._get_skills_state()

        # Get installed skills
        manifests = list_installed_skills(
            base_dir=self._get_base_dir(),
            install_dir=self._get_install_dir(),
        )

        if not manifests:
            return "No skills installed."

        # Find the skill by number or name
        skill_name = None
        try:
            idx = int(ref) - 1  # Convert to 0-indexed
            if 0 <= idx < len(manifests):
                skill_name = manifests[idx].name
        except ValueError:
            # Not a number, try by name
            ref_lower = ref.lower()
            for m in manifests:
                if m.name.lower() == ref_lower:
                    skill_name = m.name
                    break

        if not skill_name:
            return f"Skill '{ref}' not found. Use `/skills remove` to see installed skills."

        # Remove the skill
        success, message = remove_skill(
            skill_name,
            base_dir=self._get_base_dir(),
            install_dir=self._get_install_dir(),
        )

        if success:
            # Clear pending state
            state.pending_remove_skills = None

            # Rebuild system prompt
            await self._rebuild_after_skill_change()

            return f"✓ Removed skill: **{skill_name}**\n\nSystem prompt updated."
        else:
            return f"✗ Failed to remove skill: **{skill_name}**\n\n{message}"

    async def _handle_skills_selection(self, selection: str) -> str:
        """Handle a numeric selection when there's a pending operation."""
        state = self._get_skills_state()

        if state.pending_add_marketplace:
            return await self._install_skill_by_ref(selection)
        elif state.pending_remove_skills:
            return await self._remove_skill_by_ref(selection)
        else:
            return f"Unknown skills command: {selection}\n\nUse `/skills` to list installed skills."

    async def _rebuild_after_skill_change(self) -> None:
        """
        Rebuild instruction templates after a skill change.

        This updates the system prompt to include/exclude the changed skill.
        """
        # Check if the agent has rebuild_instruction_templates method
        rebuild_fn = getattr(self, "rebuild_instruction_templates", None)
        if callable(rebuild_fn):
            try:
                await rebuild_fn()
            except Exception as e:
                logger.warning(
                    "Failed to rebuild instruction templates after skill change",
                    data={"error": str(e)},
                )
