import os
import sys

from fast_agent.cli.asyncio_utils import set_asyncio_exception_handler
from fast_agent.cli.constants import (
    GO_SPECIFIC_OPTIONS,
    KNOWN_SUBCOMMANDS,
    normalize_resume_flag_args,
)
from fast_agent.cli.main import LAZY_SUBCOMMANDS, app
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV
from fast_agent.utils.async_utils import configure_uvloop, ensure_event_loop

# if the arguments would work with "go" we'll just route to it


def main():
    """Main entry point that handles auto-routing to 'go' command."""
    if os.getenv(FAST_AGENT_SHELL_CHILD_ENV):
        print(
            "fast-agent is already running inside a fast-agent shell command. "
            "Exit the shell or unset FAST_AGENT_SHELL_CHILD to continue.",
            file=sys.stderr,
        )
        sys.exit(1)
    requested_uvloop, enabled_uvloop = configure_uvloop()
    if requested_uvloop and not enabled_uvloop:
        print(
            "FAST_AGENT_UVLOOP is set but uvloop is unavailable; falling back to asyncio.",
            file=sys.stderr,
        )
    try:
        loop = ensure_event_loop()

        set_asyncio_exception_handler(loop)
    except RuntimeError:
        # No running loop yet (rare for sync entry), safe to ignore
        pass
    normalize_resume_flag_args(sys.argv, start_index=1)

    # Check if we should auto-route to 'go'
    if len(sys.argv) > 1:
        # Detect explicit subcommands even when global options (like --env)
        # appear before the command name.
        known_commands = set(KNOWN_SUBCOMMANDS) | set(LAZY_SUBCOMMANDS.keys())
        explicit_subcommand = next(
            (arg for arg in sys.argv[1:] if arg in known_commands),
            None,
        )

        # Only auto-route if any known go-specific options are present
        has_go_options = any(
            (arg in GO_SPECIFIC_OPTIONS) or any(arg.startswith(opt + "=") for opt in GO_SPECIFIC_OPTIONS)
            for arg in sys.argv[1:]
        )

        if (explicit_subcommand not in known_commands) and has_go_options:
            # Find where to insert 'go' - before the first go-specific option
            insert_pos = 1
            for i, arg in enumerate(sys.argv[1:], 1):
                if (arg in GO_SPECIFIC_OPTIONS) or any(
                    arg.startswith(opt + "=") for opt in GO_SPECIFIC_OPTIONS
                ):
                    insert_pos = i
                    break
            # Auto-route to go command
            sys.argv.insert(insert_pos, "go")

    app()


if __name__ == "__main__":
    main()
