import sys

from mcp_agent.cli.constants import GO_SPECIFIC_OPTIONS, KNOWN_SUBCOMMANDS
from mcp_agent.cli.main import app

# if the arguments would work with "go" we'll just route to it


def main():
    """Main entry point that handles auto-routing to 'go' command."""
    # Check if we should auto-route to 'go'
    if len(sys.argv) > 1:
        # Check if first arg is not already a subcommand
        first_arg = sys.argv[1]

        if first_arg not in KNOWN_SUBCOMMANDS and any(
            arg in sys.argv or any(arg.startswith(opt + "=") for opt in GO_SPECIFIC_OPTIONS)
            for arg in sys.argv
        ):
            # Find where to insert 'go' - before the first go-specific option
            insert_pos = 1
            for i, arg in enumerate(sys.argv[1:], 1):
                if arg in GO_SPECIFIC_OPTIONS or any(arg.startswith(opt + "=") for opt in GO_SPECIFIC_OPTIONS):
                    insert_pos = i
                    break
            # Auto-route to go command
            sys.argv.insert(insert_pos, "go")

    app()


if __name__ == "__main__":
    main()
