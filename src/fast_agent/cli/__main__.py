import asyncio
import json
import sys

from fast_agent.cli.constants import GO_SPECIFIC_OPTIONS, KNOWN_SUBCOMMANDS
from fast_agent.cli.main import app

# if the arguments would work with "go" we'll just route to it


def main():
    """Main entry point that handles auto-routing to 'go' command."""
    try:
        loop = asyncio.get_event_loop()

        def _log_asyncio_exception(loop: asyncio.AbstractEventLoop, context: dict) -> None:
            import logging

            logger = logging.getLogger("fast_agent.asyncio")

            message = context.get("message", "(no message)")
            task = context.get("task")
            future = context.get("future")
            handle = context.get("handle")
            source_traceback = context.get("source_traceback")
            exception = context.get("exception")

            details = {
                "message": message,
                "task": repr(task) if task else None,
                "future": repr(future) if future else None,
                "handle": repr(handle) if handle else None,
                "source_traceback": [str(frame) for frame in source_traceback] if source_traceback else None,
            }

            logger.error("Unhandled asyncio error: %s", message)
            logger.error("Asyncio context: %s", json.dumps(details, indent=2))

            if exception:
                logger.exception("Asyncio exception", exc_info=exception)

        loop.set_exception_handler(_log_asyncio_exception)
    except RuntimeError:
        # No running loop yet (rare for sync entry), safe to ignore
        pass
    # Check if we should auto-route to 'go'
    if len(sys.argv) > 1:
        # Check if first arg is not already a subcommand
        first_arg = sys.argv[1]

        # Only auto-route if any known go-specific options are present
        has_go_options = any(
            (arg in GO_SPECIFIC_OPTIONS) or any(arg.startswith(opt + "=") for opt in GO_SPECIFIC_OPTIONS)
            for arg in sys.argv[1:]
        )

        if first_arg not in KNOWN_SUBCOMMANDS and has_go_options:
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
