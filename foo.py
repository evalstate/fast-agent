import asyncio
import os
import sys


def _trigger_blocking_error(stdout_fd: int) -> None:
    # Redirect stdout to a nonblocking pipe to trigger BlockingIOError quickly
    # without spamming the terminal.
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    os.dup2(write_fd, stdout_fd)
    os.close(write_fd)

    chunk = b"x" * 32768
    for _ in range(3):
        os.write(stdout_fd, chunk)

    raise RuntimeError("Did not trigger BlockingIOError after filling pipe buffer")


def main() -> None:
    #    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stdin_fd = sys.stdin.fileno()
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    print(
        "before:",
        os.get_blocking(stdin_fd),
        os.get_blocking(stdout_fd),
        os.get_blocking(stderr_fd),
    )

    loop.add_reader(stdin_fd, lambda: None)

    print(
        "after add_reader:",
        os.get_blocking(stdin_fd),
        os.get_blocking(stdout_fd),
        os.get_blocking(stderr_fd),
    )

    loop.remove_reader(stdin_fd)

    print(
        "after remove_reader:",
        os.get_blocking(stdin_fd),
        os.get_blocking(stdout_fd),
        os.get_blocking(stderr_fd),
    )

    try:
        _trigger_blocking_error(stdout_fd)
    except BlockingIOError:
        # Ensure traceback is visible even if stderr is non-blocking.
        try:
            os.set_blocking(stderr_fd, True)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
