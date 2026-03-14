"""Utilities for detecting keyring availability and write access."""

from __future__ import annotations

import os
import secrets
import sys
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class KeyringStatus:
    name: str
    available: bool
    writable: bool


_KEYRING_ACCESS_NOTICE_LOCK = threading.Lock()
_KEYRING_ACCESS_NOTICE_SHOWN = False


def format_keyring_access_notice(*, purpose: str | None = None) -> str:
    """Return the standard one-time keyring access notice."""
    message = (
        "fast-agent is accessing the OS keyring for stored tokens. "
        "Some platforms may pause and show a prompt."
    )
    if purpose:
        return f"{message} ({purpose})"
    return message


def _keyring_access_notice_enabled() -> bool:
    suppress_notice = os.getenv("FAST_AGENT_KEYRING_NOTICE", "1").strip().lower()
    return suppress_notice not in {"0", "false", "no", "off"}


def emit_keyring_access_notice(
    *,
    purpose: str | None = None,
    emitter: Callable[[str], None] | None = None,
) -> bool:
    """Emit the one-time keyring access notice through the supplied emitter."""
    global _KEYRING_ACCESS_NOTICE_SHOWN

    if _KEYRING_ACCESS_NOTICE_SHOWN or not _keyring_access_notice_enabled():
        return False

    with _KEYRING_ACCESS_NOTICE_LOCK:
        if _KEYRING_ACCESS_NOTICE_SHOWN or not _keyring_access_notice_enabled():
            return False

        message = format_keyring_access_notice(purpose=purpose)

        if emitter is None:
            try:
                if not sys.stderr.isatty():
                    return False
            except Exception:
                return False

            def _stderr_emitter(text: str) -> None:
                sys.stderr.write(f"{text}\n")
                sys.stderr.flush()

            emitter = _stderr_emitter

        try:
            emitter(message)
        except Exception:
            return False

        _KEYRING_ACCESS_NOTICE_SHOWN = True
        return True


def maybe_print_keyring_access_notice(*, purpose: str | None = None) -> None:
    """Print a one-time note before first keyring access in interactive sessions."""
    emit_keyring_access_notice(purpose=purpose)


def _probe_keyring_write(service: str) -> bool:
    try:
        maybe_print_keyring_access_notice(purpose="checking keyring availability")
        import keyring

        probe_key = f"probe:{secrets.token_urlsafe(8)}"
        keyring.set_password(service, probe_key, "probe")
        try:
            keyring.delete_password(service, probe_key)
        except Exception:
            # If deletion fails but set succeeded, still treat as writable.
            pass
        return True
    except Exception:
        return False


def get_keyring_status() -> KeyringStatus:
    try:
        maybe_print_keyring_access_notice(purpose="checking keyring backend")
        import keyring

        backend = keyring.get_keyring()
        name = getattr(backend, "name", backend.__class__.__name__)
        available = True
        try:
            from keyring.backends.fail import Keyring as FailKeyring

            available = not isinstance(backend, FailKeyring)
        except Exception:
            available = True
        writable = _probe_keyring_write("fast-agent-keyring-probe") if available else False
        return KeyringStatus(name=name, available=available, writable=writable)
    except Exception:
        return KeyringStatus(name="unavailable", available=False, writable=False)
