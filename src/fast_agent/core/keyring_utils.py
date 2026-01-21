"""Utilities for detecting keyring availability and write access."""

from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyringStatus:
    name: str
    available: bool
    writable: bool


def _probe_keyring_write(service: str) -> bool:
    try:
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
        import keyring

        backend = keyring.get_keyring()
        name = getattr(backend, "name", backend.__class__.__name__)
        available = True
        try:
            from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

            available = not isinstance(backend, FailKeyring)
        except Exception:
            available = True
        writable = _probe_keyring_write("fast-agent-keyring-probe") if available else False
        return KeyringStatus(name=name, available=available, writable=writable)
    except Exception:
        return KeyringStatus(name="unavailable", available=False, writable=False)

