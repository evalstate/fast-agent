"""Shared display helpers for Codex OAuth token status."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_CODEX_OAUTH_SOURCE_LABELS: dict[str, str] = {
    "keyring": "Keyring OAuth",
    "auth.json": "Codex auth.json",
}


def codex_oauth_source_label(source: object) -> str:
    if isinstance(source, str):
        return _CODEX_OAUTH_SOURCE_LABELS.get(source, "OAuth token")
    return "OAuth token"


def codex_oauth_source_display(codex_status: Mapping[str, object]) -> str:
    return f"[green]{codex_oauth_source_label(codex_status.get('source'))}[/green]"


def codex_oauth_expiry_display(
    codex_status: Mapping[str, object],
    *,
    datetime_type: type[datetime] = datetime,
) -> str:
    expires_at = codex_status.get("expires_at")
    if not isinstance(expires_at, int | float):
        return "[green]unknown[/green]"

    expires_display = datetime_type.fromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M")
    if codex_status.get("expired"):
        return f"[red]expired {expires_display}[/red]"
    return f"[green]{expires_display}[/green]"
