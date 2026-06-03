from __future__ import annotations

from datetime import datetime

from fast_agent.cli.codex_oauth_display import (
    codex_oauth_expiry_display,
    codex_oauth_source_display,
    codex_oauth_source_label,
)


def test_codex_oauth_source_label_handles_known_sources() -> None:
    assert codex_oauth_source_label("keyring") == "Keyring OAuth"
    assert codex_oauth_source_label("auth.json") == "Codex auth.json"


def test_codex_oauth_source_label_falls_back_for_unknown_sources() -> None:
    assert codex_oauth_source_label("env") == "OAuth token"
    assert codex_oauth_source_label(None) == "OAuth token"


def test_codex_oauth_source_display_wraps_label_in_success_style() -> None:
    assert codex_oauth_source_display({"source": "auth.json"}) == "[green]Codex auth.json[/green]"


def test_codex_oauth_expiry_display_formats_present_and_expired_tokens() -> None:
    timestamp = datetime(2026, 6, 2, 13, 45).timestamp()

    assert codex_oauth_expiry_display({"expires_at": timestamp}) == "[green]2026-06-02 13:45[/green]"
    assert (
        codex_oauth_expiry_display({"expires_at": timestamp, "expired": True})
        == "[red]expired 2026-06-02 13:45[/red]"
    )


def test_codex_oauth_expiry_display_uses_unknown_for_missing_or_invalid_values() -> None:
    assert codex_oauth_expiry_display({}) == "[green]unknown[/green]"
    assert codex_oauth_expiry_display({"expires_at": "bad"}) == "[green]unknown[/green]"
