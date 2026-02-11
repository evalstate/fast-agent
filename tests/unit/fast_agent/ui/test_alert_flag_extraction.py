import json
from typing import Literal

from mcp.types import TextContent

from fast_agent.constants import (
    FAST_AGENT_ALERT_CHANNEL,
    FAST_AGENT_ERROR_CHANNEL,
    FAST_AGENT_REMOVED_METADATA_CHANNEL,
)
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.enhanced_prompt import (
    _extract_alert_flags_from_alert,
    _extract_alert_flags_from_meta,
    _resolve_alert_flags_from_history,
)


def _json_text_block(payload: dict[str, object]) -> TextContent:
    return TextContent(type="text", text=json.dumps(payload))


def test_extract_alert_flags_from_alert_prefers_structured_flags() -> None:
    blocks = [
        _json_text_block(
            {
                "type": "unsupported_content_removed",
                "flags": ["V"],
                "categories": ["vision"],
                "handled": True,
            }
        )
    ]

    assert _extract_alert_flags_from_alert(blocks) == {"V"}


def test_extract_alert_flags_from_alert_uses_category_fallback() -> None:
    blocks = [
        _json_text_block(
            {
                "type": "unsupported_content_removed",
                "categories": ["document"],
                "handled": True,
            }
        )
    ]

    assert _extract_alert_flags_from_alert(blocks) == {"D"}


def test_extract_alert_flags_from_meta_remains_backward_compatible() -> None:
    blocks = [
        _json_text_block(
            {
                "type": "fast-agent-removed",
                "category": "vision",
            }
        )
    ]

    assert _extract_alert_flags_from_meta(blocks) == {"V"}

def _message_with_channels(
    *,
    role: Literal["user", "assistant"] = "user",
    channels: dict[str, list[TextContent]] | None = None,
) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role,
        content=[TextContent(type="text", text="payload")],
        channels=channels,
    )


def test_resolve_alert_flags_prefers_new_alert_channel_over_legacy_meta() -> None:
    messages = [
        _message_with_channels(
            channels={
                FAST_AGENT_ERROR_CHANNEL: [TextContent(type="text", text="err")],
                FAST_AGENT_ALERT_CHANNEL: [
                    _json_text_block(
                        {
                            "type": "unsupported_content_removed",
                            "flags": ["V"],
                            "categories": ["vision"],
                            "handled": True,
                        }
                    )
                ],
                FAST_AGENT_REMOVED_METADATA_CHANNEL: [
                    _json_text_block(
                        {
                            "type": "fast-agent-removed",
                            "category": "document",
                        }
                    )
                ],
            }
        )
    ]

    assert _resolve_alert_flags_from_history(messages) == {"V"}


def test_resolve_alert_flags_falls_back_to_legacy_meta_when_alert_missing() -> None:
    messages = [
        _message_with_channels(
            channels={
                FAST_AGENT_ERROR_CHANNEL: [TextContent(type="text", text="err")],
                FAST_AGENT_REMOVED_METADATA_CHANNEL: [
                    _json_text_block(
                        {
                            "type": "fast-agent-removed",
                            "category": "document",
                        }
                    )
                ],
            }
        )
    ]

    assert _resolve_alert_flags_from_history(messages) == {"D"}


def test_resolve_alert_flags_falls_back_to_t_for_unclassified_error() -> None:
    messages = [
        _message_with_channels(
            role="assistant",
            channels={
                FAST_AGENT_ERROR_CHANNEL: [TextContent(type="text", text="generic error")],
            },
        )
    ]

    assert _resolve_alert_flags_from_history(messages) == {"T"}


def test_resolve_alert_flags_handles_mixed_categories() -> None:
    messages = [
        _message_with_channels(
            channels={
                FAST_AGENT_ALERT_CHANNEL: [
                    _json_text_block(
                        {
                            "type": "unsupported_content_removed",
                            "flags": ["V", "D"],
                            "categories": ["vision", "document"],
                            "handled": True,
                        }
                    )
                ]
            }
        )
    ]

    assert _resolve_alert_flags_from_history(messages) == {"V", "D"}

