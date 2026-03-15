from fast_agent.llm.provider.openai.tool_stream_state import OpenAIToolStreamState


def test_openai_tool_stream_state_resolves_by_item_id_alias() -> None:
    tool_state = OpenAIToolStreamState()

    entry = tool_state.register(
        tool_use_id="call_123",
        name="weather",
        index=1,
        item_id="fc_123",
        item_type="function_call",
    )

    assert tool_state.resolve_open(item_id="fc_123") is entry
    assert tool_state.resolve_open(tool_use_id="call_123") is entry


def test_openai_tool_stream_state_rekeys_placeholder_item_identity() -> None:
    tool_state = OpenAIToolStreamState()

    placeholder = tool_state.register(
        tool_use_id="fc_123",
        name="weather",
        index=-1,
        item_id="fc_123",
        item_type="function_call",
    )

    updated = tool_state.register(
        tool_use_id="call_123",
        name="weather",
        index=1,
        item_id="fc_123",
        item_type="function_call",
    )

    assert updated is placeholder
    assert updated.tool_use_id == "call_123"
    assert tool_state.resolve_open(item_id="fc_123") is updated
    assert tool_state.resolve_open(tool_use_id="call_123") is updated


def test_openai_tool_stream_state_completed_uses_item_id_aliases() -> None:
    tool_state = OpenAIToolStreamState()
    tool_state.register(
        tool_use_id="call_123",
        name="weather",
        index=1,
        item_id="fc_123",
        item_type="function_call",
    )

    tool_state.close(item_id="fc_123")

    assert tool_state.resolve_open(item_id="fc_123") is None
    assert tool_state.is_completed(item_id="fc_123")
