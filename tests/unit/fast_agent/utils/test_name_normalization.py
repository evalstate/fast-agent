from fast_agent.utils.name_normalization import normalize_provider_key


def test_normalize_provider_key_collapses_common_separators() -> None:
    assert normalize_provider_key(" Anthropic-Vertex ") == "anthropicvertex"
    assert normalize_provider_key("codex_responses") == "codexresponses"
    assert normalize_provider_key("Open Responses") == "openresponses"


def test_normalize_provider_key_collapses_all_whitespace() -> None:
    assert normalize_provider_key("Open\tResponses\n") == "openresponses"


def test_normalize_provider_key_collapses_repeated_mixed_separators() -> None:
    assert normalize_provider_key("Anthropic - _ Vertex") == "anthropicvertex"
