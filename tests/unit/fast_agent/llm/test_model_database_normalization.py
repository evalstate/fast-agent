from fast_agent.llm.model_database import ModelDatabase


def test_model_database_normalizes_effort_suffix() -> None:
    assert ModelDatabase.get_max_output_tokens("gpt-5-mini.low") == ModelDatabase.get_max_output_tokens(
        "gpt-5-mini"
    )


def test_model_database_normalizes_provider_prefix_dot() -> None:
    assert ModelDatabase.get_max_output_tokens("openai.gpt-4.1") == ModelDatabase.get_max_output_tokens(
        "gpt-4.1"
    )


def test_model_database_normalizes_provider_prefix_slash() -> None:
    assert ModelDatabase.get_max_output_tokens("openai/gpt-4.1") == ModelDatabase.get_max_output_tokens(
        "gpt-4.1"
    )


def test_model_database_normalizes_aliases() -> None:
    assert ModelDatabase.get_max_output_tokens("sonnet") == ModelDatabase.get_max_output_tokens(
        "claude-sonnet-4-6"
    )
    assert ModelDatabase.get_max_output_tokens("claude") == ModelDatabase.get_max_output_tokens(
        "claude-sonnet-4-6"
    )
    assert ModelDatabase.get_max_output_tokens("codexspark") == ModelDatabase.get_max_output_tokens(
        "gpt-5.3-codex-spark"
    )


def test_model_database_strips_hf_routing_suffix() -> None:
    assert ModelDatabase.get_max_output_tokens(
        "hf.moonshotai/Kimi-K2-Instruct-0905:groq"
    ) == ModelDatabase.get_max_output_tokens("moonshotai/kimi-k2-instruct-0905")


def test_model_database_preserves_known_slash_keys() -> None:
    # Some model ids are canonical slash paths (e.g. HF model repos).
    assert ModelDatabase.get_max_output_tokens("openai/gpt-oss-20b") == ModelDatabase.get_max_output_tokens(
        "hf.openai/gpt-oss-20b"
    )
