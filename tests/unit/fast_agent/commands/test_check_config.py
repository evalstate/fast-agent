from fast_agent.cli.commands.check_config import (
    API_KEY_HINT_TEXT,
    DEFAULT_OPENRESPONSES_BASE_URL,
    Provider,
    _adjacent_settings_pairs,
    _effective_environment_override,
    _extract_skills_directories,
    _format_provider_row,
    _format_step_interval,
    _should_warn_for_provider,
    _split_model_specs,
    _split_provider_status_rows,
    _truncate_server_display,
    _truncate_summary_text,
    check_api_keys,
)


def make_secrets_summary(azure_cfg):
    return {"status": "parsed", "error": None, "secrets": {"azure": azure_cfg}}


def test_check_api_keys_only_api_key():
    azure_cfg = {
        "api_key": "test-azure-key",
        "resource_name": "test-resource",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    assert results["azure"]["config"] == "...e-key"


def test_check_api_keys_only_default_cred():
    azure_cfg = {
        "use_default_azure_credential": True,
        "base_url": "https://mydemo.openai.azure.com/",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    assert results["azure"]["config"] == "DefaultAzureCredential"


def test_check_api_keys_both_modes():
    azure_cfg = {
        "api_key": "test-azure-key",
        "use_default_azure_credential": True,
        "base_url": "https://mydemo.openai.azure.com/",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # When use_default_azure_credential=True, Azure LLM ignores api_key and only uses DefaultAzureCredential
    assert results["azure"]["config"] == "DefaultAzureCredential"


def test_check_api_keys_invalid_config():
    azure_cfg = {
        "use_default_azure_credential": True,
        # missing base_url
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # Should not mark as DefaultAzureCredential if base_url missing
    assert results["azure"]["config"] == ""


def test_check_api_keys_google_vertex_adc():
    config_summary = {
        "status": "parsed",
        "config": {"google": {"vertex_ai": {"enabled": True}}},
    }

    results = check_api_keys({}, config_summary)

    assert results["google"]["config"] == "Vertex AI ADC"


def test_check_api_keys_detects_anthropic_sdk_auth_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")

    results = check_api_keys({}, {})

    assert results["anthropic"]["env"] == ""
    assert results["anthropic"]["config"] == "ANTHROPIC_AUTH_TOKEN"


def test_check_api_keys_hint_text():
    azure_cfg = {
        "api_key": API_KEY_HINT_TEXT,
        "resource_name": "test-resource",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # Should not show API_KEY_HINT_TEXT as a valid key
    assert results["azure"]["config"] == ""


def test_format_provider_row_openresponses_shows_default_without_key():
    row = _format_provider_row("openresponses", {"env": "", "config": ""})

    assert row[3] == "[green]none (default)[/green]"


def test_adjacent_settings_pairs_keeps_odd_tail() -> None:
    assert _adjacent_settings_pairs([("a", "1"), ("b", "2"), ("c", "3")]) == [
        (("a", "1"), ("b", "2")),
        (("c", "3"), None),
    ]


def test_split_provider_status_rows_balances_columns() -> None:
    rows = _split_provider_status_rows(
        {
            "anthropic": {"env": "env-a", "config": ""},
            "openai": {"env": "", "config": "cfg-o"},
            "google": {"env": "", "config": ""},
        }
    )

    assert rows == [
        (("anthropic", {"env": "env-a", "config": ""}), ("google", {"env": "", "config": ""})),
        (("openai", {"env": "", "config": "cfg-o"}), None),
    ]


def test_format_step_interval_compacts_common_durations():
    assert _format_step_interval(0) == "0s"
    assert _format_step_interval(59) == "59s"
    assert _format_step_interval(75) == "1m15s"
    assert _format_step_interval(120) == "2m"
    assert _format_step_interval(3661) == "61m01s"
    assert _format_step_interval(7200) == "2h"
    assert _format_step_interval(172800) == "2d"
    assert _format_step_interval("bad") == "bad"


def test_check_config_truncation_helpers_share_suffix_rule():
    value = "x" * 65

    assert _truncate_server_display(value) == f"{'x' * 57}..."
    assert _truncate_summary_text(value, 60) == _truncate_server_display(value)


def test_split_model_specs_normalizes_comma_separated_models():
    assert _split_model_specs(" opus, , sonnet ,") == ["opus", "sonnet"]


def test_extract_skills_directories_normalizes_list_values():
    assert _extract_skills_directories(
        {"skills": {"directories": [" ./skills ", "", "  ", 123]}}
    ) == ["./skills", "123"]


def test_effective_environment_override_normalizes_env_and_config_values(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT_DIR", "  /tmp/fast-agent-env  ")

    assert _effective_environment_override(env_dir=None, config_summary={}) == "/tmp/fast-agent-env"

    monkeypatch.setenv("ENVIRONMENT_DIR", "   ")
    assert (
        _effective_environment_override(
            env_dir=None,
            config_summary={"config": {"environment_dir": "  .fast-agent-local  "}},
        )
        == ".fast-agent-local"
    )


def test_should_warn_for_openresponses_provider_normalizes_base_url(monkeypatch):
    monkeypatch.setenv("OPENRESPONSES_BASE_URL", "   ")
    assert not _should_warn_for_provider(
        Provider.OPENRESPONSES,
        {
            "status": "parsed",
            "config": {"openresponses": {"base_url": f"  {DEFAULT_OPENRESPONSES_BASE_URL}/  "}},
        },
    )
