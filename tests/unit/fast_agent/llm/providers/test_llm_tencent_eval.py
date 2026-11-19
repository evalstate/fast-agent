from pathlib import Path

import pytest
from mcp_types import TextContent

from fast_agent.cli.commands.check_config import check_api_keys
from fast_agent.config import OpenAISettings, Settings, TencentEvalSettings, get_settings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.provider.openai.llm_tencent_eval import (
    DEFAULT_TENCENT_EVAL_MODEL,
    TENCENT_EVAL_BASE_URL,
    TencentEvalLLM,
)
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortSetting
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams
from fast_agent.ui.model_picker_common import PICKER_PROVIDER_ORDER, build_snapshot


def _request(llm: TencentEvalLLM) -> dict[str, object]:
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        RequestParams(model=llm.default_request_params.model),
    )


@pytest.mark.parametrize(
    ("model_spec", "expected_model"),
    [
        ("tencent-eval.hy4-flash-dev", "hy4-flash-dev"),
        ("tencent-eval/hy4-flash-dev", "hy4-flash-dev"),
        ("hy4-flash-dev", "hy4-flash-dev"),
        ("tencent-eval", ""),
    ],
)
def test_tencent_eval_model_specs_resolve_to_native_provider(
    model_spec: str,
    expected_model: str,
) -> None:
    parsed = ModelFactory.parse_model_string(model_spec)

    assert parsed.provider == Provider.TENCENT_EVAL
    assert parsed.model_name == expected_model
    assert ModelFactory._load_provider_class(Provider.TENCENT_EVAL) is TencentEvalLLM


def test_tencent_eval_defaults_to_hy4_flash_dev_and_official_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENCENT_EVAL_BASE_URL", raising=False)
    llm = TencentEvalLLM(context=Context(config=Settings()), model="")

    assert llm.default_request_params.model == DEFAULT_TENCENT_EVAL_MODEL == "hy4-flash-dev"
    assert llm._base_url() == TENCENT_EVAL_BASE_URL


@pytest.mark.parametrize("config_key", ["tencent-eval", "tencent_eval"])
def test_tencent_eval_config_overrides_endpoint_model_and_headers(
    monkeypatch: pytest.MonkeyPatch,
    config_key: str,
) -> None:
    monkeypatch.setenv("TENCENT_EVAL_BASE_URL", "https://env-gateway.example/v2")
    settings = Settings.model_validate(
        {
            config_key: {
                "api_key": "config-test-key",
                "base_url": "https://config-gateway.example/v2",
                "default_model": "hy4-custom",
                "default_headers": {"X-Test": "value"},
            }
        }
    )
    llm = TencentEvalLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "hy4-custom"
    assert llm._base_url() == "https://config-gateway.example/v2"
    assert llm._default_headers() == {"X-Test": "value"}


def test_tencent_eval_base_url_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TENCENT_EVAL_BASE_URL", "https://env-gateway.example/v2")
    llm = TencentEvalLLM(context=Context(config=Settings()), model="hy4-flash-dev")

    assert llm._base_url() == "https://env-gateway.example/v2"


def test_tencent_eval_api_key_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TENCENT_EVAL_API_KEY", raising=False)
    monkeypatch.setenv("HY4_API_KEY", "hy4-env-key")

    llm_hy4 = TencentEvalLLM(context=Context(config=Settings()), model="hy4-flash-dev")
    assert llm_hy4._api_key() == "hy4-env-key"

    monkeypatch.setenv("TENCENT_EVAL_API_KEY", "tencent-eval-env-key")
    llm_env = TencentEvalLLM(context=Context(config=Settings()), model="hy4-flash-dev")
    assert llm_env._api_key() == "tencent-eval-env-key"

    settings_hyphen = Settings.model_validate({"tencent-eval": {"api_key": "config-hyphen-key"}})
    llm_cfg_hyphen = TencentEvalLLM(context=Context(config=settings_hyphen), model="hy4-flash-dev")
    assert llm_cfg_hyphen._api_key() == "config-hyphen-key"

    settings_typed = Settings(tencent_eval=TencentEvalSettings(api_key="config-typed-key"))
    llm_cfg_typed = TencentEvalLLM(context=Context(config=settings_typed), model="hy4-flash-dev")
    assert llm_cfg_typed._api_key() == "config-typed-key"


def test_tencent_eval_reasoning_defaults_to_high_and_ignores_openai_config() -> None:
    settings = Settings(openai=OpenAISettings(reasoning_effort="low"))
    llm = TencentEvalLLM(context=Context(config=settings), model="hy4-flash-dev")

    arguments = _request(llm)

    assert llm.reasoning_effort is None
    assert arguments["reasoning_effort"] == "high"
    assert arguments["stream_options"] == {"include_usage": True}


def test_tencent_eval_omits_max_tokens_by_default_and_forwards_explicit_override() -> None:
    llm = TencentEvalLLM(context=Context(config=Settings()), model="hy4-flash-dev")

    assert llm.default_request_params.max_tokens is None

    default_args = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.get_request_params(),
    )
    assert "max_tokens" not in default_args
    assert "max_completion_tokens" not in default_args

    explicit_args = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.get_request_params(RequestParams(max_tokens=1024)),
    )
    assert explicit_args["max_tokens"] == 1024

    alias_args = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.get_request_params(RequestParams.model_validate({"maxTokens": 2048})),
    )
    assert alias_args["max_tokens"] == 2048


@pytest.mark.parametrize(
    ("setting", "expected_wire"),
    [
        (ReasoningEffortSetting(kind="effort", value="high"), "high"),
        (ReasoningEffortSetting(kind="effort", value="none"), "no_think"),
        (ReasoningEffortSetting(kind="toggle", value=False), "no_think"),
        (ReasoningEffortSetting(kind="toggle", value=True), "high"),
    ],
)
def test_tencent_eval_reasoning_wire_mapping(
    setting: ReasoningEffortSetting,
    expected_wire: str,
) -> None:
    llm = TencentEvalLLM(
        context=Context(config=Settings()),
        model="hy4-flash-dev",
        reasoning_effort=setting,
    )

    arguments = _request(llm)

    assert arguments["reasoning_effort"] == expected_wire


def test_tencent_eval_reasoning_replay_preserves_raw_reasoning_content() -> None:
    llm = TencentEvalLLM(context=Context(config=Settings()), model="hy4-flash-dev")
    message = Prompt.assistant("answer")
    message.channels = {
        REASONING: [
            TextContent(type="text", text="first."),
            TextContent(type="text", text="Second"),
        ]
    }

    converted = llm._convert_extended_messages_to_provider([message])

    assert converted[0].get("reasoning_content") == "first.Second"


def test_tencent_eval_model_database_entry() -> None:
    params = ModelDatabase.get_model_params("hy4-flash-dev", provider=Provider.TENCENT_EVAL)

    assert params is not None
    assert params.context_window == 960000
    assert params.max_output_tokens == 65536
    assert params.tokenizes == ModelDatabase.OPENAI_VISION
    assert params.json_mode == "schema"
    assert params.reasoning == "reasoning_content"
    assert params.stream_mode == "manual"
    assert params.default_provider == Provider.TENCENT_EVAL
    assert params.fast is True
    assert params.reasoning_effort_spec is not None
    assert params.reasoning_effort_spec.allowed_efforts == ["none", "high"]
    assert params.reasoning_effort_spec.allow_toggle_disable is True
    assert params.reasoning_effort_spec.default == ReasoningEffortSetting(
        kind="effort", value="high"
    )


def test_tencent_eval_is_excluded_from_interactive_model_picker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENCENT_EVAL_API_KEY", "test-key")
    monkeypatch.setenv("HY4_API_KEY", "test-hy4-key")

    assert Provider.TENCENT_EVAL not in PICKER_PROVIDER_ORDER
    assert Provider.TENCENT_EVAL not in ModelSelectionCatalog.CATALOG_ENTRIES_BY_PROVIDER

    snapshot = build_snapshot()
    provider_keys = {option.option_key for option in snapshot.providers}
    assert Provider.TENCENT_EVAL.config_name not in provider_keys


def test_check_api_keys_detects_tencent_eval_env_and_hy4_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENCENT_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("HY4_API_KEY", raising=False)

    empty = check_api_keys({"status": "not_found", "secrets": {}}, {"status": "not_found"})
    assert empty["tencent-eval"]["env"] == ""

    monkeypatch.setenv("HY4_API_KEY", "hy4-fallback-12345")
    with_hy4 = check_api_keys({"status": "not_found", "secrets": {}}, {"status": "not_found"})
    assert with_hy4["tencent-eval"]["env"] == "...12345"

    monkeypatch.setenv("TENCENT_EVAL_API_KEY", "tencent-primary-99999")
    with_primary = check_api_keys({"status": "not_found", "secrets": {}}, {"status": "not_found"})
    assert with_primary["tencent-eval"]["env"] == "...99999"


@pytest.mark.parametrize(
    ("config_key", "secrets_key"),
    [
        ("tencent-eval", "tencent_eval"),
        ("tencent_eval", "tencent-eval"),
    ],
)
def test_tencent_eval_mixed_aliases_merge_across_config_and_secrets_files(
    tmp_path: Path,
    config_key: str,
    secrets_key: str,
) -> None:
    config_file = tmp_path / "fast-agent.yaml"
    secrets_file = tmp_path / "fast-agent.secrets.yaml"
    config_file.write_text(
        f"""{config_key}:
  base_url: "https://config-gateway.example/v2"
  api_key: "base-key"
  default_model: "hy4-custom"
""",
        encoding="utf-8",
    )
    secrets_file.write_text(
        f"""{secrets_key}:
  api_key: "secret-key"
""",
        encoding="utf-8",
    )

    settings = get_settings(config_path=config_file, no_home=True)

    assert settings.tencent_eval is not None
    assert settings.tencent_eval.base_url == "https://config-gateway.example/v2"
    assert settings.tencent_eval.default_model == "hy4-custom"
    assert settings.tencent_eval.api_key == "secret-key"


def test_tencent_eval_mixed_aliases_merge_in_model_validate() -> None:
    merged_a = Settings.model_validate(
        {
            "tencent-eval": {"base_url": "https://a.example/v2", "api_key": "first-key"},
            "tencent_eval": {"api_key": "second-key", "default_model": "hy4-custom"},
        }
    )
    assert merged_a.tencent_eval is not None
    assert merged_a.tencent_eval.base_url == "https://a.example/v2"
    assert merged_a.tencent_eval.api_key == "second-key"
    assert merged_a.tencent_eval.default_model == "hy4-custom"

    merged_b = Settings.model_validate(
        {
            "tencent_eval": {"base_url": "https://b.example/v2", "api_key": "canonical-key"},
            "tencent-eval": {"api_key": "alias-key", "default_model": "hy4-from-alias"},
        }
    )
    assert merged_b.tencent_eval is not None
    assert merged_b.tencent_eval.base_url == "https://b.example/v2"
    assert merged_b.tencent_eval.default_model == "hy4-from-alias"
    assert merged_b.tencent_eval.api_key == "canonical-key"
