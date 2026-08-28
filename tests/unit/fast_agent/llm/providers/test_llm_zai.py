from typing import cast

import pytest
from mcp_types import TextContent
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from fast_agent.commands.model_capabilities import (
    resolve_reasoning_effort,
    resolve_reasoning_effort_spec,
)
from fast_agent.config import OpenAISettings, Settings, ZaiSettings
from fast_agent.constants import REASONING
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider.openai.llm_zai import ZAI_BASE_URL, ZaiLLM
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.reasoning_effort import ReasoningEffortLevel, ReasoningEffortSetting
from fast_agent.mcp.prompt import Prompt
from fast_agent.types import RequestParams
from fast_agent.ui.model_shortcuts import cycle_reasoning_setting

_TOOL: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "echo_value",
        "description": "Echo a value.",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}


def _request(
    llm: ZaiLLM,
    *,
    tools: list[ChatCompletionToolParam] | None = None,
) -> dict[str, object]:
    return llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        tools,
        RequestParams(model=llm.default_request_params.model),
    )


def test_zai_model_spec_loads_native_provider() -> None:
    parsed = ModelFactory.parse_model_string("zai.glm-5.2")
    bare = ModelFactory.parse_model_string("glm-5.2")
    hf_alias = ModelFactory.parse_model_string("glm52")

    assert parsed.provider == Provider.ZAI
    assert parsed.model_name == "glm-5.2"
    assert bare.provider == Provider.ZAI
    assert bare.model_name == "glm-5.2"
    assert hf_alias.provider == Provider.HUGGINGFACE
    assert ModelFactory._load_provider_class(Provider.ZAI) is ZaiLLM


@pytest.mark.parametrize(
    ("model_spec", "model_name"),
    [
        ("zai.glm-5.3", "glm-5.3"),
        ("glm-5.3", "glm-5.3"),
        ("zaiglm53", "glm-5.3"),
        ("zai.glm-5.3-flash", "glm-5.3-flash"),
        ("glm-5.3-flash", "glm-5.3-flash"),
        ("zaiglm53flash", "glm-5.3-flash"),
    ],
)
def test_zai_glm_5_3_model_specs_use_native_provider(
    model_spec: str,
    model_name: str,
) -> None:
    parsed = ModelFactory.parse_model_string(model_spec)

    assert parsed.provider == Provider.ZAI
    assert parsed.model_name == model_name


def test_zai_defaults_to_glm_5_2_and_official_endpoint() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="")

    assert llm.default_request_params.model == "glm-5.2"
    assert llm._base_url() == ZAI_BASE_URL


def test_zai_config_overrides_default_model_and_endpoint() -> None:
    settings = Settings(
        zai=ZaiSettings(
            default_model="glm-custom",
            base_url="https://gateway.example/v4/",
        )
    )
    llm = ZaiLLM(context=Context(config=settings), model="")

    assert llm.default_request_params.model == "glm-custom"
    assert llm._base_url() == "https://gateway.example/v4/"


def test_zai_glm_5_2_capabilities_are_native() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")
    params = ModelDatabase.get_model_params("glm-5.2", provider=Provider.ZAI)
    hf_params = ModelDatabase.get_model_params(
        "zai-org/glm-5.2",
        provider=Provider.HUGGINGFACE,
    )

    assert llm.provider == Provider.ZAI
    assert params is not None
    assert hf_params is not None
    assert params.context_window == 1_000_000
    assert params.max_output_tokens == 131_072
    assert params.tokenizes == ["text/plain"]
    assert params.default_provider == Provider.ZAI
    assert params.process_poll_default_wait_seconds == 240
    assert hf_params.process_poll_default_wait_seconds == 0
    assert llm._get_model_stream_mode("glm-5.2") == "manual"
    assert llm._get_model_json_mode("glm-5.2") == "object"


def test_zai_glm_5_3_capabilities_are_native() -> None:
    text = ModelDatabase.get_model_params("glm-5.3", provider=Provider.ZAI)
    flash = ModelDatabase.get_model_params("glm-5.3-flash", provider=Provider.ZAI)

    assert text is not None
    assert flash is not None
    assert text.context_window == flash.context_window == 1_000_000
    assert text.max_output_tokens == flash.max_output_tokens == 131_072
    assert text.tokenizes == ["text/plain"]
    assert flash.tokenizes == [
        "text/plain",
        "image/jpeg",
        "image/png",
        "application/pdf",
        "video/quicktime",
    ]
    assert text.default_provider == flash.default_provider == Provider.ZAI
    assert text.process_poll_default_wait_seconds == flash.process_poll_default_wait_seconds == 0
    assert text.json_mode == flash.json_mode == "object"
    assert text.stream_mode == flash.stream_mode == "manual"
    assert flash.fast is True


def test_zai_glm_5_3_flash_limits_link_only_media() -> None:
    assert ModelDatabase.supports_mime(
        "glm-5.3-flash",
        "application/pdf",
        provider=Provider.ZAI,
        resource_source="link",
    )
    assert ModelDatabase.supports_mime(
        "glm-5.3-flash",
        "video/quicktime",
        provider=Provider.ZAI,
        resource_source="link",
    )
    assert not ModelDatabase.supports_mime(
        "glm-5.3-flash",
        "application/pdf",
        provider=Provider.ZAI,
        resource_source="embedded",
    )
    assert not ModelDatabase.supports_mime(
        "glm-5.3-flash",
        "video/quicktime",
        provider=Provider.ZAI,
        resource_source="embedded",
    )
    assert not ModelDatabase.supports_mime(
        "glm-5.3-flash",
        "image/webp",
        provider=Provider.ZAI,
    )


def test_zai_reasoning_defaults_to_max_and_can_be_disabled() -> None:
    enabled = _request(ZaiLLM(context=Context(config=Settings()), model="glm-5.2"))
    disabled = _request(
        ZaiLLM(
            context=Context(config=Settings()),
            model="glm-5.2",
            reasoning_effort=False,
        )
    )

    assert enabled["reasoning_effort"] == "max"
    assert enabled["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "reasoning_effort" not in disabled
    assert disabled["extra_body"] == {"thinking": {"type": "disabled"}}


def test_zai_reasoning_default_ignores_openai_config() -> None:
    openai_configs = (
        OpenAISettings(api_key="openai-key"),
        OpenAISettings(reasoning="none"),
    )

    for openai_config in openai_configs:
        settings = Settings(openai=openai_config)
        request = _request(ZaiLLM(context=Context(config=settings), model="glm-5.2"))

        assert request["reasoning_effort"] == "max"
        assert request["extra_body"] == {"thinking": {"type": "enabled"}}


def test_zai_reasoning_none_disables_thinking() -> None:
    request = _request(
        ZaiLLM(
            context=Context(config=Settings()),
            model="glm-5.2",
            reasoning_effort="none",
        )
    )

    assert "reasoning_effort" not in request
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_zai_reasoning_effort_is_forwarded() -> None:
    settings = Settings(openai=OpenAISettings(reasoning="none"))
    request = _request(
        ZaiLLM(
            context=Context(config=settings),
            model="glm-5.2",
            reasoning_effort="high",
        )
    )

    assert request["reasoning_effort"] == "high"
    assert request["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
@pytest.mark.parametrize("effort", ["low", "high", "max"])
def test_zai_glm_5_3_forces_documented_reasoning(
    model: str,
    effort: str,
) -> None:
    request = _request(
        ZaiLLM(
            context=Context(config=Settings()),
            model=model,
            reasoning_effort=effort,
        )
    )

    assert request["reasoning_effort"] == effort
    assert request["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "clear_thinking": False,
        }
    }


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
@pytest.mark.parametrize("reasoning", [False, "none", "medium"])
def test_zai_glm_5_3_invalid_reasoning_falls_back_to_max(
    model: str,
    reasoning: bool | str,
) -> None:
    request = _request(
        ZaiLLM(
            context=Context(config=Settings()),
            model=model,
            reasoning_effort=reasoning,
        )
    )

    assert request["reasoning_effort"] == "max"
    assert request["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "clear_thinking": False,
        }
    }


def test_zai_reasoning_request_uses_max_tokens() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        RequestParams(model="glm-5.2", max_tokens=48_000),
    )

    assert request["max_tokens"] == 48_000


def test_zai_glm_5_3_uses_provider_sampling_defaults_unless_overridden() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3")

    default = _request(llm)
    temperature = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        RequestParams(model="glm-5.3", temperature=0.8),
    )
    top_p = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        RequestParams(model="glm-5.3", top_p=0.8),
    )

    assert "temperature" not in default
    assert "top_p" not in default
    assert temperature["temperature"] == 0.8
    assert "top_p" not in temperature
    assert top_p["top_p"] == 0.8
    assert "temperature" not in top_p


def test_zai_enables_incremental_tool_streaming_only_for_streamed_tool_requests() -> None:
    llm = ZaiLLM(
        context=Context(config=Settings()),
        model="glm-5.2",
        reasoning_effort=False,
    )

    streamed = _request(llm, tools=[_TOOL])
    without_tools = _request(llm)
    non_streamed = llm._prepare_non_streaming_request(streamed)

    assert streamed["extra_body"] == {
        "thinking": {"type": "disabled"},
        "tool_stream": True,
    }
    assert without_tools["extra_body"] == {"thinking": {"type": "disabled"}}
    assert non_streamed["stream"] is False
    assert non_streamed["extra_body"] == {"thinking": {"type": "disabled"}}


def test_zai_glm_5_3_preserves_forced_reasoning_in_tool_fallback() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3")

    streamed = _request(llm, tools=[_TOOL])
    non_streamed = llm._prepare_non_streaming_request(streamed)

    assert streamed["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "clear_thinking": False,
        },
        "tool_stream": True,
    }
    assert non_streamed["extra_body"] == {
        "thinking": {
            "type": "enabled",
            "clear_thinking": False,
        }
    }


def test_zai_config_credentials_and_headers_take_precedence(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "environment-key")
    settings = Settings(
        zai=ZaiSettings(
            api_key="configured-key",
            default_headers={"X-Test": "zai"},
        )
    )
    llm = ZaiLLM(context=Context(config=settings), model="glm-5.2")

    assert llm._api_key() == "configured-key"
    assert llm._default_headers() == {"X-Test": "zai"}


def test_zai_structured_output_uses_json_object_mode() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")

    messages, params = llm._prepare_structured_request(
        [Prompt.user("return json")],
        RequestParams(model="glm-5.2", structured_schema=schema),
    )

    assert params.response_format == {"type": "json_object"}
    assert "YOU MUST RESPOND WITH A JSON OBJECT" in messages[0].all_text()


def test_zai_glm_5_3_flash_structured_output_uses_json_object_mode() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3-flash")

    messages, params = llm._prepare_structured_request(
        [Prompt.user("return json")],
        RequestParams(model="glm-5.3-flash", structured_schema=schema),
    )

    assert params.response_format == {"type": "json_object"}
    assert "YOU MUST RESPOND WITH A JSON OBJECT" in messages[0].all_text()


@pytest.mark.asyncio
async def test_zai_converts_remote_openai_file_parts_to_native_file_urls() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3-flash")
    messages = [
        cast(
            "ChatCompletionMessageParam",
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "document.pdf",
                            "file_url": "https://example.test/document.pdf",
                        },
                    },
                ],
            },
        )
    ]

    client = AsyncOpenAI(api_key="test")
    try:
        normalized = await llm._normalize_chat_completion_files(client, messages)
    finally:
        await client.close()

    assert normalized == [
        {
            "role": "user",
            "content": [
                {
                    "type": "file_url",
                    "file_url": {"url": "https://example.test/document.pdf"},
                }
            ],
        }
    ]


def test_zai_glm_5_3_f6_uses_only_documented_effort_levels() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3")
    spec = resolve_reasoning_effort_spec(llm)

    assert spec is not None
    assert spec.allowed_efforts == ["low", "high", "max"]
    assert spec.allow_toggle_disable is False


def test_zai_glm_5_3_reasoning_replay_does_not_insert_block_separators() -> None:
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.3")
    message = Prompt.assistant("answer")
    message.channels = {
        REASONING: [
            TextContent(type="text", text="first."),
            TextContent(type="text", text="Second"),
        ]
    }

    converted = llm._convert_extended_messages_to_provider([message])

    assert converted[0].get("reasoning_content") == "first.Second"


def test_zai_reasoning_none_preserves_effort_kind_for_f6_rotation() -> None:
    """set_reasoning_effort('none') must store an effort setting, not a toggle.

    The F6 cycle builds candidates from effort-kind settings.  If 'none' is
    converted to a toggle(False) at storage time, the cycle can never match it
    again and gets stuck returning 'none' on every press.
    """
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="none"))

    assert llm.reasoning_effort == ReasoningEffortSetting(kind="effort", value="none")


def test_zai_f6_rotates_through_all_glm_52_effort_levels() -> None:
    """F6 must rotate through every GLM-5.2 effort level without getting stuck."""
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")
    spec = resolve_reasoning_effort_spec(llm)
    assert spec is not None

    # The default is 'max'; F6 should rotate through all 7 effort levels.
    expected: list[ReasoningEffortLevel] = [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    current = resolve_reasoning_effort(llm)
    assert current == ReasoningEffortSetting(kind="effort", value="max")

    seen: list[str] = []
    for _ in expected:
        next_setting = cycle_reasoning_setting(current, spec)
        assert next_setting is not None
        llm.set_reasoning_effort(next_setting)
        current = llm.reasoning_effort
        assert current is not None
        seen.append(str(current.value))

    assert seen == expected
    # Next press wraps back to the first level.
    assert cycle_reasoning_setting(current, spec) == ReasoningEffortSetting(
        kind="effort", value=expected[0]
    )


def test_zai_reasoning_none_disables_thinking_via_set_reasoning_effort() -> None:
    """Setting 'none' through set_reasoning_effort must still disable thinking."""
    llm = ZaiLLM(context=Context(config=Settings()), model="glm-5.2")
    llm.set_reasoning_effort(ReasoningEffortSetting(kind="effort", value="none"))

    request = _request(llm)

    assert "reasoning_effort" not in request
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
