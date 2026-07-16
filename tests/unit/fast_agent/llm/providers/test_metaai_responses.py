import json
from types import SimpleNamespace

from mcp.types import TextContent

from fast_agent.config import (
    MetaAISettings,
    MetaAIWebSearchSettings,
    OpenAIUserLocationSettings,
    Settings,
)
from fast_agent.context import Context
from fast_agent.llm.provider.openai.metaai_responses import (
    DEFAULT_META_AI_MODEL,
    RESPONSE_INCLUDE_WEB_SEARCH_RESULTS,
    MetaAIResponsesLLM,
)
from fast_agent.llm.provider.openai.responses_output import ResponsesOutputMixin
from fast_agent.llm.provider.openai.web_tools import normalize_web_search_call_payload
from fast_agent.llm.provider_types import Provider


def test_metaai_responses_provider_defaults_to_sse_transport() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="muse-spark-1.1",
    )

    assert llm.provider == Provider.META_AI
    assert llm.configured_transport == "sse"


def test_metaai_responses_default_model_used_when_model_missing() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="",
    )

    assert llm.default_request_params.model == DEFAULT_META_AI_MODEL


def test_metaai_responses_uses_metaai_config_fallback() -> None:
    settings = Settings(
        metaai=MetaAISettings(
            api_key="meta-key",
            base_url="https://gateway.example/metaai/v1",
            default_headers={"X-Test": "1"},
            default_model="muse-spark-1.1",
        )
    )
    llm = MetaAIResponsesLLM(context=Context(config=settings), model="")

    assert llm._api_key() == "meta-key"
    assert llm._base_url() == "https://gateway.example/metaai/v1"
    assert llm._default_headers() == {"X-Test": "1"}
    assert llm.default_request_params.model == "muse-spark-1.1"


def test_metaai_responses_advertises_web_search() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="muse-spark-1.1",
    )

    assert llm.web_search_supported is True
    assert llm.web_search_enabled is False


def test_metaai_responses_builds_web_search_tool_when_enabled() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(
            config=Settings(
                metaai=MetaAISettings(
                    api_key="test-key",
                    web_search=MetaAIWebSearchSettings(
                        enabled=True,
                        search_context_size="high",
                        user_location=OpenAIUserLocationSettings(
                            country="GB",
                            region="London",
                            city="London",
                            timezone="Europe/London",
                        ),
                    ),
                )
            )
        ),
        model="muse-spark-1.1",
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [
        {
            "type": "web_search",
            "search_context_size": "high",
            "user_location": {
                "type": "approximate",
                "country": "GB",
                "region": "London",
                "city": "London",
                "timezone": "Europe/London",
            },
        }
    ]
    assert args["include"] == [RESPONSE_INCLUDE_WEB_SEARCH_RESULTS]
    assert "service_tier" not in args
    assert llm.web_search_enabled is True


def test_metaai_web_search_override_enables_tool() -> None:
    llm = MetaAIResponsesLLM(
        context=Context(config=Settings(metaai=MetaAISettings(api_key="test-key"))),
        model="muse-spark-1.1",
        web_search=True,
    )
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    ]

    args = llm._build_response_args(input_items, llm.default_request_params, tools=None)

    assert args["tools"] == [{"type": "web_search"}]
    assert args["include"] == [RESPONSE_INCLUDE_WEB_SEARCH_RESULTS]


def test_normalize_web_search_call_payload_includes_meta_results() -> None:
    tool_payload, citations = normalize_web_search_call_payload(
        SimpleNamespace(
            type="web_search_call",
            id="ws_789",
            status="completed",
            results=[
                SimpleNamespace(
                    type="text_result",
                    title="2026 British Grand Prix",
                    url="https://en.wikipedia.org/wiki/2026_British_Grand_Prix",
                    snippet="Leclerc took his ninth Formula One victory...",
                )
            ],
        )
    )

    assert tool_payload == {
        "type": "server_tool_use",
        "name": "web_search",
        "id": "ws_789",
        "status": "completed",
    }
    assert citations == [
        {
            "type": "web_search_result_location",
            "url": "https://en.wikipedia.org/wiki/2026_British_Grand_Prix",
            "title": "2026 British Grand Prix",
            "snippet": "Leclerc took his ninth Formula One victory...",
        }
    ]


def test_extract_web_search_metadata_captures_meta_results_and_citations() -> None:
    class _Harness(ResponsesOutputMixin):
        pass

    harness = _Harness()
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="web_search_call",
                id="ws_789",
                status="completed",
                results=[
                    SimpleNamespace(
                        type="text_result",
                        title="Muse Spark debut",
                        url="https://example.com/muse-spark",
                        snippet="Meta announced Muse Spark...",
                    )
                ],
            ),
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text="Muse Spark was announced.",
                        annotations=[
                            SimpleNamespace(
                                type="url_citation",
                                start_index=0,
                                end_index=10,
                                title="Muse Spark debut",
                                url="https://example.com/muse-spark",
                            )
                        ],
                    )
                ],
            ),
        ]
    )

    web_tools, citations = harness._extract_web_search_metadata(response)

    assert len(web_tools) == 1
    tool_block = web_tools[0]
    assert isinstance(tool_block, TextContent)
    assert '"name": "web_search"' in tool_block.text

    citation_urls = {
        json.loads(citation.text).get("url")
        for citation in citations
        if isinstance(citation, TextContent)
    }
    assert "https://example.com/muse-spark" in citation_urls
