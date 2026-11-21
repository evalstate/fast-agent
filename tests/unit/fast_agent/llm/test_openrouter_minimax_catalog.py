"""Exercise discovery and explicit requests across the OpenRouter HTTP boundary."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from fast_agent.config import OpenRouterSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_selection import ModelSelectionCatalog
from fast_agent.llm.openrouter_model_lookup import (
    clear_openrouter_model_cache,
    list_openrouter_model_specs_sync,
)
from fast_agent.llm.provider.openai.llm_openrouter import OpenRouterLLM
from fast_agent.llm.provider_types import Provider

FLASH = "minimax/minimax-m3.1-flash:free"
M3 = "minimax/minimax-m3"
LEGACY = "minimax/minimax-m2.7"
MEDIA = "minimax/image-test"
OTHER = "openai/gpt-4.1-mini"


@pytest.fixture
def catalog_http(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[list[dict], list[httpx.Request]]]:
    models = [
        {
            "id": model,
            "context_length": 123456,
            "top_provider": {"max_completion_tokens": 8192},
            "architecture": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["image"] if model == MEDIA else ["text"],
            },
        }
        for model in [LEGACY, OTHER, M3, MEDIA, FLASH]
    ]
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            assert request.url.path.endswith("/models/user")
            return httpx.Response(200, json={"data": models})
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        chunk = {
            "id": "fixture-completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    class MockHTTPClient(httpx.AsyncClient):
        def __init__(self, **kwargs) -> None:
            super().__init__(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr("fast_agent.llm.openrouter_model_lookup.httpx.AsyncClient", MockHTTPClient)
    monkeypatch.setattr(
        "fast_agent.llm.provider.openai.llm_openai.DefaultAioHttpClient", MockHTTPClient
    )
    clear_openrouter_model_cache()
    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)
    yield models, requests
    clear_openrouter_model_cache()
    ModelDatabase.clear_runtime_model_params(Provider.OPENROUTER)


def test_catalog_refresh_preserves_metadata_without_resurfacing_legacy(catalog_http) -> None:
    config = {"openrouter": {"api_key": "fixture-key"}}
    for _ in range(2):
        models = ModelSelectionCatalog.list_all_models(Provider.OPENROUTER, config)
        assert models == [f"openrouter.{model}" for model in [FLASH, M3, OTHER, MEDIA]]
    legacy = ModelDatabase.get_model_params(f"openrouter.{LEGACY}")
    assert legacy is not None
    assert legacy.context_window == 123456
    assert legacy.max_output_tokens == 8192
    assert "image/png" in legacy.tokenizes
    _, requests = catalog_http
    assert len(requests) == 1  # The second catalog read uses real discovery cache data.


def test_saved_overlay_remains_available(catalog_http, tmp_path: Path) -> None:
    home = tmp_path / "saved-config"
    overlays = home / "model-overlays"
    overlays.mkdir(parents=True)
    (overlays / "saved-legacy.yaml").write_text(
        f"name: saved-legacy\nprovider: openrouter\nmodel: {LEGACY}\n", encoding="utf-8"
    )
    models = ModelSelectionCatalog.list_all_models(
        Provider.OPENROUTER,
        {"openrouter": {"api_key": "fixture-key"}},
        home=home,
        start_path=tmp_path,
    )
    assert models == [f"openrouter.{model}" for model in [LEGACY, FLASH, M3, OTHER, MEDIA]]


def test_missing_current_routes_are_not_synthesized(catalog_http) -> None:
    models, _ = catalog_http
    models[:] = [entry for entry in models if entry["id"] not in {FLASH, M3}]
    assert list_openrouter_model_specs_sync("fixture-key") == [
        f"openrouter.{OTHER}",
        f"openrouter.{MEDIA}",
    ]


def test_custom_gateway_retains_its_own_catalog(catalog_http) -> None:
    models, _ = catalog_http
    assert list_openrouter_model_specs_sync("fixture-key", "https://custom.invalid/v1") == [
        f"openrouter.{entry['id']}" for entry in models
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection", ["catalog", M3, LEGACY, "minimax/private-custom", OTHER, "saved", "preset"]
)
async def test_selected_and_explicit_models_reach_sdk_unchanged(
    catalog_http, selection: str
) -> None:
    config = {"openrouter": {"api_key": "fixture-key"}}
    catalog = ModelSelectionCatalog.list_all_models(Provider.OPENROUTER, config)
    expected = (
        FLASH
        if selection == "catalog"
        else LEGACY
        if selection in {"saved", "preset"}
        else selection
    )
    context = Context(
        config=Settings(openrouter=OpenRouterSettings(api_key="fixture-key", default_model=LEGACY))
    )
    if selection == "saved":
        llm = OpenRouterLLM(context=context, model="")
    else:
        spec = catalog[0] if selection == "catalog" else f"openrouter.{expected}"
        presets = {"saved-choice": spec} if selection == "preset" else None
        factory = ModelFactory.create_factory("saved-choice" if presets else spec, presets=presets)
        llm = factory(MagicMock(instruction=None), context=context)
        assert isinstance(llm, OpenRouterLLM)
    await llm._openai_completion(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            },
        ]
    )
    _, requests = catalog_http
    outgoing = [request for request in requests if request.method == "POST"]
    assert len(outgoing) == 1
    assert str(outgoing[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    body = json.loads(outgoing[0].content)
    assert body["model"] == expected
    assert (
        body["messages"][-1]["content"][1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
    )
    assert "service_tier" not in body
    assert "thinking" not in body
    assert "reasoning_effort" not in body
    assert "reasoning_split" not in body
