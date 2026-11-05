"""HF/Together's legacy schema envelope must retain schema constraints and route scope."""

from copy import deepcopy
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["fruit", "vegetable"]
    quantity: int


class Inventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Item]


def _llm(model: str, settings: HuggingFaceSettings | None = None) -> HuggingFaceLLM:
    return HuggingFaceLLM(
        context=Context(config=Settings(hf=settings or HuggingFaceSettings())),
        model=model,
    )


@pytest.mark.parametrize("api", ["typed", "schema"])
@pytest.mark.parametrize("provider_source", ["suffix", "config"])
def test_kimi_together_preserves_generated_schema(api: str, provider_source: str) -> None:
    llm = _llm(
        "moonshotai/Kimi-K3.1-Pro:together"
        if provider_source == "suffix"
        else "moonshotai/Kimi-K3.1-Pro",
        HuggingFaceSettings(default_provider="together" if provider_source == "config" else None),
    )
    response_format = (
        llm.model_to_response_format(Inventory)
        if api == "typed"
        else llm.schema_to_response_format(Inventory.model_json_schema())
    )
    assert response_format is not None
    original = deepcopy(response_format)
    params = llm.default_request_params.model_copy(update={"response_format": response_format})

    arguments = llm._prepare_api_request([{"role": "user", "content": "Inventory?"}], [], params)

    assert llm._structured_json_mode(params) == "schema"
    assert arguments["model"] == "moonshotai/Kimi-K3.1-Pro:together"
    assert arguments["response_format"] == {
        "type": "json_object",
        "schema": original["json_schema"]["schema"],
    }
    assert "tools" not in arguments
    assert params.response_format == original


@pytest.mark.parametrize(
    ("model", "settings"),
    [
        ("moonshotai/Kimi-K3.1-Pro:novita", HuggingFaceSettings(default_provider="together")),
        ("moonshotai/Kimi-K3.1-Pro", HuggingFaceSettings(default_provider="auto")),
        ("moonshotai/Kimi-K3:together", HuggingFaceSettings()),
        (
            "moonshotai/Kimi-K3.1-Pro:together",
            HuggingFaceSettings(base_url="https://endpoint.example/v1"),
        ),
        (
            "moonshotai/Kimi-K3.1-Pro:together",
            HuggingFaceSettings(base_url="https://router.huggingface.co/together/v1"),
        ),
    ],
)
def test_other_routes_keep_standard_schema_format(
    model: str, settings: HuggingFaceSettings
) -> None:
    llm = _llm(model, settings)
    response_format = llm.schema_to_response_format(Inventory.model_json_schema())
    params = llm.default_request_params.model_copy(update={"response_format": response_format})

    arguments = llm._prepare_api_request([{"role": "user", "content": "Inventory?"}], None, params)

    assert arguments["response_format"] == response_format


@pytest.mark.parametrize("response_format", [None, {"type": "text"}, {"type": "json_object"}])
def test_non_schema_requests_are_unchanged(response_format: dict[str, str] | None) -> None:
    llm = _llm("moonshotai/Kimi-K3.1-Pro:together")
    params = llm.default_request_params.model_copy(update={"response_format": response_format})

    arguments = llm._prepare_api_request([{"role": "user", "content": "Hello"}], None, params)

    assert arguments.get("response_format") == response_format
