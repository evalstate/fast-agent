from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
import typer

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.cli.commands.go import _resolve_model_url_model, _resolve_model_url_options
from fast_agent.cli.constants import GO_SPECIFIC_OPTIONS
from fast_agent.config import HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import (
    clear_ephemeral_model_overlays,
    register_ephemeral_model_overlay,
    serialize_model_overlay_manifest,
)
from fast_agent.llm.model_url_discovery import (
    ModelUrlDiscoveryError,
    build_ephemeral_model_url_overlay,
    discover_model_url_catalog,
    interrogate_model_url_model,
    is_huggingface_endpoint_url,
    normalize_model_url,
    resolve_model_url_auth,
)
from fast_agent.llm.provider.openai.llm_huggingface import HuggingFaceLLM
from fast_agent.llm.provider_types import Provider

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _clean_ephemeral_overlays() -> Iterator[None]:
    clear_ephemeral_model_overlays()
    yield
    clear_ephemeral_model_overlays()


def _mock_transport(
    routes: dict[str, tuple[int, object]],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = request.url.path
        if request.url.query:
            query_key = f"{key}?{request.url.query.decode('utf-8')}"
            if query_key in routes:
                status, body = routes[query_key]
                return httpx.Response(status, json=body)
        if key in routes:
            status, body = routes[key]
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handler), seen


def test_normalize_model_url_and_hf_endpoint_detection() -> None:
    bare = normalize_model_url("https://abc123.endpoints.huggingface.cloud")
    with_v1 = normalize_model_url("https://abc123.endpoints.huggingface.cloud/v1/")
    assert bare.server_url == "https://abc123.endpoints.huggingface.cloud"
    assert bare.request_base_url == "https://abc123.endpoints.huggingface.cloud/v1"
    assert with_v1.request_base_url == "https://abc123.endpoints.huggingface.cloud/v1"

    assert is_huggingface_endpoint_url("https://abc123.endpoints.huggingface.cloud/v1") is True
    assert is_huggingface_endpoint_url("https://ABC123.ENDPOINTS.HUGGINGFACE.CLOUD") is True
    assert is_huggingface_endpoint_url("http://localhost:8000/v1") is False
    assert is_huggingface_endpoint_url("https://router.huggingface.co/v1") is False


def test_auth_precedence_for_hf_endpoints_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://6b020222f37d8c631304ae17.endpoints.huggingface.cloud/v1"
    settings = Settings(hf=HuggingFaceSettings(api_key="hf_from_secrets_yaml"))
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    monkeypatch.setenv("CUSTOM_EP_KEY", "hf_from_custom_env")

    # 1. --model-auth wins and strips optional Bearer prefix.
    auth_explicit = resolve_model_url_auth(
        url,
        model_auth="Bearer hf_from_cli_flag",
        api_key_env="CUSTOM_EP_KEY",
        config=settings,
    )
    assert auth_explicit.provider is Provider.HUGGINGFACE
    assert auth_explicit.source == "model_auth"
    assert auth_explicit.token == "hf_from_cli_flag"
    assert auth_explicit.overlay_connection_auth() == (None, None, "hf_from_cli_flag")
    assert "hf_from_cli_flag" not in repr(auth_explicit)

    # 2. --model-api-key-env wins over HF_TOKEN and secrets file.
    auth_env = resolve_model_url_auth(
        url,
        api_key_env="CUSTOM_EP_KEY",
        config=settings,
    )
    assert auth_env.provider is Provider.HUGGINGFACE
    assert auth_env.source == "api_key_env"
    assert auth_env.token == "hf_from_custom_env"
    assert auth_env.overlay_connection_auth() == ("env", "CUSTOM_EP_KEY", None)

    # 3. HF_TOKEN wins over hf.api_key in fastagent.secrets.yaml.
    auth_hf_env = resolve_model_url_auth(url, config=settings)
    assert auth_hf_env.provider is Provider.HUGGINGFACE
    assert auth_hf_env.source == "hf_token_env"
    assert auth_hf_env.token == "hf_from_env"
    assert auth_hf_env.overlay_connection_auth() == ("env", "HF_TOKEN", None)

    # 4. hf.api_key in fastagent.secrets.yaml is used when HF_TOKEN is absent.
    monkeypatch.delenv("HF_TOKEN")
    auth_hf_file = resolve_model_url_auth(url, config=settings)
    assert auth_hf_file.provider is Provider.HUGGINGFACE
    assert auth_hf_file.source == "hf_secrets_file"
    assert auth_hf_file.token == "hf_from_secrets_yaml"
    assert auth_hf_file.overlay_connection_auth() == (None, None, "hf_from_secrets_yaml")

    # 5. Never silently fall back to ~/.cache/huggingface/token.
    monkeypatch.setattr(
        "fast_agent.utils.huggingface_hub.get_huggingface_hub_token",
        lambda: "hf_from_cache_must_not_be_used",
    )
    auth_none = resolve_model_url_auth(url, config=Settings())
    assert auth_none.provider is Provider.HUGGINGFACE
    assert auth_none.source == "none"
    assert auth_none.token is None
    assert auth_none.overlay_connection_auth() == ("none", None, None)


def test_auth_precedence_for_non_hf_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://localhost:8000/v1"
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)

    unauth = resolve_model_url_auth(url)
    assert unauth.provider is Provider.GENERIC
    assert unauth.source == "none"
    assert unauth.token is None
    assert unauth.overlay_connection_auth() == ("none", None, None)

    monkeypatch.setenv("GENERIC_API_KEY", "generic_secret")
    with_env = resolve_model_url_auth(url)
    assert with_env.provider is Provider.GENERIC
    assert with_env.source == "generic_api_key_env"
    assert with_env.token == "generic_secret"
    assert with_env.overlay_connection_auth() == ("env", "GENERIC_API_KEY", None)

    with pytest.raises(ModelUrlDiscoveryError, match="MISSING_VAR"):
        resolve_model_url_auth(url, api_key_env="MISSING_VAR")


@pytest.mark.asyncio
async def test_discover_hf_vllm_endpoint_builds_custom_inkling_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    url = "https://6b020222f37d8c631304ae17.endpoints.huggingface.cloud"
    transport, requests = _mock_transport(
        {
            "/v1/models": (
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "thinkingmachines/Inkling-2-Mini",
                            "owned_by": "vllm",
                            "root": "thinkingmachines/Inkling-2-Mini",
                            "max_model_len": 196608,
                        }
                    ],
                },
            ),
            "/version": (200, {"version": "0.11.2"}),
        }
    )

    catalog = await discover_model_url_catalog(
        url,
        model_auth="Bearer hf_live_secret_token",
        transport=transport,
    )
    assert catalog.server_kind == "vllm"
    assert catalog.server_version == "0.11.2"
    assert catalog.auth.provider is Provider.HUGGINGFACE
    assert requests[0].headers["Authorization"] == "Bearer hf_live_secret_token"

    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="thinkingmachines/Inkling-2-Mini",
        transport=transport,
    )
    assert discovered.matched_database_model == "thinkingmachines/inkling-2-mini"
    assert discovered.effective_context_window == 196_608

    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    assert overlay.provider is Provider.HUGGINGFACE
    assert overlay.manifest.profile == "thinkingmachines/inkling-2-mini"
    assert "hf_live_secret_token" not in serialize_model_overlay_manifest(overlay.manifest)
    assert "hf_live_secret_token" not in repr(overlay)

    register_ephemeral_model_overlay(overlay)
    resolved = ModelFactory.resolve_model_spec(f"{overlay.name}?reasoning=medium")
    assert resolved.source == "overlay"
    assert resolved.provider is Provider.HUGGINGFACE
    assert resolved.context_window == 196_608
    assert resolved.wire_model_name == "thinkingmachines/Inkling-2-Mini"

    llm = ModelFactory.create_factory_for_resolved_model(resolved)(
        agent=LlmAgent(AgentConfig(name="vllm-hf")),
        context=Context(config=Settings()),
    )
    assert isinstance(llm, HuggingFaceLLM)
    assert llm._base_url() == "https://6b020222f37d8c631304ae17.endpoints.huggingface.cloud/v1"
    assert llm._api_key() == "hf_live_secret_token"
    assert llm._get_model_stream_mode("thinkingmachines/Inkling-2-Mini") == "openai"
    assert llm._preserves_raw_reasoning_history("thinkingmachines/Inkling-2-Mini") is True

    request = llm._prepare_api_request(
        [{"role": "user", "content": "hello"}],
        None,
        llm.default_request_params,
    )
    assert request["model"] == "thinkingmachines/Inkling-2-Mini"
    assert request["reasoning_effort"] == "medium"
    assert "max_tokens" not in request


@pytest.mark.asyncio
async def test_discover_vllm_matches_model_database_by_root_and_overrides_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)
    transport, _ = _mock_transport(
        {
            "/v1/models": (
                200,
                {
                    "data": [
                        {
                            "id": "my-served-inkling",
                            "owned_by": "vllm",
                            "root": "/models/thinkingmachines/Inkling-2-Mini",
                            "max_model_len": 131072,
                        }
                    ]
                },
            ),
        }
    )

    catalog = await discover_model_url_catalog("http://localhost:8000", transport=transport)
    assert catalog.server_kind == "vllm"
    assert catalog.server_version is None
    assert catalog.auth.provider is Provider.GENERIC

    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="my-served-inkling",
        transport=transport,
    )
    assert discovered.matched_database_model == "thinkingmachines/inkling-2-mini"

    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    params = overlay.build_model_parameters()
    assert params is not None
    assert params.context_window == 131_072
    assert params.default_provider is Provider.GENERIC
    base_params = ModelDatabase.get_model_params("thinkingmachines/inkling-2-mini")
    assert base_params is not None
    assert params.reasoning == base_params.reasoning


@pytest.mark.asyncio
async def test_discover_llamacpp_via_model_url_reads_props_and_n_ctx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)
    transport, _ = _mock_transport(
        {
            "/v1/models": (
                200,
                {
                    "data": [
                        {
                            "id": "unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
                            "owned_by": "llamacpp",
                            "meta": {"n_ctx_train": 262144},
                        }
                    ]
                },
            ),
            "/props": (
                200,
                {
                    "default_generation_settings": {
                        "n_ctx": 65536,
                        "params": {"n_predict": 4096},
                    },
                    "model_alias": "Qwen3.5-9B",
                    "modalities": {"vision": True, "audio": False},
                },
            ),
        }
    )

    catalog = await discover_model_url_catalog("http://localhost:8080/v1", transport=transport)
    assert catalog.server_kind == "llamacpp"

    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="unsloth/Qwen3.5-9B-GGUF:Q4_K_M",
        transport=transport,
    )
    assert discovered.effective_context_window == 65_536
    assert discovered.max_output_tokens == 4096
    assert discovered.tokenizes is not None
    assert "image/png" in discovered.tokenizes


@pytest.mark.asyncio
async def test_discover_generic_openai_compatible_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)
    transport, _ = _mock_transport(
        {
            "/v1/models": (
                200,
                {"data": [{"id": "acme/custom-model-v1", "owned_by": "acme"}]},
            ),
        }
    )

    catalog = await discover_model_url_catalog("http://127.0.0.1:9000", transport=transport)
    assert catalog.server_kind == "generic"
    assert catalog.auth.provider is Provider.GENERIC

    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="acme/custom-model-v1",
        transport=transport,
    )
    assert discovered.matched_database_model is None
    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    params = overlay.build_model_parameters()
    assert params is not None
    assert params.context_window == 32_768
    assert params.default_provider is Provider.GENERIC


@pytest.mark.asyncio
async def test_discover_401_errors_are_actionable_and_never_leak_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)
    transport, _ = _mock_transport({"/v1/models": (401, {"error": "Unauthorized"})})

    with pytest.raises(ModelUrlDiscoveryError) as hf_exc:
        await discover_model_url_catalog(
            "https://demo.endpoints.huggingface.cloud/v1",
            model_auth="hf_super_secret_should_not_appear",
            transport=transport,
        )
    hf_msg = str(hf_exc.value)
    assert "401" in hf_msg
    assert "HF_TOKEN" in hf_msg
    assert "hf.api_key" in hf_msg
    assert "fastagent.secrets.yaml" in hf_msg
    assert "--model-auth" in hf_msg
    assert "--model-api-key-env" in hf_msg
    assert "hf_super_secret_should_not_appear" not in hf_msg

    with pytest.raises(ModelUrlDiscoveryError) as generic_exc:
        await discover_model_url_catalog(
            "https://llm.example.test/v1",
            transport=transport,
        )
    generic_msg = str(generic_exc.value)
    assert "401" in generic_msg
    assert "GENERIC_API_KEY" in generic_msg
    assert "--model-auth" in generic_msg
    assert "--model-api-key-env" in generic_msg


def test_go_cli_model_url_option_validation_and_routing() -> None:
    assert "--model-url" in GO_SPECIFIC_OPTIONS
    assert "--model-auth" in GO_SPECIFIC_OPTIONS
    assert "--model-api-key-env" in GO_SPECIFIC_OPTIONS

    with pytest.raises(typer.BadParameter, match="Requires --model-url"):
        _resolve_model_url_options(
            None,
            model_auth="tok",
            model_api_key_env=None,
            local=False,
            base_url=None,
            model=None,
            force_model_picker=False,
            resume=None,
        )

    with pytest.raises(typer.BadParameter, match="Requires --model-url"):
        _resolve_model_url_options(
            None,
            model_auth=None,
            model_api_key_env="MY_KEY",
            local=False,
            base_url=None,
            model=None,
            force_model_picker=False,
            resume=None,
        )

    with pytest.raises(typer.BadParameter, match="Cannot be combined with --local"):
        _resolve_model_url_options(
            "http://localhost:8000",
            model_auth=None,
            model_api_key_env=None,
            local=True,
            base_url=None,
            model=None,
            force_model_picker=False,
            resume=None,
        )

    with pytest.raises(typer.BadParameter, match="Cannot be combined with --base-url"):
        _resolve_model_url_options(
            "http://localhost:8000",
            model_auth=None,
            model_api_key_env=None,
            local=False,
            base_url="http://localhost:8000/v1",
            model=None,
            force_model_picker=False,
            resume=None,
        )

    assert (
        _resolve_model_url_options(
            " https://demo.endpoints.huggingface.cloud ",
            model_auth="Bearer secret",
            model_api_key_env=None,
            local=False,
            base_url=None,
            model="?reasoning=high",
            force_model_picker=False,
            resume=None,
        )
        == "https://demo.endpoints.huggingface.cloud"
    )


def test_resolve_model_url_model_preserves_query_suffix_and_does_not_print_secret(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    transport, _ = _mock_transport(
        {
            "/v1/models": (
                200,
                {
                    "data": [
                        {
                            "id": "thinkingmachines/Inkling-2-Mini",
                            "owned_by": "vllm",
                            "root": "thinkingmachines/Inkling-2-Mini",
                            "max_model_len": 262144,
                        }
                    ]
                },
            ),
            "/version": (200, {"version": "0.11.2"}),
        }
    )

    real_discover = discover_model_url_catalog

    async def _discover_with_mock(
        url: str,
        *,
        model_auth: str | None = None,
        api_key_env: str | None = None,
        config: object | None = None,
    ):
        return await real_discover(
            url,
            model_auth=model_auth,
            api_key_env=api_key_env,
            config=config,
            transport=transport,
        )

    monkeypatch.setattr(
        "fast_agent.llm.model_url_discovery.discover_model_url_catalog",
        _discover_with_mock,
    )

    model_spec = _resolve_model_url_model(
        model_url="https://demo.endpoints.huggingface.cloud",
        model_auth="Bearer hf_never_echo_this_token",
        model_api_key_env=None,
        requested_model="?reasoning=xhigh",
        config_path=None,
        quiet=False,
    )
    assert model_spec == "vllm-inkling-2-mini?reasoning=xhigh"

    captured = capsys.readouterr()
    assert "Endpoint (vllm 0.11.2):" in captured.err
    assert "ctx 262,144" in captured.err
    assert "profile: thinkingmachines/inkling-2-mini" in captured.err
    assert "hf_never_echo_this_token" not in captured.err
    assert "hf_never_echo_this_token" not in captured.out
    assert json.dumps({"spec": model_spec})  # sanity check non-empty string
