from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import typer
from openai._models import FinalRequestOptions

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.cli.commands.go import (
    _resolve_model_url_model,
    _resolve_model_url_options,
    _select_model_url_listing,
)
from fast_agent.cli.constants import GO_SPECIFIC_OPTIONS
from fast_agent.config import GenericSettings, HuggingFaceSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.model_database import ModelDatabase
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.model_overlays import (
    LoadedModelOverlay,
    ModelOverlayConnection,
    ModelOverlayManifest,
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
from fast_agent.llm.provider.openai.llm_generic import GenericLLM
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


@pytest.mark.asyncio
async def test_no_auth_model_url_overlay_constructs_openai_client_without_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_name in ("OPENAI_API_KEY", "GENERIC_API_KEY", "HF_TOKEN"):
        monkeypatch.delenv(env_name, raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "local-qwen", "owned_by": "vllm", "max_model_len": 16384}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog(
        "http://localhost:8000",
        transport=transport,
    )
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="local-qwen",
        transport=transport,
    )
    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    register_ephemeral_model_overlay(overlay)

    context = Context(config=Settings())
    agent = LlmAgent(AgentConfig("no-auth-agent"), context=context)
    llm = ModelFactory.create_factory(overlay.name)(agent)
    assert isinstance(llm, GenericLLM)

    client = llm._openai_client()
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"
    request_args = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )
    assert request_args["model"] == "local-qwen"

    built_request = client._build_request(
        FinalRequestOptions(method="post", url="/chat/completions", json_data=request_args)
    )
    lower_headers = {k.lower(): v for k, v in built_request.headers.items()}
    assert "authorization" not in lower_headers


@pytest.mark.asyncio
async def test_model_url_overlay_suppresses_inherited_provider_and_sdk_ambient_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GENERIC_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-OpenAI-Gateway: sdk-leak, X-API-Key: sdk-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "org-leak")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-leak")

    settings = Settings(
        generic=GenericSettings(default_headers={"X-API-Key": "generic-gateway-secret"}),
        hf=HuggingFaceSettings(default_headers={"X-Gateway-Key": "hf-gateway-secret"}),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "served-model", "owned_by": "vllm", "max_model_len": 32768}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    cases: tuple[tuple[str, str | None, str, str | None], ...] = (
        ("http://localhost:8000", None, "ephemeral-generic-no-auth", None),
        (
            "https://llm.example.com/v1",
            "explicit_model_auth_tok",
            "ephemeral-generic-explicit-auth",
            "Bearer explicit_model_auth_tok",
        ),
        (
            "https://demo.endpoints.huggingface.cloud",
            None,
            "ephemeral-hf-origin",
            "Bearer hf_test_token",
        ),
    )
    for url, model_auth, overlay_name, expected_auth in cases:
        catalog = await discover_model_url_catalog(
            url,
            model_auth=model_auth,
            config=settings,
            transport=transport,
        )
        discovered = await interrogate_model_url_model(
            catalog=catalog,
            model_id="served-model",
            transport=transport,
        )
        overlay = build_ephemeral_model_url_overlay(
            discovered=discovered,
            catalog=catalog,
            overlay_name=overlay_name,
        )
        register_ephemeral_model_overlay(overlay)

        context = Context(config=settings)
        agent = LlmAgent(AgentConfig(f"agent-{overlay_name}"), context=context)
        llm = ModelFactory.create_factory(overlay.name)(agent, context=context)
        assert isinstance(llm, (GenericLLM, HuggingFaceLLM))
        assert llm._default_headers() == {}
        client = llm._openai_client()
        built = client._build_request(
            FinalRequestOptions(
                method="post",
                url="/chat/completions",
                json_data={"model": "served-model", "messages": []},
            )
        )
        lower_headers = {k.lower(): v for k, v in built.headers.items()}
        assert "x-api-key" not in lower_headers
        assert "x-gateway-key" not in lower_headers
        assert "x-openai-gateway" not in lower_headers
        assert "openai-organization" not in lower_headers
        assert "openai-project" not in lower_headers
        if expected_auth is None:
            assert "authorization" not in lower_headers
        else:
            assert lower_headers.get("authorization") == expected_auth


def test_schemeless_hf_endpoint_defaults_to_https_and_handles_trailing_dot() -> None:
    assert is_huggingface_endpoint_url("demo.endpoints.huggingface.cloud")
    assert is_huggingface_endpoint_url("demo.endpoints.huggingface.cloud.")
    assert is_huggingface_endpoint_url("https://demo.endpoints.huggingface.cloud./v1")
    assert not is_huggingface_endpoint_url("https://demo.endpoints.huggingface.cloud.evil.com/v1")

    endpoints = normalize_model_url("demo.endpoints.huggingface.cloud")
    assert endpoints.request_base_url == "https://demo.endpoints.huggingface.cloud/v1"

    trailing = normalize_model_url("demo.endpoints.huggingface.cloud.")
    assert trailing.request_base_url == "https://demo.endpoints.huggingface.cloud./v1"


def test_explicit_http_hf_endpoint_refuses_cleartext_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_never_send_over_http")
    with pytest.raises(ModelUrlDiscoveryError, match="require https://"):
        resolve_model_url_auth("http://demo.endpoints.huggingface.cloud/v1")


def test_remote_cleartext_http_does_not_attach_ambient_generic_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERIC_API_KEY", "generic_ambient_secret")

    remote_http = resolve_model_url_auth("http://gpu-box.internal:8000")
    assert remote_http.source == "none"
    assert remote_http.token is None

    local_http = resolve_model_url_auth("http://localhost:8000")
    assert local_http.source == "generic_api_key_env"
    assert local_http.token == "generic_ambient_secret"

    remote_https = resolve_model_url_auth("https://gpu-box.internal:8000")
    assert remote_https.source == "generic_api_key_env"
    assert remote_https.token == "generic_ambient_secret"


def test_model_url_rejects_userinfo_without_leaking_credentials() -> None:
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        normalize_model_url("https://alice:super-secret-pw@demo.endpoints.huggingface.cloud/v1")
    message = str(exc_info.value)
    assert "userinfo" in message.lower()
    assert "super-secret-pw" not in message
    assert "alice" not in message


def test_model_url_rejects_control_chars_in_tokens_without_leaking_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_secret = "tok_secret_value\r\nX-Injected: 1"
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        resolve_model_url_auth(
            "https://demo.endpoints.huggingface.cloud",
            model_auth=bad_secret,
        )
    assert "control characters" in str(exc_info.value)
    assert "tok_secret_value" not in str(exc_info.value)

    monkeypatch.setenv("BAD_CUSTOM_KEY", bad_secret)
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        resolve_model_url_auth(
            "https://llm.example.com/v1",
            api_key_env="BAD_CUSTOM_KEY",
        )
    assert "tok_secret_value" not in str(exc_info.value)

    monkeypatch.setenv("HF_TOKEN", bad_secret)
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        resolve_model_url_auth("https://demo.endpoints.huggingface.cloud")
    assert "tok_secret_value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_transport_error_does_not_leak_raw_exception_secret() -> None:
    def failing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed with Authorization: Bearer super-secret-transport-token")

    transport = httpx.MockTransport(failing_handler)
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        await discover_model_url_catalog(
            "https://demo.endpoints.huggingface.cloud",
            model_auth="hf_real_token",
            transport=transport,
        )
    message = str(exc_info.value)
    assert "ConnectError" in message
    assert "super-secret-transport-token" not in message
    assert "hf_real_token" not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_ephemeral_model_url_preserves_exact_wire_id_and_hf_profile_route() -> None:
    # Case A: lowercase wire ID must NOT be rewritten to Inkling-2-Mini casing
    def lowercase_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "thinkingmachines/inkling-2-mini",
                            "owned_by": "vllm",
                            "max_model_len": 131072,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport_a = httpx.MockTransport(lowercase_handler)
    catalog_a = await discover_model_url_catalog(
        "https://demo.endpoints.huggingface.cloud",
        model_auth="hf_test_token",
        transport=transport_a,
    )
    discovered_a = await interrogate_model_url_model(
        catalog=catalog_a,
        model_id="thinkingmachines/inkling-2-mini",
        transport=transport_a,
    )
    overlay_a = build_ephemeral_model_url_overlay(
        discovered=discovered_a,
        catalog=catalog_a,
        overlay_name="hf-lowercase-inkling",
    )
    register_ephemeral_model_overlay(overlay_a)

    context = Context(config=Settings())
    agent_a = LlmAgent(AgentConfig("hf-lowercase"), context=context)
    llm_a = ModelFactory.create_factory(overlay_a.name)(agent_a)
    assert isinstance(llm_a, HuggingFaceLLM)
    args_a = llm_a._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm_a.default_request_params,
    )
    assert args_a["model"] == "thinkingmachines/inkling-2-mini"
    assert args_a["reasoning_effort"] == "high"
    assert "max_tokens" not in args_a

    # Case B: renamed HF endpoint matched via `root` retains Inkling custom-endpoint profile
    def renamed_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "acme/custom-inkling-alias",
                            "root": "thinkingmachines/Inkling-2-Mini",
                            "owned_by": "vllm",
                            "max_model_len": 131072,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport_b = httpx.MockTransport(renamed_handler)
    catalog_b = await discover_model_url_catalog(
        "https://demo.endpoints.huggingface.cloud",
        model_auth="hf_test_token",
        transport=transport_b,
    )
    discovered_b = await interrogate_model_url_model(
        catalog=catalog_b,
        model_id="acme/custom-inkling-alias",
        transport=transport_b,
    )
    assert discovered_b.matched_database_model == "thinkingmachines/inkling-2-mini"
    overlay_b = build_ephemeral_model_url_overlay(
        discovered=discovered_b,
        catalog=catalog_b,
        overlay_name="hf-renamed-inkling",
    )
    register_ephemeral_model_overlay(overlay_b)

    agent_b = LlmAgent(AgentConfig("hf-renamed"), context=context)
    llm_b = ModelFactory.create_factory(overlay_b.name)(agent_b)
    assert isinstance(llm_b, HuggingFaceLLM)
    route_profile_b = llm_b._route_profile(llm_b.default_request_params.model)
    assert route_profile_b is not None
    assert route_profile_b.preserve_raw_reasoning is True
    assert route_profile_b.omit_default_max_tokens is True
    args_b = llm_b._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm_b.default_request_params,
    )
    assert args_b["model"] == "acme/custom-inkling-alias"
    assert args_b["reasoning_effort"] == "high"
    assert "max_tokens" not in args_b


@pytest.mark.asyncio
async def test_select_model_url_listing_exact_id_beats_earlier_alias_and_rejects_ambiguous() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "model-alpha",
                            "aliases": ["target", "shared-alias"],
                            "owned_by": "vllm",
                            "max_model_len": 8192,
                        },
                        {
                            "id": "target",
                            "aliases": ["shared-alias"],
                            "owned_by": "vllm",
                            "max_model_len": 16384,
                        },
                    ]
                },
            )
        return httpx.Response(404)

    catalog = await discover_model_url_catalog(
        "http://localhost:8000",
        transport=httpx.MockTransport(handler),
    )

    # Exact wire ID ("target") must win over earlier item's alias ("target")
    selected = _select_model_url_listing(catalog, requested_model="target")
    assert selected.model_id == "target"

    # Ambiguous alias match must fail cleanly
    with pytest.raises(typer.BadParameter, match="matched multiple models"):
        _select_model_url_listing(catalog, requested_model="shared-alias")


@pytest.mark.asyncio
async def test_multimodel_llamacpp_never_falls_back_to_unqualified_props() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(str(request.url))
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "model-a", "owned_by": "llamacpp", "meta": {"n_ctx_train": 131072}},
                        {"id": "model-b", "owned_by": "llamacpp", "meta": {"n_ctx_train": 16384}},
                    ]
                },
            )
        if request.url.path == "/props" and request.url.query == b"":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": 65536},
                    "modalities": {"vision": True, "audio": False},
                    "model_alias": "model-a-alias",
                },
            )
        if request.url.path == "/props" and request.url.query == b"model=model-b":
            return httpx.Response(404)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog("http://localhost:9931", transport=transport)
    assert catalog.server_kind == "llamacpp"

    seen_paths.clear()
    discovered_b = await interrogate_model_url_model(
        catalog=catalog,
        model_id="model-b",
        transport=transport,
    )
    assert seen_paths == ["http://localhost:9931/props?model=model-b"]
    assert discovered_b.runtime_context_window is None
    assert discovered_b.effective_context_window == 16384
    assert discovered_b.tokenizes is None
    assert discovered_b.model_alias is None


@pytest.mark.asyncio
async def test_llamacpp_owner_takes_precedence_over_max_model_len_heuristic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "qwen-gguf",
                            "owned_by": "llamacpp",
                            "max_model_len": 16384,
                        }
                    ]
                },
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={"default_generation_settings": {"n_ctx": 32768}},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog("http://localhost:9931", transport=transport)
    assert catalog.server_kind == "llamacpp"
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="qwen-gguf",
        transport=transport,
    )
    assert discovered.effective_context_window == 32768


@pytest.mark.asyncio
async def test_models_payload_tolerates_string_status_and_null_aliases() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "vendor-model",
                            "owned_by": "custom",
                            "status": "available",
                            "aliases": None,
                            "meta": {"n_ctx_train": "not-an-int"},
                        }
                    ]
                },
            )
        return httpx.Response(404)

    catalog = await discover_model_url_catalog(
        "http://localhost:8000",
        transport=httpx.MockTransport(handler),
    )
    assert len(catalog.models) == 1
    assert catalog.models[0].model_id == "vendor-model"
    assert catalog.models[0].status == "available"
    assert catalog.models[0].aliases == ()


@pytest.mark.asyncio
async def test_live_n_predict_is_clamped_to_context_window() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "tiny-ctx-model", "owned_by": "llamacpp"}]},
            )
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {
                        "n_ctx": 8192,
                        "params": {"n_predict": 32768},
                    }
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog("http://localhost:9931", transport=transport)
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="tiny-ctx-model",
        transport=transport,
    )
    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    assert overlay.override_model_parameters is not None
    assert overlay.override_model_parameters.context_window == 8192
    assert overlay.override_model_parameters.max_output_tokens == 8192
    assert overlay.manifest.defaults.max_tokens == 8192
    assert overlay.manifest.metadata.max_output_tokens == 8192


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://[::1",
        "http://localhost:99999",
        "https://user:topsecret@example.com/v1",
    ],
)
def test_resolve_model_url_model_handles_malformed_url_and_userinfo_cleanly(
    bad_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(typer.Exit) as exc_info:
        _resolve_model_url_model(
            model_url=bad_url,
            model_auth=None,
            model_api_key_env=None,
            requested_model=None,
            config_path=None,
            quiet=False,
        )
    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "topsecret" not in captured.err
    assert "topsecret" not in captured.out


def test_malformed_nfkc_host_with_userinfo_does_not_leak_credentials() -> None:
    with pytest.raises(ModelUrlDiscoveryError) as exc_info:
        normalize_model_url("https://alice:super-secret-nfkc@bad℀host.example/v1")
    message = str(exc_info.value)
    assert "super-secret-nfkc" not in message
    assert "alice" not in message
    assert exc_info.value.__cause__ is None


def test_discover_model_url_catalog_does_not_expose_unvalidated_auth_override() -> None:
    params = inspect.signature(discover_model_url_catalog).parameters
    assert "auth" not in params


def test_cli_rejects_combining_model_auth_and_model_api_key_env() -> None:
    with pytest.raises(typer.BadParameter, match="--model-api-key-env"):
        _resolve_model_url_options(
            model_url="https://llm.example.com/v1",
            model_auth="Bearer secret",
            model_api_key_env="MY_ENDPOINT_KEY",
            local=False,
            base_url=None,
            model=None,
            force_model_picker=False,
            resume=None,
        )


@pytest.mark.asyncio
async def test_model_url_discovery_matches_google_gemma_4_without_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "google/gemma-4-31b-it",
                            "owned_by": "vllm",
                            "max_model_len": 131072,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog("http://localhost:8000", transport=transport)
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="google/gemma-4-31b-it",
        transport=transport,
    )
    assert discovered.matched_database_model == "google/gemma-4-31b-it"
    overlay = build_ephemeral_model_url_overlay(discovered=discovered, catalog=catalog)
    assert overlay.manifest.profile == "google/gemma-4-31b-it"
    assert overlay.manifest.model == "google/gemma-4-31b-it"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "opaque_id",
    [
        "generic/foo",
        "generic.foo",
        "myalias.high",
        "model?v=2",
    ],
)
async def test_opaque_discovered_wire_model_ids_survive_spec_resolution(opaque_id: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": opaque_id, "owned_by": "vllm", "max_model_len": 16384}]},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog("http://localhost:8000", transport=transport)
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id=opaque_id,
        transport=transport,
    )
    overlay = build_ephemeral_model_url_overlay(
        discovered=discovered,
        catalog=catalog,
        overlay_name="opaque-wire-test",
    )
    register_ephemeral_model_overlay(overlay)

    resolved = ModelFactory.resolve_model_spec(overlay.name)
    assert resolved.wire_model_name == opaque_id
    assert resolved.model_config.model_name == opaque_id

    context = Context(config=Settings())
    agent = LlmAgent(AgentConfig("opaque-agent"), context=context)
    llm = ModelFactory.create_factory_for_resolved_model(resolved)(agent)
    assert isinstance(llm, GenericLLM)
    args = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )
    assert args["model"] == opaque_id


@pytest.mark.asyncio
async def test_dedicated_hf_opaque_id_with_colon_retains_wire_id_and_custom_profile() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "acme/custom-inkling:fp8",
                            "root": "thinkingmachines/Inkling-2-Mini",
                            "owned_by": "vllm",
                            "max_model_len": 131072,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    catalog = await discover_model_url_catalog(
        "https://demo.endpoints.huggingface.cloud",
        model_auth="hf_test_tok",
        transport=transport,
    )
    discovered = await interrogate_model_url_model(
        catalog=catalog,
        model_id="acme/custom-inkling:fp8",
        transport=transport,
    )
    assert discovered.matched_database_model == "thinkingmachines/inkling-2-mini"
    overlay = build_ephemeral_model_url_overlay(
        discovered=discovered,
        catalog=catalog,
        overlay_name="hf-colon-inkling",
    )
    register_ephemeral_model_overlay(overlay)

    context = Context(config=Settings())
    agent = LlmAgent(AgentConfig("hf-colon"), context=context)
    llm = ModelFactory.create_factory(overlay.name)(agent)
    assert isinstance(llm, HuggingFaceLLM)
    assert llm._hf_provider_suffix is None
    profile = llm._route_profile(llm.default_request_params.model)
    assert profile is not None
    assert profile.preserve_raw_reasoning is True
    assert profile.omit_default_max_tokens is True
    args = llm._prepare_api_request(
        [{"role": "user", "content": "hi"}],
        None,
        llm.default_request_params,
    )
    assert args["model"] == "acme/custom-inkling:fp8"
    assert args["reasoning_effort"] == "high"
    assert "max_tokens" not in args


def test_saved_overlay_preserves_same_origin_headers_and_isolates_different_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "X-OpenAI-Leak: should-not-leak")
    settings = Settings(
        generic=GenericSettings(
            base_url="https://gateway.corp/v1",
            default_headers={"X-Corp-Key": "keep-same-origin"},
        )
    )
    context = Context(config=settings)

    same_origin = LoadedModelOverlay(
        manifest=ModelOverlayManifest(
            name="saved-same-origin",
            provider=Provider.GENERIC,
            model="corp-model",
            connection=ModelOverlayConnection(
                base_url="https://gateway.corp/v1",
                auth="none",
            ),
        ),
        manifest_path=Path("saved-same-origin.yaml"),
    )
    register_ephemeral_model_overlay(same_origin)
    agent_same = LlmAgent(AgentConfig("same-origin"), context=context)
    llm_same = ModelFactory.create_factory(same_origin.name)(agent_same, context=context)
    assert isinstance(llm_same, GenericLLM)
    assert llm_same._default_headers() == {"X-Corp-Key": "keep-same-origin"}
    req_same = llm_same._openai_client()._build_request(
        FinalRequestOptions(
            method="post",
            url="/chat/completions",
            json_data={"model": "corp-model", "messages": []},
        )
    )
    same_headers = {k.lower(): v for k, v in req_same.headers.items()}
    assert same_headers.get("x-corp-key") == "keep-same-origin"
    assert "authorization" not in same_headers

    diff_origin = LoadedModelOverlay(
        manifest=ModelOverlayManifest(
            name="saved-diff-origin",
            provider=Provider.GENERIC,
            model="other-model",
            connection=ModelOverlayConnection(
                base_url="https://other-host.example/v1",
                auth="none",
            ),
        ),
        manifest_path=Path("saved-diff-origin.yaml"),
    )
    register_ephemeral_model_overlay(diff_origin)
    agent_diff = LlmAgent(AgentConfig("diff-origin"), context=context)
    llm_diff = ModelFactory.create_factory(diff_origin.name)(agent_diff, context=context)
    assert isinstance(llm_diff, GenericLLM)
    assert llm_diff._default_headers() == {}
    req_diff = llm_diff._openai_client()._build_request(
        FinalRequestOptions(
            method="post",
            url="/chat/completions",
            json_data={"model": "other-model", "messages": []},
        )
    )
    diff_headers = {k.lower(): v for k, v in req_diff.headers.items()}
    assert "x-corp-key" not in diff_headers
    assert "x-openai-leak" not in diff_headers
    assert "authorization" not in diff_headers
