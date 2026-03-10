from __future__ import annotations

import argparse
import asyncio
import json
from typing import Literal

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.llm_agent import LlmAgent
from fast_agent.core import Core
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.request_params import RequestParams
from fast_agent.types.llm_stop_reason import LlmStopReason

ProviderName = Literal["responses", "codexresponses"]
TierName = Literal["standard", "fast", "flex"]

DEFAULT_PROVIDERS: tuple[ProviderName, ...] = ("responses", "codexresponses")
DEFAULT_TIERS: tuple[TierName, ...] = ("standard", "fast", "flex")
DEFAULT_PROMPT = "Reply with exactly: ok"
DEFAULT_MODEL = "gpt-5.4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe service-tier support with fast-agent library calls. "
            "Responses uses OPENAI_API_KEY/openai.api_key; codexresponses uses "
            "CODEX_API_KEY/codexresponses.api_key or `fast-agent auth codex-login`."
        )
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=("responses", "codexresponses", "both"),
        help="Provider(s) to probe. Repeat to specify multiple providers. Defaults to both.",
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=("standard", "fast", "flex", "all"),
        help="Service tier(s) to probe. Repeat to specify multiple tiers. Defaults to all.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to probe.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent for each probe.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="maxTokens value for each request.",
    )
    return parser.parse_args()


def _resolve_providers(values: list[str] | None) -> tuple[ProviderName, ...]:
    if not values:
        return DEFAULT_PROVIDERS

    resolved: list[ProviderName] = []
    for value in values:
        if value == "both":
            for provider in DEFAULT_PROVIDERS:
                if provider not in resolved:
                    resolved.append(provider)
            continue
        provider = value
        if provider not in resolved:
            resolved.append(provider)
    return tuple(resolved)


def _resolve_tiers(values: list[str] | None) -> tuple[TierName, ...]:
    if not values:
        return DEFAULT_TIERS

    resolved: list[TierName] = []
    for value in values:
        if value == "all":
            for tier in DEFAULT_TIERS:
                if tier not in resolved:
                    resolved.append(tier)
            continue
        tier = value
        if tier not in resolved:
            resolved.append(tier)
    return tuple(resolved)


def _build_request_params(tier: TierName, max_tokens: int) -> RequestParams:
    service_tier: Literal["fast", "flex"] | None = None
    if tier == "fast":
        service_tier = "fast"
    elif tier == "flex":
        service_tier = "flex"

    return RequestParams(maxTokens=max_tokens, service_tier=service_tier, use_history=False)


async def _probe_once(
    *,
    core: Core,
    provider: ProviderName,
    model: str,
    tier: TierName,
    prompt: str,
    max_tokens: int,
) -> dict[str, object]:
    model_spec = f"{provider}.{model}"
    agent = LlmAgent(AgentConfig(name=f"probe-{provider}-{tier}", model=model_spec), core.context)
    try:
        await agent.attach_llm(ModelFactory.create_factory(model_spec))
        result = await agent.generate(
            prompt,
            request_params=_build_request_params(tier=tier, max_tokens=max_tokens),
        )
        text = (result.last_text() or "").strip()
        ok = result.stop_reason is not LlmStopReason.ERROR
        return {
            "ok": ok,
            "provider": provider,
            "model": model,
            "tier": tier,
            "text": text,
            "stop_reason": str(result.stop_reason),
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "tier": tier,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


async def _run() -> int:
    args = _parse_args()
    providers = _resolve_providers(args.provider)
    tiers = _resolve_tiers(args.tier)

    core = Core()
    await core.initialize()
    try:
        results: list[dict[str, object]] = []
        for provider in providers:
            for tier in tiers:
                result = await _probe_once(
                    core=core,
                    provider=provider,
                    model=args.model,
                    tier=tier,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))

        failures = [result for result in results if not bool(result.get("ok"))]
        if failures:
            print(
                json.dumps(
                    {
                        "summary": "one or more probes failed",
                        "failures": len(failures),
                        "total": len(results),
                    },
                    ensure_ascii=False,
                )
            )
            return 1

        print(
            json.dumps(
                {
                    "summary": "all probes succeeded",
                    "total": len(results),
                },
                ensure_ascii=False,
            )
        )
        return 0
    finally:
        await core.cleanup()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run()))
