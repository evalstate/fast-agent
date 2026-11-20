from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol

from fast_agent.llm.router_profiles import RouterProfileRegistry, RouterProfileRule

if TYPE_CHECKING:
    from fast_agent.llm.reasoning_effort import ReasoningEffortSetting, ReasoningEffortSpec


class HuggingFaceReasoningProfile(Protocol):
    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None: ...


def _effective_setting(
    setting: ReasoningEffortSetting | None,
    spec: ReasoningEffortSpec | None,
) -> ReasoningEffortSetting | None:
    return setting or (spec.default if spec else None)


def _extra_body(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("extra_body")
    return raw if isinstance(raw, dict) else {}


def _commit_extra_body(arguments: dict[str, Any], extra_body: dict[str, Any]) -> None:
    if extra_body:
        arguments["extra_body"] = extra_body
    else:
        arguments.pop("extra_body", None)


def _set_chat_template_kwarg(
    extra_body: dict[str, Any],
    key: str,
    value: object,
) -> None:
    raw = extra_body.get("chat_template_kwargs")
    chat_kwargs = raw if isinstance(raw, dict) else {}
    chat_kwargs[key] = value
    extra_body["chat_template_kwargs"] = chat_kwargs


@dataclass(frozen=True, slots=True)
class TopLevelReasoningEffort:
    default_effort: str | None
    allowed_efforts: frozenset[str] | None = None
    disabled_efforts: frozenset[str] = frozenset()
    effort_map: Mapping[str, str] | None = None
    toggle_disable: bool = True
    chat_template_toggle_field: str | None = None
    chat_template_kwargs: Mapping[str, object] | None = None
    cleanup_extra_body: frozenset[str] = frozenset()

    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        if effective is not None and effective.kind == "toggle" and self.chat_template_toggle_field:
            extra_body = _extra_body(arguments)
            _set_chat_template_kwarg(
                extra_body,
                self.chat_template_toggle_field,
                bool(effective.value),
            )
            _commit_extra_body(arguments, extra_body)
            if effective.value is False:
                arguments.pop("reasoning_effort", None)
                return

        effort = self.default_effort
        disabled = False
        if effective is not None:
            if effective.kind == "toggle":
                disabled = self.toggle_disable and effective.value is False
            elif effective.kind == "effort" and isinstance(effective.value, str):
                disabled = effective.value in self.disabled_efforts
                if self.allowed_efforts is None or effective.value in self.allowed_efforts:
                    effort = effective.value

        if disabled:
            effort = "none"
        elif self.effort_map and effort is not None:
            effort = self.effort_map.get(effort, effort)
        if effort is None:
            arguments.pop("reasoning_effort", None)
        else:
            arguments["reasoning_effort"] = effort

        if self.chat_template_kwargs:
            extra_body = _extra_body(arguments)
            for key, value in self.chat_template_kwargs.items():
                _set_chat_template_kwarg(extra_body, key, value)
            _commit_extra_body(arguments, extra_body)

        if not self.cleanup_extra_body:
            return
        raw = arguments.get("extra_body")
        if not isinstance(raw, dict):
            return
        for key in self.cleanup_extra_body:
            raw.pop(key, None)
        _commit_extra_body(arguments, raw)


@dataclass(frozen=True, slots=True)
class ThinkingWithReasoningEffort:
    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        effort = "max"
        disabled = False
        if effective is not None:
            if effective.kind == "toggle":
                disabled = effective.value is False
            elif effective.kind == "effort" and isinstance(effective.value, str):
                disabled = effective.value in {"none", "minimal"}
                effort = effective.value

        extra_body = _extra_body(arguments)
        if disabled:
            extra_body["thinking"] = {"type": "disabled"}
            arguments.pop("reasoning_effort", None)
        else:
            extra_body["thinking"] = {"type": "enabled", "clear_thinking": False}
            arguments["reasoning_effort"] = effort
        arguments["extra_body"] = extra_body


@dataclass(frozen=True, slots=True)
class ProviderDefaultReasoningToggle:
    disable_api: Literal["thinking", "chat_template_thinking"]

    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        if effective is None or effective.kind != "toggle" or effective.value is not False:
            return

        extra_body = _extra_body(arguments)
        if self.disable_api == "thinking":
            extra_body["thinking"] = {"type": "disabled"}
        else:
            _set_chat_template_kwarg(extra_body, "thinking", False)
        arguments["extra_body"] = extra_body


@dataclass(frozen=True, slots=True)
class ChatTemplateReasoningToggle:
    field: str

    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        if effective is None or effective.kind != "toggle":
            return
        extra_body = _extra_body(arguments)
        _set_chat_template_kwarg(extra_body, self.field, bool(effective.value))
        arguments["extra_body"] = extra_body


@dataclass(frozen=True, slots=True)
class GenericDisableReasoningToggle:
    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        if effective is None or effective.kind != "toggle":
            return
        if setting is None and effective.value is not False:
            return
        extra_body = _extra_body(arguments)
        extra_body["disable_reasoning"] = not bool(effective.value)
        arguments["extra_body"] = extra_body


@dataclass(frozen=True, slots=True)
class HuggingFaceRouteProfile:
    reasoning: HuggingFaceReasoningProfile | None = None
    # Replay streamed reasoning chunks verbatim in history instead of normalizing them.
    preserve_raw_reasoning: bool = False
    structured_json_mode: Literal["schema", "object"] | None = None
    # Together's legacy schema envelope avoids HF's incompatible json_schema flattening.
    use_legacy_json_schema: bool = False
    # Backend-specific tool-call fragments the SDK accumulator rejects (explicit null type).
    stream_mode: Literal["openai", "manual"] | None = None
    omit_default_max_tokens: bool = False
    max_output_tokens: int | None = None
    default_max_tokens: int | None = None
    context_window: int | None = None
    prompt_context_window: int | None = None


HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND = "custom-endpoint"

_DEEPSEEK_V4_FLASH = "deepseek-ai/deepseek-v4-flash-0731"
_DEEPSEEK_V41_FLASH = "deepseek-ai/deepseek-v4.1-flash"
_DEEPSEEK_V41_FLASH_1017 = "deepseek-ai/deepseek-v4.1-flash-1017"
_GLM_52 = "zai-org/glm-5.2"
_KIMI_K3 = "moonshotai/kimi-k3"
_GEMMA_4 = "google/gemma-4-31b-it"
_MUSE_GLIMMER = "meta-models/muse-glimmer-30b"
_QWEN_38 = "qwen/qwen3.8-27b"
_INKLING_2_SMALL = "thinkingmachines/inkling-2-small"
_BEAM_1 = "reflection/beam-1.0-501b-a23b"
_MUSE_GLIMMER_PROMPT_CONTEXT_WINDOW = 131_072 - 32_768

# The HF router now forwards these routes through OpenRouter (2026-11-15).
_ROUTED_CLEANUP = frozenset({"thinking", "chat_template_kwargs", "disable_reasoning"})
_ROUTED_TOGGLE = TopLevelReasoningEffort(default_effort=None, cleanup_extra_body=_ROUTED_CLEANUP)
_DEEPSEEK_V41_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"low", "high", "max"}),
    toggle_disable=False,
)
_DEEPSEEK_REASONING = TopLevelReasoningEffort(default_effort="max")
# Both V4.1 Flash checkpoints on Novita: backend output default instead of max_tokens.
_DEEPSEEK_NOVITA_PROFILE = HuggingFaceRouteProfile(
    reasoning=_DEEPSEEK_REASONING,
    omit_default_max_tokens=True,
)
_KIMI_K3_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"low", "high", "max"}),
    toggle_disable=False,
    cleanup_extra_body=frozenset({"thinking", "chat_template_kwargs"}),
)
# GLM-5.3 and Flash share the common low/high/max effort contract.
# Retain Together's clear_thinking until its multi-turn effect is established.
# Z.ai's thinking object (including clear_thinking) was dropped on single-turn
# evidence; replay still worked in a post-change tool loop.
_GLM_53_TOGETHER_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"low", "high", "max"}),
    toggle_disable=False,
    chat_template_kwargs={"clear_thinking": False},
    cleanup_extra_body=frozenset({"thinking"}),
)
# Novita accepts top-level effort; do not assume Z.ai-specific thinking controls.
_GLM_53_NOVITA_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"low", "high", "max"}),
    toggle_disable=False,
    cleanup_extra_body=frozenset({"thinking"}),
)
_GLM_52_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    disabled_efforts=frozenset({"none", "minimal"}),
    cleanup_extra_body=_ROUTED_CLEANUP,
)
_GLM_52_DEEPINFRA_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    disabled_efforts=frozenset({"none", "minimal"}),
    effort_map={"max": "xhigh"},
    cleanup_extra_body=frozenset({"thinking"}),
)
_MUSE_GLIMMER_REASONING = TopLevelReasoningEffort(
    default_effort="high",
    allowed_efforts=frozenset({"low", "medium", "high", "xhigh"}),
    toggle_disable=False,
    cleanup_extra_body=_ROUTED_CLEANUP,
)
_BEAM_1_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"minimal", "low", "medium", "high", "xhigh", "max"}),
    toggle_disable=False,
    cleanup_extra_body=_ROUTED_CLEANUP,
)
# OpenRouter requires reasoning on both Inkling routes; "none" is rejected.
_INKLING_2_REASONING = TopLevelReasoningEffort(
    default_effort="high",
    allowed_efforts=frozenset({"minimal", "low", "medium", "high", "xhigh", "max"}),
    toggle_disable=False,
)
_INKLING_2_PROFILE = HuggingFaceRouteProfile(
    reasoning=_INKLING_2_REASONING,
    preserve_raw_reasoning=True,
    omit_default_max_tokens=True,
)
# DeepInfra continuation fragments send "type": null, which the OpenAI SDK accumulator rejects.
# It also ends every response at 65,536 output tokens whatever max_tokens asks for.
_INKLING_2_DEEPINFRA_PROFILE = replace(
    _INKLING_2_PROFILE, stream_mode="manual", max_output_tokens=65_536
)
_QWEN_38_REASONING = TopLevelReasoningEffort(
    default_effort="medium",
    allowed_efforts=frozenset({"low", "medium", "xhigh"}),
    toggle_disable=False,
    chat_template_toggle_field="enable_thinking",
)
_HY4_REASONING = TopLevelReasoningEffort(
    default_effort="high",
    allowed_efforts=frozenset({"none", "high"}),
    disabled_efforts=frozenset({"none"}),
    toggle_disable=True,
)

HUGGINGFACE_ROUTE_PROFILES = RouterProfileRegistry(
    (
        # HF stopped serving Beam via Nebius (2026-11-16); Together is the evidenced upstream.
        RouterProfileRule(
            model=_BEAM_1,
            backends=frozenset({"together"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_BEAM_1_REASONING,
                structured_json_mode="schema",
                omit_default_max_tokens=True,
                max_output_tokens=131_072,
            ),
        ),
        # Dedicated endpoints retain their direct-backend contracts.
        *(
            RouterProfileRule(
                model=model,
                backends=frozenset({HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND}),
                profile=HuggingFaceRouteProfile(reasoning=reasoning, preserve_raw_reasoning=raw),
            )
            for model, reasoning, raw in (
                ("moonshotai/kimi-k2.5", ProviderDefaultReasoningToggle("thinking"), False),
                (
                    "moonshotai/kimi-k2.6",
                    ProviderDefaultReasoningToggle("chat_template_thinking"),
                    False,
                ),
                ("qwen/qwen3.5-397b-a17b", ChatTemplateReasoningToggle("enable_thinking"), False),
                ("qwen/qwen3.6-35b-a3b", ChatTemplateReasoningToggle("enable_thinking"), False),
                ("zai-org/glm-5", GenericDisableReasoningToggle(), False),
                (_GLM_52, ThinkingWithReasoningEffort(), True),
                ("zai-org/glm-5.3", ThinkingWithReasoningEffort(), True),
                ("zai-org/glm-5.3-flash", ThinkingWithReasoningEffort(), True),
            )
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V41_FLASH_1017,
            backends=frozenset({"novita"}),
            profile=_DEEPSEEK_NOVITA_PROFILE,
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            backends=frozenset({"baseten"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_DEEPSEEK_REASONING,
                max_output_tokens=384_000,
            ),
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            backends=frozenset({"scaleway"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_DEEPSEEK_REASONING,
                max_output_tokens=32_768,
                context_window=262_144,
            ),
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            backends=frozenset({"together"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_DEEPSEEK_REASONING,
                omit_default_max_tokens=True,
            ),
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            backends=frozenset({"deepinfra"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_DEEPSEEK_REASONING,
                default_max_tokens=131_072,
            ),
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V41_FLASH,
            backends=frozenset({"novita"}),
            profile=replace(_DEEPSEEK_NOVITA_PROFILE, reasoning=_DEEPSEEK_V41_REASONING),
        ),
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            profile=HuggingFaceRouteProfile(reasoning=_DEEPSEEK_REASONING),
        ),
        # Keep the existing DeepSeek chat API contract across HF backends.
        # V4.1's raw-weights numeric effort is not an API effort mapping.
        RouterProfileRule(
            model=_DEEPSEEK_V41_FLASH,
            profile=HuggingFaceRouteProfile(reasoning=_DEEPSEEK_V41_REASONING),
        ),
        RouterProfileRule(
            model=_GLM_52,
            backends=frozenset({"deepinfra"}),
            profile=HuggingFaceRouteProfile(reasoning=_GLM_52_DEEPINFRA_REASONING),
        ),
        RouterProfileRule(
            model=_GLM_52,
            # Preserve reasoning replay while changing the routed control field.
            profile=HuggingFaceRouteProfile(
                reasoning=_GLM_52_REASONING, preserve_raw_reasoning=True
            ),
        ),
        *(
            RouterProfileRule(
                model=model,
                backends=frozenset({backend}),
                profile=HuggingFaceRouteProfile(reasoning=reasoning, preserve_raw_reasoning=True),
            )
            for model in ("zai-org/glm-5.3", "zai-org/glm-5.3-flash")
            for backend, reasoning in (
                ("together", _GLM_53_TOGETHER_REASONING),
                ("novita", _GLM_53_NOVITA_REASONING),
            )
        ),
        # Z.ai and auto routes use the same effort contract as Novita.
        *(
            RouterProfileRule(
                model=model,
                profile=HuggingFaceRouteProfile(
                    reasoning=_GLM_53_NOVITA_REASONING, preserve_raw_reasoning=True
                ),
            )
            for model in ("zai-org/glm-5.3", "zai-org/glm-5.3-flash")
        ),
        RouterProfileRule(
            model="moonshotai/kimi-k3.1-pro",
            backends=frozenset({"together"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_KIMI_K3_REASONING,
                preserve_raw_reasoning=True,
                use_legacy_json_schema=True,
            ),
        ),
        RouterProfileRule(
            model="moonshotai/kimi-k3.1-pro",
            profile=HuggingFaceRouteProfile(
                reasoning=_KIMI_K3_REASONING, preserve_raw_reasoning=True
            ),
        ),
        RouterProfileRule(
            model=_KIMI_K3,
            profile=HuggingFaceRouteProfile(reasoning=_KIMI_K3_REASONING),
        ),
        RouterProfileRule(
            model=_MUSE_GLIMMER,
            backends=frozenset({"together"}),
            profile=HuggingFaceRouteProfile(
                reasoning=_MUSE_GLIMMER_REASONING,
                omit_default_max_tokens=True,
                # Together reserves 32,768 output tokens when max_tokens is omitted.
                prompt_context_window=_MUSE_GLIMMER_PROMPT_CONTEXT_WINDOW,
            ),
        ),
        RouterProfileRule(
            model=_MUSE_GLIMMER,
            profile=HuggingFaceRouteProfile(
                omit_default_max_tokens=True,
                prompt_context_window=_MUSE_GLIMMER_PROMPT_CONTEXT_WINDOW,
            ),
        ),
        RouterProfileRule(
            model=_INKLING_2_SMALL,
            backends=frozenset({"together"}),
            profile=replace(_INKLING_2_PROFILE, use_legacy_json_schema=True),
        ),
        RouterProfileRule(
            model=_INKLING_2_SMALL,
            backends=frozenset({"deepinfra"}),
            profile=_INKLING_2_DEEPINFRA_PROFILE,
        ),
        # PROVISIONAL: un-pinned router requests only (no backend suffix, default router
        # URL). The router may serve these from any backend; DeepInfra was selected when
        # tested, so they get its profile. If the router picks Together, standard schema
        # output fails with HTTP 400. Explicit untested backends and custom endpoints get
        # no Inkling profile.
        RouterProfileRule(
            model=_INKLING_2_SMALL,
            backends=frozenset({None}),
            profile=_INKLING_2_DEEPINFRA_PROFILE,
        ),
        # Novita rejects json_schema for GLM 5 and MiniMax M2.5 (HTTP 400, json_object
        # only) and accepts but does not enforce it for Kimi K2.6 (fenced JSON, invented
        # keys). JSON object mode adds fast-agent's schema instructions instead.
        *(
            RouterProfileRule(
                model=model,
                backends=frozenset({"novita"}),
                profile=HuggingFaceRouteProfile(reasoning=reasoning, structured_json_mode="object"),
            )
            for model, reasoning in (
                ("zai-org/glm-5", _ROUTED_TOGGLE),
                ("minimaxai/minimax-m2.5", None),
                ("moonshotai/kimi-k2.6", _ROUTED_TOGGLE),
            )
        ),
        *(
            RouterProfileRule(
                model=model,
                profile=HuggingFaceRouteProfile(reasoning=_ROUTED_TOGGLE),
            )
            for model in (
                "moonshotai/kimi-k2.5",
                "moonshotai/kimi-k2.6",
                "zai-org/glm-5",
                # Provisional: Qwen 3.5 has one post-change off check; use the Qwen 3.6 contract.
                "qwen/qwen3.5-397b-a17b",
                "qwen/qwen3.6-35b-a3b",
            )
        ),
        RouterProfileRule(
            model=_QWEN_38,
            profile=HuggingFaceRouteProfile(reasoning=_QWEN_38_REASONING),
        ),
        RouterProfileRule(
            model=_GEMMA_4,
            profile=HuggingFaceRouteProfile(
                reasoning=ChatTemplateReasoningToggle("enable_thinking")
            ),
        ),
        *(
            RouterProfileRule(
                model=model,
                backends=backends,
                profile=HuggingFaceRouteProfile(
                    reasoning=_HY4_REASONING,
                    preserve_raw_reasoning=True,
                    omit_default_max_tokens=True,
                ),
            )
            for model, backends in (
                ("tencent/hy4", frozenset({"tencent", "together", None})),
                ("tencent/hy4-flash", frozenset({"tencent", None})),
            )
        ),
    )
)

GENERIC_REASONING_TOGGLE = GenericDisableReasoningToggle()
