from __future__ import annotations

from dataclasses import dataclass
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
    default_effort: str
    allowed_efforts: frozenset[str] | None = None
    disabled_efforts: frozenset[str] = frozenset()
    effort_map: Mapping[str, str] | None = None
    toggle_disable: bool = True
    chat_template_toggle_field: str | None = None
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
        elif self.effort_map:
            effort = self.effort_map.get(effort, effort)
        arguments["reasoning_effort"] = effort

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
class ChatTemplateReasoningStrength:
    default_strength: str

    def apply(
        self,
        arguments: dict[str, Any],
        *,
        setting: ReasoningEffortSetting | None,
        spec: ReasoningEffortSpec | None,
    ) -> None:
        effective = _effective_setting(setting, spec)
        strength = self.default_strength
        if (
            effective is not None
            and effective.kind == "effort"
            and isinstance(effective.value, str)
        ):
            strength = effective.value

        extra_body = _extra_body(arguments)
        _set_chat_template_kwarg(extra_body, "reasoning_strength", strength)
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
    structured_json_mode: Literal["schema", "object"] | None = None


HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND = "custom-endpoint"

_DEEPSEEK_V4_FLASH = "deepseek-ai/deepseek-v4-flash-0731"
_GLM_52 = "zai-org/glm-5.2"
_KIMI_K3 = "moonshotai/kimi-k3"
_GEMMA_4 = "google/gemma-4-31b-it"
_MUSE_GLIMMER = "meta-models/muse-glimmer-30b"
_QWEN_38 = "qwen/qwen3.8-27b"

_DEEPSEEK_REASONING = TopLevelReasoningEffort(default_effort="max")
_KIMI_K3_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    allowed_efforts=frozenset({"low", "high", "max"}),
    toggle_disable=False,
    cleanup_extra_body=frozenset({"thinking", "chat_template_kwargs"}),
)
_GLM_52_REASONING = ThinkingWithReasoningEffort()
_GLM_52_DEEPINFRA_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    disabled_efforts=frozenset({"none", "minimal"}),
    effort_map={"max": "xhigh"},
    cleanup_extra_body=frozenset({"thinking"}),
)
_GLM_52_FIREWORKS_REASONING = TopLevelReasoningEffort(
    default_effort="max",
    disabled_efforts=frozenset({"none", "minimal"}),
    cleanup_extra_body=frozenset({"thinking"}),
)
_GEMMA_4_CEREBRAS_REASONING = TopLevelReasoningEffort(
    default_effort="none",
    cleanup_extra_body=frozenset({"chat_template_kwargs"}),
)
_MUSE_GLIMMER_REASONING = ChatTemplateReasoningStrength(default_strength="high")
_QWEN_38_REASONING = TopLevelReasoningEffort(
    default_effort="medium",
    allowed_efforts=frozenset({"low", "medium", "xhigh"}),
    toggle_disable=False,
    chat_template_toggle_field="enable_thinking",
)

HUGGINGFACE_ROUTE_PROFILES = RouterProfileRegistry(
    (
        RouterProfileRule(
            model=_DEEPSEEK_V4_FLASH,
            backends=frozenset(
                {
                    "baseten",
                    "deepinfra",
                    HUGGINGFACE_CUSTOM_ENDPOINT_BACKEND,
                }
            ),
            profile=HuggingFaceRouteProfile(reasoning=_DEEPSEEK_REASONING),
        ),
        RouterProfileRule(
            model=_GLM_52,
            backends=frozenset({"deepinfra"}),
            profile=HuggingFaceRouteProfile(reasoning=_GLM_52_DEEPINFRA_REASONING),
        ),
        RouterProfileRule(
            model=_GLM_52,
            backends=frozenset({"fireworks-ai"}),
            profile=HuggingFaceRouteProfile(reasoning=_GLM_52_FIREWORKS_REASONING),
        ),
        RouterProfileRule(
            model=_GLM_52,
            profile=HuggingFaceRouteProfile(reasoning=_GLM_52_REASONING),
        ),
        RouterProfileRule(
            model=_KIMI_K3,
            profile=HuggingFaceRouteProfile(reasoning=_KIMI_K3_REASONING),
        ),
        RouterProfileRule(
            model=_GEMMA_4,
            backends=frozenset({"cerebras"}),
            profile=HuggingFaceRouteProfile(reasoning=_GEMMA_4_CEREBRAS_REASONING),
        ),
        RouterProfileRule(
            model=_MUSE_GLIMMER,
            backends=frozenset({"together"}),
            profile=HuggingFaceRouteProfile(reasoning=_MUSE_GLIMMER_REASONING),
        ),
        RouterProfileRule(
            model="moonshotai/kimi-k2.5",
            profile=HuggingFaceRouteProfile(reasoning=ProviderDefaultReasoningToggle("thinking")),
        ),
        RouterProfileRule(
            model="moonshotai/kimi-k2.6",
            profile=HuggingFaceRouteProfile(
                reasoning=ProviderDefaultReasoningToggle("chat_template_thinking")
            ),
        ),
        RouterProfileRule(
            model="qwen/qwen3.5-397b-a17b",
            profile=HuggingFaceRouteProfile(
                reasoning=ChatTemplateReasoningToggle("enable_thinking")
            ),
        ),
        RouterProfileRule(
            model="qwen/qwen3.6-35b-a3b",
            profile=HuggingFaceRouteProfile(
                reasoning=ChatTemplateReasoningToggle("enable_thinking")
            ),
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
    )
)

GENERIC_REASONING_TOGGLE = GenericDisableReasoningToggle()
