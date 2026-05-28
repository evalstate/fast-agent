from __future__ import annotations

from dataclasses import dataclass

from fast_agent.utils.type_narrowing import is_str_object_dict

_ALLOWED_KEYS = {"entrypoint", "variant", "code_arg", "language"}
_ALLOWED_VARIANTS = {"code"}


@dataclass(frozen=True, slots=True)
class FunctionToolSpec:
    entrypoint: str
    variant: str | None = None
    code_arg: str | None = None
    language: str | None = None

    def metadata(self) -> dict[str, str] | None:
        metadata: dict[str, str] = {}
        if self.variant:
            metadata["variant"] = self.variant
        if self.code_arg:
            metadata["code_arg"] = self.code_arg
        if self.language:
            metadata["language"] = self.language
        return metadata or None


def parse_function_tool_card_entry(raw: object, *, field_path: str) -> str | FunctionToolSpec:
    if isinstance(raw, str):
        return raw
    if not is_str_object_dict(raw):
        raise ValueError(f"'{field_path}' entries must be strings or objects")

    unknown_keys = sorted(set(raw) - _ALLOWED_KEYS)
    if unknown_keys:
        extras = ", ".join(unknown_keys)
        raise ValueError(f"'{field_path}' entries contain unsupported keys: {extras}")

    entrypoint = raw.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.strip():
        raise ValueError(f"'{field_path}.entrypoint' must be a non-empty string")

    variant = raw.get("variant")
    if variant is not None and not isinstance(variant, str):
        raise ValueError(f"'{field_path}.variant' must be a string")
    if isinstance(variant, str) and variant not in _ALLOWED_VARIANTS:
        allowed = ", ".join(sorted(_ALLOWED_VARIANTS))
        raise ValueError(f"'{field_path}.variant' must be one of: {allowed}")

    code_arg = raw.get("code_arg")
    if code_arg is not None and not isinstance(code_arg, str):
        raise ValueError(f"'{field_path}.code_arg' must be a string")

    language = raw.get("language")
    if language is not None and not isinstance(language, str):
        raise ValueError(f"'{field_path}.language' must be a string")

    if variant == "code":
        code_arg = code_arg or "code"
        language = language or "python"

    return FunctionToolSpec(
        entrypoint=entrypoint,
        variant=variant,
        code_arg=code_arg,
        language=language,
    )


def function_tool_entrypoint(spec: object) -> str | None:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, FunctionToolSpec):
        return spec.entrypoint
    if is_str_object_dict(spec):
        entrypoint = spec.get("entrypoint")
        if isinstance(entrypoint, str):
            return entrypoint
    return None


def serialize_function_tool_entry(spec: object) -> str | dict[str, str] | None:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, FunctionToolSpec):
        payload: dict[str, str] = {"entrypoint": spec.entrypoint}
        if spec.variant:
            payload["variant"] = spec.variant
        if spec.code_arg:
            payload["code_arg"] = spec.code_arg
        if spec.language:
            payload["language"] = spec.language
        return payload
    if is_str_object_dict(spec):
        return {key: str(value) for key, value in spec.items() if isinstance(value, str)}
    return None


def serialize_function_tools(value: object) -> list[str | dict[str, str]] | None:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return None

    serialized: list[str | dict[str, str]] = []
    for entry in value:
        rendered = serialize_function_tool_entry(entry)
        if rendered is not None:
            serialized.append(rendered)
    return serialized or None
