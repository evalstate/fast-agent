from __future__ import annotations

from typing import Collection, Dict, List, Literal, Optional, Set, TypedDict, cast

# Lightweight, runtime-only loader for AWS Bedrock models.
# - Fetches once per process via boto3 (region from session; env override supported)
# - Memory cache only; no disk persistence
# - Provides filtering and optional prefixing (default 'bedrock.') for model IDs

try:
    import boto3
except Exception:  # pragma: no cover - import error path
    boto3 = None  # type: ignore[assignment]


Modality = Literal["TEXT", "IMAGE", "VIDEO", "SPEECH", "EMBEDDING"]
Lifecycle = Literal["ACTIVE", "LEGACY"]
InferenceType = Literal["ON_DEMAND", "PROVISIONED", "INFERENCE_PROFILE"]


class ModelSummary(TypedDict, total=False):
    modelId: str
    modelName: str
    providerName: str
    inputModalities: List[Modality]
    outputModalities: List[Modality]
    responseStreamingSupported: bool
    customizationsSupported: List[str]
    inferenceTypesSupported: List[InferenceType]
    modelLifecycle: Dict[str, Lifecycle]


_MODELS_CACHE_BY_REGION: Dict[str, Dict[str, ModelSummary]] = {}


def _resolve_region(region: Optional[str]) -> str:
    if region:
        return region
    import os

    env_region = os.getenv("BEDROCK_REGION")
    if env_region:
        return env_region
    if boto3 is None:
        raise RuntimeError(
            "boto3 is required to load Bedrock models. Install boto3 or provide a static list."
        )
    session = boto3.Session()
    if not session.region_name:
        raise RuntimeError(
            "AWS region could not be resolved. Configure your AWS SSO/profile or set BEDROCK_REGION."
        )
    return session.region_name


def _strip_prefix(model_id: str, prefix: str) -> str:
    return model_id[len(prefix) :] if prefix and model_id.startswith(prefix) else model_id


def _ensure_loaded(region: Optional[str] = None) -> Dict[str, ModelSummary]:
    resolved_region = _resolve_region(region)
    cache = _MODELS_CACHE_BY_REGION.get(resolved_region)
    if cache is not None:
        return cache

    if boto3 is None:
        raise RuntimeError("boto3 is required to load Bedrock models. Install boto3.")

    try:
        client = boto3.client("bedrock", region_name=resolved_region)
        resp = client.list_foundation_models()
        summaries: List[ModelSummary] = resp.get("modelSummaries", [])  # type: ignore[assignment]
    except Exception as exc:  # keep error simple and actionable
        raise RuntimeError(
            f"Failed to list Bedrock foundation models in region '{resolved_region}'. "
            f"Ensure AWS credentials (SSO) and permissions (bedrock:ListFoundationModels) are configured. "
            f"Original error: {exc}"
        )

    cache = {s.get("modelId", ""): s for s in summaries if s.get("modelId")}
    _MODELS_CACHE_BY_REGION[resolved_region] = cache
    return cache


def refresh_bedrock_models(region: Optional[str] = None) -> None:
    resolved_region = _resolve_region(region)
    # drop and reload on next access
    _MODELS_CACHE_BY_REGION.pop(resolved_region, None)
    _ensure_loaded(resolved_region)


def _matches_modalities(model_modalities: List[Modality], requested: Collection[Modality]) -> bool:
    # include if all requested are present in the model's modalities
    return set(requested).issubset(set(model_modalities))


def all_model_summaries(
    input_modalities: Optional[Collection[Modality]] = None,
    output_modalities: Optional[Collection[Modality]] = None,
    include_legacy: bool = False,
    providers: Optional[Collection[str]] = None,
    inference_types: Optional[Collection[InferenceType]] = None,
    direct_invocation_only: bool = True,
    region: Optional[str] = None,
) -> List[ModelSummary]:
    """Return filtered Bedrock model summaries.

    Defaults: input_modalities={"TEXT"}, output_modalities={"TEXT"}, include_legacy=False,
    inference_types={"ON_DEMAND"}, direct_invocation_only=True.
    """

    cache = _ensure_loaded(region)
    results: List[ModelSummary] = []

    effective_output: Set[Modality] = (
        set(output_modalities) if output_modalities is not None else {cast("Modality", "TEXT")}
    )
    effective_input: Optional[Set[Modality]] = (
        set(input_modalities) if input_modalities is not None else {cast("Modality", "TEXT")}
    )
    provider_filter: Optional[Set[str]] = set(providers) if providers is not None else None
    effective_inference: Set[InferenceType] = (
        set(inference_types)
        if inference_types is not None
        else {cast("InferenceType", "ON_DEMAND")}
    )

    for summary in cache.values():
        lifecycle = (summary.get("modelLifecycle") or {}).get("status")
        if not include_legacy and lifecycle == "LEGACY":
            continue

        if provider_filter is not None and summary.get("providerName") not in provider_filter:
            continue

        # direct invocation only: exclude profile variants like :0:24k or :mm
        if direct_invocation_only:
            mid = summary.get("modelId") or ""
            if mid.count(":") > 1:
                continue

        # modalities
        model_inputs: List[Modality] = summary.get("inputModalities", [])  # type: ignore[assignment]
        model_outputs: List[Modality] = summary.get("outputModalities", [])  # type: ignore[assignment]

        if effective_input is not None and not _matches_modalities(model_inputs, effective_input):
            continue
        if effective_output and not _matches_modalities(model_outputs, effective_output):
            continue

        # inference types
        model_inference: List[InferenceType] = summary.get("inferenceTypesSupported", [])  # type: ignore[assignment]
        if effective_inference and not set(effective_inference).issubset(set(model_inference)):
            continue

        results.append(summary)

    return results


def all_bedrock_models(
    input_modalities: Optional[Collection[Modality]] = None,
    output_modalities: Optional[Collection[Modality]] = None,
    include_legacy: bool = False,
    providers: Optional[Collection[str]] = None,
    prefix: str = "bedrock.",
    inference_types: Optional[Collection[InferenceType]] = None,
    direct_invocation_only: bool = True,
    region: Optional[str] = None,
) -> List[str]:
    """Return model IDs (optionally prefixed) filtered by the given criteria.

    Defaults: output_modalities={"TEXT"}, exclude LEGACY,
    inference_types={"ON_DEMAND"}, direct_invocation_only=True.
    """

    summaries = all_model_summaries(
        input_modalities=input_modalities,
        output_modalities=output_modalities,
        include_legacy=include_legacy,
        providers=providers,
        inference_types=inference_types,
        direct_invocation_only=direct_invocation_only,
        region=region,
    )
    ids: List[str] = []
    for s in summaries:
        mid = s.get("modelId")
        if mid:
            ids.append(mid)
    if prefix:
        return [f"{prefix}{mid}" for mid in ids]
    return ids


def get_model_metadata(model_id: str, region: Optional[str] = None) -> Optional[ModelSummary]:
    cache = _ensure_loaded(region)
    # Accept either prefixed or plain model IDs
    plain_id = _strip_prefix(model_id, "bedrock.")
    return cache.get(plain_id)


def list_providers(region: Optional[str] = None) -> List[str]:
    cache = _ensure_loaded(region)
    providers = {s.get("providerName") for s in cache.values() if s.get("providerName")}
    return sorted(providers)  # type: ignore[arg-type]


__all__ = [
    "Modality",
    "Lifecycle",
    "ModelSummary",
    "all_bedrock_models",
    "all_model_summaries",
    "get_model_metadata",
    "list_providers",
    "refresh_bedrock_models",
]
