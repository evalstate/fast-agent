from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.internal_resources import (
    _parse_manifest_entry,
    get_internal_resource,
    list_internal_resources,
    read_internal_resource,
)
from fast_agent.tools.environment_config import (
    CustomEnvironmentSpec,
    DockerEnvironmentSpec,
    EnvironmentMountSpec,
    HuggingFaceBucketMountSpec,
    HuggingFaceEnvironmentSpec,
    HuggingFaceVolumeMountSpec,
    LocalEnvironmentSpec,
)


def test_parse_manifest_entry_normalizes_text_fields() -> None:
    resource = _parse_manifest_entry(
        0,
        {
            "uri": "  internal://fast-agent/demo  ",
            "title": "  Demo  ",
            "description": "  Example resource  ",
            "why": "  Useful for tests  ",
            "source": "  docs/demo.md  ",
            "mime_type": "  text/markdown  ",
            "tags": [" docs ", " ", "tests"],
        },
    )

    assert resource.uri == "internal://fast-agent/demo"
    assert resource.title == "Demo"
    assert resource.description == "Example resource"
    assert resource.why == "Useful for tests"
    assert resource.source == "docs/demo.md"
    assert resource.mime_type == "text/markdown"
    assert resource.tags == ("docs", "tests")


def test_parse_manifest_entry_rejects_blank_required_text() -> None:
    with pytest.raises(AgentConfigError, match="missing non-empty 'title'"):
        _parse_manifest_entry(
            0,
            {
                "uri": "internal://fast-agent/demo",
                "title": " ",
                "description": "Example resource",
                "why": "Useful for tests",
                "source": "docs/demo.md",
            },
        )


def test_get_internal_resource_rejects_blank_uri() -> None:
    with pytest.raises(AgentConfigError, match="URI must not be empty"):
        get_internal_resource("   ")


def test_execution_environment_internal_resource_is_listed_and_readable() -> None:
    resources = list_internal_resources()
    uris = {resource.uri for resource in resources}

    assert "internal://fast-agent/execution-environments" in uris

    content = read_internal_resource("internal://fast-agent/execution-environments")

    assert "volume_mounts" in content
    assert "hf://buckets/username/my-bucket:/workspace:rw" in content
    assert "container_cli" in content


def test_execution_environment_internal_resource_covers_schema_fields() -> None:
    content = read_internal_resource("internal://fast-agent/execution-environments")
    models = (
        LocalEnvironmentSpec,
        DockerEnvironmentSpec,
        EnvironmentMountSpec,
        HuggingFaceEnvironmentSpec,
        HuggingFaceVolumeMountSpec,
        HuggingFaceBucketMountSpec,
        CustomEnvironmentSpec,
    )

    for model in models:
        for field_name, field in model.model_fields.items():
            display_name = field.alias if isinstance(field.alias, str) else field_name
            assert f"`{display_name}`" in content


def test_execution_environment_internal_resource_matches_generator() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    generator_path = repo_root / "docs" / "generate_reference_docs.py"
    spec = importlib.util.spec_from_file_location("generate_reference_docs_for_test", generator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    generator = getattr(module, "generate_execution_environments_internal_resource")
    generated = generator()
    checked_in = (
        repo_root / "resources" / "shared" / "execution_environments.md"
    ).read_text(encoding="utf-8")

    assert checked_in == generated
