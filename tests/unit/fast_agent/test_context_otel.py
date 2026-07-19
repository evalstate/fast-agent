from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest


def test_context_import_does_not_emit_opentelemetry_deprecations() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import fast_agent.context",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("sample_rate", "span_exported"), [(1.0, True), (0.0, False)])
def test_configure_otel_samples_and_flushes_spans(
    sample_rate: float, span_exported: bool
) -> None:
    script = textwrap.dedent(
        f"""
        import asyncio
        import warnings
        from unittest.mock import patch

        from opentelemetry import trace
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        import fast_agent.context as context_module
        from fast_agent.config import OpenTelemetrySettings, Settings

        exporter = InMemorySpanExporter()

        settings = Settings(
            otel=OpenTelemetrySettings(
                enabled=True,
                otlp_endpoint="http://collector.invalid/v1/traces",
                sample_rate={sample_rate!r},
            )
        )

        with (
            patch.object(context_module, "OTLPSpanExporter", return_value=exporter),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            asyncio.run(context_module.configure_otel(settings))

        tracer = trace.get_tracer("otel-contract-test")
        with tracer.start_as_current_span("working-span"):
            pass
        asyncio.run(context_module.cleanup_context())

        spans = exporter.get_finished_spans()
        assert bool(spans) is {span_exported!r}
        if spans:
            assert [span.name for span in spans] == ["working-span"]
            assert spans[0].resource.attributes["service.name"] == "fast-agent"
        deprecations = [
            (str(warning.message), warning.filename)
            for warning in caught
            if issubclass(warning.category, DeprecationWarning)
            and "/opentelemetry/" in warning.filename
        ]
        assert not deprecations, deprecations
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
