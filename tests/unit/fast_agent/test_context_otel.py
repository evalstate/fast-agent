from __future__ import annotations

import subprocess
import sys
import textwrap


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


def test_configure_otel_samples_and_flushes_spans() -> None:
    script = textwrap.dedent(
        """
        import asyncio
        import warnings
        from unittest.mock import patch

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        import fast_agent.context as context_module
        from fast_agent.config import OpenTelemetrySettings, Settings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for sample_rate, span_exported in ((1.0, True), (0.0, False)):
                exporter = InMemorySpanExporter()
                settings = Settings(
                    otel=OpenTelemetrySettings(
                        enabled=True,
                        otlp_endpoint="http://collector.invalid/v1/traces",
                        sample_rate=sample_rate,
                    )
                )

                with (
                    patch(
                        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
                        return_value=exporter,
                    ),
                    patch.object(context_module.trace, "set_tracer_provider"),
                    patch.object(OpenAIInstrumentor, "instrument"),
                    patch.object(GoogleGenAiSdkInstrumentor, "instrument"),
                ):
                    asyncio.run(context_module.configure_otel(settings))

                provider = context_module._otel_tracer_provider
                assert provider is not None
                tracer = provider.get_tracer("otel-contract-test")
                with tracer.start_as_current_span("working-span"):
                    pass
                asyncio.run(context_module.cleanup_context())

                spans = exporter.get_finished_spans()
                assert bool(spans) is span_exported
                if spans:
                    assert [span.name for span in spans] == ["working-span"]
                    assert spans[0].resource.attributes["service.name"] == "fast-agent"

                provider.shutdown()
                context_module._otel_tracer_provider = None

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


def test_configure_otel_instruments_installed_provider_sdks() -> None:
    """Keep OpenAI tracing active and make the accepted Google SDK 3 gap explicit."""
    script = textwrap.dedent(
        """
        import asyncio
        from unittest.mock import patch

        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        import fast_agent.context as context_module
        from fast_agent.config import OpenTelemetrySettings, Settings

        exporter = InMemorySpanExporter()
        settings = Settings(
            otel=OpenTelemetrySettings(
                enabled=True,
                otlp_endpoint="http://collector.invalid/v1/traces",
            )
        )
        with patch(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
            return_value=exporter,
        ):
            asyncio.run(context_module.configure_otel(settings))

        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentors = (GoogleGenAiSdkInstrumentor(), OpenAIInstrumentor())
        try:
            # SDK 3 is required for current Gemini models, but the Google
            # instrumentor only supports SDK <3. Do not bypass its safety check.
            google_instrumentor, openai_instrumentor = instrumentors
            assert not google_instrumentor.is_instrumented_by_opentelemetry
            assert openai_instrumentor.is_instrumented_by_opentelemetry
        finally:
            for instrumentor in instrumentors:
                if instrumentor.is_instrumented_by_opentelemetry:
                    instrumentor.uninstrument()
            asyncio.run(context_module.cleanup_context())
            context_module._otel_tracer_provider.shutdown()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert 'GoogleGenAiSdkInstrumentor only instruments "google-genai>=1.32.0,<3"' in result.stderr


def test_disabled_telemetry_does_not_load_sdk() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import sys

from fast_agent.config import OpenTelemetrySettings, Settings
from fast_agent.context import configure_otel

asyncio.run(configure_otel(Settings(otel=OpenTelemetrySettings(enabled=False))))
loaded = {name for name in sys.modules if name.startswith("opentelemetry.sdk")}
assert not loaded, loaded
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
