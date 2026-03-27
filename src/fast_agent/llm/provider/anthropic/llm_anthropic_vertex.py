from __future__ import annotations

from typing import Any, cast

from anthropic import AsyncAnthropicVertex

from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.vertex_config import (
    anthropic_vertex_config,
    detect_google_adc,
    resolve_anthropic_vertex_location,
    resolve_anthropic_vertex_project_id,
)
from fast_agent.llm.provider_types import Provider


class AnthropicVertexLLM(AnthropicLLM):
    @classmethod
    def provider_identity(cls) -> Provider:
        return Provider.ANTHROPIC_VERTEX

    def _vertex_cfg(self):
        return anthropic_vertex_config(getattr(self.context, "config", None))

    def _provider_base_url(self) -> str | None:
        return self._vertex_cfg().base_url

    def _provider_api_key(self) -> str:
        return ""

    def _vertex_project_id(self) -> str:
        project_id = resolve_anthropic_vertex_project_id(getattr(self.context, "config", None))
        if project_id is None:
            raise ProviderKeyError(
                "Google Cloud project not configured",
                "Set anthropic.vertex_ai.project_id or configure "
                "GOOGLE_CLOUD_PROJECT before using Anthropic via Vertex.",
            )
        return project_id

    def _vertex_location(self) -> str:
        location = resolve_anthropic_vertex_location(getattr(self.context, "config", None))
        if location is None:
            raise ProviderKeyError(
                "Google Cloud location not configured",
                "Set anthropic.vertex_ai.location before using Anthropic via Vertex.",
            )
        return location

    def _vertex_credentials(self) -> object:
        adc_status = detect_google_adc()
        if not adc_status.available or adc_status.credentials is None:
            raise ProviderKeyError(
                "Google ADC not found",
                "Anthropic via Vertex uses Google Application Default Credentials.\n"
                "Run `gcloud auth application-default login` or configure a service account.",
            )
        return adc_status.credentials

    def _initialize_anthropic_client(self) -> AsyncAnthropicVertex:
        return AsyncAnthropicVertex(
            project_id=self._vertex_project_id(),
            region=self._vertex_location(),
            credentials=cast("Any", self._vertex_credentials()),
            base_url=self._base_url(),
            default_headers=self._default_headers(),
        )

    def supports_files_api(self) -> bool:
        return False

    def supports_document_uploads(self) -> bool:
        return False

    def supports_web_tools(self) -> bool:
        return True

    def supports_direct_anthropic_beta(self, feature: str) -> bool:
        return feature in {
            "interleaved_thinking",
            "long_context",
            "fine_grained_tool_streaming",
            "web_tools",
        }
