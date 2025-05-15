from openai import AuthenticationError, AzureOpenAI, OpenAI

from mcp_agent.core.exceptions import ProviderKeyError
from mcp_agent.llm.provider_types import Provider
from mcp_agent.llm.providers.augmented_llm_openai import OpenAIAugmentedLLM

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None


def _extract_resource_name(url: str) -> str | None:
    from urllib.parse import urlparse

    host = urlparse(url).hostname or ""
    suffix = ".openai.azure.com"
    return host.replace(suffix, "") if host.endswith(suffix) else None


DEFAULT_AZURE_API_VERSION = "2023-05-15"


class AzureOpenAIAugmentedLLM(OpenAIAugmentedLLM):
    """
    Azure OpenAI implementation extending OpenAIAugmentedLLM.
    Handles both API Key and DefaultAzureCredential authentication.
    """

    def __init__(self, provider: Provider = Provider.AZURE, *args, **kwargs):
        # Set provider to AZURE, pass through to base
        super().__init__(provider=provider, *args, **kwargs)

        # Context/config extraction
        context = getattr(self, "context", None)
        config = getattr(context, "config", None) if context else None
        azure_cfg = getattr(config, "azure", None) if config else None

        if azure_cfg is None:
            raise ProviderKeyError(
                "Missing Azure configuration",
                "Azure provider requires configuration section 'azure' in your config file.",
            )

        self.use_default_cred = getattr(azure_cfg, "use_default_azure_credential", False)
        default_request_params = getattr(self, "default_request_params", None)
        self.deployment_name = getattr(default_request_params, "model", None) or getattr(
            azure_cfg, "azure_deployment", None
        )
        self.api_version = getattr(azure_cfg, "api_version", None) or DEFAULT_AZURE_API_VERSION

        if self.use_default_cred:
            self.base_url = getattr(azure_cfg, "base_url", None)
            if not self.base_url:
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "When using 'use_default_azure_credential', 'base_url' is required in azure config.",
                )
            if DefaultAzureCredential is None:
                raise ProviderKeyError(
                    "azure-identity not installed",
                    "You must install 'azure-identity' to use DefaultAzureCredential authentication.",
                )
            self.credential = DefaultAzureCredential()

            def get_azure_token():
                token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
                return token.token

            self.get_azure_token = get_azure_token
        else:
            self.api_key = getattr(azure_cfg, "api_key", None)
            self.resource_name = getattr(azure_cfg, "resource_name", None)
            self.base_url = getattr(azure_cfg, "base_url", None) or (
                f"https://{self.resource_name}.openai.azure.com/" if self.resource_name else None
            )
            if not self.api_key:
                raise ProviderKeyError(
                    "Missing Azure OpenAI credentials",
                    "Field 'api_key' is required in azure config.",
                )
            if not (self.resource_name or self.base_url):
                raise ProviderKeyError(
                    "Missing Azure endpoint",
                    "Provide either 'resource_name' or 'base_url' under azure config.",
                )
            if not self.deployment_name:
                raise ProviderKeyError(
                    "Missing deployment name",
                    "Set 'azure_deployment' in config or pass model=<deployment>.",
                )
            # If resource_name was missing, try to extract it from base_url
            if not self.resource_name and self.base_url:
                self.resource_name = _extract_resource_name(self.base_url)

    def _openai_client(self) -> OpenAI:
        """
        Returns an AzureOpenAI client, handling both API Key and DefaultAzureCredential.
        """
        try:
            if self.use_default_cred:
                if self.base_url is None:
                    raise ProviderKeyError(
                        "Missing Azure endpoint",
                        "azure_endpoint (base_url) is None at client creation time.",
                    )
                return AzureOpenAI(
                    azure_ad_token_provider=self.get_azure_token,
                    azure_endpoint=self.base_url,
                    api_version=self.api_version,
                    azure_deployment=self.deployment_name,
                )
            else:
                if self.base_url is None:
                    raise ProviderKeyError(
                        "Missing Azure endpoint",
                        "azure_endpoint (base_url) is None at client creation time.",
                    )
                return AzureOpenAI(
                    api_key=self.api_key,
                    azure_endpoint=self.base_url,
                    api_version=self.api_version,
                    azure_deployment=self.deployment_name,
                )
        except AuthenticationError as e:
            if self.use_default_cred:
                raise ProviderKeyError(
                    "Invalid Azure AD credentials",
                    "The configured Azure AD credentials were rejected.\n"
                    "Please check your Azure identity setup.",
                ) from e
            else:
                raise ProviderKeyError(
                    "Invalid Azure OpenAI API key",
                    "The configured Azure OpenAI API key was rejected.\n"
                    "Please check that your API key is valid and not expired.",
                ) from e
