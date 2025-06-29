import types
from typing import Optional

import pytest

from mcp_agent.llm.providers.augmented_llm_azure import AzureOpenAIAugmentedLLM


class DummyLogger:
    enable_markup = True


class DummyAzureConfig:
    def __init__(self):
        self.api_key: Optional[str] = "test-key"
        self.resource_name: Optional[str] = "test-resource"
        self.azure_deployment: Optional[str] = "test-deployment"
        self.api_version: Optional[str] = "2023-05-15"
        self.base_url: Optional[str] = None
        self.use_default_azure_credential: bool = False


class DummyConfig:
    def __init__(self, azure_cfg=None):
        self.azure = azure_cfg or DummyAzureConfig()
        self.logger = DummyLogger()
        self.openai = None  # For compatibility with OpenAIAugmentedLLM


class DummyContext:
    def __init__(self, azure_cfg=None):
        self.config = DummyConfig(azure_cfg=azure_cfg)
        self.executor = None


def test_openai_client_with_base_url_only():
    cfg = DummyAzureConfig()
    cfg.base_url = "https://mydemo.openai.azure.com/"
    cfg.resource_name = None
    ctx = DummyContext(azure_cfg=cfg)
    llm = AzureOpenAIAugmentedLLM(context=ctx)
    client = llm._openai_client()
    assert hasattr(client, "chat")
    # Should be AzureOpenAI instance


@pytest.mark.asyncio
async def test_openai_client_with_default_azure_credential(monkeypatch):
    """
    Test AzureOpenAIAugmentedLLM with use_default_azure_credential: True.
    Mocks DefaultAzureCredential and AzureOpenAI to ensure correct integration.
    """

    class DummyToken:
        def __init__(self, token):
            self.token = token

    class DummyCredential:
        def get_token(self, scope):
            assert scope == "https://cognitiveservices.azure.com/.default"
            return DummyToken("dummy-token")

    import mcp_agent.llm.providers.augmented_llm_azure as azure_mod

    monkeypatch.setattr(azure_mod, "DefaultAzureCredential", DummyCredential)

    class DummyAzureOpenAI:
        def __init__(self, **kwargs):
            assert "azure_ad_token_provider" in kwargs
            self.token_provider = kwargs["azure_ad_token_provider"]
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(
                    create=lambda **kw: types.SimpleNamespace(
                        choices=[
                            types.SimpleNamespace(
                                message=types.SimpleNamespace(content="tokenpong")
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(azure_mod, "AzureOpenAI", DummyAzureOpenAI)
    monkeypatch.setattr(azure_mod, "AsyncAzureOpenAI", DummyAzureOpenAI)

    class DACfg:
        def __init__(self):
            self.api_key = None
            self.resource_name = None
            self.azure_deployment = "test-deployment"
            self.api_version = "2023-05-15"
            self.base_url = "https://mydemo.openai.azure.com/"
            self.use_default_azure_credential = True

    dacfg = DACfg()
    ctx = DummyContext(azure_cfg=dacfg)
    llm = AzureOpenAIAugmentedLLM(context=ctx)
    client = llm._openai_client()
    # Just checking that the client is created and has chat
    assert hasattr(client, "chat")
