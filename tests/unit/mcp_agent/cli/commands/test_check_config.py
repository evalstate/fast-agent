from mcp_agent.cli.commands.check_config import API_KEY_HINT_TEXT, check_api_keys


def make_secrets_summary(azure_cfg):
    return {"status": "parsed", "error": None, "secrets": {"azure": azure_cfg}}


def test_check_api_keys_only_api_key():
    azure_cfg = {
        "api_key": "test-azure-key",
        "resource_name": "test-resource",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    assert results["azure"]["config"] == "...e-key"


def test_check_api_keys_only_default_cred():
    azure_cfg = {
        "use_default_azure_credential": True,
        "base_url": "https://mydemo.openai.azure.com/",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    assert results["azure"]["config"] == "DefaultAzureCredential"


def test_check_api_keys_both_modes():
    azure_cfg = {
        "api_key": "test-azure-key",
        "use_default_azure_credential": True,
        "base_url": "https://mydemo.openai.azure.com/",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # When use_default_azure_credential=True, Azure LLM ignores api_key and only uses DefaultAzureCredential
    assert results["azure"]["config"] == "DefaultAzureCredential"


def test_check_api_keys_invalid_config():
    azure_cfg = {
        "use_default_azure_credential": True,
        # missing base_url
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # Should not mark as DefaultAzureCredential if base_url missing
    assert results["azure"]["config"] == ""


def test_check_api_keys_hint_text():
    azure_cfg = {
        "api_key": API_KEY_HINT_TEXT,
        "resource_name": "test-resource",
        "azure_deployment": "test-deployment",
        "api_version": "2023-05-15",
    }
    summary = make_secrets_summary(azure_cfg)
    results = check_api_keys(summary, {})
    # Should not show API_KEY_HINT_TEXT as a valid key
    assert results["azure"]["config"] == ""
