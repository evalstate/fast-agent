from mcp_types import SERVER_INFO_META_KEY, DiscoverResult, Implementation


def test_final_discover_schema_reads_server_identity_from_metadata() -> None:
    result = DiscoverResult.model_validate(
        {
            "supportedVersions": ["2026-07-28"],
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
            },
            "resultType": "complete",
            "ttlMs": 0,
            "cacheScope": "private",
            "_meta": {
                SERVER_INFO_META_KEY: {
                    "name": "@huggingface/mcp-services",
                    "version": "0.4.1",
                    "title": "Hugging Face",
                    "websiteUrl": "https://huggingface.co/mcp",
                }
            },
        }
    )

    assert result.supported_versions == ["2026-07-28"]
    assert result.meta is not None
    server_info = Implementation.model_validate(result.meta[SERVER_INFO_META_KEY])
    assert server_info.name == "@huggingface/mcp-services"
    assert server_info.version == "0.4.1"
