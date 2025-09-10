import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from fast_agent.config import get_settings

# Absolute path for the test directory to ensure files are created in a known location
TEST_DIR = Path(__file__).parent.resolve()
TEMP_CONFIG_DIR = TEST_DIR / "temp_config_data"


@pytest.fixture(scope="function")
def temp_config_files():
    # Create a temporary directory for config files if it doesn't exist
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file_path = TEMP_CONFIG_DIR / "fastagent.config.yaml"
    secrets_file_path = TEMP_CONFIG_DIR / "fastagent.secrets.yaml"

    yield config_file_path, secrets_file_path

    # Clean up created files
    if config_file_path.exists():
        config_file_path.unlink()
    if secrets_file_path.exists():
        secrets_file_path.unlink()
    # Attempt to remove the temporary directory if it's empty
    try:
        if TEMP_CONFIG_DIR.exists() and not any(TEMP_CONFIG_DIR.iterdir()):
            TEMP_CONFIG_DIR.rmdir()
    except OSError:
        # Directory might not be empty if other tests use it or if unlinking failed
        pass


def write_config(file_path: Path, data: dict):
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def test_resolve_simple_env_var(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {"api_key": "${TEST_API_KEY}"}
    write_config(config_file, config_data)

    with patch.dict(os.environ, {"TEST_API_KEY": "actual_key_from_env"}):
        settings = get_settings(str(config_file))
        assert settings.api_key == "actual_key_from_env"


def test_resolve_env_var_with_default_when_set(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {"service_url": "${SERVICE_URL:http://default.url}"}
    write_config(config_file, config_data)

    with patch.dict(os.environ, {"SERVICE_URL": "http://env.url"}):
        settings = get_settings(str(config_file))
        assert settings.service_url == "http://env.url"


def test_resolve_env_var_with_default_when_not_set(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {"service_url": "${NON_EXISTENT_URL:http://default.url}"}
    write_config(config_file, config_data)

    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings(str(config_file))
        assert settings.service_url == "http://default.url"


def test_resolve_env_var_no_default_not_set(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {"another_key": "${UNSET_KEY_NO_DEFAULT}"}
    write_config(config_file, config_data)

    with patch.dict(os.environ, {}, clear=True):
        settings = get_settings(str(config_file))
        assert settings.another_key == "${UNSET_KEY_NO_DEFAULT}"


def test_nested_env_var_resolution(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {
        "parent": {
            "child_plain": "value",
            "child_env": "${NESTED_ENV_VAR}",
            "child_env_default": "${NESTED_DEFAULT_ENV:default_child_val}",
        }
    }
    write_config(config_file, config_data)

    with patch.dict(os.environ, {"NESTED_ENV_VAR": "nested_from_env"}):
        settings = get_settings(str(config_file))
        assert settings.parent["child_plain"] == "value"
        assert settings.parent["child_env"] == "nested_from_env"
        assert settings.parent["child_env_default"] == "default_child_val"


def test_env_var_in_list(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {
        "items": [
            "item1",
            "${LIST_ITEM_ENV}",
            "${LIST_ITEM_DEFAULT_ENV:default_list_item}",
        ]
    }
    write_config(config_file, config_data)
    with patch.dict(os.environ, {"LIST_ITEM_ENV": "list_item_from_env"}):
        settings = get_settings(str(config_file))
        assert isinstance(settings.items, list)
        assert settings.items[0] == "item1"
        assert settings.items[1] == "list_item_from_env"
        assert settings.items[2] == "default_list_item"


def test_mixed_config_and_secrets_with_env_vars(temp_config_files):
    config_file, secrets_file = temp_config_files
    config_data = {
        "general_setting": "from_config_file",
        "config_env": "${CONFIG_VAR:default_config_val}",
    }
    secrets_data = {
        "secret_key": "${SECRET_ENV_KEY}",
        "db_password": "${DB_PASS:default_db_pass}",
    }
    write_config(config_file, config_data)
    write_config(secrets_file, secrets_data)

    with patch.dict(
        os.environ,
        {"CONFIG_VAR": "env_config_val", "SECRET_ENV_KEY": "actual_secret"},
    ):
        settings = get_settings(str(config_file))
        assert settings.general_setting == "from_config_file"
        assert settings.config_env == "env_config_val"
        assert settings.secret_key == "actual_secret"
        assert settings.db_password == "default_db_pass"


def test_env_var_in_mcp_server_settings(temp_config_files):
    config_file, _ = temp_config_files
    config_data = {
        "mcp": {
            "servers": {
                "my_server": {
                    "command": "${MCP_COMMAND:default_command}",
                    "url": "${MCP_URL}",
                    "env": {"SERVER_SPECIFIC_ENV": "${SPECIFIC_VAR:specific_default}"},
                }
            }
        }
    }
    write_config(config_file, config_data)
    with patch.dict(
        os.environ,
        {"MCP_URL": "http://mcp.env.url", "SPECIFIC_VAR": "value_for_specific"},
    ):
        settings = get_settings(str(config_file))
        assert settings.mcp is not None
        assert settings.mcp.servers is not None
        assert "my_server" in settings.mcp.servers
        server_settings = settings.mcp.servers["my_server"]
        assert server_settings.command == "default_command"
        assert server_settings.url == "http://mcp.env.url"
        assert server_settings.env is not None
        assert server_settings.env["SERVER_SPECIFIC_ENV"] == "value_for_specific"
