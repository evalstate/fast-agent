"""Tests for server helper functions."""

from fast_agent.cli.commands.server_helpers import generate_server_name


class TestGenerateServerName:
    """Test cases for generate_server_name function."""

    def test_npm_package_with_org(self):
        """Test npm package names with organization prefix."""
        assert (
            generate_server_name("@modelcontextprotocol/server-filesystem") == "server_filesystem"
        )
        assert generate_server_name("@npmorg/mcp-server") == "mcp_server"
        assert generate_server_name("@my-org/my-mcp-server") == "my_mcp_server"

    def test_simple_package_names(self):
        """Test simple package names without org prefix."""
        assert generate_server_name("my-mcp-server") == "my_mcp_server"
        assert generate_server_name("server") == "server"
        assert generate_server_name("mcp_server") == "mcp_server"

    def test_file_paths(self):
        """Test file paths with extensions."""
        assert generate_server_name("./src/my-server.py") == "src_my_server"
        assert generate_server_name("server.py") == "server"
        assert generate_server_name("./mcp-server.js") == "mcp_server"
        assert generate_server_name("app/server.ts") == "app_server"

    def test_special_characters(self):
        """Test handling of special characters."""
        assert generate_server_name("my.server.name") == "my_server_name"
        assert generate_server_name("server-with-dashes") == "server_with_dashes"
        assert generate_server_name("server/with/slashes") == "slashes"
        assert generate_server_name("server@host") == "server_host"

    def test_multiple_underscores(self):
        """Test cleanup of multiple underscores."""
        assert generate_server_name("server--name") == "server_name"
        assert generate_server_name("my___server") == "my_server"

    def test_edge_cases(self):
        """Test edge cases."""
        assert generate_server_name("") == ""
        assert generate_server_name("@") == ""
        assert generate_server_name("./") == ""
        assert generate_server_name("---") == ""
        assert generate_server_name("123-server") == "123_server"

    def test_leading_trailing_cleanup(self):
        """Test removal of leading/trailing underscores."""
        assert generate_server_name("-server-") == "server"
        assert generate_server_name("_server_") == "server"
        assert generate_server_name("@-server-@") == "server"
