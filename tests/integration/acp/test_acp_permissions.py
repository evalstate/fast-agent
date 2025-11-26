"""
Tests for ACP tool permission handling.

Tests for PermissionStore file-based persistence and PermissionResult types.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.tool_permission_handler import (
    PERMISSION_DENIED_MESSAGE,
    PermissionDecision,
    PermissionResult,
)


class TestPermissionStore:
    """Unit tests for the PermissionStore file-based persistence."""

    @pytest.mark.asyncio
    async def test_store_creates_file_on_set(self) -> None:
        """Test that the store creates the auths.md file when a permission is set."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            store = PermissionStore(base_path)

            # File should not exist initially
            auths_file = base_path / ".fast-agent" / "auths.md"
            assert not auths_file.exists()

            # Set a permission
            await store.set("test_server", "test_tool", "allow_always")

            # File should now exist
            assert auths_file.exists()

            # Verify content format
            content = auths_file.read_text()
            assert "test_server" in content
            assert "test_tool" in content
            assert "allow_always" in content

    @pytest.mark.asyncio
    async def test_store_get_returns_none_for_unknown(self) -> None:
        """Test that get returns None for unknown tools."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = PermissionStore(Path(tmp_dir))

            result = await store.get("unknown_server", "unknown_tool")
            assert result is None

    @pytest.mark.asyncio
    async def test_store_get_returns_stored_permission(self) -> None:
        """Test that get returns the stored permission."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = PermissionStore(Path(tmp_dir))

            await store.set("server", "tool", "reject_always")
            result = await store.get("server", "tool")
            assert result == "reject_always"

    @pytest.mark.asyncio
    async def test_store_persists_across_instances(self) -> None:
        """Test that permissions persist when a new store instance is created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)

            # Create first store and set permission
            store1 = PermissionStore(base_path)
            await store1.set("my_server", "my_tool", "allow_always")

            # Create new store instance and verify permission is loaded
            store2 = PermissionStore(base_path)
            result = await store2.get("my_server", "my_tool")
            assert result == "allow_always"

    @pytest.mark.asyncio
    async def test_store_handles_multiple_permissions(self) -> None:
        """Test that multiple permissions can be stored and retrieved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = PermissionStore(Path(tmp_dir))

            await store.set("server1", "tool1", "allow_always")
            await store.set("server2", "tool2", "reject_always")
            await store.set("server1", "tool3", "allow_always")

            assert await store.get("server1", "tool1") == "allow_always"
            assert await store.get("server2", "tool2") == "reject_always"
            assert await store.get("server1", "tool3") == "allow_always"
            assert await store.get("server1", "tool2") is None

    @pytest.mark.asyncio
    async def test_store_remove_permission(self) -> None:
        """Test that permissions can be removed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = PermissionStore(Path(tmp_dir))

            await store.set("server", "tool", "allow_always")
            assert await store.get("server", "tool") == "allow_always"

            removed = await store.remove("server", "tool")
            assert removed is True
            assert await store.get("server", "tool") is None

    @pytest.mark.asyncio
    async def test_store_clear_all(self) -> None:
        """Test that all permissions can be cleared."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = Path(tmp_dir)
            store = PermissionStore(base_path)

            await store.set("server1", "tool1", "allow_always")
            await store.set("server2", "tool2", "reject_always")

            await store.clear()

            # Permissions should be cleared
            assert await store.get("server1", "tool1") is None
            assert await store.get("server2", "tool2") is None

            # File should be deleted
            auths_file = base_path / ".fast-agent" / "auths.md"
            assert not auths_file.exists()


class TestPermissionResult:
    """Unit tests for the PermissionResult helper class."""

    def test_allow_factory(self) -> None:
        """Test the allow() factory method."""
        result = PermissionResult.allow(remember=True)
        assert result.decision == PermissionDecision.ALLOW
        assert result.allowed is True
        assert result.remember is True

    def test_deny_factory(self) -> None:
        """Test the deny() factory method."""
        result = PermissionResult.deny(remember=False, message="test message")
        assert result.decision == PermissionDecision.DENY
        assert result.allowed is False
        assert result.remember is False
        assert result.message == "test message"

    def test_default_denied_message(self) -> None:
        """Test that the default denial message is available."""
        assert PERMISSION_DENIED_MESSAGE == "The User declined this operation."
