"""Tests for spark_code.permissions.PermissionManager."""

import pytest
from unittest.mock import patch, MagicMock

from spark_code.permissions import PermissionManager


class TestTrustMode:
    def test_trust_mode_allows_everything(self):
        pm = PermissionManager(mode="trust")
        assert pm.check("write_file", is_read_only=False, details="writing") is True
        assert pm.check("delete_file", is_read_only=False, details="deleting") is True
        assert pm.check("read_file", is_read_only=True, details="reading") is True


class TestAlwaysAllow:
    def test_always_allow_list_works(self):
        pm = PermissionManager(mode="ask", always_allow=["read_file", "list_dir"])
        assert pm.check("read_file", is_read_only=True, details="reading") is True
        assert pm.check("list_dir", is_read_only=True, details="listing") is True


class TestSessionAllow:
    def test_session_allow_works(self):
        pm = PermissionManager(mode="ask")
        pm.session_allow.add("write_file")
        assert pm.check("write_file", is_read_only=False, details="writing") is True


class TestAutoMode:
    def test_auto_mode_allows_read_only(self):
        pm = PermissionManager(mode="auto")
        assert pm.check("read_file", is_read_only=True, details="reading") is True

    def test_auto_mode_blocks_writes_when_prompt_denied(self):
        pm = PermissionManager(mode="auto")
        with patch.object(pm, "_prompt_user", return_value=False):
            result = pm.check("write_file", is_read_only=False, details="writing")
            assert result is False


class TestAskMode:
    def test_ask_mode_blocks_when_prompt_denied(self):
        pm = PermissionManager(mode="ask")
        with patch.object(pm, "_prompt_user", return_value=False):
            result = pm.check("dangerous_tool", is_read_only=False, details="danger")
            assert result is False

    def test_ask_mode_allows_when_prompt_accepted(self):
        pm = PermissionManager(mode="ask")
        with patch.object(pm, "_prompt_user", return_value=True):
            result = pm.check("write_file", is_read_only=False, details="writing")
            assert result is True


class TestModeSwitching:
    def test_mode_switching_works(self):
        pm = PermissionManager(mode="trust")
        assert pm.check("write_file", is_read_only=False, details="writing") is True

        pm.mode = "auto"
        assert pm.check("read_file", is_read_only=True, details="reading") is True

        with patch.object(pm, "_prompt_user", return_value=False):
            result = pm.check("write_file", is_read_only=False, details="writing")
            assert result is False
