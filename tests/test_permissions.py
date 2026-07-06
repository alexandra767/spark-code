"""Tests for spark_code.permissions.PermissionManager."""

from unittest.mock import patch

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


class TestNonInteractive:
    """Review fix (2026-07-06): a non-interactive PermissionManager (used by
    background workers on the shared event loop) must never reach
    _prompt_user/Prompt.ask, AND must never auto-approve everything either —
    that made every worker equivalent to trust mode regardless of the
    inherited mode. It must fail safe: allow exactly what the mode would
    allow without a prompt, and deny anything that would have prompted."""

    def test_ask_mode_denies_write_with_explanatory_message(self):
        pm = PermissionManager(mode="ask", interactive=False)
        with patch("spark_code.permissions.Prompt.ask",
                   side_effect=AssertionError("must not prompt")):
            allowed = pm.check("write_file", is_read_only=False,
                              details={"file_path": "/tmp/x"})
        assert allowed is False
        assert pm.last_denial_reason is not None
        assert "non-interactive worker" in pm.last_denial_reason
        assert "mode=ask" in pm.last_denial_reason
        assert "write_file" in pm.last_denial_reason

    def test_ask_mode_denies_everything_not_always_allowed(self):
        pm = PermissionManager(mode="ask", interactive=False)
        with patch("spark_code.permissions.Prompt.ask",
                   side_effect=AssertionError("must not prompt")):
            assert pm.check("read_file", is_read_only=True,
                            details={"file_path": "/tmp/x"}) is False
            assert pm.check("bash", is_read_only=False,
                            details={"command": "ls"}) is False

    def test_auto_mode_allows_read_only_denies_write(self):
        pm = PermissionManager(mode="auto", interactive=False)
        with patch("spark_code.permissions.Prompt.ask",
                   side_effect=AssertionError("must not prompt")):
            assert pm.check("read_file", is_read_only=True,
                            details={"file_path": "/tmp/x"}) is True
            allowed = pm.check("bash", is_read_only=False,
                              details={"command": "rm -rf /"})
        assert allowed is False
        assert pm.last_denial_reason is not None
        assert "mode=auto" in pm.last_denial_reason

    def test_trust_mode_allows_everything(self):
        pm = PermissionManager(mode="trust", interactive=False)
        with patch("spark_code.permissions.Prompt.ask",
                   side_effect=AssertionError("must not prompt")):
            assert pm.check("write_file", is_read_only=False, details="writing") is True
            assert pm.check("bash", is_read_only=False, details="danger") is True
            assert pm.check("delete_file", is_read_only=False, details="deleting") is True
        assert pm.last_denial_reason is None

    def test_always_allow_list_still_allowed_when_non_interactive(self):
        pm = PermissionManager(mode="ask", always_allow=["read_file"],
                               interactive=False)
        with patch("spark_code.permissions.Prompt.ask",
                   side_effect=AssertionError("must not prompt")):
            assert pm.check("read_file", is_read_only=True,
                            details={"file_path": "/tmp/x"}) is True

    def test_never_reaches_prompt_user(self):
        """P0 property: no code path in a non-interactive manager may ever
        reach _prompt_user, in any mode."""
        for mode in ("ask", "auto", "trust"):
            pm = PermissionManager(mode=mode, interactive=False)
            with patch.object(pm, "_prompt_user",
                             side_effect=AssertionError("must not prompt")):
                pm.check("write_file", is_read_only=False, details="writing")
                pm.check("read_file", is_read_only=True, details="reading")
                pm.check("bash", is_read_only=False, details="danger")
