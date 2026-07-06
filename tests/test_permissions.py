"""Tests for spark_code.permissions.PermissionManager."""

from unittest.mock import patch

import pytest

from spark_code.permissions import PermissionManager, resolve_mode_name


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


class TestAllowsWithoutPromptExtraction:
    """Task 5 (Phase 1): the no-prompt policy used to live in two places —
    check()'s pre-prompt short-circuits AND agent.py's inline parallel-path
    predicate — and that duplication is exactly how Phase 0's worker
    auto-approve bug happened. These tests pin down that
    allows_without_prompt() is a drop-in equivalent for check()'s "would this
    be allowed WITHOUT a prompt" decision, for every mode x tool-type
    combination, so both callers can share one implementation."""

    @pytest.mark.parametrize("mode", ["ask", "auto", "trust"])
    @pytest.mark.parametrize("tool,is_read_only", [
        ("read_file", True),
        ("glob", True),
        ("write_file", False),
        ("bash", False),
    ])
    def test_matches_noninteractive_check(self, mode, tool, is_read_only):
        # interactive=False means check() can never fall through to a prompt:
        # it either returns what allows_without_prompt would, or denies. That
        # makes this comparison a precise equivalence check across mode x
        # tool-type, without needing to mock the prompt at all.
        pm = PermissionManager(mode=mode, interactive=False)
        assert pm.allows_without_prompt(tool, is_read_only) == pm.check(tool, is_read_only)

    @pytest.mark.parametrize("mode", ["ask", "auto", "trust"])
    @pytest.mark.parametrize("is_read_only", [True, False])
    def test_matches_always_allow_list(self, mode, is_read_only):
        pm = PermissionManager(mode=mode, always_allow=["special_tool"],
                               interactive=False)
        assert (pm.allows_without_prompt("special_tool", is_read_only)
                == pm.check("special_tool", is_read_only))

    @pytest.mark.parametrize("mode", ["ask", "auto", "trust"])
    @pytest.mark.parametrize("is_read_only", [True, False])
    def test_matches_session_allow(self, mode, is_read_only):
        pm = PermissionManager(mode=mode, interactive=False)
        pm.session_allow.add("write_file")
        assert (pm.allows_without_prompt("write_file", is_read_only)
                == pm.check("write_file", is_read_only))

    def test_true_result_never_reaches_prompt_in_interactive_mode_either(self):
        """Whenever allows_without_prompt() says True, check() must return
        True without ever calling _prompt_user — including in INTERACTIVE
        mode, since that's the actual pre-prompt short-circuit path both
        callers rely on."""
        pm = PermissionManager(mode="trust", interactive=True)
        with patch.object(pm, "_prompt_user",
                          side_effect=AssertionError("must not prompt")):
            assert pm.allows_without_prompt("bash", False) is True
            assert pm.check("bash", False) is True

        pm2 = PermissionManager(mode="auto", always_allow=["glob"], interactive=True)
        with patch.object(pm2, "_prompt_user",
                          side_effect=AssertionError("must not prompt")):
            assert pm2.allows_without_prompt("read_file", True) is True
            assert pm2.check("read_file", True) is True
            assert pm2.allows_without_prompt("glob", False) is True
            assert pm2.check("glob", False) is True


def test_allows_without_prompt_false_falls_through_to_prompt_when_interactive():
    """The inverse of the above: when allows_without_prompt() says False and
    the manager IS interactive, check() must still consult _prompt_user (the
    extraction must not swallow the prompt path)."""
    pm = PermissionManager(mode="ask", interactive=True)
    assert pm.allows_without_prompt("write_file", False) is False
    with patch.object(pm, "_prompt_user", return_value=True) as mock_prompt:
        assert pm.check("write_file", False) is True
    mock_prompt.assert_called_once()


class TestResolveModeName:
    """Task 5 (Phase 1): Claude Code mode-name parity. resolve_mode_name()
    is the single place aliasing lives — used at every mode-parsing site
    (/mode, /config set permissions.mode, config file load) so they can't
    drift apart the way the no-prompt policy did."""

    def test_claude_code_mode_aliases(self):
        assert resolve_mode_name("default") == "ask"
        assert resolve_mode_name("acceptEdits") == "auto"
        assert resolve_mode_name("bypassPermissions") == "trust"
        assert resolve_mode_name("ask") == "ask"          # native names pass through
        assert resolve_mode_name("nonsense") is None

    def test_case_insensitive(self):
        assert resolve_mode_name("DEFAULT") == "ask"
        assert resolve_mode_name("ACCEPTEDITS") == "auto"
        assert resolve_mode_name("BypassPermissions") == "trust"
        assert resolve_mode_name("AUTO") == "auto"
        assert resolve_mode_name("Trust") == "trust"
        assert resolve_mode_name("Plan") == "plan"

    def test_native_names_all_pass_through(self):
        for name in ("ask", "auto", "trust", "plan"):
            assert resolve_mode_name(name) == name

    def test_none_and_empty_are_not_valid(self):
        assert resolve_mode_name("") is None
        assert resolve_mode_name("   ") is None


class TestPromptSuspendsEscWatcher:
    """The Critical final-review finding: the permission prompt reads stdin on
    the main thread while the Esc watcher's background thread holds cbreak and
    drains the same fd — swallowing the user's "y". _prompt_user must suspend
    any active watcher (pause_all) around Prompt.ask so the prompt owns stdin.
    """

    def test_prompt_user_pauses_active_watcher_around_ask(self):
        from spark_code.ui import esc_watcher

        events = []

        class _FakeWatcher:
            def pause(self):
                events.append("pause")

            def resume(self):
                events.append("resume")

        fake = _FakeWatcher()
        esc_watcher._register(fake)
        try:
            pm = PermissionManager(mode="ask")

            def _fake_ask(*a, **k):
                events.append("ask")
                return "y"

            with patch("spark_code.permissions.Prompt.ask", _fake_ask):
                assert pm._prompt_user("write_file", "writing") is True

            # Watcher paused BEFORE the prompt read stdin, resumed AFTER.
            assert events == ["pause", "ask", "resume"]
        finally:
            esc_watcher._unregister(fake)
