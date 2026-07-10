"""Regression tests for branch-name sanitization (audit finding #3).

Branch names were used raw in ``join(branch_dir, f"{name}.json")``, so a name
with a path separator ("feature/login") crashed with FileNotFoundError and a
traversal ("../x") escaped the per-project branch dir. sanitize_branch_name now
slugifies separators and rejects traversal/empty/hidden names at every entry
point, with friendly errors instead of tracebacks.
"""

import os

import pytest

from spark_code.branches import BranchManager, sanitize_branch_name
from spark_code.context import Context


def _ctx():
    c = Context(system_prompt="x")
    c.add_user("hello")
    c.add_assistant("hi")
    return c


class TestSanitizeBranchName:
    def test_slash_is_slugified(self):
        assert sanitize_branch_name("feature/login") == "feature-login"

    def test_backslash_is_slugified(self):
        assert sanitize_branch_name("feature\\login") == "feature-login"

    def test_plain_name_unchanged_and_idempotent(self):
        assert sanitize_branch_name("main") == "main"
        assert sanitize_branch_name("feature-login") == "feature-login"

    def test_surrounding_whitespace_stripped(self):
        assert sanitize_branch_name("  main  ") == "main"

    @pytest.mark.parametrize("bad", ["", "   ", "\t"])
    def test_empty_rejected(self, bad):
        with pytest.raises(ValueError):
            sanitize_branch_name(bad)

    @pytest.mark.parametrize("bad", ["..", "../x", "a/../b", "../../etc/passwd"])
    def test_traversal_rejected(self, bad):
        with pytest.raises(ValueError):
            sanitize_branch_name(bad)

    @pytest.mark.parametrize("bad", [".", ".hidden", ".."])
    def test_leading_dot_rejected(self, bad):
        with pytest.raises(ValueError):
            sanitize_branch_name(bad)


class TestBranchManagerSanitization:
    def test_save_with_slash_does_not_crash_and_writes_single_file(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        msg = bm.save_branch("feature/login", _ctx())
        assert "saved" in msg.lower()
        # A single slugified file inside the project dir — no sub-directory.
        files = os.listdir(bm.branch_dir)
        assert files == ["feature-login.json"]

    def test_saved_slash_name_round_trips_by_same_name(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        bm.save_branch("feature/login", _ctx())
        # Switching/deleting by the ORIGINAL "feature/login" must resolve.
        ok, _ = bm.switch_branch("feature/login", Context(system_prompt="y"))
        assert ok is True
        # Not the current branch anymore, so it is deletable.
        bm.save_branch("main", _ctx())
        ok_del, msg_del = bm.delete_branch("feature/login")
        assert ok_del is True
        assert "feature-login.json" not in os.listdir(bm.branch_dir)

    def test_switch_invalid_name_is_friendly_error_not_traceback(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        ok, msg = bm.switch_branch("../escape", Context(system_prompt="y"))
        assert ok is False
        assert "traversal" in msg.lower() or "invalid" in msg.lower()

    def test_delete_invalid_name_is_friendly_error(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        ok, msg = bm.delete_branch("..")
        assert ok is False
        assert msg  # a message, not a raised exception

    def test_merge_invalid_name_is_friendly_error(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        ok, msg = bm.merge_branch("../x", Context(system_prompt="y"))
        assert ok is False
        assert msg

    def test_create_invalid_name_raises_valueerror(self, tmp_path):
        bm = BranchManager(branch_dir=str(tmp_path), project_dir="/proj/x")
        with pytest.raises(ValueError):
            bm.create_branch("../evil", _ctx())

    def test_traversal_does_not_escape_branch_dir(self, tmp_path):
        # Even attempting a traversal must never create a file outside the
        # per-project branch dir.
        outside = tmp_path / "outside"
        outside.mkdir()
        bm = BranchManager(branch_dir=str(tmp_path / "branches"),
                           project_dir="/proj/x")
        with pytest.raises(ValueError):
            bm.save_branch("../outside/evil", _ctx())
        assert os.listdir(outside) == []
