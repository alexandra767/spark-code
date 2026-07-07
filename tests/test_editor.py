"""Tests for editor detection + file handoff (Phase 3 Task 4) and the
diff-in-editor option (Phase 3 Task 5).

Everything here is fully injected: fake env dicts, fake path_checkers, and
patched ``subprocess.Popen`` / ``shutil.which``. No test ever touches the
real environment or launches a real editor. ``open_diff_in_editor`` tests DO
use real ``tempfile.mkdtemp`` directories (so file CONTENTS can be asserted)
but every one of those is registered in ``spark_code.editor._diff_temp_dirs``
and swept by the ``_clean_diff_temp_dirs`` autouse fixture below.
"""

import io
import os
import re
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

import spark_code.editor as editor_mod
from spark_code.editor import (
    cleanup_diff_temp_dirs,
    detect_editor,
    file_link,
    maybe_preview_diff_in_editor,
    open_diff_in_editor,
    open_in_editor,
)
from spark_code.ui.output import (
    _C_PATH,
    _path_style,
    _should_linkify,
    render_tool_call,
)


@pytest.fixture(autouse=True)
def _clean_diff_temp_dirs():
    """Every test starts and ends with an empty tracking list so a test
    that forgets to clean up its own diff temp dirs can't leak into (or
    inherit stray state from) another test."""
    editor_mod._diff_temp_dirs.clear()
    yield
    cleanup_diff_temp_dirs()


def _which_only(*names: str):
    """Fake path_checker: returns a fake path for the given name(s), None
    for everything else — never touches the real PATH/shutil.which."""
    def _checker(name: str) -> str | None:
        return f"/usr/local/bin/{name}" if name in names else None
    return _checker


def _which_none(name: str) -> None:
    return None


# ---------------------------------------------------------------------------
# detect_editor — matrix, fully injected (no real `which` calls)
# ---------------------------------------------------------------------------

class TestDetectEditor:
    def test_vscode_with_cursor_on_path_prefers_cursor(self):
        env = {"TERM_PROGRAM": "vscode"}
        assert detect_editor(env=env, path_checker=_which_only("cursor")) == "cursor"

    def test_vscode_with_only_code_on_path_falls_back_to_code(self):
        env = {"TERM_PROGRAM": "vscode"}
        assert detect_editor(env=env, path_checker=_which_only("code")) == "code"

    def test_vscode_with_neither_on_path_still_returns_code(self):
        # TERM_PROGRAM=vscode is itself strong evidence code/cursor exists;
        # matches the spec precedence ("cursor" if on PATH else "code").
        env = {"TERM_PROGRAM": "vscode"}
        assert detect_editor(env=env, path_checker=_which_none) == "code"

    def test_no_term_program_with_xed_on_path_returns_xed(self):
        env: dict = {}
        with patch("spark_code.editor.sys.platform", "darwin"):
            assert detect_editor(env=env, path_checker=_which_only("xed")) == "xed"

    def test_nothing_detected_returns_none(self):
        env: dict = {}
        with patch("spark_code.editor.sys.platform", "darwin"):
            assert detect_editor(env=env, path_checker=_which_none) is None

    def test_xed_ignored_on_non_macos(self):
        env: dict = {}
        with patch("spark_code.editor.sys.platform", "linux"):
            assert detect_editor(env=env, path_checker=_which_only("xed")) is None


# ---------------------------------------------------------------------------
# open_in_editor — argv shapes, Popen patched (never launches anything real)
# ---------------------------------------------------------------------------

class TestOpenInEditor:
    def test_cursor_argv_with_line(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_in_editor("cursor", "/tmp/foo.py", line=42)
        assert ok is True
        args, kwargs = mock_popen.call_args
        assert args[0] == ["cursor", "--goto", "/tmp/foo.py:42"]
        assert kwargs.get("start_new_session") is True

    def test_cursor_argv_without_line(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_in_editor("cursor", "/tmp/foo.py")
        args, _ = mock_popen.call_args
        assert args[0] == ["cursor", "--goto", "/tmp/foo.py"]

    def test_code_argv_with_line(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_in_editor("code", "/tmp/bar.py", line=7)
        args, _ = mock_popen.call_args
        assert args[0] == ["code", "--goto", "/tmp/bar.py:7"]

    def test_xed_argv_with_line(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_in_editor("xed", "/tmp/baz.swift", line=3)
        args, _ = mock_popen.call_args
        assert args[0] == ["xed", "--line", "3", "/tmp/baz.swift"]

    def test_xed_argv_without_line(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_in_editor("xed", "/tmp/baz.swift")
        args, _ = mock_popen.call_args
        assert args[0] == ["xed", "/tmp/baz.swift"]

    def test_line_zero_is_not_dropped(self):
        # `line` is guarded with `is not None`, not truthiness — 0 is a
        # value the caller passed, not "no line" (review minor).
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_in_editor("cursor", "/tmp/foo.py", line=0)
        args, _ = mock_popen.call_args
        assert args[0] == ["cursor", "--goto", "/tmp/foo.py:0"]

    def test_unrecognized_editor_returns_false_no_popen(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_in_editor("vim", "/tmp/foo.py")
        assert ok is False
        mock_popen.assert_not_called()

    def test_popen_failure_returns_false_not_raise(self):
        with patch("spark_code.editor.subprocess.Popen", side_effect=OSError("nope")):
            ok = open_in_editor("cursor", "/tmp/foo.py")
        assert ok is False

    def test_popen_missing_binary_returns_false(self):
        with patch("spark_code.editor.subprocess.Popen", side_effect=FileNotFoundError()):
            ok = open_in_editor("code", "/tmp/foo.py")
        assert ok is False


# ---------------------------------------------------------------------------
# file_link — absolute + percent-encoded
# ---------------------------------------------------------------------------

class TestFileLink:
    def test_absolute_path_unchanged(self):
        link = file_link("/Users/alex/project/main.py")
        assert link == "file:///Users/alex/project/main.py"

    def test_relative_path_made_absolute(self):
        link = file_link("main.py")
        assert link.startswith("file:///")
        assert link.endswith("/main.py")

    def test_home_expansion(self):
        link = file_link("~/notes.txt")
        assert "~" not in link
        assert link.startswith("file:///")
        assert link.endswith("/notes.txt")

    def test_spaces_percent_encoded(self):
        link = file_link("/Users/alex/some file.py")
        assert link == "file:///Users/alex/some%20file.py"
        assert " " not in link

    def test_slashes_not_encoded(self):
        link = file_link("/Users/alex/a b/c.py")
        assert link == "file:///Users/alex/a%20b/c.py"


# ---------------------------------------------------------------------------
# OSC 8 linkification wiring (ui/output.py) — style-layer only. The gate
# (_should_linkify) takes an injectable `isatty` so no test touches the real
# stdout/tty; render_tool_call defaults `editor=None`, which is what every
# EXISTING call site (and every existing rendering test) still uses — so
# their recorded output is byte-for-byte unchanged.
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;[^\x1b]*\x1b\\")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _record_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=100)


class TestShouldLinkify:
    def test_none_editor_never_links(self):
        assert _should_linkify(None, isatty=lambda: True) is False

    def test_explicit_none_string_never_links(self):
        assert _should_linkify("none", isatty=lambda: True) is False

    def test_non_tty_never_links_even_with_editor(self):
        assert _should_linkify("cursor", isatty=lambda: False) is False

    def test_tty_plus_editor_links(self):
        assert _should_linkify("cursor", isatty=lambda: True) is True

    def test_isatty_exception_is_swallowed_to_false(self):
        def boom():
            raise RuntimeError("no tty")
        assert _should_linkify("cursor", isatty=boom) is False


class TestPathStyle:
    def test_no_editor_returns_plain_style_unchanged(self):
        assert _path_style("/tmp/x.py", None) == _C_PATH

    def test_editor_plus_tty_layers_in_the_link(self):
        style = _path_style("/tmp/x.py", "cursor", isatty=lambda: True)
        assert style.startswith(_C_PATH)
        assert "link file:///tmp/x.py" in style

    def test_editor_without_tty_stays_plain(self):
        assert _path_style("/tmp/x.py", "cursor", isatty=lambda: False) == _C_PATH

    def test_empty_path_stays_plain_even_with_editor_and_tty(self):
        assert _path_style("", "cursor", isatty=lambda: True) == _C_PATH


class TestRenderToolCallLinkify:
    """render_tool_call's ``editor`` kwarg is opt-in — every existing call
    site (and every pre-Task-4 rendering test) omits it, so this class
    verifies the new feature ADDITIVELY rather than by editing old tests."""

    def test_default_omits_editor_output_has_no_osc8(self):
        console = _record_console()
        render_tool_call(console, "read_file", {"file_path": "/tmp/x.py"})
        raw = console.file.getvalue()
        assert "\x1b]8" not in raw
        assert "/tmp/x.py" in raw

    def test_explicit_editor_none_kwarg_matches_default_byte_for_byte(self):
        c1 = _record_console()
        render_tool_call(c1, "read_file", {"file_path": "/tmp/x.py"})
        c2 = _record_console()
        render_tool_call(c2, "read_file", {"file_path": "/tmp/x.py"}, editor=None)
        assert c1.file.getvalue() == c2.file.getvalue()

    def test_editor_plus_tty_adds_osc8_link_style_only(self, monkeypatch):
        import spark_code.ui.output as output_mod
        monkeypatch.setattr(output_mod.sys.stdout, "isatty", lambda: True)

        plain = _record_console()
        render_tool_call(plain, "read_file", {"file_path": "/tmp/x.py"})
        plain_raw = plain.file.getvalue()

        linked = _record_console()
        render_tool_call(linked, "read_file", {"file_path": "/tmp/x.py"}, editor="cursor")
        linked_raw = linked.file.getvalue()

        assert "\x1b]8" not in plain_raw
        assert "\x1b]8;id=" in linked_raw
        assert "file:///tmp/x.py" in linked_raw
        # Style-layer only: the VISIBLE text is identical either way.
        assert _strip_ansi(plain_raw) == _strip_ansi(linked_raw)

    def test_ui_editor_none_string_never_links_even_on_a_real_tty(self, monkeypatch):
        import spark_code.ui.output as output_mod
        monkeypatch.setattr(output_mod.sys.stdout, "isatty", lambda: True)
        console = _record_console()
        render_tool_call(console, "read_file", {"file_path": "/tmp/x.py"}, editor="none")
        assert "\x1b]8" not in console.file.getvalue()

    def test_non_tty_stdout_never_links_even_with_editor_resolved(self, monkeypatch):
        import spark_code.ui.output as output_mod
        monkeypatch.setattr(output_mod.sys.stdout, "isatty", lambda: False)
        console = _record_console()
        render_tool_call(console, "read_file", {"file_path": "/tmp/x.py"}, editor="cursor")
        assert "\x1b]8" not in console.file.getvalue()


# ---------------------------------------------------------------------------
# Path-label escape-injection hardening (review follow-up). The OSC 8 URL
# side was always percent-encoded (file_link), but the VISIBLE label text
# was appended raw — a filename containing ESC/]8;; bytes injected live
# terminal escape sequences. _abbreviate_path is the single choke point all
# six rendered-path sites route through; it now replaces non-printable
# characters with U+FFFD.
# ---------------------------------------------------------------------------

class TestPathLabelSanitization:
    def test_abbreviate_path_replaces_control_chars(self):
        from spark_code.ui.output import _abbreviate_path
        assert _abbreviate_path("/tmp/a\x1bb.py") == "/tmp/a�b.py"

    def test_abbreviate_path_printable_paths_unchanged(self):
        from spark_code.ui.output import _abbreviate_path
        assert _abbreviate_path("/tmp/x.py") == "/tmp/x.py"
        assert _abbreviate_path("/tmp/with space.py") == "/tmp/with space.py"

    def test_osc8_injection_via_filename_is_neutralized(self):
        console = _record_console()
        evil = "/tmp/\x1b]8;;http://evil\x1b\\click.py"
        render_tool_call(console, "read_file", {"file_path": evil})
        raw = console.file.getvalue()
        # The injected OSC 8 open-sequence must not survive into the stream.
        assert "\x1b]8;;http://evil" not in raw
        # The printable remainder of the name still renders.
        assert "click.py" in raw

    def test_injection_neutralized_at_every_path_site(self):
        evil = "/tmp/\x1b]8;;evil\x1b\\f"
        cases = [
            ("write_file", {"file_path": evil, "content": "x"}),
            ("edit_file", {"file_path": evil, "old_string": "a", "new_string": "b"}),
            ("grep", {"pattern": "p", "path": evil}),
            ("glob", {"pattern": "*.py", "path": evil}),
            ("list_dir", {"path": evil}),
        ]
        for name, args in cases:
            console = _record_console()
            render_tool_call(console, name, args)
            assert "\x1b]8;;evil" not in console.file.getvalue(), name


# ---------------------------------------------------------------------------
# Agent(editor=...) wiring — a resolved editor reaches render_tool_call.
# ---------------------------------------------------------------------------

class TestAgentEditorWiring:
    def test_default_editor_is_none(self):
        from unittest.mock import MagicMock

        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry

        agent = Agent(
            model=MagicMock(), context=Context(), tools=ToolRegistry(),
            permissions=PermissionManager(mode="trust", always_allow=[]),
        )
        assert agent.editor is None

    def test_editor_kwarg_is_stored(self):
        from unittest.mock import MagicMock

        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry

        agent = Agent(
            model=MagicMock(), context=Context(), tools=ToolRegistry(),
            permissions=PermissionManager(mode="trust", always_allow=[]),
            editor="cursor",
        )
        assert agent.editor == "cursor"


# ---------------------------------------------------------------------------
# /open command (spark_code/cli.py) — resolves the editor (config +
# detection), calls open_in_editor, and never blocks or raises. Every real
# editor launch is patched out (spark_code.cli.open_in_editor /
# spark_code.cli.detect_editor) — this suite never runs a real editor.
# ---------------------------------------------------------------------------

def _open_command_deps(editor: str = "none"):
    """Minimal handle_slash_command dependencies for /open tests (mirrors
    tests/test_cli_helpers.py's _init_command_deps pattern)."""
    from unittest.mock import MagicMock

    from spark_code.context import Context
    from spark_code.model import ModelClient
    from spark_code.permissions import PermissionManager
    from spark_code.skills.base import SkillRegistry

    config = {
        "model": {
            "endpoint": "http://localhost:11434", "name": "test-model",
            "temperature": 0.7, "max_tokens": 4096, "api_key": "", "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
        "ui": {"editor": editor},
    }
    return {
        "context": Context(),
        "console": Console(record=True, width=120),
        "config": config,
        "skills": SkillRegistry(),
        "model": MagicMock(spec=ModelClient),
        "permissions": PermissionManager(mode="auto"),
    }


class TestResolveEditor:
    def test_explicit_config_wins_without_detection(self, monkeypatch):
        import spark_code.cli as cli_mod

        def _fail():
            raise AssertionError("detect_editor must not be called on explicit config")
        monkeypatch.setattr(cli_mod, "detect_editor", _fail)
        assert cli_mod._resolve_editor({"ui": {"editor": "code"}}) == "code"
        assert cli_mod._resolve_editor({"ui": {"editor": "cursor"}}) == "cursor"
        assert cli_mod._resolve_editor({"ui": {"editor": "xed"}}) == "xed"

    def test_auto_delegates_to_detect_editor(self, monkeypatch):
        import spark_code.cli as cli_mod
        monkeypatch.setattr(cli_mod, "detect_editor", lambda: "cursor")
        assert cli_mod._resolve_editor({"ui": {"editor": "auto"}}) == "cursor"

    def test_missing_key_defaults_to_auto(self, monkeypatch):
        import spark_code.cli as cli_mod
        monkeypatch.setattr(cli_mod, "detect_editor", lambda: "xed")
        assert cli_mod._resolve_editor({}) == "xed"

    def test_none_and_unrecognized_resolve_to_none(self, monkeypatch):
        import spark_code.cli as cli_mod
        monkeypatch.setattr(cli_mod, "detect_editor", lambda: "cursor")
        assert cli_mod._resolve_editor({"ui": {"editor": "none"}}) is None
        assert cli_mod._resolve_editor({"ui": {"editor": "emacs"}}) is None


class TestOpenCommand:
    def test_no_args_prints_usage(self):
        from spark_code.cli import handle_slash_command

        deps = _open_command_deps()
        result = handle_slash_command(
            "/open", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "Usage" in deps["console"].export_text()

    def test_missing_file_reports_no_such_file(self, tmp_path):
        from spark_code.cli import handle_slash_command

        deps = _open_command_deps(editor="cursor")
        missing = tmp_path / "nope.py"
        result = handle_slash_command(
            f"/open {missing}", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "No such file" in deps["console"].export_text()

    def test_ui_editor_none_prints_dim_note_without_launching_anything(self, tmp_path, monkeypatch):
        import spark_code.cli as cli_mod
        from spark_code.cli import handle_slash_command

        target = tmp_path / "a.py"
        target.write_text("x = 1")
        called = []
        monkeypatch.setattr(cli_mod, "open_in_editor", lambda *a, **kw: called.append((a, kw)) or True)

        deps = _open_command_deps(editor="none")
        result = handle_slash_command(
            f"/open {target}", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "No editor detected" in deps["console"].export_text()
        assert called == []  # open_in_editor never invoked — editor resolved to None

    def test_bare_file_opens_with_no_line(self, tmp_path, monkeypatch):
        import spark_code.cli as cli_mod
        from spark_code.cli import handle_slash_command

        target = tmp_path / "a.py"
        target.write_text("x = 1")
        calls = []
        monkeypatch.setattr(
            cli_mod, "open_in_editor",
            lambda editor, path, line=None: calls.append((editor, path, line)) or True,
        )

        deps = _open_command_deps(editor="cursor")
        result = handle_slash_command(
            f"/open {target}", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("cursor", str(target), None)]
        assert "No editor detected" not in deps["console"].export_text()

    def test_file_colon_line_parses_line_number(self, tmp_path, monkeypatch):
        import spark_code.cli as cli_mod
        from spark_code.cli import handle_slash_command

        target = tmp_path / "a.py"
        target.write_text("x = 1\ny = 2\n")
        calls = []
        monkeypatch.setattr(
            cli_mod, "open_in_editor",
            lambda editor, path, line=None: calls.append((editor, path, line)) or True,
        )

        deps = _open_command_deps(editor="cursor")
        result = handle_slash_command(
            f"/open {target}:2", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("cursor", str(target), 2)]

    def test_auto_config_calls_detect_editor(self, tmp_path, monkeypatch):
        import spark_code.cli as cli_mod
        from spark_code.cli import handle_slash_command

        target = tmp_path / "a.py"
        target.write_text("x = 1")
        monkeypatch.setattr(cli_mod, "detect_editor", lambda: "code")
        calls = []
        monkeypatch.setattr(
            cli_mod, "open_in_editor",
            lambda editor, path, line=None: calls.append((editor, path, line)) or True,
        )

        deps = _open_command_deps(editor="auto")
        result = handle_slash_command(
            f"/open {target}", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert calls == [("code", str(target), None)]

    def test_open_in_editor_failure_prints_dim_note(self, tmp_path, monkeypatch):
        import spark_code.cli as cli_mod
        from spark_code.cli import handle_slash_command

        target = tmp_path / "a.py"
        target.write_text("x = 1")
        monkeypatch.setattr(cli_mod, "open_in_editor", lambda *a, **kw: False)

        deps = _open_command_deps(editor="cursor")
        result = handle_slash_command(
            f"/open {target}", deps["context"], deps["console"], deps["config"],
            deps["skills"], deps["model"], permissions=deps["permissions"],
        )
        assert result is None
        assert "No editor detected" in deps["console"].export_text()


# ---------------------------------------------------------------------------
# open_diff_in_editor (Phase 3 Task 5) — writes old/new to temp files under
# one mkdtemp and launches `cursor|code --diff before after`, fire-and-forget.
# Popen is patched; file WRITES are real (temp-dir only) so contents can be
# asserted; nothing here ever launches a real editor.
# ---------------------------------------------------------------------------

class TestOpenDiffInEditor:
    def test_code_launches_diff_with_matching_contents(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_diff_in_editor(
                "code", "/some/proj/app.py", "old content\n", "new content\n")
        assert ok is True
        args, kwargs = mock_popen.call_args
        argv = args[0]
        assert argv[0] == "code"
        assert argv[1] == "--diff"
        before_path, after_path = argv[2], argv[3]
        assert os.path.basename(before_path) == "app.py.before"
        assert os.path.basename(after_path) == "app.py.after"
        with open(before_path, encoding="utf-8") as f:
            assert f.read() == "old content\n"
        with open(after_path, encoding="utf-8") as f:
            assert f.read() == "new content\n"
        assert kwargs.get("start_new_session") is True
        assert kwargs.get("stdin") == subprocess.DEVNULL

    def test_cursor_launches_diff(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_diff_in_editor("cursor", "/some/proj/app.py", "a", "b")
        assert ok is True
        argv = mock_popen.call_args[0][0]
        assert argv[0] == "cursor"
        assert argv[1] == "--diff"

    def test_xed_is_a_noop_no_popen_no_temp_dir(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_diff_in_editor("xed", "/some/proj/app.py", "a", "b")
        assert ok is False
        mock_popen.assert_not_called()
        assert editor_mod._diff_temp_dirs == []

    def test_unrecognized_editor_returns_false_no_popen(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            ok = open_diff_in_editor("vim", "/some/proj/app.py", "a", "b")
        assert ok is False
        mock_popen.assert_not_called()
        assert editor_mod._diff_temp_dirs == []

    def test_temp_dir_registered_for_cleanup(self):
        with patch("spark_code.editor.subprocess.Popen"):
            open_diff_in_editor("code", "/some/proj/app.py", "a", "b")
        assert len(editor_mod._diff_temp_dirs) == 1
        assert os.path.isdir(editor_mod._diff_temp_dirs[0])

    def test_basename_sanitized_stays_under_temp_dir(self):
        evil = "/some/proj/../../etc/evil name!\x1b.py"
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_diff_in_editor("code", evil, "a", "b")
        before_path = mock_popen.call_args[0][0][2]
        tmp_dir = editor_mod._diff_temp_dirs[0]
        # Strictly under the mkdtemp dir — no ../ or separators smuggled in.
        assert os.path.dirname(before_path) == tmp_dir
        name = os.path.basename(before_path)
        assert "/" not in name
        assert ".." not in name
        assert name.endswith(".before")

    def test_empty_file_path_falls_back_to_generic_name(self):
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            open_diff_in_editor("code", "", "a", "b")
        before_path = mock_popen.call_args[0][0][2]
        assert os.path.basename(before_path) == "file.before"

    def test_popen_failure_returns_false_but_files_stay_tracked(self):
        with patch("spark_code.editor.subprocess.Popen",
                   side_effect=OSError("nope")):
            ok = open_diff_in_editor("code", "/some/proj/app.py", "a", "b")
        assert ok is False
        # The temp files were already written before the launch attempt —
        # cleanup still owns them even though the launch itself failed.
        assert len(editor_mod._diff_temp_dirs) == 1


# ---------------------------------------------------------------------------
# cleanup_diff_temp_dirs — removes exactly what THIS module tracked, never
# an arbitrary path, and is safe to call repeatedly / with nothing tracked.
# ---------------------------------------------------------------------------

class TestCleanupDiffTempDirs:
    def test_removes_tracked_dirs(self):
        with patch("spark_code.editor.subprocess.Popen"):
            open_diff_in_editor("code", "/some/proj/app.py", "a", "b")
        tmp_dir = editor_mod._diff_temp_dirs[0]
        assert os.path.isdir(tmp_dir)
        cleanup_diff_temp_dirs()
        assert not os.path.isdir(tmp_dir)
        assert editor_mod._diff_temp_dirs == []

    def test_noop_when_nothing_tracked(self):
        cleanup_diff_temp_dirs()  # must not raise
        assert editor_mod._diff_temp_dirs == []

    def test_idempotent_double_call(self):
        with patch("spark_code.editor.subprocess.Popen"):
            open_diff_in_editor("code", "/some/proj/app.py", "a", "b")
        cleanup_diff_temp_dirs()
        cleanup_diff_temp_dirs()  # must not raise the second time


# ---------------------------------------------------------------------------
# maybe_preview_diff_in_editor — the pure(ish) gate the agent call site uses:
# disabled → never touches Popen; enabled+cursor/code → launches the diff;
# enabled+xed → no Popen, one-time dim note (state threaded via a bool the
# caller stores back on itself, same pattern as the agent's other
# fire-once flags); enabled+None editor → silent no-op (no note invented
# for an editor /open itself doesn't recognize as diff-capable).
# ---------------------------------------------------------------------------

class TestMaybePreviewDiffInEditor:
    def test_disabled_never_calls_popen(self):
        console = Console(record=True, width=100)
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            noted = maybe_preview_diff_in_editor(
                console, "code", False, "/tmp/a.py", "old", "new", False)
        mock_popen.assert_not_called()
        assert noted is False

    def test_enabled_code_launches_diff(self):
        console = Console(record=True, width=100)
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            noted = maybe_preview_diff_in_editor(
                console, "code", True, "/tmp/a.py", "old", "new", False)
        argv = mock_popen.call_args[0][0]
        assert argv[1] == "--diff"
        assert noted is False  # xed-note flag untouched for non-xed editors

    def test_enabled_cursor_launches_diff(self):
        console = Console(record=True, width=100)
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            maybe_preview_diff_in_editor(
                console, "cursor", True, "/tmp/a.py", "old", "new", False)
        argv = mock_popen.call_args[0][0]
        assert argv[0] == "cursor"
        assert argv[1] == "--diff"

    def test_enabled_none_editor_is_silent_noop(self):
        console = Console(record=True, width=100)
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            noted = maybe_preview_diff_in_editor(
                console, None, True, "/tmp/a.py", "old", "new", False)
        mock_popen.assert_not_called()
        assert noted is False
        assert console.export_text() == ""

    def test_xed_notes_once(self):
        console = Console(record=True, width=100)
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            noted = maybe_preview_diff_in_editor(
                console, "xed", True, "/tmp/a.py", "old", "new", False)
        mock_popen.assert_not_called()
        assert noted is True
        assert "xed" in console.export_text().lower()

    def test_xed_second_call_does_not_repeat_note(self):
        console = Console(record=True, width=100)
        noted = maybe_preview_diff_in_editor(
            console, "xed", True, "/tmp/a.py", "old", "new", True)
        assert noted is True
        assert console.export_text() == ""  # already noted this session


# ---------------------------------------------------------------------------
# Agent wiring — ``diff_in_editor`` config read once at construction (same
# pattern as ``editor``) and threaded through the real edit_file preview
# call site in ``_execute_single_tool``.
# ---------------------------------------------------------------------------

class TestAgentDiffInEditorWiring:
    def test_default_diff_in_editor_is_false(self):
        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry

        agent = Agent(
            model=MagicMock(), context=Context(), tools=ToolRegistry(),
            permissions=PermissionManager(mode="trust", always_allow=[]),
        )
        assert agent.diff_in_editor is False

    def test_diff_in_editor_kwarg_is_stored(self):
        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry

        agent = Agent(
            model=MagicMock(), context=Context(), tools=ToolRegistry(),
            permissions=PermissionManager(mode="trust", always_allow=[]),
            diff_in_editor=True,
        )
        assert agent.diff_in_editor is True

    @pytest.mark.asyncio
    async def test_edit_file_preview_opens_diff_when_enabled_and_code(self, tmp_path):
        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry
        from spark_code.tools.edit_file import EditFileTool

        target = tmp_path / "app.py"
        target.write_text("old content\n")

        tools = ToolRegistry()
        tools.register(EditFileTool())
        # auto + non-interactive: denies edit_file (a write) WITHOUT ever
        # blocking on a real prompt — exactly what a unit test needs, and
        # the diff preview fires unconditionally before that check runs.
        permissions = PermissionManager(mode="auto", interactive=False)

        agent = Agent(
            model=MagicMock(), context=Context(), tools=tools,
            permissions=permissions, console=Console(record=True, width=100),
            editor="code", diff_in_editor=True,
        )

        tc = {
            "id": "1", "name": "edit_file",
            "arguments": {
                "file_path": str(target),
                "old_string": "old content\n",
                "new_string": "new content\n",
            },
        }
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            await agent._execute_single_tool(tc)

        mock_popen.assert_called_once()
        argv = mock_popen.call_args[0][0]
        assert argv[0] == "code"
        assert argv[1] == "--diff"
        with open(argv[2], encoding="utf-8") as f:
            assert f.read() == "old content\n"
        with open(argv[3], encoding="utf-8") as f:
            assert f.read() == "new content\n"

    @pytest.mark.asyncio
    async def test_edit_file_preview_no_popen_when_disabled(self, tmp_path):
        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry
        from spark_code.tools.edit_file import EditFileTool

        target = tmp_path / "app.py"
        target.write_text("old content\n")

        tools = ToolRegistry()
        tools.register(EditFileTool())
        permissions = PermissionManager(mode="auto", interactive=False)

        agent = Agent(
            model=MagicMock(), context=Context(), tools=tools,
            permissions=permissions, console=Console(record=True, width=100),
            editor="code", diff_in_editor=False,
        )

        tc = {
            "id": "1", "name": "edit_file",
            "arguments": {
                "file_path": str(target),
                "old_string": "old content\n",
                "new_string": "new content\n",
            },
        }
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            await agent._execute_single_tool(tc)

        mock_popen.assert_not_called()

    @pytest.mark.asyncio
    async def test_edit_file_preview_xed_notes_once_across_two_edits(self, tmp_path):
        from spark_code.agent import Agent
        from spark_code.context import Context
        from spark_code.permissions import PermissionManager
        from spark_code.tools.base import ToolRegistry
        from spark_code.tools.edit_file import EditFileTool

        target = tmp_path / "app.py"
        target.write_text("old content\n")

        tools = ToolRegistry()
        tools.register(EditFileTool())
        permissions = PermissionManager(mode="auto", interactive=False)
        console = Console(record=True, width=100)

        agent = Agent(
            model=MagicMock(), context=Context(), tools=tools,
            permissions=permissions, console=console,
            editor="xed", diff_in_editor=True,
        )

        tc = {
            "id": "1", "name": "edit_file",
            "arguments": {
                "file_path": str(target),
                "old_string": "old content\n",
                "new_string": "new content\n",
            },
        }
        with patch("spark_code.editor.subprocess.Popen") as mock_popen:
            await agent._execute_single_tool(dict(tc))
            first_output = console.export_text()
            await agent._execute_single_tool(dict(tc))
            second_output = console.export_text()[len(first_output):]

        mock_popen.assert_not_called()
        assert "xed" in first_output.lower()
        assert "xed" not in second_output.lower()
