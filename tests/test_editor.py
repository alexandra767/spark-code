"""Tests for editor detection + file handoff (Phase 3 Task 4).

Everything here is fully injected: fake env dicts, fake path_checkers, and
patched ``subprocess.Popen`` / ``shutil.which``. No test ever touches the
real environment or launches a real editor.
"""

import io
import re
from unittest.mock import patch

from rich.console import Console

from spark_code.editor import detect_editor, file_link, open_in_editor
from spark_code.ui.output import (
    _C_PATH,
    _path_style,
    _should_linkify,
    render_tool_call,
)


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
