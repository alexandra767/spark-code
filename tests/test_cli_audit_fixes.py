"""Regression tests for the cli.py audit fixes.

Covers findings:
  #2  /image on a directory → friendly error, not IsADirectoryError
  #4  /checkpoint tells the truth when `git stash apply` fails
  #5  usage strings escape Rich-swallowed brackets
  #6  EOF (Ctrl+D) at in-command sub-prompts cancels instead of crashing
  #7  /config set permissions.mode plan routes through /mode plan semantics
  #9  /history escapes cwd/label before Rich markup
  #11 the /loop Esc handler also cancels the in-flight generation
"""

from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

import spark_code.cli as cli_mod
from spark_code.cli import _loop_escape_action, handle_slash_command
from spark_code.context import Context
from spark_code.model import ModelClient
from spark_code.permissions import PermissionManager
from spark_code.plan_mode import PlanState
from spark_code.skills.base import SkillRegistry


def _console():
    buf = StringIO()
    # A non-terminal Console writes PLAIN text (no ANSI style codes), so
    # substring assertions on rendered brackets are exact; width=200 avoids
    # any line wrapping that could split a token across lines.
    return Console(file=buf, width=200), buf


def _model():
    m = MagicMock(spec=ModelClient)
    m.total_input_tokens = 0
    m.total_output_tokens = 0
    return m


def _call(cmd, *, console, config=None, permissions=None, plan_state=None,
          team_manager=None):
    return handle_slash_command(
        cmd, Context(system_prompt="x"), console,
        config if config is not None else {"model": {}, "permissions": {}},
        SkillRegistry(), _model(),
        permissions=permissions, team_manager=team_manager, plan_state=plan_state,
    )


# ---------------------------------------------------------------------------
# #2 — /image on a directory
# ---------------------------------------------------------------------------

def test_image_on_directory_does_not_crash(tmp_path):
    console, buf = _console()
    # Must not raise IsADirectoryError; returns None and reports "not a file".
    result = _call(f"/image {tmp_path}", console=console)
    assert result is None
    assert "Not a file" in buf.getvalue()


def test_image_on_regular_file_is_accepted(tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    console, _ = _console()
    result = _call(f"/image {img} describe it", console=console)
    assert result == "__IMAGE_SENT__"


# ---------------------------------------------------------------------------
# #5 — bracketed usage strings survive Rich rendering
# ---------------------------------------------------------------------------

def test_image_usage_shows_prompt_bracket():
    console, buf = _console()
    _call("/image", console=console)
    assert "[prompt]" in buf.getvalue()


def test_new_usage_shows_description_bracket():
    console, buf = _console()
    _call("/new", console=console)
    assert "[description]" in buf.getvalue()


def test_team_usage_shows_id_bracket():
    console, buf = _console()
    _call("/team", console=console, team_manager=MagicMock())
    assert "[id]" in buf.getvalue()


# ---------------------------------------------------------------------------
# #6 — EOF (Ctrl+D) at in-command sub-prompts
# ---------------------------------------------------------------------------

def test_setkey_eof_cancels_without_crash(monkeypatch):
    def _raise_eof(*a, **k):
        raise EOFError
    monkeypatch.setattr(cli_mod.Prompt, "ask", staticmethod(_raise_eof))
    console, buf = _console()
    result = _call("/setkey openrouter", console=console)
    assert result is None
    assert "Cancelled" in buf.getvalue()


def test_setkey_keyboardinterrupt_cancels_without_crash(monkeypatch):
    def _raise_ki(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli_mod.Prompt, "ask", staticmethod(_raise_ki))
    console, buf = _console()
    result = _call("/setkey openrouter", console=console)
    assert result is None
    assert "Cancelled" in buf.getvalue()


def test_resume_picker_eof_cancels_without_crash(monkeypatch):
    fake = [{"path": "/h/s1.json", "name": "s1", "label": "", "age": "1h ago",
             "turns": 3, "cwd": "/proj", "_meta_loaded": True}]
    monkeypatch.setattr(cli_mod, "_list_sessions", lambda *a, **k: fake)

    def _raise_eof(*a, **k):
        raise EOFError
    monkeypatch.setattr(cli_mod.Prompt, "ask", staticmethod(_raise_eof))
    console, _ = _console()
    # No exception escapes; command returns None.
    assert _call("/resume", console=console) is None


def test_rewind_menu_eof_cancels_without_crash(monkeypatch):
    def _raise_eof(*a, **k):
        raise EOFError
    monkeypatch.setattr(cli_mod.Prompt, "ask", staticmethod(_raise_eof))
    console, _ = _console()
    assert _call("/rewind", console=console) is None


# ---------------------------------------------------------------------------
# #7 — /config set permissions.mode plan
# ---------------------------------------------------------------------------

def test_config_set_mode_plan_routes_through_plan_semantics(monkeypatch):
    recorded = {}

    def _fake_set_config(config, key_path, value, **kwargs):
        recorded["key"] = key_path
        recorded["value"] = value
        # mimic real behaviour: mutate the in-memory config dict
        config.setdefault("permissions", {})["mode"] = value
        return True, f"Set {key_path} = {value}"

    monkeypatch.setattr(cli_mod, "set_config", _fake_set_config)

    config = {"model": {}, "permissions": {"mode": "ask"}}
    perms = PermissionManager(mode="ask")
    plan = PlanState()
    console, buf = _console()

    result = handle_slash_command(
        "/config set permissions.mode plan", Context(system_prompt="x"), console,
        config, SkillRegistry(), _model(), permissions=perms, plan_state=plan,
    )
    assert result is None
    # Live session: plan mode active, permissions dropped to a VALID live mode.
    assert plan.active is True
    assert perms.mode == "auto"
    # Persisted value is the valid "auto", never the unrunnable "plan".
    assert recorded == {"key": "permissions.mode", "value": "auto"}
    assert "plan mode" in buf.getvalue().lower()


def test_config_set_mode_trust_still_works_normally(monkeypatch):
    def _fake_set_config(config, key_path, value, **kwargs):
        config.setdefault("permissions", {})["mode"] = value
        return True, f"Set {key_path} = {value}"

    monkeypatch.setattr(cli_mod, "set_config", _fake_set_config)
    config = {"model": {}, "permissions": {"mode": "ask"}}
    perms = PermissionManager(mode="ask")
    plan = PlanState()
    console, _ = _console()
    handle_slash_command(
        "/config set permissions.mode trust", Context(system_prompt="x"), console,
        config, SkillRegistry(), _model(), permissions=perms, plan_state=plan,
    )
    assert perms.mode == "trust"
    assert plan.active is False


# ---------------------------------------------------------------------------
# #9 — /history escapes cwd and label
# ---------------------------------------------------------------------------

def test_history_escapes_bracketed_cwd_and_label(monkeypatch):
    fake = [{
        "path": "/h/20260101_000000.json", "name": "20260101_000000",
        "label": "fix [urgent] bug", "age": "2h ago", "turns": 5,
        "cwd": "/data/pr[oj]ect", "_meta_loaded": True,
    }]
    monkeypatch.setattr(cli_mod, "_list_sessions", lambda *a, **k: fake)
    console, buf = _console()
    result = _call("/history", console=console)
    assert result is None
    out = buf.getvalue()
    # Escaping preserves the literal brackets in the rendered output; without
    # it Rich would treat "[oj]" / "[urgent]" as (invalid) markup tags.
    assert "pr[oj]ect" in out
    assert "[urgent]" in out


# ---------------------------------------------------------------------------
# #1 — the REPL slash-command dispatch is wrapped in a crash-containing
# umbrella. Driving the whole async REPL is impractical, so assert the wiring
# structurally (same approach as test_polish_features' patched-stdout guard).
# ---------------------------------------------------------------------------

def test_repl_slash_dispatch_has_crash_umbrella():
    import inspect

    source = inspect.getsource(cli_mod.run_interactive)
    # The slash block opens a try immediately and closes with an
    # Exception-catching handler that reports a friendly error and continues.
    assert 'if user_input.startswith("/"):' in source
    assert "except Exception as _cmd_exc:" in source
    assert "Command error" in source
    # The umbrella catches Exception (not BaseException), so /exit's SystemExit
    # and Ctrl+C's KeyboardInterrupt still propagate — assert it stays that way.
    idx = source.index("except Exception as _cmd_exc:")
    tail = source[idx:idx + 700]
    assert "continue" in tail


# ---------------------------------------------------------------------------
# #4 — /checkpoint tells the truth when `git stash apply` fails
# ---------------------------------------------------------------------------

class _Proc:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _checkpoint_runner(apply_returncode, apply_stderr=""):
    """Fake subprocess.run: stash push succeeds; apply returns as scripted."""
    def _run(cmd, *a, **k):
        if cmd[:3] == ["git", "stash", "push"]:
            return _Proc(0, stdout="Saved working directory and index state")
        if cmd[:3] == ["git", "stash", "apply"]:
            return _Proc(apply_returncode, stderr=apply_stderr)
        return _Proc(0)
    return _run


def test_checkpoint_apply_failure_does_not_claim_preserved(monkeypatch):
    monkeypatch.setattr(cli_mod.subprocess, "run",
                        _checkpoint_runner(1, apply_stderr="merge conflict"))
    console, buf = _console()
    result = _call("/checkpoint", console=console)
    assert result is None
    out = buf.getvalue()
    assert "Working state preserved" not in out  # the false claim is gone
    assert "stash" in out.lower()                 # recovery guidance is shown
    assert "merge conflict" in out


def test_checkpoint_apply_success_reports_preserved(monkeypatch):
    monkeypatch.setattr(cli_mod.subprocess, "run", _checkpoint_runner(0))
    console, buf = _console()
    result = _call("/checkpoint", console=console)
    assert result is None
    assert "Working state preserved" in buf.getvalue()


# ---------------------------------------------------------------------------
# #11 — /loop Esc handler cancels the in-flight round
# ---------------------------------------------------------------------------

def test_loop_escape_action_stops_and_cancels():
    engine = MagicMock()
    agent = MagicMock()
    _loop_escape_action(engine, agent)
    engine.request_stop.assert_called_once_with()
    agent.cancel.assert_called_once_with()
