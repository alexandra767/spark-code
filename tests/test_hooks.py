"""Regression tests for hooks.py security fix.

Substituted context values ({path}, {command}, ...) must never be re-parsed
as shell syntax, so a malicious filename can't execute arbitrary shell in
the hook command.
"""

from spark_code.hooks import Hook, HookManager


async def test_hostile_path_does_not_execute_injected_command(tmp_path):
    """The canonical injection proof (Task 6): a filename that embeds a
    shell command separator must never cause that command to run. Sentinel
    file must NOT be created.
    """
    sentinel = tmp_path / "pwned_HOOKTEST"
    hostile_path = f"foo.py; touch {sentinel}"
    hook = Hook(command="echo {path}")

    success, output = await hook.run({"path": hostile_path})

    assert not sentinel.exists(), (
        "hook substitution executed injected shell command from a hostile "
        "filename"
    )


async def test_hook_path_substitution_is_quoted():
    # If {path} were substituted raw, the command substitution `$(...)` would run.
    hook = Hook(command="true {path}")
    success, output = await hook.run({"path": "x; echo INJECTED_HK"})
    assert "INJECTED_HK" not in output


async def test_hook_command_substitution_not_executed():
    hook = Hook(command="echo {path}")
    # The dangerous filename must be echoed literally, never evaluated: `echo`
    # emits the verbatim `$(echo EVALUATED)` string rather than `EVALUATED`.
    success, output = await hook.run({"path": "$(echo EVALUATED)"})
    assert output.strip() == "$(echo EVALUATED)"


async def test_hook_normal_path_still_works():
    hook = Hook(command="echo checking {path}")
    success, output = await hook.run({"path": "main.py"})
    assert success
    assert "main.py" in output


# ---------------------------------------------------------------------------
# Hardening (Task 6): a failing or hanging hook must be reported as a
# (False, message) result, never raised — that's what lets the call site
# swallow it without special-casing exceptions.
# ---------------------------------------------------------------------------


async def test_failing_hook_does_not_raise():
    hook = Hook(command="false")
    success, output = await hook.run({})
    assert success is False


async def test_hook_timeout_is_caught_not_raised():
    hook = Hook(command="sleep 5", timeout=0.2)
    success, output = await hook.run({})
    assert success is False
    assert "timed out" in output


async def test_hook_manager_run_hooks_with_failing_hook_returns_result_list():
    hm = HookManager({
        "hooks": {"after_write_file": [{"command": "false", "pattern": "*.py"}]},
    })
    results = await hm.run_hooks("after_write_file", {"path": "x.py"})
    assert results == [(False, "")]


async def test_hook_manager_matches_bash_command_by_pattern():
    """A before_bash hook with a pattern scopes itself to matching commands
    (e.g. a commit gate) instead of firing on every bash call — the
    docs/hooks.md commit-gate recipe depends on this."""
    hm = HookManager({
        "hooks": {"before_bash": [{"command": "echo gate", "pattern": "*commit*"}]},
    })

    matching = await hm.run_hooks(
        "before_bash", {"tool": "bash", "command": "git commit -m x"})
    assert len(matching) == 1

    non_matching = await hm.run_hooks(
        "before_bash", {"tool": "bash", "command": "git status"})
    assert len(non_matching) == 0


async def test_hook_manager_run_hooks_with_malformed_template_does_not_raise():
    """A hook command that isn't valid shlex syntax (unbalanced quotes) must
    be reported as a failure, not raise out of run_hooks."""
    hm = HookManager({
        "hooks": {"after_write_file": [{"command": "echo 'unterminated"}]},
    })
    results = await hm.run_hooks("after_write_file", {"path": "x.py"})
    assert len(results) == 1
    assert results[0][0] is False


# ---------------------------------------------------------------------------
# /hooks command (Task 6) — read-only listing.
# ---------------------------------------------------------------------------


def test_hooks_command_lists_configured_hooks():
    import io

    from rich.console import Console

    from spark_code.cli import handle_slash_command
    from spark_code.context import Context
    from spark_code.model import ModelClient
    from spark_code.permissions import PermissionManager
    from spark_code.skills.base import SkillRegistry

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    context = Context()
    skills = SkillRegistry()
    from unittest.mock import MagicMock
    model = MagicMock(spec=ModelClient)
    model.total_input_tokens = 0
    model.total_output_tokens = 0
    permissions = PermissionManager(mode="auto")

    config = {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "test-model",
            "temperature": 0.7,
            "max_tokens": 4096,
            "context_window": 32768,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
        "hooks": {
            "after_write_file": [
                {"command": "ruff check --fix {path}", "pattern": "*.py"},
            ],
            "before_git_commit": [
                {"command": "pytest -q", "timeout": 120},
            ],
        },
    }

    result = handle_slash_command(
        "/hooks", context, console, config, skills, model,
        permissions=permissions,
    )

    assert result is None  # read-only, no agent turn
    output = buf.getvalue()
    assert "after_write_file" in output
    assert "before_git_commit" in output
    assert "*.py" in output
    assert "ruff check" in output
    assert "pytest" in output


def test_hooks_command_with_no_hooks_configured():
    import io

    from rich.console import Console

    from spark_code.cli import handle_slash_command
    from spark_code.context import Context
    from spark_code.model import ModelClient
    from spark_code.permissions import PermissionManager
    from spark_code.skills.base import SkillRegistry

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    context = Context()
    skills = SkillRegistry()
    from unittest.mock import MagicMock
    model = MagicMock(spec=ModelClient)
    model.total_input_tokens = 0
    model.total_output_tokens = 0
    permissions = PermissionManager(mode="auto")

    config = {
        "model": {
            "endpoint": "http://localhost:11434",
            "name": "test-model",
            "temperature": 0.7,
            "max_tokens": 4096,
            "context_window": 32768,
            "api_key": "",
            "provider": "ollama",
        },
        "permissions": {"mode": "auto", "always_allow": []},
    }

    result = handle_slash_command(
        "/hooks", context, console, config, skills, model,
        permissions=permissions,
    )

    assert result is None
    assert "No hooks configured" in buf.getvalue()
