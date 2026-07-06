"""Regression tests for hooks.py security fix.

Substituted context values ({path}, {command}, ...) must be shell-quoted so a
malicious filename can't execute arbitrary shell in the hook command.
"""


from spark_code.hooks import Hook


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
