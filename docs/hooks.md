# Hooks

Hooks run a shell command before or after a tool call — auto-lint a file you
just wrote, gate a risky bash command, log what the agent is doing. They're
observational by default: a hook's exit code and output never block or
change what the agent does next (see [Guarantees](#guarantees) below).

## Config shape

Hooks live under a top-level `hooks:` key in `~/.spark/config.yaml`
(global) or `.spark/config.yaml` (project, merges over global):

```yaml
hooks:
  <event>:
    - pattern: "<glob, optional — defaults to \"*\">"
      command: "<shell command, with {placeholders}>"
      timeout: <seconds, optional — defaults to 30>
```

- `<event>` is `before_<tool>` or `after_<tool>`, where `<tool>` is any
  registered tool name — `write_file`, `edit_file`, `bash`, `read_file`,
  `grep`, `glob`, etc. Each event maps to a **list** of hooks; all of them
  run (in list order) when the event fires.
- `pattern` is a glob. For tool calls that carry a file path
  (`write_file`, `edit_file`, ...) it's matched against the path's
  **basename** (`*.py` matches `foo/bar.py`). For tool calls with no path
  but a `command` (`bash`) it's matched against the **raw command text**
  instead (`*commit*` matches `git commit -m "x"` anywhere in the string).
  Omit `pattern` (or use `"*"`) to match everything for that event.
- `command` is the shell command template. `{key}` placeholders are
  substituted from the tool call's arguments — `{path}` (or `{file_path}`),
  `{command}`, `{old_string}`, `{new_string}`, etc., whatever that tool call
  actually received. See [Substitution is argv-safe](#substitution-is-argv-safe)
  for what "substituted" means here — it is **not** string-pasted into a
  shell.
- `timeout` (seconds, default 30) bounds how long the hook is allowed to
  run before it's killed and reported as timed out.

List configured hooks any time with **`/hooks`** (read-only — prints a
table of event / pattern / command / timeout, doesn't run anything).

## Events

A hook fires around every tool call, keyed by the tool's name:

| Event | Fires |
|---|---|
| `before_<tool>` | Immediately before the tool executes (after the permission check). |
| `after_<tool>` | Immediately after the tool executes, with the result already known. |

`before_*` hooks are pure observers — even if a `before_write_file` hook
fails, the write still happens. There's no way to have a hook veto a tool
call today; hooks are for side effects (linting, logging, gating on
*informational* grounds you notice from the output), not for permissioning
(use `permissions:` / `/mode` for that).

## Recipe 1 — auto-lint Python files after every write

Runs `ruff check --fix` on every `.py` file the agent writes, so obvious
lint issues get cleaned up without you asking:

```yaml
hooks:
  after_write_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
  after_edit_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
```

`{path}` is substituted with the file the agent just wrote/edited. Because
substitution is argv-safe (see below), this is fine even for a file whose
name is adversarial or just weird — it's passed to `ruff` as a single
literal argument, never re-interpreted.

## Recipe 2 — a pytest gate before commits

Bash is the only tool git commands go through (there's no dedicated
`git_commit` tool), so a "before commit" hook is a `before_bash` hook
scoped with a `pattern` that matches the **command text** rather than a
path:

```yaml
hooks:
  before_bash:
    - pattern: "*commit*"
      command: "pytest -q"
      timeout: 120
```

This fires only when the bash command the agent is about to run contains
`commit` (e.g. `git commit -m "..."`) — not on every bash call. It's a
*gate* in the loose sense described in [Guarantees](#guarantees): the
test suite runs and its pass/fail is printed inline, but (today) a failing
run does not stop the commit — treat the printed result as a nudge to
`/undo`/fix before you actually push, the same way the agent's own
verification-habit nudge works. If you want a harder gate with extra
markers, note that **shell operators like `&&`/`||` do not work in a hook
command** (no shell runs it — see [Substitution is argv-safe](#substitution-is-argv-safe));
put the logic in a script instead: `command: "./scripts/commit-gate.sh"`.
Spark prints a one-time dim warning if a hook command contains a bare
`&&`/`||`/`;`/`|`, so this degradation never happens silently.

## Guarantees

These are enforced by the implementation (`spark_code/hooks.py`,
`Agent._run_hooks_safe` in `spark_code/agent.py`), not just documented
behavior:

- **A hook never blocks the tool call it's attached to.** Pre- and
  post-hook results are collected and printed but never checked against
  the tool's execution — a `before_write_file` hook that fails does not
  stop the write, and an `after_write_file` hook that fails does not undo
  it or mark the tool call as an error.
- **A hook never crashes the agent loop.** A nonzero exit code, a timeout,
  or an exception anywhere in the hook plumbing is caught and surfaced as
  a dim console note (`hook error: ...`) — the tool call and the rest of
  the turn continue normally.
- **A hook that hangs is killed at its `timeout`** (default 30s) and
  reported as `Hook timed out after Ns`, not left to hang the tool call.

### Substitution is argv-safe

`{path}`/`{command}`/etc. are **not** pasted into a shell string. The
command template is tokenized once with `shlex.split()` (so your own
quoting in the template, e.g. `"echo 'Running: {command}'"`, is respected),
then each `{key}` placeholder is substituted with the raw context value
*inside its token*, and the resulting argument list is executed directly
(`create_subprocess_exec`) — no shell is ever invoked. A value that
contains shell metacharacters (a filename like `; rm -rf ~` or
`$(curl evil)`) becomes a single literal argv string; it is never
re-parsed as shell syntax, so a hostile filename can't inject a second
command into the hook. This is covered by a regression test
(`tests/test_hooks.py::test_hostile_path_does_not_execute_injected_command`)
that proves a sentinel file is never created via an injected `; touch ...`
in a substituted path.

One consequence: **no shell is invoked for the hook author either.**
`command: "black {path} && ruff check {path}"` will NOT chain — the
template is tokenized with `shlex.split` into
`['black', '{path}', '&&', 'ruff', 'check', '{path}']`, and `&&` is passed
as a literal (nonsensical) argument to `black`, not interpreted as a shell
operator. Spark detects this and prints a **one-time dim warning** when a
hook command contains a bare `&&`/`||`/`;`/`|` token, so the dropped second
half never fails silently behind a green `hook: ...` line. If you need
multi-step logic:

- put it in a script and call the script — `command: "./scripts/lint.sh {path}"`
  keeps the chaining inside a program you control, or
- invoke a shell explicitly and pass the substituted value as a positional
  argument rather than splicing it into the script text —
  `command: "sh -c 'ruff check --fix \"$0\"' {path}"` hands the path to the
  inline script as `$0`, so it's still never re-parsed as shell syntax even
  though a shell is now running the fixed, author-written script body.
