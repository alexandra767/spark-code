"""Hooks system — run commands before/after tool calls.

Configuration in ~/.spark/config.yaml or .spark/config.yaml:

hooks:
  after_write_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
    - pattern: "*.js"
      command: "eslint --fix {path}"
  after_edit_file:
    - pattern: "*.py"
      command: "ruff check --fix {path}"
  before_bash:
    - command: "echo 'Running: {command}'"
    - pattern: "*commit*"          # matches the bash COMMAND text, not a
      command: "pytest -q"         # path, when there's no path in context

See docs/hooks.md for the full config shape, event list, and worked
recipes. Full read-only listing: the `/hooks` CLI command.

A hook never blocks or crashes the tool call it's attached to — a nonzero
exit, a timeout, or an outright bug in the hook plumbing all surface as a
result/console note, never an exception that stops the agent loop (see
Agent._run_hooks_safe). Command-template substitution ({path}, {command},
...) is argv-safe, not shell-interpolated — see Hook.run.
"""

import asyncio
import fnmatch
import logging
import os
import shlex

logger = logging.getLogger(__name__)

# Shell operators that DON'T work in a hook command: hooks run argv-exec
# (no shell), so a bare `&&`/`||`/`;`/`|` token becomes a literal argument
# to the first program and the "second half" silently never runs. We warn
# on these so that silent degradation is visible (wrap in `sh -c '...'`).
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "|"})


class Hook:
    """A single hook definition."""

    def __init__(self, command: str, pattern: str = "*", timeout: int = 30):
        self.command = command
        self.pattern = pattern
        self.timeout = timeout
        # One-shot latch so the shell-operator degradation warning fires at
        # most once per hook (per session), not on every matching tool call.
        self._warned_shell_op = False

    def has_shell_operator(self) -> bool:
        """True if the command template, once tokenized, contains a bare
        shell operator (&&, ||, ;, |) as its own token. Such a command
        silently drops everything after the operator (no shell runs it) —
        callers use this to surface the degradation. Never raises: a
        malformed template (unbalanced quotes) just returns False."""
        try:
            tokens = shlex.split(self.command)
        except ValueError:
            return False
        return any(t in _SHELL_OPERATORS for t in tokens)

    def matches(self, path: str) -> bool:
        """Check if hook pattern matches the given path."""
        if self.pattern == "*":
            return True
        basename = os.path.basename(path)
        return fnmatch.fnmatch(basename, self.pattern)

    def matches_text(self, text: str) -> bool:
        """Like matches(), but against raw text (e.g. a bash command string)
        rather than a file path — no basename stripping, since a shell
        command has no directory structure to strip and a `/` inside it
        (a quoted path argument, a URL, ...) isn't a path separator for
        matching purposes. Lets a `before_bash`/`after_bash` hook scope
        itself with e.g. `pattern: "*commit*"` (a commit gate) instead of
        firing on every bash call."""
        if self.pattern == "*":
            return True
        return fnmatch.fnmatch(text, self.pattern)

    async def run(self, context: dict[str, str]) -> tuple[bool, str]:
        """Execute the hook command with context substitution.

        context keys: path, command, old_string, new_string, pattern, etc.

        SECURITY: substitution is argv-safe, not shell-interpolated. The
        command TEMPLATE (author-controlled, from config) is tokenized with
        shlex.split() — this respects the author's own quoting, e.g.
        `"echo 'Running: {command}'"` still lands as one argv element — and
        THEN each `{key}` placeholder is substituted with the raw context
        value inside its token. The resulting argv list is executed
        directly via create_subprocess_exec, so no shell is ever invoked:
        a value that embeds shell metacharacters (a filename like
        `; rm -rf ~` or `$(curl evil)`) becomes one literal argv string —
        it is never re-parsed as shell syntax, so it can't inject a second
        command. (Previously this shell-quoted each value with shlex.quote
        before handing the whole string to a shell; that was already safe
        against injection, but avoiding the shell entirely removes the
        class of bug rather than relying on correct quoting.)

        Returns (success, output).
        """
        try:
            argv = shlex.split(self.command)
        except ValueError as e:
            return False, f"Hook command template error: {e}"
        if not argv:
            return False, "Hook command is empty"

        substituted = []
        for token in argv:
            for key, value in context.items():
                token = token.replace(f"{{{key}}}", str(value))
            substituted.append(token)

        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *substituted,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=os.getcwd(),
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=self.timeout
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            return process.returncode == 0, output
        except asyncio.TimeoutError:
            if process is not None:
                try:
                    process.kill()
                except Exception:
                    pass
            return False, f"Hook timed out after {self.timeout}s"
        except FileNotFoundError:
            return False, f"Hook error: command not found: {substituted[0]!r}"
        except Exception as e:
            return False, f"Hook error: {e}"


class HookManager:
    """Manages pre/post hooks for tool calls."""

    def __init__(self, config: dict | None = None):
        self._hooks: dict[str, list[Hook]] = {}
        if config:
            self.load(config)

    def load(self, config: dict):
        """Load hooks from config dict."""
        hooks_conf = config.get("hooks", {})
        if not hooks_conf:
            return

        for event_name, hook_list in hooks_conf.items():
            if not isinstance(hook_list, list):
                continue
            self._hooks[event_name] = []
            for h in hook_list:
                if isinstance(h, dict) and "command" in h:
                    self._hooks[event_name].append(Hook(
                        command=h["command"],
                        pattern=h.get("pattern", "*"),
                        timeout=h.get("timeout", 30),
                    ))

    def has_hooks(self, event: str) -> bool:
        return bool(self._hooks.get(event))

    async def run_hooks(self, event: str, context: dict[str, str],
                        console=None) -> list[tuple[bool, str]]:
        """Run all hooks for an event. Returns list of (success, output).

        Pattern matching: events carrying a file path (write_file, edit_file,
        ...) match `pattern` against the path's basename. Events with no
        path but a `command` (bash) fall back to matching `pattern` against
        the raw command text — this is what lets a `before_bash` hook scope
        itself to e.g. `pattern: "*commit*"` as a commit gate instead of
        firing on every bash call. An event with neither always matches
        (pattern is effectively ignored), same as before this fallback.
        """
        hooks = self._hooks.get(event, [])
        if not hooks:
            return []

        results = []
        path = context.get("path", context.get("file_path", ""))
        command_text = context.get("command", "")

        for hook in hooks:
            if path:
                if not hook.matches(path):
                    continue
            elif command_text:
                if not hook.matches_text(command_text):
                    continue

            # Surface silent degradation: a hook whose command uses a bare
            # shell operator (&&, ||, ;, |) drops everything after it since
            # no shell runs it — warn once per hook so a green "hook: ..."
            # line isn't mistaken for the whole command having succeeded.
            if (console and not hook._warned_shell_op
                    and hook.has_shell_operator()):
                hook._warned_shell_op = True
                from rich.text import Text
                console.print(Text(
                    "  hook: shell operators (&&, ||, ;, |) don't work — "
                    "hooks run without a shell; wrap in sh -c '...' "
                    "(see docs/hooks.md)",
                    style="dim"))

            success, output = await hook.run(context)
            results.append((success, output))

            if console and output:
                from rich.text import Text
                style = "#a3be8c" if success else "#ebcb8b"
                console.print(Text(f"  hook: {output[:120]}", style=style))

        return results

    def get_events(self) -> list[str]:
        """List all configured hook events."""
        return list(self._hooks.keys())

    def get_hooks(self, event: str) -> list["Hook"]:
        """Return the Hook objects configured for one event (read-only
        listing, e.g. for `/hooks`). Empty list if none configured."""
        return list(self._hooks.get(event, []))

    @property
    def count(self) -> int:
        return sum(len(hooks) for hooks in self._hooks.values())
