"""Host-editor detection and file handoff (Phase 3 Task 4).

Spark runs INSIDE (or next to) a host editor — Cursor's or VS Code's
integrated terminal, or a plain Terminal.app tab next to Xcode. This module
lets Spark hand a file (and optionally a line) off to whichever editor is
actually there, so `/open` and clickable OSC 8 file links do something
useful instead of nothing.

Hard rule: editor launches are fire-and-forget and NEVER block or crash the
CLI. ``open_in_editor`` swallows every failure and returns ``False`` — the
caller turns that into a dim console note, never an exception.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote

from rich.console import Console

# The three editors Spark knows how to hand a file to. Kept in sync with the
# `ui.editor` config values ("auto" | "cursor" | "code" | "xed" | "none") —
# "auto" and "none" are resolved by the CALLER (config override wins before
# detect_editor is ever consulted), not by this module.
_KNOWN_EDITORS = ("cursor", "code", "xed")


def detect_editor(
    env: dict | None = None,
    path_checker=shutil.which,
) -> str | None:
    """Detect the host editor from the environment — pure and injectable.

    ``env`` defaults to the real process environment only when ``None`` is
    passed (never read implicitly beyond that one fallback), and
    ``path_checker`` defaults to ``shutil.which`` but tests pass a fake that
    never touches the real ``PATH``. No real filesystem/PATH lookups happen
    unless the caller lets the defaults through.

    Priority (config override is NOT handled here — the caller only calls
    this when ``ui.editor`` is "auto"):
      1. ``TERM_PROGRAM == "vscode"`` (both Cursor and VS Code's integrated
         terminal set this — Cursor is a VS Code fork) → "cursor" if the
         `cursor` CLI shim is on PATH, else "code". PATH is the only signal
         that tells the two apart when both are installed.
      2. macOS with `xed` (Xcode's CLI file-opener) on PATH → "xed".
      3. Otherwise ``None`` — nothing detected.
    """
    active_env = os.environ if env is None else env

    if active_env.get("TERM_PROGRAM") == "vscode":
        return "cursor" if path_checker("cursor") else "code"

    if sys.platform == "darwin" and path_checker("xed"):
        return "xed"

    return None


def open_in_editor(editor: str, path: str, line: int | None = None) -> bool:
    """Launch ``editor`` on ``path`` (optionally at ``line``), detached.

    Fire-and-forget: ``subprocess.Popen`` is started and never waited on, so
    a slow-to-launch editor can't block the CLI. Returns ``True`` once the
    process has been launched (not once it's done anything useful) and
    ``False`` — never raising — for an unrecognized editor name or any
    launch failure (binary missing, permission error, etc.).
    """
    if editor in ("cursor", "code"):
        target = f"{path}:{line}" if line is not None else path
        argv = [editor, "--goto", target]
    elif editor == "xed":
        argv = [editor]
        if line is not None:
            argv += ["--line", str(line)]
        argv.append(path)
    else:
        return False

    try:
        subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False


def file_link(path: str) -> str:
    """Build an absolute, percent-encoded ``file://`` URL for OSC 8 links.

    ``path`` is expanded (``~``) and made absolute against cwd if it wasn't
    already. Percent-encoding (``urllib.parse.quote``, keeping ``/``
    unescaped) matters because unescaped spaces or other reserved
    characters in a hyperlink target break terminal OSC 8 parsing.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    return "file://" + quote(abs_path)


# ---------------------------------------------------------------------------
# Diff-in-editor (Phase 3 Task 5)
# ---------------------------------------------------------------------------
#
# Opt-in (``ui.diff_in_editor``, default False). When enabled AND the
# resolved editor is "cursor" or "code", the edit-permission preview
# ADDITIONALLY writes the proposed before/after content to two temp files
# and launches `cursor|code --diff before after` — the existing inline
# terminal diff (spark_code/ui/diff.py) still renders regardless; this is
# purely an extra view, never a replacement. xed has no `--diff` flag, so
# it gets a one-time dim note instead of a launch attempt.

# Every mkdtemp() directory created by ``open_diff_in_editor`` this process,
# so it can be swept at session exit. Cleanup only ever removes paths FROM
# this list (never an arbitrary caller-supplied path), so a bug elsewhere
# can't turn "clean up my own temp files" into deleting something a user
# cares about.
_diff_temp_dirs: list[str] = []


def open_diff_in_editor(editor: str, file_path: str, old_string: str,
                        new_string: str) -> bool:
    """Write ``old_string``/``new_string`` to temp files and launch a
    side-by-side diff in ``editor``, detached — fire-and-forget like
    ``open_in_editor``.

    Only "cursor" and "code" support `--diff before after`; anything else
    (xed, or an unrecognized name) returns False WITHOUT creating any
    files or launching anything — callers decide how (or whether) to note
    that (see ``maybe_preview_diff_in_editor`` for the xed one-time note).

    The two files live under a single fresh ``tempfile.mkdtemp()``
    directory, named ``<basename>.before`` / ``<basename>.after`` where
    ``basename`` is derived from ``file_path`` but stripped down to
    alnum/dot/dash/underscore (falling back to "file" if that leaves
    nothing) — deliberately boring and safe, so a hostile or unusual
    ``file_path`` (path separators, escape bytes, ``..`` segments) can
    never smuggle anything past the temp directory it's confined to. The
    directory is registered in ``_diff_temp_dirs`` for cleanup regardless
    of whether the editor launch itself succeeds — the files exist either
    way and still need to be swept.
    """
    if editor not in ("cursor", "code"):
        return False

    raw_basename = os.path.basename(file_path) or "file"
    safe_basename = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in raw_basename
    ) or "file"

    try:
        tmp_dir = tempfile.mkdtemp(prefix="spark-diff-")
        before_path = os.path.join(tmp_dir, f"{safe_basename}.before")
        after_path = os.path.join(tmp_dir, f"{safe_basename}.after")
        with open(before_path, "w", encoding="utf-8") as f:
            f.write(old_string)
        with open(after_path, "w", encoding="utf-8") as f:
            f.write(new_string)
        _diff_temp_dirs.append(tmp_dir)
    except Exception:
        return False

    try:
        subprocess.Popen(
            [editor, "--diff", before_path, after_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except Exception:
        return False


def cleanup_diff_temp_dirs() -> None:
    """Remove every temp dir ``open_diff_in_editor`` created this session.

    Safe to call multiple times (registered with ``atexit`` AND called
    explicitly from the CLI's session-teardown ``finally`` block, so
    whichever runs first empties the list) — each dir is popped before
    being removed, and ``ignore_errors=True`` means an already-gone or
    unreadable dir is skipped rather than raising during shutdown.
    """
    while _diff_temp_dirs:
        tmp_dir = _diff_temp_dirs.pop()
        shutil.rmtree(tmp_dir, ignore_errors=True)


atexit.register(cleanup_diff_temp_dirs)


def maybe_preview_diff_in_editor(console: Console, editor: str | None,
                                 diff_in_editor: bool, file_path: str,
                                 old_string: str, new_string: str,
                                 xed_noted: bool) -> bool:
    """Gate + dispatch for the edit-permission preview's editor-side diff.

    Called unconditionally alongside (never instead of) the inline
    terminal diff. Returns the ``xed_noted`` flag the caller should store
    back on itself for the rest of the session:

      * ``diff_in_editor`` False → no-op, flag unchanged.
      * ``editor`` "cursor"/"code" → launches the diff (``open_diff_in_editor``
        already swallows every failure); flag unchanged.
      * ``editor`` "xed" → xed has no ``--diff`` flag, so this never
        launches anything; prints a ONE-TIME dim note the first time
        (``xed_noted`` False → True), stays silent on every call after.
      * ``editor`` None (or anything else unrecognized) → silent no-op —
        no note is invented for an editor this feature doesn't know how
        to diff in.
    """
    if not diff_in_editor:
        return xed_noted

    if editor in ("cursor", "code"):
        open_diff_in_editor(editor, file_path, old_string, new_string)
    elif editor == "xed" and not xed_noted:
        xed_noted = True
        console.print(
            "[dim]xed has no side-by-side diff view — showing inline diff only[/dim]"
        )

    return xed_noted
