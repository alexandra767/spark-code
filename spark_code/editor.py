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

import os
import shutil
import subprocess
import sys
from urllib.parse import quote

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
