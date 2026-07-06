"""Esc-to-interrupt: raw stdin watcher active only while the model generates."""

from __future__ import annotations

import os
import select
import threading

_TICK = 0.05  # seconds; also the grace window for escape-sequence continuation


def classify_escape(buf: bytes) -> bool:
    """True iff ``buf`` is a lone Esc keypress (not an escape sequence)."""
    return buf == b"\x1b"


class EscWatcher:
    """Context manager: fires ``on_escape`` once if the user presses Esc.

    ``raw_mode=False`` (tests) skips termios manipulation and tty checks,
    reading the given fd directly. In real use (fd=0, raw_mode=True) the
    watcher no-ops unless stdin is a tty, puts it in cbreak for the duration
    (ISIG stays on, so Ctrl+C keeps working), and restores attrs on exit.
    """

    def __init__(self, on_escape, fd: int = 0, raw_mode: bool = True):
        self._on_escape = on_escape
        self._fd = fd
        self._raw_mode = raw_mode
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_attrs = None

    def __enter__(self):
        if self._raw_mode:
            if not os.isatty(self._fd):
                return self
            import termios
            import tty
            self._saved_attrs = termios.tcgetattr(self._fd)
            # TCSANOW, not tty.setcbreak's default TCSAFLUSH: TCSAFLUSH blocks
            # until pending stdout has drained AND discards any not-yet-read
            # input before applying. If the terminal is momentarily not draining
            # output (a fast burst filling the pty buffer), that block delays the
            # watcher from starting and the flush eats an Esc the user already
            # pressed — so the interrupt is silently lost. TCSANOW applies
            # immediately without draining or discarding; already-typed input
            # stays queued for the watcher to read (and swallow if not an Esc).
            tty.setcbreak(self._fd, when=termios.TCSANOW)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved_attrs is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        return False

    def _watch(self):
        fired = False
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], _TICK)
                if not ready:
                    continue
                buf = os.read(self._fd, 32)
                if not buf:
                    break  # EOF: fd closed — stop cleanly (no busy-loop)
                if not buf.startswith(b"\x1b"):
                    continue  # swallow stray typing during generation
                if buf == b"\x1b":
                    ready, _, _ = select.select([self._fd], [], [], _TICK)
                    if ready:
                        buf += os.read(self._fd, 32)
            except (OSError, ValueError):
                break  # fd closed out from under the thread — exit quietly
            if classify_escape(buf) and not fired:
                fired = True
                self._on_escape()
