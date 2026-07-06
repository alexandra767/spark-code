"""Esc-to-interrupt: raw stdin watcher active only while the model generates."""

from __future__ import annotations

import contextlib
import os
import select
import threading

_TICK = 0.05  # seconds; also the grace window for escape-sequence continuation

# Module-level registry of active watchers so an interactive prompt anywhere in
# the process can suspend Esc-watching for its duration without threading a
# reference through the call stack. There is at most one active watcher in
# practice (a single generation turn), but the registry is a set so concurrent
# scopes never clobber each other.
_active_watchers: set = set()
_registry_lock = threading.Lock()


def _register(watcher) -> None:
    with _registry_lock:
        _active_watchers.add(watcher)


def _unregister(watcher) -> None:
    with _registry_lock:
        _active_watchers.discard(watcher)


@contextlib.contextmanager
def pause_all():
    """Suspend every active Esc watcher for the duration of the block.

    While paused, watchers stop reading stdin AND restore the terminal to its
    saved (cooked, echo-on) attributes, so an interactive ``Prompt.ask`` can
    read and echo the user's keystrokes normally instead of the watcher thread
    swallowing them from the shared fd. ``pause()`` blocks until each watcher
    has acknowledged that it stopped reading and restored termios, so the prompt
    never races the watcher for the same byte. On exit, watchers re-enter cbreak
    and resume watching. No-op when no watcher is active.
    """
    with _registry_lock:
        watchers = list(_active_watchers)
    for w in watchers:
        w.pause()
    try:
        yield
    finally:
        for w in watchers:
            w.resume()


def classify_escape(buf: bytes) -> bool:
    """True iff ``buf`` is a lone Esc keypress (not an escape sequence)."""
    return buf == b"\x1b"


class EscWatcher:
    """Context manager: fires ``on_escape`` once if the user presses Esc.

    ``raw_mode=False`` (tests) skips termios manipulation and tty checks,
    reading the given fd directly. In real use (fd=stdin, raw_mode=True) the
    watcher no-ops unless stdin is a tty, puts it in cbreak for the duration
    (ISIG stays on, so Ctrl+C keeps working), and restores attrs on exit.

    An active watcher can be suspended via the module-level ``pause_all()``
    (used around interactive prompts): while paused it stops reading the fd and
    restores the saved cooked/echo termios, then re-enters cbreak on resume.
    """

    def __init__(self, on_escape, fd: int = 0, raw_mode: bool = True):
        self._on_escape = on_escape
        self._fd = fd
        self._raw_mode = raw_mode
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._saved_attrs = None
        # Pause coordination (see pause_all). _pause_requested is set by the
        # main thread; _paused_ack is set by the watch thread once it has left
        # its read path and restored termios so pause() can return safely.
        self._pause_requested = threading.Event()
        self._paused_ack = threading.Event()
        self._wake = threading.Event()
        # True while stdin is currently in cbreak (raw) mode. The watch thread
        # flips this as it pauses (restore→False) and resumes (cbreak→True);
        # __exit__ reads it (after join) to avoid a redundant second restore.
        self._in_cbreak = False

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
            self._in_cbreak = True
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        _register(self)
        return self

    def __exit__(self, *exc):
        _unregister(self)
        self._stop.set()
        self._wake.set()  # wake the thread if it's idling in a paused wait
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Restore only if stdin is still in cbreak. If we were paused at exit,
        # the watch thread already restored the saved attrs (and left cbreak),
        # so restoring again would be a redundant second tcsetattr.
        if self._saved_attrs is not None and self._in_cbreak:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
            self._in_cbreak = False
        return False

    def _restore_termios(self):
        """Put stdin back to its saved cooked/echo attributes (immediate)."""
        if self._saved_attrs is None:
            return
        import termios
        # TCSANOW: apply now without draining/discarding, so a waiting prompt can
        # read and echo the user's input immediately.
        termios.tcsetattr(self._fd, termios.TCSANOW, self._saved_attrs)
        self._in_cbreak = False

    def _enter_cbreak(self):
        """Re-enter cbreak (echo off), matching __enter__'s TCSANOW semantics."""
        if self._saved_attrs is None:
            return
        import termios
        import tty
        tty.setcbreak(self._fd, when=termios.TCSANOW)
        self._in_cbreak = True

    def pause(self):
        """Stop reading stdin and restore cooked termios; block until done.

        Called on the main thread (via pause_all) right before an interactive
        prompt reads stdin. No-op if the watcher isn't running (non-tty real
        mode, or never entered). Returns only after the watch thread confirms it
        has stopped reading and restored termios, so the prompt can't lose the
        first byte to the watcher.
        """
        if self._thread is None or not self._thread.is_alive():
            return
        self._paused_ack.clear()
        self._pause_requested.set()
        self._wake.set()
        # The thread may be mid-select (≤ _TICK) when the request lands; the
        # generous timeout is only a deadlock guard, not the expected wait.
        self._paused_ack.wait(timeout=2.0)

    def resume(self):
        """Re-arm the watcher after a pause: re-enter cbreak and read again."""
        if self._thread is None or not self._thread.is_alive():
            return
        self._pause_requested.clear()
        self._wake.set()

    def _watch(self):
        fired = False
        while not self._stop.is_set():
            if self._pause_requested.is_set():
                # Suspend: restore cooked termios, acknowledge, then idle until
                # resumed or stopped. pause() blocks on _paused_ack, so the main
                # thread won't start reading stdin until we're parked here.
                self._restore_termios()
                self._paused_ack.set()
                while self._pause_requested.is_set() and not self._stop.is_set():
                    self._wake.wait(timeout=_TICK)
                    self._wake.clear()
                if self._stop.is_set():
                    break
                self._enter_cbreak()
                continue

            try:
                ready, _, _ = select.select([self._fd], [], [], _TICK)
                if self._pause_requested.is_set():
                    # A pause landed during the select wait — handle it at the
                    # top before consuming any byte the prompt may want.
                    continue
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
