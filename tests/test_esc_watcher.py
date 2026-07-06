import os
import select
import threading
import time

from spark_code.ui.esc_watcher import EscWatcher, classify_escape, pause_all


def _pty_drainer(master, stop_drain):
    """Keep the pty master drained so the slave never blocks on echoed output
    (and a TCSADRAIN restore can complete). Mirrors a real terminal."""
    def _drain():
        while not stop_drain.is_set():
            r, _, _ = select.select([master], [], [], 0.05)
            if r:
                try:
                    os.read(master, 1024)
                except OSError:
                    break
    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    return t


def test_lone_escape_is_interrupt():
    assert classify_escape(b"\x1b") is True


def test_escape_sequence_is_not_interrupt():
    assert classify_escape(b"\x1b[A") is False   # up arrow
    assert classify_escape(b"\x1bOP") is False   # F1


def test_watcher_fires_on_lone_escape_from_pipe():
    r, w = os.pipe()
    fired = threading.Event()
    watcher = EscWatcher(fired.set, fd=r, raw_mode=False)
    with watcher:
        os.write(w, b"\x1b")
        assert fired.wait(timeout=2.0)
    os.close(r)
    os.close(w)


def test_watcher_ignores_arrow_key():
    r, w = os.pipe()
    fired = threading.Event()
    with EscWatcher(fired.set, fd=r, raw_mode=False):
        os.write(w, b"\x1b[A")
        time.sleep(0.3)
        assert not fired.is_set()
    os.close(r)
    os.close(w)


def test_noop_on_non_tty():
    # default fd=0 under pytest is not a tty; enter/exit must not raise
    with EscWatcher(lambda: None):
        pass


def test_entering_watcher_does_not_flush_already_typed_escape():
    """Root cause of the acceptance FAIL: an Esc already sitting in the tty
    input queue when the watcher starts must still fire.

    ``tty.setcbreak`` defaults to ``when=TCSAFLUSH``, which discards unread
    input (and blocks until stdout drains) before applying — so an Esc the
    user pressed while output was mid-burst got flushed away and the interrupt
    was silently lost. The watcher must use ``TCSANOW`` (no flush, no block).
    Verified over a real pty so ``os.isatty`` is true and the cbreak path runs.
    """
    import pty

    master, slave = pty.openpty()
    fired = threading.Event()
    stop_drain = threading.Event()

    # Keep the master drained so the slave never blocks on echoed output (and
    # __exit__'s TCSADRAIN can complete). Mirrors a real terminal, which always
    # reads its pty.
    def _drain():
        while not stop_drain.is_set():
            r, _, _ = __import__("select").select([master], [], [], 0.05)
            if r:
                try:
                    os.read(master, 1024)
                except OSError:
                    break

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    # Type-ahead an Esc into the slave's input queue BEFORE the watcher enters.
    os.write(master, b"\x1b")
    time.sleep(0.1)  # let the byte land in the tty input queue
    try:
        with EscWatcher(fired.set, fd=slave, raw_mode=True):
            # With TCSAFLUSH the queued Esc is discarded at cbreak time and this
            # never fires; with TCSANOW it survives and the watcher reads it.
            assert fired.wait(timeout=2.0)
    finally:
        stop_drain.set()
        drainer.join(timeout=1.0)
        os.close(master)
        os.close(slave)


# ---------------------------------------------------------------------------
# pause_all: suspend the watcher for the duration of an interactive prompt.
# The Critical final-review finding: while the watcher held cbreak (echo off)
# and drained every byte from stdin every 50 ms, a permission prompt's
# Prompt.ask read from the same fd — so the user's "y" was swallowed, nothing
# echoed, and the prompt hung. pause_all() must free stdin (stop reading AND
# restore cooked/echo termios) before the prompt reads, and re-arm after.
# ---------------------------------------------------------------------------

def test_pause_all_frees_stdin_for_prompt_and_restores_termios():
    """With an active watcher on a pty, pause_all() must (a) restore ECHO and
    ICANON so a prompt echoes, (b) stop consuming stdin so a typed "y" is
    readable by an ordinary read, and (c) re-arm on resume so Esc still fires.
    """
    import pty
    import termios

    master, slave = pty.openpty()
    fired = threading.Event()
    stop_drain = threading.Event()
    drainer = _pty_drainer(master, stop_drain)

    try:
        with EscWatcher(fired.set, fd=slave, raw_mode=True):
            # cbreak is active while watching: echo off, non-canonical.
            attrs = termios.tcgetattr(slave)
            assert not (attrs[3] & termios.ECHO)
            assert not (attrs[3] & termios.ICANON)

            with pause_all():
                # Paused: termios restored to cooked (echo + canonical back).
                attrs = termios.tcgetattr(slave)
                assert attrs[3] & termios.ECHO
                assert attrs[3] & termios.ICANON
                # The watcher is no longer reading; a byte typed now stays in
                # the tty queue for the prompt to read instead of being
                # swallowed by the watcher thread.
                os.write(master, b"y\n")
                r, _, _ = select.select([slave], [], [], 2.0)
                assert r, "input was not delivered to the freed stdin"
                assert os.read(slave, 16) == b"y\n"

            # Resumed: cbreak re-applied, Esc interrupt works again.
            os.write(master, b"\x1b")
            assert fired.wait(timeout=2.0)
    finally:
        stop_drain.set()
        drainer.join(timeout=1.0)
        os.close(master)
        os.close(slave)


def test_pause_all_is_noop_without_active_watcher():
    # No watcher registered → pause_all is a no-op and must not raise.
    with pause_all():
        pass


def test_exit_while_paused_restores_termios_exactly_once(monkeypatch):
    """If the watcher is torn down while paused, it has already restored the
    saved (cooked) attrs at pause time — __exit__ must NOT restore a second
    time. Count restores by object identity of the saved attrs; setcbreak uses
    tty's own tcsetattr reference and is not counted here."""
    import pty
    import termios

    master, slave = pty.openpty()
    stop_drain = threading.Event()
    drainer = _pty_drainer(master, stop_drain)

    real_tcsetattr = termios.tcsetattr
    restore_attrs = []

    def _counting_tcsetattr(fd, when, attrs):
        restore_attrs.append(attrs)
        return real_tcsetattr(fd, when, attrs)

    monkeypatch.setattr(termios, "tcsetattr", _counting_tcsetattr)

    try:
        watcher = EscWatcher(lambda: None, fd=slave, raw_mode=True)
        watcher.__enter__()
        saved = watcher._saved_attrs
        watcher.pause()  # restores the saved attrs once
        assert sum(1 for a in restore_attrs if a is saved) == 1
        watcher.__exit__()  # exit while paused: must NOT restore again
        assert sum(1 for a in restore_attrs if a is saved) == 1
    finally:
        stop_drain.set()
        drainer.join(timeout=1.0)
        os.close(master)
        os.close(slave)


# ---------------------------------------------------------------------------
# Hardening branches (final-review Finding 2): the watch loop must stop cleanly
# on EOF (write end closed → read returns b"") and on the fd being closed out
# from under the thread (select/read raise OSError/ValueError) — never busy-loop
# or crash.
# ---------------------------------------------------------------------------

def test_watcher_stops_on_eof_when_write_end_closed():
    r, w = os.pipe()
    fired = threading.Event()
    watcher = EscWatcher(fired.set, fd=r, raw_mode=False)
    with watcher:
        os.close(w)  # EOF on r: select reports readable, os.read returns b""
        time.sleep(0.2)
        assert watcher._thread is not None
        assert not watcher._thread.is_alive()  # broke out on EOF, no busy-loop
    assert not fired.is_set()
    os.close(r)


def test_watcher_stops_when_fd_closed_underneath():
    r, w = os.pipe()
    fired = threading.Event()
    watcher = EscWatcher(fired.set, fd=r, raw_mode=False)
    watcher.__enter__()
    os.close(r)  # fd closed out from under the thread → OSError/ValueError
    time.sleep(0.2)
    assert not watcher._thread.is_alive()
    watcher.__exit__()
    os.close(w)
    assert not fired.is_set()
