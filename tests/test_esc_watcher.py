import os
import threading
import time

from spark_code.ui.esc_watcher import EscWatcher, classify_escape


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
