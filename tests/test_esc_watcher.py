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
