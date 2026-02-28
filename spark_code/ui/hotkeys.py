"""Hotkeys and team status monitor for agent execution.

Provides:
- TeamStatusMonitor: auto-prints worker status every N seconds
- Ctrl+T signal handler: on macOS, Ctrl+T sends SIGINFO which we intercept
"""

import asyncio
import os
import platform
import signal
import sys
from typing import Callable


class TeamStatusMonitor:
    """Periodically prints team status while workers are running.

    Also intercepts Ctrl+T (SIGINFO on macOS) for on-demand status display.
    """

    def __init__(self, team_status_fn: Callable, console,
                 display_fn: Callable | None = None, interval: float = 5.0):
        self._team_status_fn = team_status_fn  # returns list of worker dicts
        self._display_fn = display_fn  # detailed display (Ctrl+T handler)
        self._console = console
        self._interval = interval
        self._task: asyncio.Task | None = None
        self._running = False
        self._last_snapshot: dict[str, str] = {}  # worker_id -> status
        self._prev_siginfo_handler = None

    def start(self):
        """Start monitoring (call before agent.run)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor())
        self._install_signal_handler()

    def stop(self):
        """Stop monitoring (call after agent.run)."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            self._task = None
        self._restore_signal_handler()

    def _install_signal_handler(self):
        """On macOS, intercept SIGINFO (Ctrl+T) to show team status."""
        if platform.system() != "Darwin":
            return
        try:
            siginfo = signal.SIGINFO  # type: ignore[attr-defined]
            self._prev_siginfo_handler = signal.getsignal(siginfo)
            signal.signal(siginfo, self._handle_siginfo)
            # Suppress the default "load:" output by disabling the
            # terminal's status character via termios directly
            self._disable_tty_status()
        except (AttributeError, OSError):
            pass

    def _restore_signal_handler(self):
        """Restore the previous SIGINFO handler."""
        if platform.system() != "Darwin":
            return
        try:
            siginfo = signal.SIGINFO  # type: ignore[attr-defined]
            if self._prev_siginfo_handler is not None:
                signal.signal(siginfo, self._prev_siginfo_handler)
                self._prev_siginfo_handler = None
            self._restore_tty_status()
        except (AttributeError, OSError):
            pass

    def _disable_tty_status(self):
        """Disable the terminal's VSTATUS character so Ctrl+T doesn't print 'load:'."""
        try:
            import termios
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            # VSTATUS index is 16 on macOS (cc array)
            VSTATUS = 16
            self._saved_vstatus = attrs[6][VSTATUS]
            attrs[6][VSTATUS] = 0  # disable
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception:
            self._saved_vstatus = None

    def _restore_tty_status(self):
        """Restore the terminal's VSTATUS character."""
        if not hasattr(self, '_saved_vstatus') or self._saved_vstatus is None:
            return
        try:
            import termios
            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            VSTATUS = 16
            attrs[6][VSTATUS] = self._saved_vstatus
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            self._saved_vstatus = None
        except Exception:
            pass

    def _handle_siginfo(self, signum, frame):
        """Called when user presses Ctrl+T during execution."""
        if self._display_fn:
            self._display_fn()
        else:
            self._print_compact_status()

    def _print_compact_status(self):
        """Print a one-line status summary."""
        workers = self._team_status_fn()
        if not workers:
            self._console.print("\n  [#666666]No workers active.[/#666666]")
            return

        parts = []
        for w in workers:
            if w["status"] == "running":
                parts.append(f"[#ebcb8b]⟳ {w['name']}[/#ebcb8b]")
            elif w["status"] == "completed":
                parts.append(f"[#a3be8c]✓ {w['name']}[/#a3be8c]")
            else:
                parts.append(f"[#bf616a]✗ {w['name']}[/#bf616a]")

        status_line = "  ".join(parts)
        self._console.print(f"\n  [#88c0d0]▸ Team:[/#88c0d0] {status_line}")

    async def _monitor(self):
        """Background loop — prints status when workers change state."""
        try:
            while self._running:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                self._check_and_print()
        except asyncio.CancelledError:
            pass

    def _check_and_print(self):
        """Print status if any workers changed state since last check."""
        workers = self._team_status_fn()
        if not workers:
            return

        # Check for state changes
        current = {w["id"]: w["status"] for w in workers}
        changed = current != self._last_snapshot
        has_running = any(w["status"] == "running" for w in workers)

        if not changed and not has_running:
            return

        self._last_snapshot = current

        # Print compact status line
        parts = []
        for w in workers:
            if w["status"] == "running":
                parts.append(f"[#ebcb8b]⟳ {w['name']}[/#ebcb8b]")
            elif w["status"] == "completed":
                parts.append(f"[#a3be8c]✓ {w['name']}[/#a3be8c]")
            else:
                parts.append(f"[#bf616a]✗ {w['name']}[/#bf616a]")

        status_line = "  ".join(parts)
        self._console.print(f"  [#88c0d0]▸ Team:[/#88c0d0] {status_line}")
