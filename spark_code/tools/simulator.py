"""iOS Simulator screenshot tool — the vision loop (Phase 4 Task 4).

Lets the model look at the booted iOS Simulator: screenshot it, feed the
image into context, iterate on SwiftUI work. Only registered on macOS with
Xcode's command line tools installed (see ``build_tools()`` in cli.py) —
absent everywhere else (e.g. the Linux/DGX box this runs on remotely), no
error.

Tool results are plain strings, subject to per-tool truncation
(``agent.MAX_TOOL_RESULT_CHARS`` / ``TOOL_RESULT_BUDGETS``) before they reach
the model — a screenshot's base64 payload would blow straight through that
budget and come back corrupted. So the actual image bytes never travel
through the tool-result string at all: on success (and a vision-capable
model), ``execute()`` returns a short sentinel — :data:`SCREENSHOT_SENTINEL_PREFIX`
plus the saved PNG's path — and ``Agent._store_tool_result`` (agent.py)
detects it, reads the file itself, and injects an
``add_user_with_image``-shaped multimodal turn (context.py:207) — the same
mechanism dragged-and-dropped images use (see cli.py's ``_is_image_drop``
handling and the ``/image`` command).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

from .base import Tool

# Prefix marking a tool result as "screenshot ready, saved at the path that
# follows" rather than plain text for the model to read directly. Detected
# by Agent._store_tool_result — see module docstring.
SCREENSHOT_SENTINEL_PREFIX = "__SPARK_SCREENSHOT_READY__:"

# xcrun/simctl are fast local calls against an already-running simulator —
# 15s comfortably covers a slow CI-ish box without hanging a turn.
DEFAULT_TIMEOUT = 15.0


class SimulatorScreenshotTool(Tool):
    """Screenshot the currently booted iOS Simulator.

    No arguments — it always targets ``simctl io booted`` (the one currently
    booted device). Requires a vision-capable model (``model.supports_vision``,
    set from config ``providers.<name>.vision`` / ``model.vision``) to actually
    look at the image; otherwise it still captures and saves the PNG, and
    tells the model where to find it.

    The saved PNG is intentionally NOT deleted after use (either path) — it
    mirrors the non-vision message's promise ("screenshot saved to <path>")
    and lets a human open the file directly. It lives in the OS temp
    directory, so it doesn't linger indefinitely.
    """

    name = "simulator_screenshot"
    description = (
        "Take a screenshot of the currently booted iOS Simulator so you can "
        "look at the UI — use this to check SwiftUI layout, colors, and "
        "content after a build. Takes no arguments; always targets whichever "
        "simulator is currently booted. Requires a vision-capable model to "
        "actually see the image — otherwise the screenshot is still saved "
        "to disk and its path is returned as text."
    )
    is_read_only = True
    requires_permission = False

    def __init__(self, model=None, timeout: float = DEFAULT_TIMEOUT):
        # Held by reference, same pattern as DispatchAgentTool — reflects a
        # later `/model` switch IF the caller mutates this same object's
        # attributes in place; a switch that rebinds `agent.model` to a brand
        # new ModelClient will not retroactively update an already-built
        # tool's reference (a pre-existing limitation shared with the other
        # model-holding tools, not something this task introduces).
        self.model = model
        self.timeout = timeout

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        booted_error = self._check_booted()
        if booted_error:
            return booted_error

        tmp_path = os.path.join(
            tempfile.gettempdir(),
            f"spark_sim_screenshot_{int(time.time() * 1000)}.png",
        )
        try:
            result = subprocess.run(
                ["xcrun", "simctl", "io", "booted", "screenshot", tmp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: simulator screenshot timed out after {self.timeout:.0f}s"
        except OSError as e:
            return f"Error: could not run xcrun simctl ({e})"

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            return ("Error: could not capture simulator screenshot"
                    + (f" ({stderr})" if stderr else ""))

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            return "Error: screenshot command reported success but produced no image"

        if not getattr(self.model, "supports_vision", False):
            # Text-only primary: use Gemini as the eyes and show the image
            # inline — unified with android_screenshot/ios_screenshot instead of
            # handing the model an unusable "saved to <path>" placeholder.
            from ..vision import DISPLAY_IMAGE_SENTINEL, VisionError, describe_image
            try:
                desc = await describe_image(tmp_path)
            except VisionError as e:
                return f"iOS Simulator screenshot saved to {tmp_path} (vision failed: {e})"
            return f"📱 iOS Simulator — Gemini sees:\n{desc}{DISPLAY_IMAGE_SENTINEL}{tmp_path}"

        return f"{SCREENSHOT_SENTINEL_PREFIX}{tmp_path}"

    def _check_booted(self) -> str | None:
        """Return a clear error message if no simulator is booted, else None.

        Checked BEFORE attempting the screenshot itself — `simctl io booted`
        targets "whichever device is booted", which is ambiguous/undefined
        with none booted; failing fast here with a clear, actionable message
        beats parsing that command's own (less predictable) failure mode.
        """
        try:
            result = subprocess.run(
                ["xcrun", "simctl", "list", "devices"],
                capture_output=True, text=True, timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: simulator status check timed out after {self.timeout:.0f}s"
        except OSError as e:
            return f"Error: could not run xcrun simctl ({e})"

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            return ("Error: could not query simulator state"
                    + (f" ({stderr})" if stderr else ""))

        if "(Booted)" not in (result.stdout or ""):
            return ("No booted iOS Simulator found. Open Simulator.app (or run "
                    "`xcrun simctl boot <device>`) and try again.")
        return None
