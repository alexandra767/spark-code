import base64
from io import StringIO

from rich.console import Console

from spark_code.ui.output import render_image_inline
from spark_code.vision import DISPLAY_IMAGE_SENTINEL


def _console():
    buf = StringIO()
    return Console(file=buf, force_terminal=True), buf


def test_render_image_inline_emits_iterm_escape(tmp_path, monkeypatch):
    monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    png = tmp_path / "s.png"
    png.write_bytes(b"PNGBYTES")
    console, buf = _console()
    assert render_image_inline(console, str(png)) is True
    out = buf.getvalue()
    assert "\033]1337;File=inline=1" in out
    assert base64.b64encode(b"PNGBYTES").decode() in out  # the image payload


def test_render_image_inline_false_off_iterm(tmp_path, monkeypatch):
    monkeypatch.delenv("LC_TERMINAL", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    png = tmp_path / "s.png"
    png.write_bytes(b"X")
    console, _ = _console()
    assert render_image_inline(console, str(png)) is False


def test_render_image_inline_false_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    console, _ = _console()
    assert render_image_inline(console, str(tmp_path / "nope.png")) is False


def _agent():
    from spark_code.agent import Agent
    from spark_code.context import Context
    from spark_code.permissions import PermissionManager
    from spark_code.tools.base import ToolRegistry
    return Agent(object(), Context(system_prompt="x"), ToolRegistry(),
                 PermissionManager(mode="trust", interactive=False))


def test_store_tool_result_strips_display_path_from_model_text(tmp_path):
    png = tmp_path / "s.png"
    png.write_bytes(b"P")
    a = _agent()
    stored = a._store_tool_result(
        {"name": "ios_screenshot", "id": "1"},
        f"A DoorDash screen.{DISPLAY_IMAGE_SENTINEL}{png}")
    # the MODEL sees only the description — never the filesystem path
    assert stored == "A DoorDash screen."
    assert str(png) not in stored
    assert str(png) in a._pending_display_images


def test_flush_display_images_renders_and_deletes_on_iterm(tmp_path, monkeypatch):
    monkeypatch.setenv("LC_TERMINAL", "iTerm2")
    png = tmp_path / "s.png"
    png.write_bytes(b"PNGDATA")
    a = _agent()
    a.console = Console(file=StringIO(), force_terminal=True)
    a._pending_display_images = [str(png)]
    a._flush_display_images()
    assert not png.exists()  # rendered inline -> cleaned up
    assert a._pending_display_images == []


def test_flush_display_images_keeps_file_and_notes_path_off_iterm(tmp_path, monkeypatch):
    monkeypatch.delenv("LC_TERMINAL", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    png = tmp_path / "s.png"
    png.write_bytes(b"PNGDATA")
    a = _agent()
    buf = StringIO()
    a.console = Console(file=buf, force_terminal=True)
    a._pending_display_images = [str(png)]
    a._flush_display_images()
    assert png.exists()  # kept so the user can open it
    assert "saved to" in buf.getvalue() and png.name in buf.getvalue()
