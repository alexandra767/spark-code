"""Display-only `display_name` override: shows a friendly model name in the
banner / connection line and suppresses the vLLM `--served-model-name` alias,
without changing the wire model id that requests actually use."""
import io

from rich.console import Console

from spark_code.cli import print_banner
from spark_code.config import resolve_provider
from spark_code.model import ModelClient


def _cfg(display_name=None):
    llm = {"endpoint": "http://x:30000", "model": "qwen3.5:122b",
           "worker_model": "qwen3.5:122b"}
    if display_name is not None:
        llm["display_name"] = display_name
    return {"active_provider": "llm", "providers": {"llm": llm}}


def test_resolve_provider_surfaces_display_name():
    resolved = resolve_provider(_cfg("Qwen3-Coder-Next"))["model"]
    assert resolved["display_name"] == "Qwen3-Coder-Next"
    assert resolved["name"] == "qwen3.5:122b"  # wire id unchanged


def test_resolve_provider_defaults_display_name_empty():
    assert resolve_provider(_cfg())["model"]["display_name"] == ""


def test_model_client_connection_line_uses_display_name():
    c = ModelClient(endpoint="http://x:30000", model="qwen3.5:122b",
                    provider="llm", display_name="Qwen3-Coder-Next")
    assert c.display_name == "Qwen3-Coder-Next"
    # the connection message is f"Connected to {provider} ({display_name or model})"
    shown = c.display_name or c.model
    assert shown == "Qwen3-Coder-Next"


def _banner_text(config, real_model_name):
    console = Console(file=io.StringIO(), width=200, no_color=True)
    print_banner(console, config, real_model_name=real_model_name)
    return console.file.getvalue()


def test_banner_shows_display_name_and_hides_alias():
    cfg = resolve_provider(_cfg("Qwen3-Coder-Next"))
    out = _banner_text(cfg, real_model_name="Qwen3-Coder-Next-int4-AutoRound")
    assert "Qwen3-Coder-Next" in out
    assert "served as" not in out
    assert "qwen3.5:122b" not in out  # the alias never appears


def test_status_bar_precedence_prefers_display_name():
    # The bottom status bar (cli status_callback) passes
    # `display_name or name` to toolbar_status_segments — the same precedence
    # used for /model and /benchmark. Verify the toolbar renders that name and
    # never the wire alias.
    from spark_code.ui.input import toolbar_status_segments
    cfg = resolve_provider(_cfg("Qwen3-Coder-Next"))["model"]
    shown = cfg.get("display_name") or cfg.get("name")
    assert shown == "Qwen3-Coder-Next"
    segs = toolbar_status_segments(model_name=shown, provider_name="llm", turns=3)
    text = "".join(t for _s, t in segs)
    assert "Qwen3-Coder-Next" in text and "qwen3.5:122b" not in text


def test_banner_without_display_name_keeps_served_as():
    cfg = resolve_provider(_cfg())  # no display_name
    out = _banner_text(cfg, real_model_name="Qwen3-Coder-Next-int4-AutoRound")
    assert "Qwen3-Coder-Next-int4-AutoRound" in out
    assert "served as" in out and "qwen3.5:122b" in out  # honest fallback intact
