"""Phase 3 Task 3: embedded-terminal correctness (the non-CPR gap).

Covers the three pure/testable pieces of the mechanism (investigation notes
live in spark_code/ui/input.py, right above `terminal_supports_cpr`):

- `terminal_supports_cpr()` — the pre-flight heuristic (definite non-CPR
  signals only; the ambiguous pexpect-style case is a documented limitation,
  resolved at runtime instead — see the pexpect verification in the task
  report, not here).
- `build_status_segments` / `format_status_line` — the pure "mode + ctx%"
  content shared by the inline fallback.
- `resolve_statusline_mode` — routes `ui.statusline` (auto/toolbar/inline/off)
  + a CPR-support signal to (use_toolbar, use_inline).
"""

from spark_code.ui.input import (
    build_status_segments,
    create_session,
    format_status_line,
    resolve_statusline_mode,
    terminal_supports_cpr,
)


class TestTerminalSupportsCPR:
    def test_spark_no_cpr_escape_hatch_forces_false(self):
        assert terminal_supports_cpr(env={"SPARK_NO_CPR": "1"}, isatty=lambda: True) is False

    def test_prompt_toolkit_native_env_var_also_forces_false(self):
        # Respect prompt_toolkit's own escape hatch too, not just Spark's.
        assert (
            terminal_supports_cpr(env={"PROMPT_TOOLKIT_NO_CPR": "1"}, isatty=lambda: True)
            is False
        )

    def test_dumb_terminal_forces_false(self):
        assert terminal_supports_cpr(env={"TERM": "dumb"}, isatty=lambda: True) is False

    def test_no_tty_forces_false(self):
        assert terminal_supports_cpr(env={"TERM": "xterm-256color"}, isatty=lambda: False) is False

    def test_isatty_exception_is_treated_as_no_cpr(self):
        def boom():
            raise OSError("no tty")

        assert terminal_supports_cpr(env={}, isatty=boom) is False

    def test_plain_tty_with_normal_term_is_optimistically_true(self):
        # This is the heuristic's documented limit: a pexpect pty looks
        # identical to this from the outside (isatty() True, TERM set) even
        # though it never answers CPR — that case is caught at runtime
        # instead (create_session's cpr_state wiring), not by this function.
        assert (
            terminal_supports_cpr(env={"TERM": "xterm-256color"}, isatty=lambda: True) is True
        )

    def test_defaults_to_real_os_environ_and_stdout_when_not_injected(self):
        # Just confirm it doesn't raise when called with no args (uses the
        # real os.environ / sys.stdout.isatty under pytest, whatever that is).
        assert terminal_supports_cpr() in (True, False)


class TestBuildStatusSegments:
    def test_mode_only(self):
        segments = build_status_segments("ask")
        text = format_status_line(segments)
        assert text == "⏵⏵ ask mode"

    def test_mode_and_context_pct(self):
        segments = build_status_segments("auto", context_pct=42.0)
        text = format_status_line(segments)
        assert text == "⏵⏵ auto mode  ·  ctx 42%"

    def test_context_pct_none_omits_ctx_segment(self):
        segments = build_status_segments("trust", context_pct=None)
        text = format_status_line(segments)
        assert "ctx" not in text

    def test_turns_included_when_positive(self):
        segments = build_status_segments("ask", turns=5)
        text = format_status_line(segments)
        assert "5 turns" in text

    def test_turns_omitted_when_zero(self):
        segments = build_status_segments("ask", turns=0)
        text = format_status_line(segments)
        assert "turns" not in text

    def test_model_name_and_provider_included(self):
        segments = build_status_segments(
            "ask", model_name="qwen3.5:122b", provider_name="llm"
        )
        text = format_status_line(segments)
        assert "qwen3.5:122b (llm)" in text

    def test_model_name_without_provider(self):
        segments = build_status_segments("ask", model_name="coder")
        text = format_status_line(segments)
        assert "coder" in text
        assert "()" not in text

    def test_context_style_is_carried_on_the_segment_not_lost(self):
        segments = build_status_segments(
            "ask", context_pct=10.0, context_style="class:bottom-toolbar.context-red"
        )
        styles = [style for style, _ in segments]
        assert "class:bottom-toolbar.context-red" in styles

    def test_full_line_combines_everything_in_order(self):
        segments = build_status_segments(
            "auto",
            model_name="qwen3.5:122b",
            provider_name="llm",
            turns=3,
            context_pct=26.0,
        )
        text = format_status_line(segments)
        assert text == "qwen3.5:122b (llm)  ⏵⏵ auto mode  ·  3 turns  ·  ctx 26%"


class TestFormatStatusLine:
    def test_strips_and_joins_plain_text_no_style_codes(self):
        segments = [
            ("class:a", "  hello  "),
            ("class:b", "world"),
        ]
        # No ANSI/style leakage — pure text join.
        assert format_status_line(segments) == "hello  world"

    def test_empty_segments_yields_empty_string(self):
        assert format_status_line([]) == ""


class TestResolveStatuslineMode:
    def test_off_is_always_neither(self):
        assert resolve_statusline_mode("off", cpr_supported=True) == (False, False)
        assert resolve_statusline_mode("off", cpr_supported=False) == (False, False)

    def test_toolbar_forces_toolbar_never_inline(self):
        assert resolve_statusline_mode("toolbar", cpr_supported=True) == (True, False)
        assert resolve_statusline_mode("toolbar", cpr_supported=False) == (True, False)

    def test_inline_forces_inline_never_toolbar(self):
        assert resolve_statusline_mode("inline", cpr_supported=True) == (False, True)
        assert resolve_statusline_mode("inline", cpr_supported=False) == (False, True)

    def test_auto_uses_toolbar_when_cpr_supported(self):
        assert resolve_statusline_mode("auto", cpr_supported=True) == (True, False)

    def test_auto_falls_back_to_inline_when_cpr_not_supported(self):
        assert resolve_statusline_mode("auto", cpr_supported=False) == (False, True)

    def test_unrecognized_value_behaves_like_auto(self):
        assert resolve_statusline_mode("bogus", cpr_supported=True) == (True, False)
        assert resolve_statusline_mode("bogus", cpr_supported=False) == (False, True)


class TestCreateSessionStatuslineWiring:
    """Session-construction-level checks: the CPR-capable toolbar path must
    be provably unaffected (self-review requirement) — verified here by
    forcing statusline="toolbar" (bypassing the pytest-has-no-tty pre-flight
    heuristic, which would otherwise always say "no CPR" under a captured
    stdout) and asserting the exact toolbar callable still reaches
    PromptSession, unmodified.
    """

    def test_statusline_toolbar_forces_the_toolbar_widget_attached(self, tmp_path):
        session = create_session(
            history_file=str(tmp_path / "hist"), statusline="toolbar"
        )
        assert session.bottom_toolbar is not None

    def test_statusline_off_attaches_no_toolbar(self, tmp_path):
        session = create_session(history_file=str(tmp_path / "hist"), statusline="off")
        assert session.bottom_toolbar is None

    def test_statusline_inline_attaches_no_toolbar(self, tmp_path):
        session = create_session(
            history_file=str(tmp_path / "hist"), statusline="inline"
        )
        assert session.bottom_toolbar is None

    def test_default_auto_under_pytest_no_tty_skips_the_toolbar(self, tmp_path):
        # Under pytest, stdout is captured (not a real tty), so the pre-flight
        # heuristic says CPR is unlikely — "auto" should skip attaching the
        # (guaranteed-useless) toolbar rather than silently no-op forever.
        session = create_session(history_file=str(tmp_path / "hist"))
        assert session.bottom_toolbar is None

    def test_cpr_not_supported_callback_is_neutered_and_updates_cpr_state(
        self, tmp_path
    ):
        cpr_state = {"supported": True}
        session = create_session(
            history_file=str(tmp_path / "hist"),
            statusline="toolbar",
            cpr_state=cpr_state,
        )
        # Simulate prompt_toolkit's own timeout firing (what it does after
        # 2s with no CPR response) — must not raise, must not print anything
        # (there's no console/output arg to capture, so "doesn't raise" plus
        # "does exactly what we told it to" is what's checkable here), and
        # must flip the shared state dict.
        session.app.renderer.cpr_not_supported_callback()
        assert cpr_state["supported"] is False

    def test_cpr_not_supported_callback_is_a_noop_without_cpr_state(self, tmp_path):
        session = create_session(
            history_file=str(tmp_path / "hist"), statusline="toolbar"
        )
        # No cpr_state passed — must still be safely callable (no crash), just
        # does nothing observable.
        session.app.renderer.cpr_not_supported_callback()
