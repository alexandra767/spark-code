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

from prompt_toolkit.renderer import CPR_Support

from spark_code.ui.input import (
    build_status_segments,
    cpr_confirmed_supported,
    create_session,
    format_status_line,
    renderer_cpr_support,
    resolve_statusline_mode,
    terminal_supports_cpr,
    toolbar_mode_segments,
    toolbar_status_segments,
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


class TestToolbarGoldens:
    """Review fix (one-source unification): the toolbar's two lines must be
    BYTE-IDENTICAL to what the pre-refactor cli.py closures produced. Every
    expected list below was captured by EXECUTING the original closure code
    (copied verbatim, state injected) before the refactor — not transcribed
    by hand. If a golden fails, the toolbar's on-screen appearance changed;
    that is a visual regression, not a test to update casually.
    """

    def test_status_line_golden_full_state(self):
        # model+provider, 3 turns, speed, cost, ctx 42% (green)
        got = toolbar_status_segments(
            model_name="qwen3.5:122b",
            provider_name="llm",
            turns=3,
            speed_str="12.3 tok/s",
            cost_str="$0.0042",
            context_pct=42.0,
            context_style="class:bottom-toolbar.context-green",
        )
        assert got == [
            ("class:bottom-toolbar.team", "  qwen3.5:122b (llm)"),
            ("class:bottom-toolbar.info", "  "),
            ("class:bottom-toolbar.info", "3 turns"),
            ("class:bottom-toolbar.info", "  12.3 tok/s"),
            ("class:bottom-toolbar.info", "  $0.0042"),
            ("class:bottom-toolbar.context-green", "          ctx 42%"),
        ]

    def test_status_line_golden_sparse_state_token_fallback(self):
        # model without provider, 0 turns (emits the EMPTY info segment the
        # original closure emitted, not nothing), no stats, no ctx% but a raw
        # token count → the "N tokens" fallback segment.
        got = toolbar_status_segments(
            model_name="coder",
            turns=0,
            context_pct=None,
            context_tokens=8389,
        )
        assert got == [
            ("class:bottom-toolbar.team", "  coder"),
            ("class:bottom-toolbar.info", "  "),
            ("class:bottom-toolbar.info", ""),
            ("class:bottom-toolbar.context", "    8,389 tokens"),
        ]

    def test_status_line_golden_no_model_red_ctx(self):
        got = toolbar_status_segments(
            turns=1,
            context_pct=14.0,
            context_style="class:bottom-toolbar.context-red",
        )
        assert got == [
            ("class:bottom-toolbar.info", "1 turns"),
            ("class:bottom-toolbar.context-red", "          ctx 14%"),
        ]

    def test_mode_line_golden_ask_no_team(self):
        got = toolbar_mode_segments("ask")
        assert got == [
            ("class:bottom-toolbar.mode", "  ⏵⏵ "),
            ("class:bottom-toolbar.mode-text", "ask mode on"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.info", "shift+tab to switch"),
        ]

    def test_mode_line_golden_plan_with_team(self):
        got = toolbar_mode_segments("plan", team_active=2, team_total=3)
        assert got == [
            ("class:bottom-toolbar.mode", "  ⏵⏵ "),
            ("class:bottom-toolbar.mode-text", "plan mode on"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.info", "shift+tab to switch"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.team", "ctrl+t "),
            ("class:bottom-toolbar.team-text", "team (2/3)"),
        ]

    def test_mode_line_golden_trust_no_team(self):
        got = toolbar_mode_segments("trust")
        assert got == [
            ("class:bottom-toolbar.mode", "  ⏵⏵ "),
            ("class:bottom-toolbar.mode-text", "trust mode on"),
            ("class:bottom-toolbar.info", "  ·  "),
            ("class:bottom-toolbar.info", "shift+tab to switch"),
        ]

    def test_inline_flavor_unchanged_by_the_knob_extension(self):
        # The knob defaults must keep the inline fallback's rendering exactly
        # as shipped (same assertion as test_full_line_combines_everything_
        # in_order, restated here as the refactor's other half of the
        # safety net).
        segments = build_status_segments(
            "auto",
            model_name="qwen3.5:122b",
            provider_name="llm",
            turns=3,
            context_pct=26.0,
        )
        assert format_status_line(segments) == (
            "qwen3.5:122b (llm)  ⏵⏵ auto mode  ·  3 turns  ·  ctx 26%"
        )


class _FakeRenderer:
    def __init__(self, support):
        self.cpr_support = support


class _FakeApp:
    def __init__(self, renderer):
        self.renderer = renderer


class _FakeSession:
    def __init__(self, support):
        self.app = _FakeApp(_FakeRenderer(support))


class TestRendererCprSupport:
    """Review fix (fast-turn CPR race): the inline decision must consult the
    renderer's LIVE cpr_support enum, because prompt_toolkit's 2s CPR timer
    is a background task of the prompt Application — a sub-2s turn cancels
    it, the neutered callback never fires, and the latch alone would never
    flip in exactly the fast scripted non-CPR pty this task targets.
    """

    def test_supported_maps_to_true(self):
        assert renderer_cpr_support(_FakeSession(CPR_Support.SUPPORTED)) is True

    def test_not_supported_maps_to_false(self):
        assert renderer_cpr_support(_FakeSession(CPR_Support.NOT_SUPPORTED)) is False

    def test_unknown_maps_to_none(self):
        assert renderer_cpr_support(_FakeSession(CPR_Support.UNKNOWN)) is None

    def test_unreadable_session_maps_to_none_not_a_crash(self):
        assert renderer_cpr_support(object()) is None

    def test_fresh_real_session_is_readable(self, tmp_path):
        # Integration sanity: a real, never-run PromptSession's renderer state
        # is readable through the helper. Under pytest (captured stdout, not
        # a tty) prompt_toolkit resolves NOT_SUPPORTED at construction; in
        # exotic capture setups it could still be UNKNOWN — either way it
        # must never read as "supported" before any prompt has run.
        session = create_session(history_file=str(tmp_path / "hist"))
        assert renderer_cpr_support(session) in (False, None)


class TestCprConfirmedSupported:
    def test_unknown_is_pessimistically_unsupported(self):
        # The review's core rule: print inline while UNKNOWN — brief cosmetic
        # duplication in good terminals beats invisible status in bad ones.
        assert cpr_confirmed_supported(None, latched_unsupported=False) is False

    def test_live_supported_and_no_latch_is_supported(self):
        assert cpr_confirmed_supported(True, latched_unsupported=False) is True

    def test_latch_wins_even_over_live_supported(self):
        # The neutered-callback flag stays authoritative as a secondary latch.
        assert cpr_confirmed_supported(True, latched_unsupported=True) is False

    def test_live_not_supported_is_unsupported(self):
        assert cpr_confirmed_supported(False, latched_unsupported=False) is False

    def test_auto_routes_inline_while_unknown(self):
        # End-to-end decision for the exact race scenario: auto mode, fast
        # turns, timer cancelled → live UNKNOWN → inline must print.
        confirmed = cpr_confirmed_supported(None, latched_unsupported=False)
        assert resolve_statusline_mode("auto", confirmed) == (False, True)

    def test_auto_skips_inline_once_supported_observed(self):
        confirmed = cpr_confirmed_supported(True, latched_unsupported=False)
        assert resolve_statusline_mode("auto", confirmed) == (True, False)


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
        # FRAGILITY NOTE (review): this depends on pytest's stdout capture NOT
        # being a tty. Running with -s/--capture=no in a real terminal makes
        # sys.stdout a tty again and would flip this assertion. If that ever
        # becomes a supported workflow, switch this to the deterministic
        # injected form used by test_auto_consumes_passed_cpr_state below.
        session = create_session(history_file=str(tmp_path / "hist"))
        assert session.bottom_toolbar is None

    def test_auto_consumes_passed_cpr_state(self, tmp_path):
        # Review minor: create_session must consume the caller's cpr_state
        # for the initial attachment decision instead of redundantly
        # re-running terminal_supports_cpr() itself. Under pytest the
        # heuristic says False (no tty) — so a passed {"supported": True}
        # attaching the toolbar proves the passed state won, deterministically.
        session = create_session(
            history_file=str(tmp_path / "hist"),
            statusline="auto",
            cpr_state={"supported": True},
        )
        assert session.bottom_toolbar is not None

    def test_auto_with_passed_unsupported_cpr_state_skips_toolbar(self, tmp_path):
        session = create_session(
            history_file=str(tmp_path / "hist"),
            statusline="auto",
            cpr_state={"supported": False},
        )
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
