"""Regression tests for finding #1: context_left() must account for messages
appended AFTER the last server-reported usage count.

record_usage() stores the server's prompt-token count, but that number reflects
the PREVIOUS request. This turn's later messages (the assistant reply + tool
results, often several KB each) are not in it. Before the fix, context_left()
trusted the stale count forever, so _maybe_compact never saw the growth and the
next request overflowed the window (400); and because an errored request
reports no usage, the stale count stuck and re-overflowed every turn until a
manual /clear. context_left() now adds a 3.5-chars/token estimate of ONLY the
messages appended since the watermark.
"""

from spark_code.context import Context


def test_context_left_drops_as_messages_appended_after_usage():
    ctx = Context(system_prompt="", max_tokens=10000)
    # Server reports the prompt was 5000 tokens at this point (half the window).
    ctx.record_usage(5000)
    before = ctx.context_left(10000)
    assert abs(before - 0.5) < 0.01

    # ~7000 chars of tool results land AFTER that usage report (as happens
    # mid-turn). The meter must reflect them, not stay frozen at 0.5.
    ctx.add_assistant("here's what I found")
    ctx.add_tool_result("call_1", "read_file", "X" * 7000)
    after = ctx.context_left(10000)

    assert after < before
    # 5000 + ~2000 (7019 chars / 3.5) ≈ 7000 tokens used → ~0.30 left.
    assert abs(after - 0.30) < 0.02


def test_context_left_stable_when_nothing_appended():
    ctx = Context(system_prompt="", max_tokens=10000)
    ctx.record_usage(5000)
    # Two reads with no intervening appends return the same value.
    assert ctx.context_left(10000) == ctx.context_left(10000) == 0.5


def test_estimate_path_unchanged_when_usage_never_recorded():
    # (b) No server usage ever recorded → pure estimate path, untouched by the
    # watermark machinery.
    ctx = Context(system_prompt="", max_tokens=10000)
    ctx.add_user("hello there")
    expected = max(0.0, min(1.0, 1 - ctx.estimate_tokens() / 10000))
    assert abs(ctx.context_left(10000) - expected) < 1e-9


def test_errored_request_stale_count_still_reflects_appends():
    """An errored request records no new usage, so last_prompt_tokens keeps its
    previous value — but appended messages must still push the meter down
    (previously the stale count stuck and the session re-overflowed every turn
    until a manual /clear)."""
    ctx = Context(system_prompt="", max_tokens=10000)
    ctx.record_usage(5000)
    # Turn errors out → no new record_usage, but it still appended messages.
    ctx.add_assistant("partial answer before the error")
    ctx.add_tool_result("c1", "bash", "Z" * 10500)  # ~3000 tokens
    left = ctx.context_left(10000)
    # 5000 + ~3000 ≈ 8000 used → ~0.2 left, NOT frozen at 0.5.
    assert left < 0.35


def test_compaction_resets_usage_watermark():
    # (c) compaction resets → context_left re-estimates instead of adding to a
    # stale server count for a history that no longer exists.
    ctx = Context(system_prompt="", max_tokens=10000)
    for i in range(20):
        ctx.add_user(f"message {i} " + "y" * 200)
        ctx.add_assistant(f"reply {i} " + "z" * 200)
    ctx.record_usage(8000)
    ctx.compact(keep_recent=6)

    assert ctx.last_prompt_tokens is None
    expected = max(0.0, min(1.0, 1 - ctx.estimate_tokens() / 10000))
    assert abs(ctx.context_left(10000) - expected) < 1e-9


def test_clear_resets_usage_watermark():
    ctx = Context(system_prompt="", max_tokens=10000)
    ctx.record_usage(5000)
    ctx.add_tool_result("c1", "read_file", "X" * 7000)
    ctx.clear()
    assert ctx.last_prompt_tokens is None
    # Fresh/empty history → nearly the whole window is free.
    assert ctx.context_left(10000) > 0.99


def test_load_resets_usage_watermark(tmp_path):
    # Loading replaces the whole history; a prior session's server count must
    # not keep reporting a window for messages that are no longer present.
    src = Context(system_prompt="", max_tokens=10000)
    src.add_user("old message " + "a" * 500)
    path = str(tmp_path / "sess.json")
    src.save(path)

    ctx = Context(system_prompt="", max_tokens=10000)
    ctx.record_usage(9000)  # stale count from a different, larger history
    assert ctx.load(path) is True
    assert ctx.last_prompt_tokens is None
    expected = max(0.0, min(1.0, 1 - ctx.estimate_tokens() / 10000))
    assert abs(ctx.context_left(10000) - expected) < 1e-9


def test_rewind_restores_consistent_usage_watermark():
    ctx = Context()
    ctx.record_usage(500)      # recorded while history was empty
    ctx.snapshot()
    ctx.add_user("hi " + "q" * 1000)
    ctx.record_usage(900)      # new count for the grown history
    assert ctx.rewind_conversation(1) is True
    # last_prompt_tokens restored (existing behavior)...
    assert ctx.last_prompt_tokens == 500
    # ...and its watermark restored to match the rewound-to (empty) history, so
    # context_left doesn't mis-count the rewound-away message as "appended".
    assert ctx._usage_msg_count == 0
    assert ctx._usage_msg_chars == 0
