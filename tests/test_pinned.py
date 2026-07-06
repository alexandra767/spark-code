"""Regression tests for spark_code.pinned — audit 2026-07-06 item 5.

Pinned files were re-injected by appending the rendered block to the system
prompt whenever it differed from what was already there. Because the block
embeds full file content, any edit produced a different string, so the stale
block stayed and a fresh copy was appended — N edits → N+1 copies. Unpinning to
zero never removed the block at all. The fix rewrites a single delimited region
each turn (idempotent), so the prompt carries exactly one current block or none.
"""

from spark_code.pinned import PinnedFiles


def _count_regions(prompt: str) -> int:
    return prompt.count(PinnedFiles.PIN_START)


def test_pin_then_edit_reinject_no_duplicate(tmp_path):
    f = tmp_path / "conf.py"
    f.write_text("VALUE = 1\n")
    pins = PinnedFiles()
    ok, _ = pins.pin(str(f))
    assert ok

    base = "SYSTEM PROMPT BODY"
    p1 = pins.apply_to_prompt(base)
    assert _count_regions(p1) == 1
    assert "VALUE = 1" in p1
    assert "SYSTEM PROMPT BODY" in p1

    # Edit the file, refresh, and re-inject onto the ALREADY-injected prompt.
    f.write_text("VALUE = 2\n")
    pins.refresh()
    p2 = pins.apply_to_prompt(p1)
    assert _count_regions(p2) == 1, "must not accumulate a second pinned block"
    assert "VALUE = 2" in p2
    assert "VALUE = 1" not in p2, "stale content must be replaced, not kept"
    assert "SYSTEM PROMPT BODY" in p2

    # Repeated re-injection with no change stays at exactly one region.
    p3 = pins.apply_to_prompt(p2)
    assert _count_regions(p3) == 1


def test_unpin_removes_region(tmp_path):
    f = tmp_path / "conf.py"
    f.write_text("X = 1\n")
    pins = PinnedFiles()
    pins.pin(str(f))

    base = "BODY"
    p1 = pins.apply_to_prompt(base)
    assert _count_regions(p1) == 1

    pins.unpin(str(f))
    p2 = pins.apply_to_prompt(p1)
    assert _count_regions(p2) == 0, "unpin must remove the region entirely"
    assert "X = 1" not in p2
    assert "BODY" in p2


def test_apply_to_prompt_no_pins_is_noop(tmp_path):
    pins = PinnedFiles()
    base = "JUST THE BODY"
    assert pins.apply_to_prompt(base).strip() == "JUST THE BODY"
