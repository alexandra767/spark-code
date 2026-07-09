# /loop Command — Design

**Date:** 2026-07-09
**Status:** Approved (design walkthrough with Alexandra; "Both flavors" + "Fix with safety net" chosen explicitly)

## Goal

A `/loop` slash command that keeps working without a human driving: either **until a goal is verifiably done** (until-done flavor) or **repeatedly on a timer** (timed flavor). Motivating use case: "keep an eye on the build every 30 minutes and fix anything that breaks while I'm out on deliveries."

## Command surface

```
/loop <goal>                        # until-done: rounds until verified done or stopped
/loop --check "<cmd>" <goal>        # until-done, goal verified by <cmd> exit 0
/loop every <interval> <task>       # timed: run task, sleep, repeat until stopped
/loop every <interval> --check "<cmd>" <task>
/loop status                        # active loop: flavor, round #, last result, next run time
/loop stop                          # graceful stop (also: Esc)
/loop                               # no args: usage help (mirrors /watch)
```

- `<interval>` accepts `30m`, `2h`, `90s` (min 60s, max 24h).
- One active loop per session (v1). Starting a second prints the active one and how to stop it.
- Interval parsing is its own small function (`_parse_interval`) with unit tests.

## Semantics

### A round (identical for both flavors)

1. **Checkpoint**: create a git checkpoint labeled `loop-<id>-round-<n>` via the existing checkpoint machinery. If the repo is dirty from the user's own work, the checkpoint still snapshots (same behavior as `/checkpoint`); the log notes it.
2. **Run**: execute the goal/task as one agentic run using the existing autonomous machinery (AGENTIC_PROMPT + trusted permissions for the duration of the round — the same trust switch `/yolo` uses), bounded by a per-round turn cap and wall-clock timeout.
3. **Verify**:
   - With `--check`: run the command; exit 0 = goal met. The model never self-grades when a check exists.
   - Without `--check`: the model states DONE/NOT-DONE in a structured marker; the log records that this was self-assessed.
4. **Log**: append a round entry to the loop log (see Observability).
5. **Continue/sleep/stop**:
   - Until-done: goal met → stop with summary; else next round immediately.
   - Timed: sleep `<interval>` (checks stop-flag every second while sleeping), then next round. `--check` failing in a timed loop just means the round did work; timed loops don't exit on success — they exit when stopped.

### Stop conditions (all flavors)

- `/loop stop` or Esc (graceful: finishes writing the log, no mid-round kill unless Esc pressed twice).
- Until-done: goal verified done.
- **Round cap**: default 20 rounds (until-done); timed loops are uncapped in rounds but respect the same per-round bounds.
- **No-progress guard**: two consecutive rounds with no file changes AND failing verify → stop with `STALLED` status (prevents infinite spinning, mirrors "loop-until-dry" discipline).
- Per-round timeout (default 15 min) and turn cap (default 40 turns) — a hung round is cancelled, logged as `TIMEOUT`, and counts as no-progress.

### Safety net (Alexandra's explicit choice: "fix it, with a safety net")

- Git checkpoint **before every round** — any round's changes are one restore away.
- Full audit log of every round: what ran, what changed (files + diffstat), verify result.
- Loops run with round-scoped trust; when the loop ends (any reason), the session's permission mode is restored to what it was before the loop started.

## Implementation shape

Follows the `/watch` precedent end to end:

- `cli.py` slash handler returns sentinel `__LOOP__<json-args>`; the main REPL consumes it (same place `__WATCH__` is consumed, cli.py ~3579) and starts the loop engine.
- **Loop engine** = new module `spark_code/loop.py`: an asyncio task inside the running CLI process (same lifecycle style as the Esc watcher / watch task). Not a system scheduler; terminal stays open (documented limitation, v1).
- Engine calls the same agent-run entry the normal prompt path uses, with agentic prompt + trusted permissions injected for the round.
- State object (`LoopState`): id, flavor, goal, check_cmd, interval, round_n, started_at, last_result, next_run_at, stop_flag. `/loop status` renders it; the input-area status line shows `🔁 round 3 · next 14:32` while active.
- Round summaries: `.spark/loops/<timestamp>-<slug>.md` in the project (git-ignored), one file per loop, one section per round.

## Out of scope (v1, deliberate)

- Surviving terminal close / running while the app is off (would need launchd/cron + headless mode; documented as a future recipe).
- Multiple concurrent loops.
- Cross-session loop resume.
- Notifications (Telegram etc.) on loop events — natural v2 hook.

## Testing

- Unit: interval parsing (valid/invalid/bounds), no-progress detection, round-cap and timeout accounting, stop-flag responsiveness during sleep, sentinel arg round-trip (JSON encode/decode), permission-mode restore.
- Integration (mock agent runner): until-done stops on verified success; `--check` gates DONE even when the model claims success; timed flavor sleeps and re-fires; STALLED after two no-change failing rounds; Esc stops gracefully; checkpoint called before every round (assert call order).
- No test invokes a real model (existing test-suite convention).
