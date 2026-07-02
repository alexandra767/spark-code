"""Plan executor — system-driven plan execution that works with any model.

Parses plan.md into numbered steps, identifies which can run in parallel,
then drives execution: parallel steps go to workers, sequential steps
go to the lead agent. No model needs to "decide" to use spawn_worker.
"""

import asyncio
import re

from rich.console import Console

# Nord palette
_C_TOOL = "#88c0d0"
_C_GREEN = "#a3be8c"
_C_RED = "#bf616a"
_C_DIM = "#4c566a"


def _parse_parallel_spec(section_lines: list[str]) -> set[int]:
    """Parse the ## Parallelization section CONSERVATIVELY.

    The old logic scraped every integer 1-50 out of the prose, which turned
    dependency descriptions (e.g. "steps 3-7") into bogus parallel batches.
    This version, in priority order:

      1. Explicit machine-readable marker — a line like
         ``Parallel: 1, 2, 3`` (or ``Parallel steps: 1 2 3``). Authoritative.
      2. A list of bare numbers, one per line (e.g. ``- 2`` / ``3``).
      3. Prose fallback — ONLY numbers explicitly joined as a group
         (``Steps 1 and 2``) inside a sentence that mentions "parallel".
         Ranges like ``3-7`` and lone numbers are ignored so dependency prose
         never creates false parallel batches.
    """
    nums: set[int] = set()

    def _add(n: int):
        if 1 <= n <= 50:
            nums.add(n)

    # 1. Explicit marker anywhere in the section.
    marker_re = re.compile(r"(?i)^\s*parallel(?:\s+steps)?\s*[:=]\s*(.+)$")
    for line in section_lines:
        m = marker_re.match(line)
        if m:
            for num in re.findall(r"\d+", m.group(1)):
                _add(int(num))
    if nums:
        return nums

    # 2. Bare-number list lines (bullets or plain numbers).
    bare_re = re.compile(r"^[\-\*•]?\s*(\d+)[\.\)]?\s*$")
    bare_found = False
    for line in section_lines:
        m = bare_re.match(line)
        if m:
            _add(int(m.group(1)))
            bare_found = True
    if bare_found:
        return nums

    # 3. Conservative prose parsing — only explicit "X and Y" groups in a
    #    sentence that actually says "parallel". Hyphenated ranges are skipped
    #    because the connector must be a comma / "and" / "&", not "-".
    text = " ".join(section_lines)
    group_re = re.compile(r"\d+(?:\s*(?:,|and|&)\s*\d+)+", re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if "parallel" not in sentence.lower():
            continue
        for grp in group_re.finditer(sentence):
            for num in re.findall(r"\d+", grp.group(0)):
                _add(int(num))
    return nums


def parse_plan(plan_text: str) -> tuple[list[dict], set[int]]:
    """Parse plan.md into steps and identify parallel step numbers.

    Returns (steps, parallel_step_numbers) where each step is:
      {"number": int, "title": str, "body": str}
    """
    steps = []
    parallel_section_lines: list[str] = []

    lines = plan_text.split("\n")
    current_step = None
    current_body_lines: list[str] = []
    in_parallel_section = False
    past_steps_section = False

    for line in lines:
        stripped = line.strip()

        # Detect section headers (## Parallelization, ## Files, ## Risks, etc.)
        if stripped.startswith("## ") or stripped.startswith("# "):
            header_lower = stripped.lower()
            if "parallel" in header_lower:
                in_parallel_section = True
                past_steps_section = True
            else:
                in_parallel_section = False
                # Check if we've moved past steps into Files/Risks/etc.
                if any(kw in header_lower for kw in
                       ["file", "risk", "consider", "depend", "document",
                        "package", "distribut"]):
                    past_steps_section = True

        # Collect the raw parallelization-section lines; parse them
        # conservatively AFTER the loop (see _parse_parallel_spec).
        if in_parallel_section and not stripped.startswith("#"):
            parallel_section_lines.append(stripped)

        # Match numbered step headers: "1. **Title:**" or "1. Title"
        step_match = re.match(
            r"^(\d+)\.?\s+\*{0,2}(.+?)[\*:]*\s*$", stripped
        )
        if step_match and not in_parallel_section and not past_steps_section:
            # Save previous step
            if current_step:
                current_step["body"] = "\n".join(current_body_lines).strip()
                steps.append(current_step)

            current_step = {
                "number": int(step_match.group(1)),
                "title": step_match.group(2).strip().rstrip("*:").strip(),
                "body": "",
            }
            current_body_lines = []
        elif current_step and not in_parallel_section and not past_steps_section:
            current_body_lines.append(line)

    # Save last step
    if current_step:
        current_step["body"] = "\n".join(current_body_lines).strip()
        steps.append(current_step)

    parallel_nums = _parse_parallel_spec(parallel_section_lines)
    return steps, parallel_nums


def parse_references(plan_text: str) -> dict[int, dict]:
    """Parse [Ref N] blocks from ## Reference Material section.

    Returns {ref_number: {"title": str, "text": str}} dict.
    """
    refs = {}
    lines = plan_text.split("\n")
    in_ref_section = False
    current_ref_num = None
    current_title = ""
    current_text_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Detect start of reference material section
        if stripped.startswith("## ") or stripped.startswith("# "):
            header_lower = stripped.lower()
            if "reference material" in header_lower:
                in_ref_section = True
                continue
            elif in_ref_section:
                # Hit the next section — save last ref and stop
                if current_ref_num is not None:
                    refs[current_ref_num] = {
                        "title": current_title,
                        "text": "\n".join(current_text_lines).strip(),
                    }
                break

        if not in_ref_section:
            continue

        # Horizontal rule = end of ref section
        if stripped == "---":
            if current_ref_num is not None:
                refs[current_ref_num] = {
                    "title": current_title,
                    "text": "\n".join(current_text_lines).strip(),
                }
            break

        # Match [Ref N] header line
        ref_match = re.match(r"\[Ref\s+(\d+)\]\s*\*{0,2}(.+?)\*{0,2}\s*(?:\(.*\))?\s*$", stripped)
        if ref_match:
            # Save previous ref
            if current_ref_num is not None:
                refs[current_ref_num] = {
                    "title": current_title,
                    "text": "\n".join(current_text_lines).strip(),
                }
            current_ref_num = int(ref_match.group(1))
            current_title = ref_match.group(2).strip()
            current_text_lines = []
        elif current_ref_num is not None and stripped:
            # Collect blockquote text (strip leading >)
            text = stripped.lstrip("> ").strip()
            if text:
                current_text_lines.append(text)

    # Save last ref if we didn't hit a break
    if current_ref_num is not None and current_ref_num not in refs:
        refs[current_ref_num] = {
            "title": current_title,
            "text": "\n".join(current_text_lines).strip(),
        }

    return refs


def extract_step_refs(title: str, body: str) -> set[int]:
    """Extract [see Ref N] numbers from a step's title and body."""
    combined = f"{title} {body}"
    return {int(m.group(1)) for m in re.finditer(r"Ref\s+(\d+)", combined)}


def build_task_desc(step: dict, refs: dict[int, dict]) -> str:
    """Build a task description for a worker, injecting matched references.

    If the step has [see Ref N] tags and those refs exist, prepends a
    '## Relevant Documentation' block. Otherwise returns a plain task desc.
    """
    ref_nums = extract_step_refs(step["title"], step["body"])
    matched = {n: refs[n] for n in ref_nums if n in refs}

    parts = []
    if matched:
        parts.append("## Relevant Documentation\n")
        for n in sorted(matched):
            r = matched[n]
            parts.append(f"**[Ref {n}] {r['title']}**")
            parts.append(f"> {r['text']}\n")
        parts.append("---\n")
        parts.append("Follow the documentation above when implementing this task.\n")

    parts.append(f"## Task: {step['title']}\n")
    parts.append(f"{step['body']}\n")
    parts.append("Instructions:")
    parts.append("- Create the file(s) described above")
    parts.append("- Write complete, working code with imports")
    parts.append("- Include docstrings and basic error handling")
    parts.append("- If the task mentions tests, write real tests with pytest")

    return "\n".join(parts)


def _make_worker_name(title: str, step_num: int) -> str:
    """Create a clean worker name from a step title."""
    name = re.sub(r"[^a-z0-9]", "-", title.lower())
    name = re.sub(r"-+", "-", name).strip("-")
    return (name[:15] if name else f"step-{step_num}")


async def execute_plan(plan_text: str, team_manager, agent, console: Console):
    """Execute a parsed plan — works with any model.

    - Parallel steps are spawned as background workers
    - Sequential steps run through the lead agent
    - After parallel batches, a summary is injected into the lead's context
    """
    steps, parallel_nums = parse_plan(plan_text)

    # Parse references if this is a projectplan with ## Reference Material
    refs = parse_references(plan_text) if "## Reference Material" in plan_text else {}

    if not steps:
        console.print(
            f"[{_C_RED}]Could not parse plan steps from plan.md[/{_C_RED}]"
        )
        console.print(
            f"[{_C_DIM}]Make sure the plan has numbered steps "
            f"(1., 2., etc.)[/{_C_DIM}]"
        )
        return

    total = len(steps)
    par_count = sum(1 for s in steps if s["number"] in parallel_nums)

    console.print(f"\n[{_C_TOOL}]▸ Executing {total} steps[/{_C_TOOL}]")
    if parallel_nums:
        par_list = ", ".join(str(n) for n in sorted(parallel_nums))
        console.print(
            f"[{_C_TOOL}]  Parallel steps: {par_list}  "
            f"({par_count} workers)[/{_C_TOOL}]"
        )
    console.print()

    i = 0
    while i < len(steps):
        step = steps[i]

        # ── Parallel batch ──────────────────────────────────────────
        if step["number"] in parallel_nums:
            batch = []
            while i < len(steps) and steps[i]["number"] in parallel_nums:
                batch.append(steps[i])
                i += 1

            console.print(
                f"[{_C_TOOL}]▸ Running {len(batch)} steps "
                f"in parallel...[/{_C_TOOL}]"
            )

            workers = []
            for s in batch:
                # Wait if at capacity. Use the team's real cap (local mode
                # allows only MAX_WORKERS_LOCAL=2) instead of a hardcoded 3,
                # which silently dropped the 3rd parallel step from tracking.
                while team_manager.active_count >= team_manager.max_workers:
                    await asyncio.sleep(1)

                task_desc = build_task_desc(s, refs)

                worker_name = _make_worker_name(s["title"], s["number"])
                worker = await team_manager.spawn(
                    task_desc, name=worker_name
                )
                if worker:
                    workers.append(worker)

            # Wait for all workers in this batch
            while any(w.status == "running" for w in workers):
                await asyncio.sleep(2)

            # Report
            succeeded = sum(1 for w in workers if w.status == "completed")
            failed_workers = [w for w in workers if w.status == "failed"]

            parts = [f"[{_C_GREEN}]  ✓ {succeeded} completed[/{_C_GREEN}]"]
            if failed_workers:
                parts.append(
                    f"  [{_C_RED}]✗ {len(failed_workers)} failed[/{_C_RED}]"
                )
            console.print("".join(parts))

            # Inject summary into lead agent's context so it knows
            # what the workers created
            summary_lines = ["Workers completed the following tasks:"]
            for w in workers:
                status = "✓" if w.status == "completed" else "✗"
                result_preview = (w.result or "")[:200].replace("\n", " ")
                summary_lines.append(
                    f"  {status} {w.name}: {result_preview}"
                )
            summary = "\n".join(summary_lines)
            agent.context.add_user(f"[System update] {summary}")
            agent.context.add_assistant(
                "Understood. Workers have completed their tasks. "
                "I'll continue with the remaining steps."
            )

        # ── Sequential step ─────────────────────────────────────────
        else:
            console.print(
                f"[{_C_TOOL}]▸ Step {step['number']}: "
                f"{step['title']}[/{_C_TOOL}]"
            )

            if refs:
                task_desc = (
                    "Execute this step of the project plan:\n\n"
                    + build_task_desc(step, refs)
                    + "\n\nComplete this step fully. "
                    "Create any files or run any commands needed."
                )
            else:
                task_desc = (
                    f"Execute this step of the project plan:\n\n"
                    f"## {step['title']}\n\n"
                    f"{step['body']}\n\n"
                    f"Complete this step fully. "
                    f"Create any files or run any commands needed."
                )

            try:
                await agent.run(task_desc)
            except KeyboardInterrupt:
                console.print(f"\n[{_C_RED}]Step interrupted[/{_C_RED}]")
                return
            except Exception as e:
                # Stop-on-failure: a failed sequential step usually invalidates
                # everything downstream (later steps build on it). Make the
                # failure loud and abort, clearly marking the skipped steps.
                console.print(
                    f"[{_C_RED}]  ✗ Step {step['number']} failed: {e}[/{_C_RED}]"
                )
                downstream = steps[i + 1:]
                if downstream:
                    skipped = ", ".join(str(s["number"]) for s in downstream)
                    console.print(
                        f"[{_C_RED}]  Aborting plan — skipping downstream "
                        f"step(s): {skipped}[/{_C_RED}]"
                    )
                console.print(
                    f"[{_C_RED}]▸ Plan halted at step {step['number']}."
                    f"[/{_C_RED}]"
                )
                return

            i += 1

    # Wait for any remaining workers (e.g. spawned by lead during sequential steps)
    if team_manager.active_count > 0:
        console.print(
            f"[{_C_DIM}]  Waiting for {team_manager.active_count} "
            f"worker(s) to finish...[/{_C_DIM}]"
        )
        while team_manager.active_count > 0:
            await asyncio.sleep(1)

    console.print(f"\n[{_C_GREEN}]▸ Plan execution complete![/{_C_GREEN}]")
