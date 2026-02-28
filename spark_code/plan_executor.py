"""Plan executor — system-driven plan execution that works with any model.

Parses plan.md into numbered steps, identifies which can run in parallel,
then drives execution: parallel steps go to workers, sequential steps
go to the lead agent. No model needs to "decide" to use spawn_worker.
"""

import asyncio
import re
from rich.console import Console
from rich.text import Text


# Nord palette
_C_TOOL = "#88c0d0"
_C_GREEN = "#a3be8c"
_C_RED = "#bf616a"
_C_DIM = "#4c566a"


def parse_plan(plan_text: str) -> tuple[list[dict], set[int]]:
    """Parse plan.md into steps and identify parallel step numbers.

    Returns (steps, parallel_step_numbers) where each step is:
      {"number": int, "title": str, "body": str}
    """
    steps = []
    parallel_nums = set()

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

        # Collect numbers from parallelization section
        if in_parallel_section and not stripped.startswith("#"):
            for m in re.finditer(r"\b(\d+)\b", stripped):
                num = int(m.group(1))
                if 1 <= num <= 50:  # sanity check
                    parallel_nums.add(num)

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

    return steps, parallel_nums


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
                # Wait if at capacity
                while team_manager.active_count >= 3:
                    await asyncio.sleep(1)

                # Build a clear, complete task prompt for the worker
                task_desc = (
                    f"## Task: {s['title']}\n\n"
                    f"{s['body']}\n\n"
                    f"Instructions:\n"
                    f"- Create the file(s) described above\n"
                    f"- Write complete, working code with imports\n"
                    f"- Include docstrings and basic error handling\n"
                    f"- If the task mentions tests, write real tests with pytest\n"
                )

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
                console.print(f"[{_C_RED}]  Error: {e}[/{_C_RED}]")

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
