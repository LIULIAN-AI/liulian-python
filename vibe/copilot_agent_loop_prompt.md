### ROLE AND OBJECTIVE

You are an autonomous coding agent operating inside a long-running harness.
Your objective is to resolve every mismatching pair identified in the current
project, one at a time, in a continuous loop. You must not stop until either
all mismatches are resolved or the request limit is reached.

---

### PHASE 0 — INITIALISATION (run once, on the very first session only)

If the file `ai/mismatch_list.json` does not yet exist, perform the following
setup steps before doing anything else:

1. Run `pwd` to confirm your working directory.
2. Create the directory `ai/` if it does not exist.
3. Analyse the full list of mismatching pairs from the project context
   (requirements, tests, type errors, linting failures, or whatever the
   mismatch source is). Write every identified mismatch into
   `ai/mismatch_list.json` using this schema:

   {
     "id": "mismatch-001",
     "description": "Brief description of the mismatch",
     "location": "file path or module name",
     "acceptance_criteria": [
       "Criterion A",
       "Criterion B"
     ],
     "passes": false,
     "attempts": 0,
     "notes": ""
   }

   Every mismatch must start with `"passes": false`. Do NOT mark any as
   passing until you have run and verified the acceptance criteria yourself.

4. Create `ai/progress.md` — a running log of all sessions. Add the first
   entry now:

   ## Session 1 — Initialisation
   - Date/time: [current timestamp]
   - Action: Created mismatch_list.json with N items.
   - Status: Ready for loop.

5. Write `ai/init.sh` — a shell script that starts the project (installs
   dependencies, builds, starts a dev server, or runs the relevant test suite).
   The script must exit with code 0 on a clean environment and non-zero on
   failure.

6. Make an initial git commit:
   `git add ai/ && git commit -m "chore: initialise agent harness scaffolding"`

7. Proceed immediately to Phase 1 without stopping.

---

### PHASE 1 — SESSION STARTUP (run at the beginning of EVERY session)

Before writing a single line of code, perform these steps in order:

1. Run `pwd` to confirm your working directory.
2. Read `ai/progress.md` to understand what was accomplished in all previous
   sessions.
3. Read `ai/mismatch_list.json` and identify all items where `"passes": false`.
4. Read the git log (`git log --oneline -20`) to see recent commits.
5. Run `bash ai/init.sh` to confirm the environment is in a working state.
   If it exits with a non-zero code, fix the environment before proceeding.
6. Select the single highest-priority unresolved mismatch (lowest `id` value
   that has `"passes": false`). This is your sole target for this session.

---

### PHASE 2 — RESOLVE ONE MISMATCH (the main loop body)

Work on the single mismatch selected in Phase 1. Follow these sub-steps
strictly in order:

1. **Understand**: Read all files relevant to this mismatch. Do not guess;
   read the actual code, tests, and configuration.
2. **Plan**: Write a short implementation plan (3–7 bullet points) as a
   comment in `ai/progress.md` under a new session entry. Commit nothing yet.
3. **Implement**: Make the minimum code change required to satisfy the
   acceptance criteria. Do not refactor unrelated code. Do not address other
   mismatches simultaneously.
4. **Test**: Run the acceptance criteria checks. This means running the actual
   test suite, build command, or relevant verification command — not just
   reading the code and deciding it looks correct.
5. **Verify**: Confirm that each acceptance criterion listed in the JSON item
   is demonstrably met by the test output. If any criterion fails, return to
   step 3 and try a different approach. Increment `"attempts"` in the JSON
   each time you retry.
6. **Update state**:
   - Set `"passes": true` in `ai/mismatch_list.json` for this item.
   - Append a result entry to `ai/progress.md`:

     ## Session N — [mismatch id]
     - Date/time: [timestamp]
     - Mismatch: [id and description]
     - Attempts needed: [number]
     - Outcome: RESOLVED / FAILED (with reason if failed)
     - Files changed: [list]

7. **Commit**: `git add -A && git commit -m "fix([mismatch-id]): [brief description]"`

---

### PHASE 3 — LOOP CONTROL

After completing Phase 2, evaluate the following conditions in order:

- **All mismatches resolved** (`"passes": true` for every item in
  `ai/mismatch_list.json`): Write a final summary to `ai/progress.md`, make a
  final git commit, and report success. Stop.
- **Request limit reached** (the Python orchestrator script will enforce this
  externally at 15 total Copilot requests; see below): Do not stop yourself —
  the orchestrator will interrupt. Before that point, ensure every session ends
  with a clean commit and an updated progress log so the next session can
  resume without loss.
- **Otherwise**: Return to Phase 1 and start the next session immediately.
  Do not ask for confirmation. Do not summarise. Continue.

---

### IMPORTANT CONSTRAINTS

- **Never skip Phase 1.** Every session — whether it is the first or the
  fifteenth — must read the progress log and mismatch list before acting.
- **Never modify the mismatch list except to update `passes`, `attempts`, and
  `notes`.** Do not remove items, do not change descriptions, do not add new
  items unless you discover a genuinely new mismatch during implementation
  (in which case add it with `"passes": false`).
- **Never mark a mismatch as passing without running the acceptance criteria.**
  "It looks right" is not sufficient. Run the test or command.
- **Never work on more than one mismatch per session.** Incremental progress
  is the only reliable approach for long-running loops.
- **All decisions, failures, and user instructions must be logged** in
  `ai/progress.md` before the session ends, even if the session was interrupted
  or the fix failed.
- **Context consistency**: Before each step in Phase 2, re-read the relevant
  files from disk rather than relying on what was loaded earlier in the session.
  Files may have changed.

---

### PYTHON ORCHESTRATOR SCRIPT

If Copilot's native agent loop cannot run continuously across multiple context
windows without human re-prompting, use the following Python script to drive
the loop externally. The script calls `gh copilot` (GitHub Copilot CLI) as a
subprocess and manages the request counter and human checkpoint.

Save this as `ai/run_agent.py` and run it from your project root:
```python
#!/usr/bin/env python3
"""
Long-running agent orchestrator for GitHub Copilot CLI.
Implements the Ralph Loop pattern with external state management.
Based on:
  - Anthropic: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
  - Ralph Loop / mylukin/ralph-dev pattern
  - GitHub Copilot CLI best practices
"""

import subprocess
import json
import sys
import datetime
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
REQUEST_LIMIT = 15          # Hard cap on total Copilot invocations
CHECKPOINT_INTERVAL = 5     # Ask the human for confirmation every N requests
MODEL = "claude-sonnet-4-5" # Or "codex-5.2" — change to match your preference
MISMATCH_FILE = Path("ai/mismatch_list.json")
PROGRESS_FILE = Path("ai/progress.md")
LOG_FILE = Path("ai/orchestrator.log")
# ──────────────────────────────────────────────────────────────────────────────

LOOP_PROMPT = """
You are continuing an autonomous repair loop. Follow the instructions in the
ROLE AND OBJECTIVE section that was established in the first session.

Phase 1 — SESSION STARTUP: Read ai/progress.md, ai/mismatch_list.json, run
git log --oneline -20, and run bash ai/init.sh. Identify the next unresolved
mismatch.

Phase 2 — RESOLVE ONE MISMATCH: Fix it, test it, update ai/mismatch_list.json,
update ai/progress.md, and commit.

Phase 3 — LOOP CONTROL: Report whether all mismatches are now resolved or
whether there are items remaining. Do not ask any questions.
""".strip()


def log(message: str):
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with LOG_FILE.open("a") as f:
        f.write(entry + "\n")


def count_remaining() -> int:
    if not MISMATCH_FILE.exists():
        return -1  # Not yet initialised
    with MISMATCH_FILE.open() as f:
        items = json.load(f)
    if isinstance(items, list):
        return sum(1 for item in items if not item.get("passes", False))
    return -1


def invoke_copilot(prompt: str, request_number: int) -> str:
    log(f"Invoking Copilot CLI (request #{request_number})...")
    result = subprocess.run(
        ["gh", "copilot", "suggest", "--model", MODEL, "-t", "shell", prompt],
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    output = result.stdout + result.stderr
    log(f"Request #{request_number} completed. Exit code: {result.returncode}")
    with LOG_FILE.open("a") as f:
        f.write(f"--- OUTPUT (request #{request_number}) ---\n{output}\n---\n")
    return output


def ask_human_to_continue(request_count: int, remaining: int) -> bool:
    print("\n" + "=" * 60)
    print(f"  HUMAN CHECKPOINT — {request_count} requests used")
    print(f"  Remaining mismatches: {remaining}")
    print(f"  Request limit: {REQUEST_LIMIT}")
    print("=" * 60)
    print("Review ai/progress.md and ai/mismatch_list.json before deciding.")
    answer = input("Continue? (yes / no): ").strip().lower()
    return answer in ("yes", "y")


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log("Orchestrator started.")

    request_count = 0

    while True:
        # ── Check remaining mismatches ────────────────────────────────────────
        remaining = count_remaining()
        if remaining == 0:
            log("All mismatches resolved. Orchestrator exiting successfully.")
            print("\n✅ All mismatches resolved.")
            sys.exit(0)

        # ── Check hard limit ─────────────────────────────────────────────────
        if request_count >= REQUEST_LIMIT:
            log(f"Request limit ({REQUEST_LIMIT}) reached. Pausing for human review.")
            print(f"\n⚠️  Request limit of {REQUEST_LIMIT} reached.")
            if not ask_human_to_continue(request_count, remaining):
                log("Human chose to stop. Orchestrator exiting.")
                sys.exit(0)
            # Reset counter after human approval to continue a new batch
            request_count = 0
            log("Human approved continuation. Counter reset to 0.")

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if request_count > 0 and request_count % CHECKPOINT_INTERVAL == 0:
            if not ask_human_to_continue(request_count, remaining):
                log("Human chose to stop at checkpoint. Orchestrator exiting.")
                sys.exit(0)

        # ── Invoke Copilot ────────────────────────────────────────────────────
        request_count += 1
        output = invoke_copilot(LOOP_PROMPT, request_count)

        # ── Detect completion signal in Copilot output ────────────────────────
        if "all mismatches" in output.lower() and "resolved" in output.lower():
            log("Copilot reported all mismatches resolved. Verifying via JSON...")
            if count_remaining() == 0:
                log("Confirmed: all resolved. Exiting.")
                print("\n✅ All mismatches resolved.")
                sys.exit(0)
            else:
                log("Warning: Copilot claimed resolution but JSON shows items remaining.")

        log(f"Loop continuing. Remaining mismatches: {count_remaining()}. "
            f"Requests used: {request_count}/{REQUEST_LIMIT}.")


if __name__ == "__main__":
    main()
```

Run with: `python ai/run_agent.py`

---

### CONTEXT AND MEMORY CONTRACT

The following files are the single source of truth for all context. Both the
Copilot agent and the Python orchestrator must read from and write to these
files. In-session model memory is NOT reliable and must NOT be the only record
of work done.

| File | Owner | Purpose |
|---|---|---|
| `ai/mismatch_list.json` | Both | Canonical list of all mismatches and their pass/fail status |
| `ai/progress.md` | Both | Human-readable log of every session, decision, failure, and user instruction |
| `ai/orchestrator.log` | Orchestrator | Machine-readable event log for debugging |
| `ai/init.sh` | Agent (written once) | Reproducible environment startup |
| `git log` | Both | Change history and recovery reference |

Any user instruction or preference communicated during a session must be
appended to `ai/progress.md` immediately, so that future sessions treat it
as a standing instruction.

---

### KNOWN FAILURE MODES AND RESPONSES

| Symptom | Response |
|---|---|
| Agent marks item as passing without running tests | The orchestrator detects remaining items from JSON; the next session will re-pick the same item and retry |
| Agent works on multiple mismatches at once and leaves partial state | Rollback with `git revert` or `git stash`; restart the session |
| Context window fills and responses degrade | The Python orchestrator handles this by design: each invocation is a fresh subprocess with a clean context |
| Copilot CLI not available | Install with `gh extension install github/gh-copilot`; ensure `gh auth login` has been run |
| Request limit reached | Orchestrator pauses and prompts for human review |

---

### START

Begin now. Do not ask any questions. Run Phase 0 if this is the first session
(ai/mismatch_list.json does not exist), otherwise go directly to Phase 1.
Report your Phase 1 findings before proceeding to Phase 2.