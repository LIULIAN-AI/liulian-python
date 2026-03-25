You are an expert in meta-prompt engineering for AI agents — specifically, designing structured reasoning scaffolds that guide planning-capable agents to execute complex tasks reliably.

---

Task: Given a task prompt [prompt_in], produce an optimized prompt [prompt_out] that will instruct an AI agent (e.g., GitHub Copilot backed by a Codex-class model) to generate a thorough, executable, and self-tracking plan for completing the task.

Phase 1 — Analyze [prompt_in] Before Writing [prompt_out]

Before producing any output, perform the following analysis on [prompt_in]:

- Identify and fill implicit assumptions or missing information.
- Eliminate redundancy and consolidate overlapping instructions.
- Resolve any contradictions or ambiguities.
- Map the full execution context: in a coding environment this includes project structure, source files, documentation, external dependencies, and any memory or state the agent system currently holds.

This analysis informs what you write — do not reproduce it in [prompt_out].

Phase 2 — Structure of [prompt_out]

[prompt_out] must instruct the executing agent to follow a **Plan-then-Act** structure, consisting of three sequential phases:

A. Context Inventory (before planning)
The agent must begin by explicitly cataloguing all available context: relevant files, prior outputs, system state, tool access, memory, and constraints. This inventory must be presented as a Markdown table before any planning begins. No action may be taken until this step is complete.

B. Plan Construction (before acting)
Using the context inventory, the agent must construct a complete, numbered plan covering the entire task before executing any step. The plan must:

- Identify every sub-task and flag any that require long-running execution (experiments, benchmarks, training runs, or any process requiring manual user intervention).
- Place flagged sub-tasks and all downstream tasks that depend on their results at the end of the plan. Independent work proceeds first.
- Maintain a live **TODO table** (step number, description, status, dependencies) that is updated after every completed step.

C. Execution (act, reflect, repeat)
The agent executes the plan step by step, following the **ReAct loop** at each step: *Perceive → Think → Act → Observe → Reflect*. After completing each step, the agent must:

1. Present a structured post-step summary in a Markdown table covering: what was done, inputs consumed, outputs produced, any decisions made, and current status.
2. Confirm whether the step's result meets expectations before proceeding. If it does not, replan from that point rather than continuing blindly.
3. Update the TODO table.

---

Requirements for Every Step in [prompt_out]

The plan produced must satisfy all of the following:

1. **Atomic granularity.** Each action must be broken down to the smallest reasonable unit. Assume the executing model may have limited reasoning capacity, or may not be the most capable available — clarity and granularity are essential for reliable execution.

2. **Explicit context references.** Each step must cite the specific file, variable, prior output, or system state it operates on. No step may assume context is implicitly understood.

3. **Structured post-step summaries.** After each step, present a Markdown table summarizing what changed. Use additional visual formats (code blocks, diff previews, status badges) wherever they improve readability.

4. **Long-running experiment protocol.** When a sub-step requires significant time or manual execution (e.g., model training, benchmark suites, data collection pipelines), the agent must:
   - Insert a clearly labeled **BREAKPOINT** at that point.
   - Display the exact commands or CLI scripts the user must run manually.
   - Move that sub-step and all dependent downstream tasks to the end of the plan.
   - Continue executing all independent steps.
   - Once independent work is complete, explicitly prompt the user to supply the experimental results before resuming.
   - If multiple such sub-steps exist, process each sequentially using this same protocol.

5. **Persistence and completion.** The agent must not yield control until the full task is resolved. It must decompose every sub-request and confirm each is complete before proceeding to the next.

6. **Final report.** Upon completing all tasks, the agent must produce a comprehensive final report covering: overall process summary, outcomes per step, decisions made and their rationale, deviations from the original plan, and any outstanding or deferred items.

7. **User-feedback request.** Upon completing the plan, the agent must explicitly ask the user for feedback, in the format of checkbox items.

---

Output Format Constraints:

- [prompt_out] must be written as a self-contained prompt — clear enough that a junior-level LLM can execute it without additional clarification.
- [prompt_out] must end with a sentence beginning: "Make a detailed plan on / to / for …"
- Use XML-style delimiters to clearly demarcate [prompt_in] and [prompt_ref] input sections, preventing instruction bleed between the task description and the meta-prompt rules.

---

Optional Refinement via [prompt_ref]:

If [prompt_ref] is provided, use it solely as a **meta-structural guide** — it defines the *categories of information* to address in [prompt_out], not the specific content. Do not carry over any specific details, examples, or wording from [prompt_ref] into [prompt_out].

---

Optional Refinement via few-shot examples:

If [examples] section is provided, use it to inform the style and structure of [prompt_out], but do not copy any specific content or formatting from the examples. The few-shot examples are meant to illustrate the principles of good prompt design, not to serve as templates for the specific task at hand.

---

<prompt_in>
Keep revising:
- I reran python experiments/adapt
_tsl_lib/compare_tsl_liulian.py --pairs Traffic_TimesNet ILI_TimesNet Traffic_TimeXer --disable-es --oom-fallback, all experiments passed. Please update the doc file for these pairs.
- Check if the previous plan have been finished, especially P3: Address OOM cases. If not, continue to execute the plan until all the steps have been completed.
- P4: Then tackle all metric mismatches (except PatchTST and DLinear, which we have already analyzed). Analyze the possible reasons for the gap first and list them in a Markdown table, then try to fix it. Please analyze from codes, docs, deep research (online if necessary), and logical analysis. Do not run the "compare_tsl_liulian.py" script for this part. Record the details of the analysis and fix in a Markdown table. Use targeted config parity checks from audit_dataset_configs.py and dataset/model configs under experiments if necessary.
</prompt_in>

---

<prompt_ref>
</prompt_ref>

--------------------------------------------------------

The following are few-shot examples:

<examples>
  <example1>
    <prompt_in>
    - P1: Add missing model config defaults in-model (21 failed runs) — Guard missing config fields with getattr(..., default) in timesnet.py, timemixer.py, and timexer.py for num_kernels, channel_independence, use_norm.
    - fix The 3 remaining failures (non-freq)
    ETTm1_NonstationaryTransformer
    ETTm2_NonstationaryTransformer
    Weather_NonstationaryTransformer
    </prompt_in>

   <prompt_out>
<task_description>
You are an AI coding agent working inside a time-series forecasting codebase. Your task
has two sequential objectives:

  1. Guard missing model configuration fields with safe defaults in timesnet.py,
     timemixer.py, and timexer.py — specifically for the attributes num_kernels,
     channel_independence, and use_norm — so that 21 previously failed runs can execute
     without AttributeError or KeyError crashes.

  2. Diagnose and fix 3 residual failures that are unrelated to frequency settings:
       - ETTm1_NonstationaryTransformer
       - ETTm2_NonstationaryTransformer
       - Weather_NonstationaryTransformer

You must follow the Plan-then-Act structure described below in full. Do not execute
any action before completing the Context Inventory and Plan Construction phases.
</task_description>

---

## Phase A — Context Inventory (complete before planning)

Before writing a single line of code or making any edit, produce the following inventory
as a Markdown table. Each row must be populated from actual inspection of the repository,
not from assumptions.

| # | Item | Location / Value | Relevance |
|---|------|-----------------|-----------|
| 1 | timesnet.py | Path in repo | Contains config fields to be guarded |
| 2 | timemixer.py | Path in repo | Contains config fields to be guarded |
| 3 | timexer.py | Path in repo | Contains config fields to be guarded |
| 4 | NonstationaryTransformer model file | Path in repo | Source of 3 residual failures |
| 5 | Run configuration files for ETTm1, ETTm2, Weather | Paths in repo | Define dataset–model pairings |
| 6 | Existing uses of num_kernels, channel_independence, use_norm | File + line numbers | Baseline for guarding strategy |
| 7 | Reference project (tsl or equivalent) | Path or URL | Used to validate intended default values |
| 8 | Error logs / failure reports for the 21 runs | Path or inline | Confirm which attribute accesses fail |
| 9 | Error logs for the 3 NonstationaryTransformer runs | Path or inline | Characterise non-freq failure mode |
| 10 | Test / evaluation runner script | Path | Used to verify fixes after each change |

Do not proceed to Phase B until every row is populated with a concrete, verified value.

---

## Phase B — Plan Construction (complete before acting)

Using the inventory above, construct a complete numbered plan before executing any step.
Maintain the TODO table below, updating the Status column after every completed step.

### TODO Table

| Step | Description | Status | Depends On |
|------|-------------|--------|------------|
| 1 | Inspect timesnet.py: locate all accesses to num_kernels, channel_independence, use_norm | Pending | Inventory |
| 2 | Inspect timemixer.py: same as Step 1 | Pending | Inventory |
| 3 | Inspect timexer.py: same as Step 1 | Pending | Inventory |
| 4 | Consult reference project to confirm correct default values for each attribute | Pending | Steps 1–3 |
| 5 | Produce fix-plan table: file × attribute × current code × proposed guarded form | Pending | Step 4 |
| 6 | Apply getattr guards in timesnet.py | Pending | Step 5 |
| 7 | Apply getattr guards in timemixer.py | Pending | Step 5 |
| 8 | Apply getattr guards in timexer.py | Pending | Step 5 |
| 9 | Run the 21 previously failing configurations and record outcomes | Pending | Steps 6–8 |
| 10 | Inspect error logs for ETTm1_NonstationaryTransformer | Pending | Inventory |
| 11 | Inspect error logs for ETTm2_NonstationaryTransformer | Pending | Inventory |
| 12 | Inspect error logs for Weather_NonstationaryTransformer | Pending | Inventory |
| 13 | Identify root cause(s) of the 3 NonstationaryTransformer failures | Pending | Steps 10–12 |
| 14 | Produce fix-plan table for NonstationaryTransformer failures | Pending | Step 13 |
| 15 | Implement fixes for NonstationaryTransformer | Pending | Step 14 |
| 16 | Run the 3 NonstationaryTransformer configurations and record outcomes | Pending | Step 15 |
| 17 | Produce final report | Pending | Steps 9, 16 |

Steps 1–9 (config-guard work) and Steps 10–16 (NonstationaryTransformer work) are
independent of each other and may be planned in parallel, but must each be fully
verified before the final report is written.

If any step requires a long-running training or benchmark execution, insert a clearly
labelled BREAKPOINT, show the exact CLI command the user must run, defer that step and
all downstream dependents to the end, and continue with independent work. Resume only
after the user supplies results.

---

## Phase C — Execution (ReAct loop: Perceive → Think → Act → Observe → Reflect)

Execute each step in the plan above. After completing every step, present a post-step
summary in the following Markdown table before proceeding:

| Field | Detail |
|-------|--------|
| Step completed | |
| Files / lines modified | |
| Inputs consumed | |
| Outputs produced | |
| Decisions made (and rationale) | |
| Result meets expectations? (Yes / No / Partial) | |
| Next step | |

If the result does not meet expectations, replan from that point explicitly — do not
continue as though the step succeeded.

---

### Embedded instructions for the config-guard sub-task (Steps 1–9)

**Step 4 — Reference lookup.** For each of the three attributes (num_kernels,
channel_independence, use_norm), check the reference project to determine:
(a) whether a default value is explicitly defined; (b) if yes, what that value is;
(c) if no, what the documented or conventional default is. Present findings as:

| Attribute | Defined in reference? | Reference default | Proposed getattr default |
|-----------|----------------------|------------------|--------------------------|
| num_kernels | | | |
| channel_independence | | | |
| use_norm | | | |

**Step 5 — Fix-plan table.** Before touching any file, present a complete table of
every proposed edit:

| File | Attribute | Current line (exact) | Proposed replacement | Rationale |
|------|-----------|---------------------|---------------------|-----------|
| timesnet.py | num_kernels | | | |
| … | … | | | |

Obtain implicit confirmation (i.e., verify the table is self-consistent) before
proceeding to Steps 6–8.

---

### Embedded instructions for the NonstationaryTransformer sub-task (Steps 10–16)

**Step 13 — Root-cause analysis.** After reading all three error logs, produce:

| Run | Error type | Failing file + line | Probable cause | Category |
|-----|-----------|---------------------|---------------|----------|
| ETTm1_NonstationaryTransformer | | | | |
| ETTm2_NonstationaryTransformer | | | | |
| Weather_NonstationaryTransformer | | | | |

If all three failures share a common root cause, consolidate them into a single fix.
If they differ, treat each independently in Step 14.

**Step 14 — Fix-plan table.** Produce the same style of table as Step 5, scoped to
the NonstationaryTransformer failures.

---

### Final Report (Step 17)

Upon completing all steps, produce a report with the following sections:

1. **Summary** — one paragraph describing what was done and the overall outcome.
2. **Step-by-step outcomes** — a table listing every step, its result, and any
   deviation from the original plan.
3. **Decisions and rationale** — a prose account of any non-trivial choices made
   (e.g., choice of default value, consolidation of fixes).
4. **Outstanding items** — any issues that remain unresolved, deferred, or flagged
   for follow-up.

---

Make a detailed plan for guarding missing configuration attributes in timesnet.py,
timemixer.py, and timexer.py with safe getattr defaults, and for diagnosing and
resolving the three residual NonstationaryTransformer failures across the ETTm1,
ETTm2, and Weather datasets.   
    </prompt_out>
 </example1>
</examples>
