# TSL vs Liulian Comparison

This page tracks the active comparison inventory in
[`experiments/adapt_tsl_lib/compare_tsl_liulian.py`](../experiments/adapt_tsl_lib/compare_tsl_liulian.py),
not the older 121-pair snapshot that used to be documented here.

## Current State

- Active inventory in code: 188 pairs.
- Persisted result rows in
  [`tsl_comparison_results.json`](../experiments/adapt_tsl_lib/tsl_comparison_results.json):
  184.
- The 4-row gap is the newly added default-backed `Exchange` and `ILI` pairs
  for `TimeMixer` and `TimeXer`. They are active in code now, but they do not
  yet have persisted full-run result rows on this CPU-only machine.
- Stored result summary at the time of this update:
  76 `checked and matched`, 26 `checked but not matched`, 28 `run failed`,
  plus tracked skip classes for models or datasets that are intentionally
  outside the bundled TSL long-term comparison scope.
- Raw artifacts:
  [`tsl_comparison_results.json`](../experiments/adapt_tsl_lib/tsl_comparison_results.json)
  and
  [`tsl_comparison_results.txt`](../experiments/adapt_tsl_lib/tsl_comparison_results.txt).

## Fix Reference Key

Use these short keys in the tables below.

| Key | Fix |
| --- | --- |
| `I1` | [`TimeMixer` inventory expansion](../experiments/adapt_tsl_lib/compare_tsl_liulian.py#L1281) |
| `I2` | [`TimeXer` inventory expansion](../experiments/adapt_tsl_lib/compare_tsl_liulian.py#L1427) |
| `T` | [`TimeMixer` generated default args (`channel_independence`, `decomp_method`, `use_norm`)](../experiments/adapt_tsl_lib/generate_configs.py#L245) |
| `F` | [`freq` backfill from liulian YAML into TSL commands when TSL leaves it implicit](../experiments/adapt_tsl_lib/compare_tsl_liulian.py#L2789) |
| `P` | [liulian reseed before model construction to align RNG order with TSL](../liulian/pipeline.py#L755) |
| `G` | [Bundled TSL `run.py` CPU fallback that clears stale `use_gpu=True`](../refer_projects/Time-Series-Library/run.py#L161) |
| `O` | [OOM fallback profiles used for `Traffic_TimesNet`, `ILI_TimesNet`, and `Traffic_TimeXer`](../experiments/adapt_tsl_lib/compare_tsl_liulian.py#L57) |
| `R` | [Rerun helper commands, including the newly active default-backed pairs](../experiments/adapt_tsl_lib/run_fixed_pairs_compare.sh#L18) |

## Exchange and ILI for TimeMixer and TimeXer

They were missing from the active comparison set because the inventory in
`compare_tsl_liulian.py` had drifted, not because there was a real model-side
reason to exclude them.

- There are no dedicated bundled TSL shell scripts for
  `Exchange_TimeMixer`, `ILI_TimeMixer`, `Exchange_TimeXer`, or
  `ILI_TimeXer`.
- That by itself is not a blocker. The comparison harness already supports
  active default-backed pairs when a dedicated TSL `.sh` script is absent.
- So the old omission was an inventory gap, not an intentional exclusion.

### Coverage Correction

| Model | Active datasets now in code | Script-backed | Default-backed active pairs | Notes |
| --- | --- | --- | --- | --- |
| `TimeMixer` | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `ECL`, `Traffic`, `Exchange`, `ILI` | 7 | `Exchange`, `ILI` | Added in `I1`. These pairs also depend on the explicit generated `TimeMixer` defaults in `T`. |
| `TimeXer` | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Weather`, `ECL`, `Traffic`, `Exchange`, `ILI` | 7 | `Exchange`, `ILI` | Added in `I2`. No separate architectural blocker was found. |

### Newly Active Pairs

These pairs are active in code now but do not yet have persisted full-run
result rows in the JSON snapshot. Representative liulian quick-test execution
was also checked for `Exchange_TimeMixer` and `ILI_TimeXer`; both built
successfully and entered training on this machine.

| Pair | Source mode | Current state | Fix refs |
| --- | --- | --- | --- |
| `Exchange_TimeMixer` | Default-backed TSL args | Added to active inventory; full rerun still pending on this machine | `I1`, `T`, `F`, `R` |
| `ILI_TimeMixer` | Default-backed TSL args | Added to active inventory; full rerun still pending on this machine | `I1`, `T`, `F`, `R` |
| `Exchange_TimeXer` | Default-backed TSL args | Added to active inventory; full rerun still pending on this machine | `I2`, `F`, `R` |
| `ILI_TimeXer` | Default-backed TSL args | Added to active inventory; full rerun still pending on this machine | `I2`, `F`, `R` |

## What The Review Labels Mean

- `Static parity audit; retained remediation path`
  means config parity and the obvious static code-path checks were completed,
  and there is still a concrete next remediation route already identified in
  code or rerun tooling. We use this when another targeted rerun or a bounded
  follow-up fix still makes sense.
- `Config parity + static audit; no safe static-only fix yet`
  means config values, script traces, and static code-path differences have
  already been reconciled as far as we can do safely in this repo. Any
  remaining gap is likely deeper training dynamics, model behavior, or
  environment/runtime effects, so another static patch would be guesswork.

## All Unmatched Pairs

These are the current rows in
[`tsl_comparison_results.json`](../experiments/adapt_tsl_lib/tsl_comparison_results.json)
with status `checked but not matched`.

| Pair | Stored gap | Review label | Current read | Fix refs |
| --- | --- | --- | --- | --- |
| `ETTh1_Informer` | 10.63% | `Config parity + static audit; no safe static-only fix yet` | Shared parity cleanup is already in place; no pair-specific static mismatch remains obvious in the current repo. | `P`, `F` |
| `ETTh2_Informer` | 21.50% | `Config parity + static audit; no safe static-only fix yet` | Same as above. The gap is too large to justify another blind config-only patch. | `P`, `F` |
| `Exchange_Informer` | 10.18% | `Config parity + static audit; no safe static-only fix yet` | Daily `freq` correctness is now enforced for future reruns, but the stored row is still pre-refresh. | `F`, `P` |
| `ETTm2_Autoformer` | 6.73% | `Config parity + static audit; no safe static-only fix yet` | No remaining low-risk static parity delta stands out after the shared cleanup work. | `P`, `F` |
| `Weather_Autoformer` | 9.76% | `Config parity + static audit; no safe static-only fix yet` | Weather is one of the datasets most exposed to the old implicit-`freq` problem; future reruns should use the explicit local value. | `F`, `P` |
| `ECL_Autoformer` | 3.36% | `Static parity audit; retained remediation path` | Stored as unmatched because this is a large-dataset 2-epoch compare path, even though final MSE is close. A fresh synced rerun is still worthwhile. | `P`, `R` |
| `Traffic_Autoformer` | 8.19% | `Static parity audit; retained remediation path` | Large-dataset per-epoch trace divergence remains the main issue; rerun tooling is still the next step. | `P`, `R` |
| `Exchange_Autoformer` | 11.31% | `Config parity + static audit; no safe static-only fix yet` | Daily `freq` backfill is now available for future reruns, but no additional safe model-side static patch is clear. | `F`, `P` |
| `Traffic_TimesNet` | 5.52% | `Static parity audit; retained remediation path` | The stored row predates the TSL CPU-fallback fix and the current rerun pass. It still needs a fresh full rerun on a machine with working CUDA or much more CPU time. | `G`, `O`, `P`, `R` |
| `ILI_TimesNet` | 6.47% | `Static parity audit; retained remediation path` | The stored row was especially suspect: the TSL side had been using an implicit hourly `freq` and was also blocked by the broken device fallback for fresh reruns. Both are fixed now, but a full rerun did not finish within this turn on CPU-only hardware. | `F`, `G`, `O`, `P`, `R` |
| `ETTh1_Transformer` | 33.51% | `Config parity + static audit; no safe static-only fix yet` | Shared parity cleanup is already applied. The remaining gap looks deeper than config plumbing. | `P`, `F` |
| `ETTh2_Transformer` | 29.06% | `Config parity + static audit; no safe static-only fix yet` | Same. No safe static-only patch is obvious. | `P`, `F` |
| `ETTm1_Transformer` | 9.60% | `Config parity + static audit; no safe static-only fix yet` | Minute-data reruns will now inherit explicit local `freq`; beyond that the gap still needs deeper work. | `F`, `P` |
| `ETTm2_Transformer` | 23.88% | `Config parity + static audit; no safe static-only fix yet` | Same as above. | `F`, `P` |
| `Weather_Transformer` | 30.11% | `Config parity + static audit; no safe static-only fix yet` | Weather reruns benefit from explicit `freq`, but the stored gap is still far beyond what a static-only patch can safely explain. | `F`, `P` |
| `Exchange_Transformer` | 53.31% | `Config parity + static audit; no safe static-only fix yet` | Daily `freq` backfill is fixed for future reruns; the stored gap is still too large to treat as a pure config miss. | `F`, `P` |
| `ILI_Transformer` | 11.89% | `Config parity + static audit; no safe static-only fix yet` | Weekly `freq` is now explicit for future reruns, but no other safe static-only fix is obvious. | `F`, `P` |
| `Traffic_TimeXer` | 2.26% | `Static parity audit; retained remediation path` | Final MSE is close, but the stored status is still unmatched because the large-dataset 2-epoch trace diverged. OOM fallback and rerun tooling are already wired for another pass. | `I2`, `O`, `P`, `R` |
| `ETTh1_NonstationaryTransformer` | 5.69% | `Config parity + static audit; no safe static-only fix yet` | This one sits just over threshold after the shared cleanup work; no safe static-only fix is obvious. | `P`, `F` |
| `ETTm2_NonstationaryTransformer` | 6.16% | `Config parity + static audit; no safe static-only fix yet` | Same. | `P`, `F` |
| `Weather_NonstationaryTransformer` | 6.89% | `Config parity + static audit; no safe static-only fix yet` | Same, with Weather now protected against implicit-hourly `freq` on future reruns. | `F`, `P` |
| `Exchange_NonstationaryTransformer` | 40.39% | `Config parity + static audit; no safe static-only fix yet` | Daily `freq` backfill is fixed for future reruns, but the model-side gap still looks deeper than config plumbing. | `F`, `P` |
| `ETTh1_Reformer` | 5.73% | `Config parity + static audit; no safe static-only fix yet` | Shared parity cleanup is already applied; the residual gap remains open. | `P`, `F` |
| `ETTm1_Reformer` | 18.34% | `Config parity + static audit; no safe static-only fix yet` | Minute-data `freq` is now explicit for future reruns; beyond that no safe static-only patch is clear. | `F`, `P` |
| `Exchange_Reformer` | 12.55% | `Config parity + static audit; no safe static-only fix yet` | Daily `freq` backfill is fixed for future reruns, but this still needs deeper investigation. | `F`, `P` |
| `ILI_ETSformer` | 14.38% | `Config parity + static audit; no safe static-only fix yet` | Weekly `freq` is now explicit for future reruns; no additional safe static-only fix is obvious from the current code trace. | `F`, `P` |

## TimesNet Follow-Up

The earlier note "Applied `--disable-es --oom-fallback`" was incomplete by
itself. That pair family still had two additional comparison bugs that could
pollute the result before we even got to true model parity:

1. The bundled TSL launcher could still try `cuda:0` on machines without a
   working CUDA runtime because it never cleared `args.use_gpu`. That is fixed
   in `G`.
2. When a TSL script omitted `--freq`, the compare runner let TSL fall back to
   run.py's hourly default even if the local liulian YAML had the corrected
   dataset cadence. That is fixed in `F`.
3. liulian's model-init RNG stream used to differ from TSL because dataset
   construction happened before model construction. The reseed fix in `P`
   closes that gap.

### Practical Outcome

- `ILI_TimesNet`:
  the stored 6.47% gap should no longer be treated as a clean model-only
  signal. The full rerun path now starts successfully with the corrected weekly
  `freq` and fixed TSL device fallback, but it did not finish within this turn
  on CPU-only hardware.
- `Traffic_TimesNet`:
  the stored 5.52% gap also predates the TSL device-fallback fix. Because it
  is a large-dataset pair and already relies on `O`, the next meaningful step
  is a fresh rerun on a machine with a working CUDA runtime.

### Recommended Rerun Path

Use the commands already listed in
[`run_fixed_pairs_compare.sh`](../experiments/adapt_tsl_lib/run_fixed_pairs_compare.sh#L26).
The most important remaining reruns are:

- `Traffic_TimesNet`
- `ILI_TimesNet`
- `Traffic_TimeXer`
- `Exchange_TimeMixer`
- `ILI_TimeMixer`
- `Exchange_TimeXer`
- `ILI_TimeXer`

## Bottom Line

- `Exchange` and `ILI` were not missing from `TimeMixer` and `TimeXer`
  because of a real technical exclusion. They were missing because the active
  comparison inventory had drifted. That is now fixed in `I1` and `I2`.
- The new unmatched-pairs ledger above replaces the older undocumented
  shorthand by explicitly explaining what each review label means.
- The remaining `TimesNet` mismatch discussion is now more precise:
  `--disable-es --oom-fallback` was not the whole story. The comparison also
  needed the TSL CPU-fallback fix in `G`, the explicit `freq` backfill in `F`,
  and the liulian RNG-order alignment in `P`.
