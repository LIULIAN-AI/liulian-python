#!/usr/bin/env python3
"""UBELIX Paygo cost tracker for the entity-identifier effort.

Records every paid (Paygo) cluster job's *projected* and *actual* cost,
reports session / daily / monthly / total / per-experiment spend, enforces a
hard 10 CHF cap (alarm at 90 %), and derives a calibration factor so future
cost estimates improve as real numbers come in.

Ledger lives in ``jobs/ubelix_cost_ledger.json`` next to this script.

Subcommands
-----------
* ``add``        — log a newly submitted Paygo job (projected cost).
* ``set-actual`` — fill in the real cost once a job has finished.
* ``set-status`` — mark a job ``running`` / ``done`` / ``cancelled``.
* ``scan-cost``  — METHOD 2: scan a job output file for an actual-cost line.
* ``report``     — print the full ledger + spend breakdown + cap status.
* ``can-afford`` — given a projected estimate, say whether it fits the cap.

Two ways to obtain a job's ACTUAL cost
--------------------------------------
**Method 1 — from elapsed time (verified).** UBELIX bills per-minute on
confirmed rates (RTX 4090 = 0.10 CHF/GPU-h). Given ``ElapsedRaw`` seconds
from ``sacct``::

    actual_CHF = GPU_RATE_CHF_PER_H × n_gpu × ElapsedRaw / 3600

This is what the completion watcher uses automatically.

**Method 2 — from job output files (UNVERIFIED).** Some clusters write a
final/billed cost line into the job ``.o<JOBID>`` file via an epilog.
``scan-cost`` greps the output file for such a line. UBELIX prints the cost
*block* at submission time (a prolog), so it may NOT write an actual-cost line
on completion — this method is implemented but must be verified once a real
Paygo job finishes. Use ``set-actual --method logfile`` if it works.

**Resume / preemption.** One experiment (``run_tag``) may consume several
SLURM job IDs — a 12 h walltime times out and is resubmitted, or a preemptable
job is killed and requeued. Each (re)submission is logged as its own job-ID
entry; the **per-experiment total is the sum over all job IDs sharing the
run_tag** (see the "Per experiment" block in ``report``). For a single job ID
that was preempted-and-requeued, sum ``ElapsedRaw`` over all its ``sacct`` rows
before applying Method 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

LEDGER_PATH = Path(__file__).resolve().parent / 'ubelix_cost_ledger.json'

HARD_CAP_CHF = 10.0
WARN_FRACTION = 0.90
WARN_CHF = HARD_CAP_CHF * WARN_FRACTION

# Confirmed UBELIX PAYG rate (intranet price page, 2026-05-15).
GPU_RATE_CHF_PER_H = {'rtx4090': 0.10, 'h100': 0.60}

# Method 2 — candidate patterns for an actual-cost line in a job .o file.
# UNVERIFIED: kept broad on purpose; tighten once a real Paygo job is seen.
_COST_LINE_PATTERNS = [
    re.compile(r'actual\s+cost[^0-9]*([0-9]+\.[0-9]+)', re.I),
    re.compile(r'final\s+cost[^0-9]*([0-9]+\.[0-9]+)', re.I),
    re.compile(r'billed[^0-9]*([0-9]+\.[0-9]+)\s*CHF', re.I),
    re.compile(r'job\s+cost[^0-9]*([0-9]+\.[0-9]+)', re.I),
    re.compile(r'cost\s+of\s+this\s+job[^0-9]*([0-9]+\.[0-9]+)', re.I),
]


def _load() -> dict[str, Any]:
    """Load the ledger JSON, creating an empty one on first use."""
    if not LEDGER_PATH.exists():
        return {
            'currency': 'CHF',
            'hard_cap': HARD_CAP_CHF,
            'warn_threshold': WARN_CHF,
            'jobs': [],
        }
    with LEDGER_PATH.open(encoding='utf-8') as fh:
        return json.load(fh)


def _save(ledger: dict[str, Any]) -> None:
    """Persist the ledger JSON."""
    with LEDGER_PATH.open('w', encoding='utf-8') as fh:
        json.dump(ledger, fh, indent=2, ensure_ascii=False)
        fh.write('\n')


def _effective_cost(job: dict[str, Any]) -> float:
    """Cost to count for a job: actual if known, else projected.

    Cancelled jobs that never ran count as 0 (Paygo bills actual usage only).
    """
    if job.get('status') == 'cancelled' and job.get('actual_chf') is None:
        return 0.0
    actual = job.get('actual_chf')
    if actual is not None:
        return float(actual)
    return float(job.get('projected_chf') or 0.0)


def _calibration_factor(jobs: list[dict[str, Any]]) -> tuple[float, int]:
    """Mean actual/projected ratio over finished jobs; (factor, n_samples)."""
    ratios = [
        float(j['actual_chf']) / float(j['projected_chf'])
        for j in jobs
        if j.get('actual_chf') is not None
        and j.get('projected_chf')
        and float(j['projected_chf']) > 0
    ]
    if not ratios:
        return 1.0, 0
    return sum(ratios) / len(ratios), len(ratios)


def _committed_total(jobs: list[dict[str, Any]]) -> float:
    """Conservative running total: actual where known, else projection."""
    return sum(_effective_cost(j) for j in jobs)


def cost_from_elapsed(elapsed_seconds: float, n_gpu: int = 1,
                      gpu: str = 'rtx4090') -> float:
    """METHOD 1 — actual cost from elapsed time and the confirmed rate."""
    rate = GPU_RATE_CHF_PER_H.get(gpu, 0.10)
    return round(rate * n_gpu * float(elapsed_seconds) / 3600.0, 4)


def actual_cost_from_text(text: str) -> tuple[float | None, str | None]:
    """METHOD 2 — scan a job's output-file text for an explicit cost figure.

    Returns ``(cost_chf, matched_line)`` or ``(None, None)``. UNVERIFIED —
    UBELIX may not emit an actual-cost line into the ``.o`` file (the cost
    block is a submit-time prolog). Verify against a finished Paygo job.
    """
    for raw in text.splitlines():
        for pat in _COST_LINE_PATTERNS:
            m = pat.search(raw)
            if m:
                return float(m.group(1)), raw.strip()
    return None, None


def cmd_add(args: argparse.Namespace) -> None:
    """Log a freshly submitted Paygo job."""
    ledger = _load()
    job = {
        'jobid': args.jobid,
        'run_tag': args.run_tag,
        'account': args.account,
        'wckey': args.wckey,
        'session': args.session,
        'submit_date': args.date or datetime.now().strftime('%Y-%m-%d'),
        'projected_chf': round(float(args.projected), 4),
        'actual_chf': None,
        'cost_method': None,
        'status': 'running',
        'notes': args.notes,
    }
    ledger['jobs'].append(job)
    _save(ledger)
    print(f'[cost-tracker] logged job {args.jobid}: projected {args.projected} CHF')
    _print_cap_status(ledger)


def cmd_set_actual(args: argparse.Namespace) -> None:
    """Fill in a job's real cost after it finished.

    Cost source: --actual <CHF> directly, or --from-log <file> to scan the
    job output file (Method 2). --method records which source was used.
    """
    actual = args.actual
    method = args.method
    if args.from_log:
        text = Path(args.from_log).read_text(encoding='utf-8', errors='replace')
        found, line = actual_cost_from_text(text)
        if found is None:
            sys.exit(
                f'[cost-tracker] Method 2: no cost line found in {args.from_log} '
                f'— fall back to --actual (Method 1, elapsed-time).'
            )
        actual = found
        method = method or 'logfile'
        print(f'[cost-tracker] Method 2 matched: "{line}" -> {actual} CHF')
    if actual is None:
        sys.exit('[cost-tracker] ERROR: provide --actual <CHF> or --from-log <file>')

    ledger = _load()
    for job in ledger['jobs']:
        if job['jobid'] == args.jobid:
            job['actual_chf'] = round(float(actual), 4)
            job['cost_method'] = method or 'manual'
            if job['status'] == 'running':
                job['status'] = 'done'
            _save(ledger)
            print(
                f'[cost-tracker] job {args.jobid} actual cost: {actual} CHF '
                f'(method: {job["cost_method"]})'
            )
            _print_cap_status(ledger)
            return
    sys.exit(f'[cost-tracker] ERROR: jobid {args.jobid} not found in ledger')


def cmd_set_status(args: argparse.Namespace) -> None:
    """Update a job's status."""
    ledger = _load()
    for job in ledger['jobs']:
        if job['jobid'] == args.jobid:
            job['status'] = args.status
            _save(ledger)
            print(f'[cost-tracker] job {args.jobid} status -> {args.status}')
            return
    sys.exit(f'[cost-tracker] ERROR: jobid {args.jobid} not found in ledger')


def cmd_scan_cost(args: argparse.Namespace) -> None:
    """METHOD 2 probe — scan an output file and report any cost line found.

    Does not modify the ledger; use it to verify whether UBELIX writes an
    actual-cost line into job ``.o`` files. If it does, feed the file to
    ``set-actual --from-log``.
    """
    text = Path(args.logfile).read_text(encoding='utf-8', errors='replace')
    found, line = actual_cost_from_text(text)
    if found is None:
        print(
            f'[cost-tracker] Method 2: NO cost line found in {args.logfile}.\n'
            f'  → UBELIX likely does not write actual cost to the .o file; '
            f'use Method 1 (elapsed-time).'
        )
        sys.exit(1)
    print(f'[cost-tracker] Method 2: found cost {found} CHF in line:\n  {line}')


def _print_cap_status(ledger: dict[str, Any]) -> None:
    """Print the cap status line and raise the alarm / stop signal."""
    total = _committed_total(ledger['jobs'])
    pct = total / HARD_CAP_CHF * 100
    print(
        f'[cost-tracker] committed total: {total:.4f} / {HARD_CAP_CHF:.2f} CHF '
        f'({pct:.1f} %)'
    )
    if total >= HARD_CAP_CHF:
        print('  🛑 STOP: hard cap reached — cancel/stop all Paygo jobs now.')
    elif total >= WARN_CHF:
        print(
            f'  ⚠️  ALARM: ≥{WARN_FRACTION:.0%} of cap spent — ask the user '
            f'whether to raise the {HARD_CAP_CHF:.0f} CHF limit before submitting more.'
        )
    else:
        print(f'  ✅ OK: {HARD_CAP_CHF - total:.4f} CHF headroom remaining.')


def cmd_report(args: argparse.Namespace) -> None:
    """Print the full ledger and spend breakdown."""
    ledger = _load()
    jobs = ledger['jobs']
    print('=' * 78)
    print('UBELIX Paygo cost ledger')
    print('=' * 78)
    if not jobs:
        print('(no Paygo jobs logged yet)')
        return

    hdr = '{:>10} {:>9} {:>9} {:>8} {:>9} {:>10} {:>12}'
    print(hdr.format('jobid', 'proj', 'actual', 'method', 'status', 'date', 'run_tag'))
    for j in jobs:
        print(
            hdr.format(
                str(j['jobid']),
                f"{j.get('projected_chf', 0):.3f}",
                'n/a' if j.get('actual_chf') is None else f"{j['actual_chf']:.3f}",
                str(j.get('cost_method') or '-'),
                str(j.get('status', '?')),
                str(j.get('submit_date', '?')),
                str(j.get('run_tag', ''))[:12],
            )
        )

    # breakdowns
    by_day: dict[str, float] = {}
    by_month: dict[str, float] = {}
    by_session: dict[str, float] = {}
    by_run_tag: dict[str, float] = {}
    for j in jobs:
        c = _effective_cost(j)
        d = str(j.get('submit_date', '?'))
        by_day[d] = by_day.get(d, 0.0) + c
        by_month[d[:7]] = by_month.get(d[:7], 0.0) + c
        by_session[str(j.get('session', 'default'))] = (
            by_session.get(str(j.get('session', 'default')), 0.0) + c
        )
        by_run_tag[str(j.get('run_tag', '?'))] = (
            by_run_tag.get(str(j.get('run_tag', '?')), 0.0) + c
        )

    print('-' * 78)
    print('Per day:    ', {k: round(v, 3) for k, v in sorted(by_day.items())})
    print('Per month:  ', {k: round(v, 3) for k, v in sorted(by_month.items())})
    print('Per session:', {k: round(v, 3) for k, v in sorted(by_session.items())})
    print('Per experiment (run_tag — sums resume/requeue job IDs):')
    for tag, v in sorted(by_run_tag.items()):
        print(f'    {tag:<40} {v:.3f} CHF')

    factor, n = _calibration_factor(jobs)
    print('-' * 78)
    print(
        f'Calibration factor: {factor:.4f}  (from {n} finished job(s); '
        f'calibrated_estimate = sbatch_projected × {factor:.4f})'
    )
    _print_cap_status(ledger)


def cmd_can_afford(args: argparse.Namespace) -> None:
    """Decide whether a new job of the given projected cost fits the cap."""
    ledger = _load()
    jobs = ledger['jobs']
    total = _committed_total(jobs)
    factor, n = _calibration_factor(jobs)
    calibrated = float(args.estimate) * factor
    after = total + calibrated
    print(
        f'[cost-tracker] current committed: {total:.3f} CHF | '
        f'new job projected {args.estimate} CHF × calib {factor:.3f} '
        f'= {calibrated:.3f} CHF | total after: {after:.3f} / {HARD_CAP_CHF:.2f}'
    )
    if after >= HARD_CAP_CHF:
        print('  🛑 DENY: would reach/exceed hard cap. Do not submit.')
        sys.exit(2)
    if after >= WARN_CHF:
        print('  ⚠️  ALARM: would cross 90 % — ask user to confirm before submitting.')
        sys.exit(1)
    print('  ✅ ALLOW: within budget.')
    sys.exit(0)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description='UBELIX Paygo cost tracker.')
    sub = p.add_subparsers(dest='cmd', required=True)

    a = sub.add_parser('add', help='log a submitted Paygo job')
    a.add_argument('--jobid', required=True)
    a.add_argument('--run-tag', default='')
    a.add_argument('--projected', required=True, type=float)
    a.add_argument('--account', default='paygo')
    a.add_argument('--wckey', default='inf_prg-research')
    a.add_argument('--session', default='entity-id-2026-05')
    a.add_argument('--date', default='')
    a.add_argument('--notes', default='')
    a.set_defaults(func=cmd_add)

    sa = sub.add_parser('set-actual', help='record a job real cost')
    sa.add_argument('--jobid', required=True)
    sa.add_argument('--actual', type=float, default=None,
                    help='Method 1: explicit CHF (e.g. from elapsed-time calc)')
    sa.add_argument('--from-log', default='',
                    help='Method 2: scan this job output file for a cost line')
    sa.add_argument('--method', default='',
                    choices=['', 'elapsed', 'logfile', 'manual'],
                    help='how the cost was obtained (recorded in the ledger)')
    sa.set_defaults(func=cmd_set_actual)

    ss = sub.add_parser('set-status', help='update a job status')
    ss.add_argument('--jobid', required=True)
    ss.add_argument('--status', required=True,
                    choices=['running', 'done', 'cancelled'])
    ss.set_defaults(func=cmd_set_status)

    sc = sub.add_parser('scan-cost', help='Method 2 probe: scan an output file')
    sc.add_argument('--logfile', required=True)
    sc.set_defaults(func=cmd_scan_cost)

    r = sub.add_parser('report', help='print ledger + breakdown')
    r.set_defaults(func=cmd_report)

    c = sub.add_parser('can-afford', help='check a projected cost against the cap')
    c.add_argument('--estimate', required=True, type=float)
    c.set_defaults(func=cmd_can_afford)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
