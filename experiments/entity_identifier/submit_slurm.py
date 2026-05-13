#!/usr/bin/env python3
"""Generate and submit Slurm jobs for entity-identifier matrix runs."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

if __package__ in (None, ''):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from experiments.entity_identifier.matrix import (
    DATASETS,
    MODELS,
    MODES,
    DEFAULT_SEEDS,
    iter_jobs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARD_CODED_MAIL_USER = 'jajupmochi@gmail.com'
HARD_CODED_SLURM_USER = 'lj24u267'


def build_sbatch_script(
    *,
    job_name: str,
    command: str,
    project_root: Path,
    use_gpu: bool,
    gres: str,
    partition_gpu: str,
    partition_cpu: str,
    qos: str,
    account: str,
    time_limit: str,
    cpus_per_task: int,
    mem_per_cpu: str,
    python_module: str,
    venv_path: str,
    mail_user: str,
) -> str:
    """Build one sbatch script body.

    Args:
        job_name: Slurm job name.
        command: Command line executed inside the job.
        project_root: Repository root.
        use_gpu: Whether to request a GPU node.
        gres: GPU generic resource spec (e.g. ``gpu:1``, ``gpu:rtx4090:1``).
        partition_gpu: GPU partition name.
        partition_cpu: CPU partition name.
        qos: Slurm QoS.
        account: Slurm account.
        time_limit: Slurm time limit.
        cpus_per_task: CPU count per task.
        mem_per_cpu: Memory per CPU.
        python_module: Cluster module to load.
        venv_path: Virtualenv activation path.
        mail_user: Notification email.

    Returns:
        Fully rendered sbatch script text.
    """
    lines: list[str] = [
        '#!/bin/bash',
        f'#SBATCH --job-name="{job_name}"',
        '#SBATCH --mail-type=ALL',
        f'#SBATCH --mail-user={mail_user}',
        f'#SBATCH --output="{project_root}/outputs/{job_name}.o%J"',
        f'#SBATCH --error="{project_root}/errors/{job_name}.e%J"',
        f'#SBATCH --account={account}',
        f'#SBATCH --qos={qos}',
        f'#SBATCH --time={time_limit}',
        f'#SBATCH --cpus-per-task={cpus_per_task}',
        f'#SBATCH --mem-per-cpu={mem_per_cpu}',
    ]

    if use_gpu:
        lines.extend(
            [
                f'#SBATCH --partition={partition_gpu}',
                f'#SBATCH --gres={gres}',
            ]
        )
    else:
        lines.append(f'#SBATCH --partition={partition_cpu}')

    lines.extend(
        [
            '',
            f'module load {python_module}',
            f'source "{venv_path}"',
            'python3 --version',
            'module list',
            f'cd "{project_root}"',
            'echo Working directory: $PWD',
            command,
            '',
        ]
    )
    return '\n'.join(lines)


def check_job_exists(script: str, user: str = HARD_CODED_SLURM_USER) -> bool:
    """Return True if a job with same --job-name is already in queue."""
    match = re.search(r'--job-name="([^"]+)"', script)
    if match is None:
        return False
    job_name = match.group(1)
    try:
        output = subprocess.check_output(['squeue', '-u', user], text=True)
    except Exception:
        return False
    return job_name in output


def _build_matrix_command(
    *,
    python_bin: str,
    run_tag: str,
    dataset: str,
    model: str,
    mode: str,
    seed: int,
    hpo_num_samples: int | None,
    hpo_max_concurrent: int | None,
    hpo_resources_gpu: float | None,
    max_train_samples: int | None,
    plot_aggregations: list[str] | None,
    plot_formats: list[str] | None,
    plot_file_stem: str | None,
) -> str:
    """Build matrix runner command for one Slurm job."""
    cmd = [
        python_bin,
        'experiments/entity_identifier/run.py',
        '--phase',
        'full',
        '--run-tag',
        run_tag,
        '--datasets',
        dataset,
        '--models',
        model,
        '--modes',
        mode,
        '--seeds',
        str(seed),
        '--resume',
        '--skip-compare',
    ]
    if hpo_num_samples is not None:
        cmd.extend(['--hpo-num-samples', str(hpo_num_samples)])
    if hpo_max_concurrent is not None:
        cmd.extend(['--hpo-max-concurrent', str(hpo_max_concurrent)])
    if hpo_resources_gpu is not None:
        cmd.extend(['--hpo-resources-gpu', str(hpo_resources_gpu)])
    if max_train_samples is not None:
        cmd.extend(['--max-train-samples', str(max_train_samples)])
    if plot_aggregations:
        cmd.extend(['--plot-aggregations', *plot_aggregations])
    if plot_formats:
        cmd.extend(['--plot-formats', *plot_formats])
    if plot_file_stem:
        cmd.extend(['--plot-file-stem', plot_file_stem])
    return ' '.join(cmd)


def _build_single_matrix_command(
    *,
    python_bin: str,
    run_tag: str,
    datasets: list[str] | None,
    models: list[str] | None,
    modes: list[str] | None,
    seeds: list[int] | None,
    max_jobs: int | None,
    hpo_num_samples: int | None,
    hpo_max_concurrent: int | None,
    hpo_resources_gpu: float | None,
    max_train_samples: int | None,
    plot_aggregations: list[str] | None,
    plot_formats: list[str] | None,
    plot_file_stem: str | None,
) -> str:
    """Build a single command that runs the full selected matrix in one job."""
    resolved_datasets = datasets if datasets else list(DATASETS)
    resolved_models = models if models else list(MODELS)
    resolved_modes = modes if modes else list(MODES)
    resolved_seeds = seeds if seeds else list(DEFAULT_SEEDS)

    cmd = [
        python_bin,
        'experiments/entity_identifier/run.py',
        '--phase',
        'full',
        '--run-tag',
        run_tag,
        '--resume',
    ]
    cmd.extend(['--datasets', *resolved_datasets])
    cmd.extend(['--models', *resolved_models])
    cmd.extend(['--modes', *resolved_modes])
    cmd.extend(['--seeds', *[str(seed) for seed in resolved_seeds]])
    if max_jobs is not None:
        cmd.extend(['--max-jobs', str(max_jobs)])
    if hpo_num_samples is not None:
        cmd.extend(['--hpo-num-samples', str(hpo_num_samples)])
    if hpo_max_concurrent is not None:
        cmd.extend(['--hpo-max-concurrent', str(hpo_max_concurrent)])
    if hpo_resources_gpu is not None:
        cmd.extend(['--hpo-resources-gpu', str(hpo_resources_gpu)])
    if max_train_samples is not None:
        cmd.extend(['--max-train-samples', str(max_train_samples)])
    if plot_aggregations:
        cmd.extend(['--plot-aggregations', *plot_aggregations])
    if plot_formats:
        cmd.extend(['--plot-formats', *plot_formats])
    if plot_file_stem:
        cmd.extend(['--plot-file-stem', plot_file_stem])
    return ' '.join(cmd)


def submit_jobs(args: argparse.Namespace) -> None:
    """Submit matrix jobs to Slurm based on CLI options."""
    (PROJECT_ROOT / 'outputs').mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'errors').mkdir(parents=True, exist_ok=True)

    if args.dispatch_mode == 'single-matrix':
        job_name = f'eid.matrix.{args.run_tag}'
        command = _build_single_matrix_command(
            python_bin=args.python_bin,
            run_tag=args.run_tag,
            datasets=args.datasets,
            models=args.models,
            modes=args.modes,
            seeds=args.seeds,
            max_jobs=args.max_jobs,
            hpo_num_samples=args.hpo_num_samples,
            hpo_max_concurrent=args.hpo_max_concurrent,
            hpo_resources_gpu=args.hpo_resources_gpu,
            max_train_samples=args.max_train_samples,
            plot_aggregations=args.plot_aggregations,
            plot_formats=args.plot_formats,
            plot_file_stem=args.plot_file_stem,
        )
        script = build_sbatch_script(
            job_name=job_name,
            command=command,
            project_root=PROJECT_ROOT,
            use_gpu=not args.cpu_only,
            gres=args.gres,
            partition_gpu=args.partition_gpu,
            partition_cpu=args.partition_cpu,
            qos=args.qos,
            account=args.account,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            python_module=args.python_module,
            venv_path=args.venv_path,
            mail_user=args.mail_user,
        )
        if args.skip_existing and check_job_exists(script, user=args.slurm_user):
            print(f'[skip] already queued: {job_name}')
            return
        if args.dry_run:
            print('\n' + '=' * 80)
            print(script)
            print('=' * 80)
            return
        result = subprocess.run(
            ['sbatch'],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f'sbatch failed for {job_name}: {stderr}')
        print(f'[submitted] {job_name}: {result.stdout.strip()}')
        print('Jobs submitted: 1')
        return

    jobs = iter_jobs(
        datasets=args.datasets,
        models=args.models,
        modes=args.modes,
        seeds=args.seeds,
    )
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    submitted = 0
    for job in jobs:
        job_name = f'eid.{job.folder_name}'
        command = _build_matrix_command(
            python_bin=args.python_bin,
            run_tag=args.run_tag,
            dataset=job.dataset,
            model=job.model,
            mode=job.mode,
            seed=job.seed,
            hpo_num_samples=args.hpo_num_samples,
            hpo_max_concurrent=args.hpo_max_concurrent,
            hpo_resources_gpu=args.hpo_resources_gpu,
            max_train_samples=args.max_train_samples,
            plot_aggregations=args.plot_aggregations,
            plot_formats=args.plot_formats,
            plot_file_stem=args.plot_file_stem,
        )
        script = build_sbatch_script(
            job_name=job_name,
            command=command,
            project_root=PROJECT_ROOT,
            use_gpu=not args.cpu_only,
            gres=args.gres,
            partition_gpu=args.partition_gpu,
            partition_cpu=args.partition_cpu,
            qos=args.qos,
            account=args.account,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            python_module=args.python_module,
            venv_path=args.venv_path,
            mail_user=args.mail_user,
        )
        if args.skip_existing and check_job_exists(script, user=args.slurm_user):
            print(f'[skip] already queued: {job_name}')
            continue

        if args.dry_run:
            print('\n' + '=' * 80)
            print(script)
            print('=' * 80)
            continue

        result = subprocess.run(
            ['sbatch'],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f'sbatch failed for {job_name}: {stderr}')
        print(f'[submitted] {job_name}: {result.stdout.strip()}')
        submitted += 1

    print(f'Jobs submitted: {submitted}')
    print(
        'After cluster jobs finish, run:\n'
        f'  {args.python_bin} experiments/entity_identifier/compare.py '
        f'--run-root artifacts/entity_identifier/{args.run_tag}'
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Slurm submission utility."""
    parser = argparse.ArgumentParser(
        description='Submit full entity-identifier matrix runs to Slurm.',
    )
    parser.add_argument('--run-tag', required=True, help='Shared run tag.')
    parser.add_argument(
        '--dispatch-mode',
        choices=('per-experiment', 'single-matrix'),
        default='per-experiment',
        help=(
            'Submission strategy: per-experiment submits one Slurm job per '
            'dataset/model/mode/seed (recommended for GPU parallelism and '
            'failure isolation); single-matrix submits one Slurm job that '
            'loops the selected matrix.'
        ),
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=None,
        help='Subset of datasets.',
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=None,
        help='Subset of models.',
    )
    parser.add_argument(
        '--modes',
        nargs='*',
        default=None,
        help='Subset of modes.',
    )
    parser.add_argument(
        '--seeds',
        nargs='*',
        type=int,
        default=list(DEFAULT_SEEDS),
        help='Seeds to submit.',
    )
    parser.add_argument('--python-bin', default='.venv/bin/python')
    parser.add_argument('--hpo-num-samples', type=int, default=None)
    parser.add_argument('--hpo-max-concurrent', type=int, default=None)
    parser.add_argument('--hpo-resources-gpu', type=float, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument(
        '--plot-aggregations',
        nargs='*',
        default=['mean', 'median', 'best', 'last'],
    )
    parser.add_argument('--plot-formats', nargs='*', default=['png'])
    parser.add_argument('--plot-file-stem', default='pred_vs_gt')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--max-jobs', type=int, default=None)

    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument(
        '--gres',
        default='gpu:1',
        help='GPU generic resource spec (e.g. gpu:1, gpu:rtx4090:1, gpu:a100:2).',
    )
    parser.add_argument('--partition-gpu', default='gpu')
    parser.add_argument('--partition-cpu', default='epyc2,bdw')
    parser.add_argument('--qos', default='job_gratis')
    parser.add_argument('--account', default='gratis')
    parser.add_argument('--time-limit', default='24:00:00')
    parser.add_argument('--cpus-per-task', type=int, default=4)
    parser.add_argument('--mem-per-cpu', default='10G')
    parser.add_argument(
        '--python-module', default='Python/3.12.3-GCCcore-13.3.0'
    )
    parser.add_argument('--venv-path', default=str(PROJECT_ROOT / '.venv' / 'bin' / 'activate'))
    parser.add_argument('--mail-user', default=HARD_CODED_MAIL_USER)
    parser.add_argument('--slurm-user', default=HARD_CODED_SLURM_USER)
    return parser


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()
    submit_jobs(args)


if __name__ == '__main__':
    main()
