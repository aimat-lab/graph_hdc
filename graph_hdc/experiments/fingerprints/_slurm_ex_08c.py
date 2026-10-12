"""
Scheduler for ex_08_c: the edit-based GED analysis (Figure 3, reviewer comment R2.7) with the HDF default of
2026-10-09: unit-modulus codebooks, bidirectional message passing and total hydrogen counts.

Same configuration as the HDF arm of ex_08_b (``_slurm_ex_08b.py``: 100 seed molecules from ZINC250k, 20
molecules at each of 1-3 edit steps, canonicalized seed SMILES, HDF depth 2 at 32, 128, 512 and 2048 dimensions);
only ``SPECTRUM`` differs (ex_08_b pins the original "gaussian" codebooks). The generated molecule pairs do not
depend on the representation, so the Morgan runs of ex_08_b serve as the Morgan reference (``analyze_ex_08c.py``
checks that the pairs are identical).

    python _slurm_ex_08c.py             # pre-load the dataset, write job scripts, submit
    python _slurm_ex_08c.py --dry-run   # write job scripts only
    python _slurm_ex_08c.py --smoke     # write a small smoke-test job script (prefix ex_08_c_smoke), print its path
"""
import os
import sys

from _slurm_ex_08b import (
    AUTOSLURM_CONFIG, CPUS, EMBEDDING_SIZES, HDC, NUM_THREADS, SMOKE_OVERRIDES, VENV, command, memory, preload,
)

PATH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PATH, '_ex08c')

PREFIX = 'ex_08_c'
EXTRA = {**HDC, 'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'GED_CANONICAL_QUERY': True, 'SPECTRUM': 'unit'}
JOB_TIME = '12:00:00'


def write_job(name: str, line: str) -> str:
    """One bash script per SLURM job: CPU only, NUM_THREADS threads, private TMPDIR, logs the code version."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc from\', graph_hdc.__file__)"\n')
        f.write(f'git -C "{PATH}" rev-parse HEAD\n')
        f.write(f'git -C "{PATH}" status --porcelain --untracked-files=no\n')
        f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def jobs(smoke: bool = False) -> list:
    """(job script path, memory) for every embedding size."""
    prefix = 'ex_08_c_smoke' if smoke else PREFIX
    return [(write_job(f'{prefix}__hdc__{size}',
                       command(prefix, 'hdc', EXTRA, size, SMOKE_OVERRIDES if smoke else None)),
             memory('hdc', size))
            for size in ([32] if smoke else EMBEDDING_SIZES)]


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        for path, _ in jobs(smoke=True):
            print(path)
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter
    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload()
    all_jobs = jobs()
    for mem in sorted({m for _, m in all_jobs}, reverse=True):
        submitter = ASlurmSubmitter(
            config_name=AUTOSLURM_CONFIG,
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'time': JOB_TIME,
                'cpus': str(CPUS),
                'num_threads': str(NUM_THREADS),
                'mem': mem,
                'job_name': 'ex08c_ged',
                'additional_sbatch_configs': '#SBATCH --nodelist=euler',
            },
            archive_path=PATH,
        )
        for path, m in all_jobs:
            if m == mem:
                submitter.add_command(f'bash {path}')
        print(f'mem {mem}: {submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}')
        submitter.submit()
