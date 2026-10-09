"""
Scheduler for Experiment 20 (ex_20): bit-depth ablation of HDF (reviewer comment R2.2).

Morgan fingerprints store one bit per dimension, HDF vectors one float32 (32 bits). This experiment measures how
many bits per dimension HDF actually needs: the HDF vectors (D=2048, L=2, bidirectional, total hydrogen counts)
are quantized to 16, 8, 4, 2 and 1 bits per dimension with equal-frequency bins fitted on the training split
(``QUANTIZE_BITS``, see ``quantize_equal_frequency``), and the MLP of the other experiments (neural_net2,
(100, 100), lr 1e-3) is trained on the quantized vectors. Every bit depth of a seed uses the same split, the same
codebooks and the same MLP seed as the float32 run (QUANTIZE_BITS=None) of that seed, so ``analyze_ex_20.py`` can
report the MAE relative to float32 seed by seed.

Any change of the inputs changes the training trajectory of the MLP, so even a nearly lossless quantization does
not reproduce the float32 MAE exactly. As a reference for this run-to-run variation, every seed has one more
float32 run with a different network seed (``NN_SEED`` = 1000 + SEED: other weight initialization, internal
validation split and batch order; same split and codebooks).

16 bits give more levels than there are training molecules in all three datasets, so the training values stay
unchanged (only validation and test values are snapped to the nearest training value). The 16-bit arm is a check of
the procedure, not a compression.

Datasets: FreeSolv, AqSolDB and BACE; seeds 0-9. Each SLURM job runs one group script from ``_ex20/jobs/`` (the
seven runs of one dataset and seed, one after the other). Job setup as in ex_15: CPU only, 2 threads, 4 CPUs per
job, private TMPDIR, caching disabled (no job writes shared cache files), datasets pre-loaded to /tmp before
submitting. The group scripts put the repository that contains this file first on PYTHONPATH, so a run from a git
worktree uses the worktree's graph_hdc package (the venv's editable install points to the main checkout).

    python _slurm_ex_20.py             # pre-load datasets, write group scripts, submit
    python _slurm_ex_20.py --dry-run   # write group scripts and job scripts only
    python _slurm_ex_20.py --smoke     # write one group script (FreeSolv, seed 0, prefix ex_20_smoke) and print
                                       # its path, to run it locally with bash
"""
import os
import sys

from _slurm_ex_14 import DATASETS, VARIANTS, _command
from _slurm_ex_15 import preload

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(PATH, '_ex20')
# the venv lives in the main checkout only (a worktree has none)
VENV = os.path.join(REPO, '.venv')
if not os.path.isdir(VENV):
    VENV = '/media/ssd2/Programming/graph_hdc/.venv'

PREFIX = 'ex_20_bits'
PREFIX_SMOKE = 'ex_20_smoke'
SEEDS = list(range(10))
BITS = [None, 16, 8, 4, 2, 1]
NN_SEED_OFFSET = 1000
SELECTED = ['freesolv_hfe', 'aqsoldb_logs', 'bace_ic50']
DATASETS_BITS = [ds for ds in DATASETS if ds[0] in SELECTED]

AUTOSLURM_CONFIG = 'euler_3'
NUM_THREADS = 2
CPUS = 4


def commands(prefix: str, ds: tuple, seed: int) -> list:
    module, params = VARIANTS['hdf']
    params = {**params, 'MODELS': ['neural_net2'], '__CACHING__': False}
    lines = [_command(module, prefix, seed, ds, {**params, 'QUANTIZE_BITS': bits}) for bits in BITS]
    # float32 with another network seed: the run-to-run variation of the MLP training itself
    lines.append(_command(module, prefix, seed, ds, {**params, 'QUANTIZE_BITS': None,
                                                     'NN_SEED': NN_SEED_OFFSET + seed}))
    return lines


def write_group(name: str, lines: list) -> str:
    """One bash script per SLURM job: CPU only, private TMPDIR, this repository first on PYTHONPATH."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export PYTHONPATH="{REPO}${{PYTHONPATH:+:$PYTHONPATH}}"\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc from\', graph_hdc.__file__)"\n')
        f.write(f'git -C "{REPO}" rev-parse HEAD\n')
        for line in lines:
            f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def groups(prefix: str, datasets: list, seeds: list) -> list:
    return [write_group(f'{prefix}__{ds[0]}__seed_{seed}', commands(prefix, ds, seed))
            for ds in datasets for seed in seeds]


if __name__ == '__main__':

    if '--smoke' in sys.argv:
        ds = [d for d in DATASETS_BITS if d[0] == 'freesolv_hfe']
        print(groups(PREFIX_SMOKE, ds, [0])[0])
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload(DATASETS_BITS)

    paths = groups(PREFIX, DATASETS_BITS, SEEDS)
    submitter = ASlurmSubmitter(
        config_name=AUTOSLURM_CONFIG,
        batch_size=1,
        randomize=False,
        dry_run=dry_run,
        overwrite_fillers={
            'venv': VENV,
            'cwd': PATH,
            'time': '12:00:00',
            'cpus': str(CPUS),
            'num_threads': str(NUM_THREADS),
            'mem': '6000mb',
            'job_name': 'ex20_bits',
            'additional_sbatch_configs': '#SBATCH --nodelist=euler',
        },
        archive_path=PATH,
    )
    for path in paths:
        submitter.add_command(f'bash {path}')

    runs = (len(BITS) + 1) * len(SEEDS) * len(DATASETS_BITS)
    print(f'{runs} runs ({len(DATASETS_BITS)} datasets x {len(SEEDS)} seeds x ({len(BITS)} bit depths + 1 re-seeded '
          f'float32)) in {submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}, venv {VENV}')
    submitter.submit()
