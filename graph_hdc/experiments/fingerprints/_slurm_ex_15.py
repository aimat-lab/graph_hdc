"""
Scheduler for Experiment 15 (ex_15): does bidirectional message passing change the HDF results?

The fingerprint experiments so far built HDF with messages along one direction of every bond only: the graph
dicts of ``graph_dict_from_mol`` store each bond once and ``HyperNet`` ran with ``bidirectional=False``, so
every atom only aggregated the neighbors with a higher atom index. This makes the fingerprint depend on the
atom order of the input SMILES. The HDF scripts now have a ``BIDIRECTIONAL`` parameter (default True).

This experiment runs the HDF configuration of ex_14 (D=2048, L=2, MLP + KNN) on the five small regression
targets, once with ``BIDIRECTIONAL=False`` (old behavior) and once with ``True`` (fixed). Both arms of a seed
use the same 80/10/10 split, the same codebooks and the same MLP seed, so ``analyze_ex_15.py`` can compare
them run by run. The splits are NOT those of ex_14: with caching disabled, the dataset order is shuffled per
seed, whereas ex_14 used a seed-independent cached order. Compare ex_15 with ex_14 only in aggregate.

Caching is disabled so that both arms encode with the current code and no job writes shared cache files.

Each SLURM job runs one group script from ``_ex15/jobs/`` (both arms of two seeds of one dataset, one after
the other). A group script trains on the CPU only (``CUDA_VISIBLE_DEVICES=``: the GPU of euler is shared with
other work), uses NUM_THREADS threads (more threads thrash on the shared machine) and its own ``TMPDIR`` (chem_mat_data rewrites its metadata file in the temp folder on every
load). The datasets are downloaded to /tmp once, sequentially, before submitting, because concurrent first
downloads of the same file can corrupt it. Jobs are pinned to euler (config ``euler_3`` with CPUS=4,
NUM_THREADS=2: six jobs side by side).

    python _slurm_ex_15.py             # pre-load datasets, write group scripts, submit
    python _slurm_ex_15.py --dry-run   # write group scripts and job scripts only
    python _slurm_ex_15.py --smoke     # write one group script (FreeSolv, seeds 0-1, prefix ex_15_smoke)
                                       # and print its path, to run it locally with bash
"""
import os
import sys

from _slurm_ex_14 import DATASETS, VARIANTS, _command

PATH = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.abspath(os.path.join(PATH, '..', '..', '..', '.venv'))
OUT = os.path.join(PATH, '_ex15')

# 'ex_15_bidir' was the first submission with 8 threads per run; it was cancelled after 24 of 100 runs because
# the MLP training thrashed (euler is shared with other CPU-heavy work: 422 ms per step with 8 threads vs. 3 ms
# with 2). Its finished runs are kept as a check of how much the thread count alone changes the results.
PREFIX = 'ex_15_bidir_t2'
PREFIX_SMOKE = 'ex_15_smoke_t2'
SEEDS = list(range(10))
# the five small experimental targets of ex_14 (the large quantum-chemistry datasets are left out)
SMALL = ['freesolv_hfe', 'aqsoldb_logs', 'lipophilicity_logD', 'bace_ic50', 'hopv15_pce']
DATASETS_SMALL = [ds for ds in DATASETS if ds[0] in SMALL]

AUTOSLURM_CONFIG = 'euler_3'
# threads per run (torch, BLAS) and CPUs per job: small jobs side by side instead of a few wide ones
NUM_THREADS = 2
CPUS = 4
# seeds per group script (each seed runs both arms)
SEEDS_PER_GROUP = 2


def commands(prefix: str, ds: tuple, seeds: list) -> list:
    """Both arms for every seed of one dataset, next to each other."""
    module, params = VARIANTS['hdf']
    return [
        # ex_15 compares the edge direction with the hydrogen count of its archives (implicit, before the
        # HYDROGEN_COUNT parameter existed), so it pins that count instead of taking the new 'total' default
        _command(module, prefix, seed, ds, {**params, 'BIDIRECTIONAL': bidirectional, 'HYDROGEN_COUNT': 'implicit',
                                            '__CACHING__': False})
        for seed in seeds
        for bidirectional in (False, True)
    ]


def write_group(name: str, lines: list) -> str:
    """One bash script per SLURM job: CPU only, private TMPDIR, commands one after the other."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        for line in lines:
            f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def groups(prefix: str, datasets: list, seeds: list) -> list:
    paths = []
    for ds in datasets:
        for i in range(0, len(seeds), SEEDS_PER_GROUP):
            chunk = seeds[i:i + SEEDS_PER_GROUP]
            name = f'{prefix}__{ds[0]}__seeds_{chunk[0]}-{chunk[-1]}'
            paths.append(write_group(name, commands(prefix, ds, chunk)))
    return paths


def preload(datasets: list):
    """Download every dataset to /tmp once, one after the other (same call as predict_molecules.py)."""
    from chem_mat_data.main import load_graph_dataset
    for name in sorted({ds[1] for ds in datasets}):
        graphs = load_graph_dataset(name, folder_path='/tmp')
        print(f'pre-loaded {name}: {len(graphs)} graphs')


if __name__ == '__main__':

    if '--smoke' in sys.argv:
        ds = [d for d in DATASETS_SMALL if d[0] == 'freesolv_hfe']
        print(groups(PREFIX_SMOKE, ds, [0, 1])[0])
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload(DATASETS_SMALL)

    paths = groups(PREFIX, DATASETS_SMALL, SEEDS)
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
            'mem': '16000mb',
            'job_name': 'ex15_bidir',
            'additional_sbatch_configs': '#SBATCH --nodelist=euler',
        },
        archive_path=PATH,
    )
    for path in paths:
        submitter.add_command(f'bash {path}')

    print(f'{2 * len(SEEDS) * len(DATASETS_SMALL)} runs ({len(DATASETS_SMALL)} datasets x {len(SEEDS)} seeds '
          f'x 2 arms) in {submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}')
    submitter.submit()
