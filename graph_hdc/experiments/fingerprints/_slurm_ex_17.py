"""
Scheduler for Experiment 17 (ex_17): exact graph edit distance vs fingerprint distances (reviewer comment R2.7).

The main text correlates fingerprint distances with the number of edit steps between seed molecules and
molecules generated from them (ex_08). This experiment correlates them with the exact GED instead, for
randomly sampled pairs of molecules small enough for an exact computation:

- QM9 (at most 9 heavy atoms): 10,000 random pairs, and the near pairs (GED <= 4) among 40,000 screened
  random pairs (2.5% hit rate in the probe, so about 1,000 near pairs).
- ZINC250k molecules with at most 12 heavy atoms (2,196 molecules): 2,000 random pairs, and the near pairs
  among 220,000 screened random pairs (about 24 CPU hours at the probed 2.6 pairs per CPU second; the probe
  found 4 near pairs in 1,564, so roughly 550 near pairs, with a wide margin).

Stage 1 (this scheduler) computes the GED values with ``ged_exact_pairs.py``, one SLURM job per chunk of a
pair set. Each job runs NUM_WORKERS single-threaded worker processes on CPUS cores of euler, with its own
TMPDIR (chem_mat_data rewrites its metadata file in the temp folder on every load). The datasets are loaded
once, sequentially, before submitting, so that no two jobs download the same file at the same time.
Stage 2 runs once all chunks are done; it takes minutes and is started manually:
``OMP_NUM_THREADS=2 python ged_exact_correlation.py --__DEBUG__="False" --__PREFIX__="'ex_17'" --PAIRS_PREFIX="'ex_17'"``.

Runtime probes (graph_hdc/experiments/fingerprints/_ged_timing_probe/, 2026-10-07/08, euler under load):
random ZINC <= 12 pairs: median 122 s, mean 334 s, max 2750 s (96 of 96 exact within 1 h); random QM9
pairs: median 2 s, max 7 s; near-pair checks with upper_bound=4: 5.4 (QM9) and 2.6 (ZINC) pairs per CPU
second, slowest check 42 s.

    python _slurm_ex_17.py             # pre-load datasets, write job scripts, submit
    python _slurm_ex_17.py --dry-run   # write job scripts only
    python _slurm_ex_17.py --smoke     # write small smoke-test job scripts (prefix ex_17_smoke), print paths
"""
import os
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.abspath(os.path.join(PATH, '..', '..', '..', '.venv'))
OUT = os.path.join(PATH, '_ex17')

PREFIX = 'ex_17'
PREFIX_SMOKE = 'ex_17_smoke'
PAIR_SEED = 0

AUTOSLURM_CONFIG = 'euler_3'
CPUS = 4
NUM_WORKERS = 4

# (dataset, max heavy atoms, pair set, number of pairs, number of chunks, timeout per pair in s, job time limit)
# Submitted in this order: the short QM9 sets first, so that the whole pipeline can be checked on them early.
PAIR_SETS = [
    ('qm9_smiles', 9, 'random', 10_000, 5, 600.0, '04:00:00'),
    ('qm9_smiles', 9, 'near', 40_000, 2, 600.0, '04:00:00'),
    ('zinc250k', 12, 'random', 2_000, 40, 7200.0, '12:00:00'),
    ('zinc250k', 12, 'near', 220_000, 10, 600.0, '08:00:00'),
]

# Small versions of every pair set for the local smoke test (same code paths, a few pairs each).
PAIR_SETS_SMOKE = [
    ('zinc250k', 12, 'random', 8, 2, 60.0, '00:30:00'),
    ('qm9_smiles', 9, 'random', 40, 2, 60.0, '00:30:00'),
    ('qm9_smiles', 9, 'near', 400, 2, 60.0, '00:30:00'),
    ('zinc250k', 12, 'near', 400, 2, 60.0, '00:30:00'),
]


def command(prefix: str, dataset: str, max_heavy: int, pair_set: str, num_pairs: int, num_chunks: int,
            chunk_index: int, timeout: float) -> str:
    """One ``python ged_exact_pairs.py --K="repr(v)" ...`` line (same format as the other schedulers)."""
    params = {
        '__DEBUG__': False, '__PREFIX__': prefix,
        'DATASET_NAME': dataset, 'MAX_HEAVY_ATOMS': max_heavy,
        'PAIR_SET': pair_set, 'NUM_PAIRS': num_pairs, 'PAIR_SEED': PAIR_SEED,
        'NUM_CHUNKS': num_chunks, 'CHUNK_INDEX': chunk_index,
        'TIMEOUT': timeout, 'UPPER_BOUND': 4.0, 'NUM_WORKERS': NUM_WORKERS,
    }
    return ' '.join(['python', 'ged_exact_pairs.py'] + [f'--{k}="{v!r}"' for k, v in params.items()])


def write_job(name: str, line: str) -> str:
    """One bash script per SLURM job: CPU only, single-threaded libraries, private TMPDIR."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write('export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def jobs(prefix: str, pair_set_spec: tuple) -> list:
    """The job script paths of all chunks of one pair set."""
    dataset, max_heavy, pair_set, num_pairs, num_chunks, timeout, _ = pair_set_spec
    paths = []
    for chunk_index in range(num_chunks):
        name = f'{prefix}__{dataset}__{pair_set}__chunk_{chunk_index:02d}_of_{num_chunks:02d}'
        line = command(prefix, dataset, max_heavy, pair_set, num_pairs, num_chunks, chunk_index, timeout)
        paths.append(write_job(name, line))
    return paths


def preload(pair_sets: list):
    """Load every dataset once, one after the other, so that the jobs find it in the chem_mat_data cache."""
    from chem_mat_data import load_smiles_dataset
    for dataset in sorted({ps[0] for ps in pair_sets}):
        df = load_smiles_dataset(dataset)
        print(f'pre-loaded {dataset}: {len(df)} molecules')


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        for spec in PAIR_SETS_SMOKE:
            for path in jobs(PREFIX_SMOKE, spec):
                print(path)
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter
    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload(PAIR_SETS)
    # one submitter per pair set, since the job time limit is a filler of the job template
    for spec in PAIR_SETS:
        submitter = ASlurmSubmitter(
            config_name=AUTOSLURM_CONFIG,
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'time': spec[6],
                'cpus': str(CPUS),
                'num_threads': '1',
                'mem': '8000mb',
                'job_name': 'ex17_ged',
                'additional_sbatch_configs': '#SBATCH --nodelist=euler',
            },
            archive_path=PATH,
        )
        for path in jobs(PREFIX, spec):
            submitter.add_command(f'bash {path}')
        print(f'{spec[0]} / {spec[2]}: {spec[3]} pairs in {submitter.count_jobs()} job(s), time limit {spec[6]}')
        submitter.submit()
