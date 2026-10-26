"""
Scheduler for ex_11_b: re-run of the similarity-based virtual screening benchmark (SI Figure S2a; original runs:
prefix ex_11_bioact, ``_slurm_ex_11_bioactivity.py``) with the current HDF.

Same protocol as ex_11: Riniker 1 and 2, MUV and DUD-E; per target, 50 repetitions with 5 query actives, targets
with at least 10 actives, seed 1, BEDROC alpha 20 and enrichment factors at 1% and 5%; Morgan fingerprints (ECFP4,
radius 2, 2048 bits, binary, Tanimoto distance) and HDF (2048 dimensions, depth 2, continuous encoders, cosine
distance). New compared to ex_11:
- HDF uses the standard configuration of the revision (bidirectional message passing, unit-modulus codebooks,
  total hydrogen counts); the ex_11 runs predate all three. They are passed explicitly here.
- The main HDF arm leaves out the encodings of the graph size and the graph diameter (GRAPH_ATTRIBUTES=False;
  user, 2026-10-13, since HDF correlates more strongly with the exact GED without them). HDF with them is run as
  a reference arm.
- Morgan is re-run unchanged (its code has not changed since ex_11), which checks that ex_11 reproduces and keeps
  all arms on the same code version.
- Each run has its own TMPDIR, because the dataset loader deletes and re-copies its folder in the temporary
  directory, which concurrent runs on the same dataset shared in ex_11. The chem_mat_data cache
  (~/.cache/chem_mat_data) has to hold all four datasets before submission (filled one by one on 2026-10-13), so
  that no two runs download the same dataset at the same time.

The runs use the code of this worktree (PYTHONPATH), not the shared checkout, and log its commit and diff. They run
on euler only, CPU only, with few threads (other load on the machine), in three SLURM jobs that each run their runs
one after the other: SLURM sees all 96 GB of euler as free, but processes outside SLURM use about 35 GB, so the
memory of all concurrent runs has to stay well below the requests SLURM would otherwise allow.

    python _slurm_ex_11b.py             # write the run and chain scripts, submit the three chains
    python _slurm_ex_11b.py --dry-run   # write the scripts only
    python _slurm_ex_11b.py --smoke     # write smoke-test run scripts (prefix ex_11_b_smoke), print their paths
"""
import os
import subprocess
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(PATH, '..', '..', '..'))  # repository (worktree) root, first on PYTHONPATH
VENV = '/media/ssd2/Programming/graph_hdc/.venv'
OUT = os.path.join(PATH, '_ex11b')

PREFIX = 'ex_11_b'
DATASETS = ['dud_e', 'riniker_1', 'muv', 'riniker_2']

# same evaluation protocol as ex_11 (_slurm_ex_11_bioactivity.py)
FIXED = {
    '__DEBUG__': False,
    '__CACHING__': False,
    'SEED': 1,
    'NUM_QUERY_ACTIVES': 5,
    'NUM_REPETITIONS': 50,
    'MIN_ACTIVES_PER_TARGET': 10,
    'BEDROC_ALPHA': 20.0,
    'EF_PERCENTAGES': [0.01, 0.05],
}
FP = {'FP_TYPE': 'ecfp4', 'FP_RADIUS': 2, 'FP_SIZE': 2048, 'USE_COUNTS': False}
HDC = {'EMBEDDING_SIZE': 2048, 'NUM_LAYERS': 2, 'ENCODING_MODE': 'continuous', 'DEVICE': 'cpu', 'BATCH_SIZE': 64,
       'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'SPECTRUM': 'unit'}
ARMS = {  # arm: (module, parameters)
    'hdc_noglobal': ('hdc', {**HDC, 'GRAPH_ATTRIBUTES': False}),
    'fp': ('fp', FP),
    'hdc_global': ('hdc', {**HDC, 'GRAPH_ATTRIBUTES': True}),
}

# Three SLURM jobs that run their (arm, dataset) runs one after the other. The main HDF arm and Morgan come first;
# the two DUD-E HDF runs (the largest) never overlap.
CHAINS = {
    'a': [('hdc_noglobal', 'dud_e'), ('hdc_global', 'riniker_1'), ('hdc_global', 'muv'), ('hdc_global', 'riniker_2')],
    'b': [('hdc_noglobal', 'riniker_1'), ('hdc_noglobal', 'muv'), ('hdc_noglobal', 'riniker_2'), ('hdc_global', 'dud_e')],
    'c': [('fp', 'dud_e'), ('fp', 'riniker_1'), ('fp', 'muv'), ('fp', 'riniker_2')],
}
# (CPUs = threads, memory) per chain. Peak memory of ex_11 (sacct MaxRSS): HDF DUD-E 42.8 GB, of which about 20 GB
# were the unused per-layer results that ex_11_b no longer keeps; Morgan DUD-E 14.1 GB.
CHAIN_RESOURCES = {'a': (4, '36000mb'), 'b': (4, '36000mb'), 'c': (2, '20000mb')}
CHAIN_TIME = '48:00:00'

# local smoke test: smallest dataset, two repetitions
SMOKE_DATASETS = ['riniker_2']
SMOKE_OVERRIDES = {'NUM_REPETITIONS': 2}


def command(prefix: str, module: str, params: dict, dataset: str, overrides: dict = None) -> str:
    """One ``python predict_bioactivity__<module>.py --K="repr(v)" ...`` line (format of _slurm_ex_11_bioactivity)."""
    values = {'__PREFIX__': prefix, **FIXED, **params, 'DATASET_NAME': dataset,
              'DATASET_NAME_ID': f'{dataset}_bioact_{module}'}
    values.update(overrides or {})
    return ' '.join(['python', f'predict_bioactivity__{module}.py'] + [f'--{k}="{v!r}"' for k, v in values.items()])


def write_run(name: str, line: str, threads: int) -> str:
    """One bash script per run: worktree code, CPU only, few threads, private TMPDIR, logs commit and diff."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('set -e\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads}\n')
        f.write(f'export PYTHONPATH="{ROOT}"\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc from\', graph_hdc.__file__)"\n')
        f.write(f'git -C "{ROOT}" rev-parse HEAD\n')
        f.write(f'git -C "{ROOT}" status --porcelain --untracked-files=no\n')
        f.write(f'git -C "{ROOT}" --no-pager diff\n')
        f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
        f.write('rm -rf "$TMPDIR"\n')
    return path


def runs(smoke: bool = False) -> dict:
    """Run script path per (arm, dataset)."""
    prefix = f'{PREFIX}_smoke' if smoke else PREFIX
    result = {}
    for chain, members in CHAINS.items():
        threads = CHAIN_RESOURCES[chain][0]
        for arm, dataset in members:
            if smoke and dataset not in SMOKE_DATASETS:
                continue
            module, params = ARMS[arm]
            line = command(prefix, module, params, dataset, SMOKE_OVERRIDES if smoke else None)
            result[(arm, dataset)] = write_run(f'{prefix}__{arm}__{dataset}', line, threads)
    return result


def write_chain(chain: str, scripts: list) -> str:
    """One sbatch script per chain: runs its run scripts one after the other, each with its own log file; a failed
    run does not stop the chain, and its exit code is logged."""
    cpus, mem = CHAIN_RESOURCES[chain]
    logs = os.path.join(OUT, 'logs')
    os.makedirs(logs, exist_ok=True)
    path = os.path.join(OUT, 'jobs', f'chain_{chain}.sbatch')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'#SBATCH --job-name=ex11b_{chain}\n')
        f.write('#SBATCH --nodelist=euler\n')
        f.write(f'#SBATCH --cpus-per-task={cpus}\n')
        f.write(f'#SBATCH --mem={mem}\n')
        f.write(f'#SBATCH --time={CHAIN_TIME}\n')
        f.write(f'#SBATCH --output={logs}/chain_{chain}_%j.out\n')
        f.write(f'source "{VENV}/bin/activate"\n')
        for script in scripts:
            name = os.path.basename(script)[:-3]
            f.write(f'echo "=== $(date \'+%F %T\') start {name}"\n')
            f.write(f'bash "{script}" > "{logs}/{name}.log" 2>&1\n')
            f.write(f'echo "=== $(date \'+%F %T\') exit $? {name}"\n')
    return path


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        for path in runs(smoke=True).values():
            print(path)
        sys.exit(0)

    scripts = runs()
    for chain, members in CHAINS.items():
        path = write_chain(chain, [scripts[member] for member in members])
        cpus, mem = CHAIN_RESOURCES[chain]
        print(f'chain {chain}: {", ".join(f"{a}/{d}" for a, d in members)} ({cpus} CPUs, {mem})')
        if '--dry-run' not in sys.argv:
            print(' ', subprocess.run(['sbatch', path], capture_output=True, text=True, check=True).stdout.strip())
