"""
Scheduler for Experiment 16 (ex_16): collisions of HDF, binary Morgan fingerprints and 1-WL (``molecule_collisions.py``).

One SLURM job per (dataset, embedding size D) on the local euler node; every job also computes the (size-free) 1-WL
hashes, which ``analyze_ex_16.py`` takes from any run of a dataset. Jobs train nothing; they encode, hash and search.

Settings follow the euler notes: CPU only (``CUDA_VISIBLE_DEVICES=``), 4 CPUs and 2 threads per job, a private
``TMPDIR`` (chem_mat_data rewrites its metadata file in the temp folder), jobs pinned to euler. The jobs wait for all
queued ``ex15_bidir`` jobs (``--dependency=afterany``) so that they do not slow ex_15 down. The SMILES datasets are
loaded once, one after the other, before submitting. The group scripts put the repository that contains this file
first on PYTHONPATH, so a run from a git worktree uses the worktree's graph_hdc package (the venv's editable install
points to the main checkout) and does not pick up later edits of the main checkout while the jobs wait in the queue.

    python _slurm_ex_16.py             # pre-load datasets, write group scripts, submit
    python _slurm_ex_16.py --dry-run   # write group scripts and job scripts only
"""
import os
import subprocess
import sys

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(PATH, '_ex16')
# the venv lives in the main checkout only (a worktree has none)
VENV = os.path.join(REPO, '.venv')
if not os.path.isdir(VENV):
    VENV = '/media/ssd2/Programming/graph_hdc/.venv'

# ex_16_collisions: first run (2026-10-08) with implicit hydrogen counts; ex_16_collisions_totalh: total H counts;
# ex_16_collisions_unitspec: total H counts and unit-modulus codebooks (the HDF default since 2026-10-09)
PREFIX = 'ex_16_collisions_unitspec'
HYDROGEN_COUNT = 'total'
SPECTRUM = 'unit'
DIMS = [32, 64, 128, 256, 512, 1024, 2048]
# (DATASET_NAME, memory per job)
DATASETS = [('qm9_smiles', '16000mb'), ('zinc250k', '24000mb')]
WAIT_FOR_JOB_NAMES = ['ex15_bidir']


def command(dataset: str, dim: int) -> str:
    params = {'__DEBUG__': False, '__PREFIX__': PREFIX, 'DATASET_NAME': dataset, 'EMBEDDING_SIZES': [dim],
              'HYDROGEN_COUNT': HYDROGEN_COUNT, 'SPECTRUM': SPECTRUM}
    return ' '.join(['python', 'molecule_collisions.py'] + [f'--{k}="{v!r}"' for k, v in params.items()])


def write_group(name: str, line: str) -> str:
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write('export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2\n')
        f.write(f'export PYTHONPATH="{REPO}${{PYTHONPATH:+:$PYTHONPATH}}"\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc from\', graph_hdc.__file__)"\n')
        f.write(f'git -C "{REPO}" rev-parse HEAD\n')
        f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def queued_job_ids(names: list) -> list:
    out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%i %j'],
                         capture_output=True, text=True).stdout
    return [line.split()[0] for line in out.splitlines() if line.split()[1:] and line.split()[1] in names]


def preload():
    from chem_mat_data import load_smiles_dataset
    for dataset, _ in DATASETS:
        print(f'pre-loaded {dataset}: {len(load_smiles_dataset(dataset))} rows')


if __name__ == '__main__':
    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload()

    wait_ids = queued_job_ids(WAIT_FOR_JOB_NAMES)
    sbatch_extra = '#SBATCH --nodelist=euler'
    if wait_ids:
        sbatch_extra += '\n#SBATCH --dependency=afterany:' + ':'.join(wait_ids)
    print(f'waiting for {len(wait_ids)} job(s): {" ".join(wait_ids)}')

    for dataset, mem in DATASETS:
        submitter = ASlurmSubmitter(
            config_name='euler_3',
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'cpus': '4',
                'num_threads': '2',
                'mem': mem,
                'time': '24:00:00',
                'job_name': 'ex16_collisions',
                'additional_sbatch_configs': sbatch_extra,
            },
            archive_path=PATH,
        )
        # largest D first: the longest jobs start first
        for dim in sorted(DIMS, reverse=True):
            name = f'{PREFIX}__{dataset}__D{dim}'
            submitter.add_command(f'bash {write_group(name, command(dataset, dim))}')
        print(f'{dataset}: {submitter.count_jobs()} job(s)')
        submitter.submit()
