"""
Scheduler for Experiment 18 (ex_18): HDF vs. "modern" fingerprints MHFP and MAP4 (reviewer comment R2.1).

Reviewer 2 asks for more recent fingerprints than Morgan/RDKit, naming MHFP and MAP4. Both were proposed as
MinHash sketches for similarity search; for machine learning their authors use the folded bit-vector forms
(SECFP for MHFP, folded MAP4), which take the same place in the pipeline as Morgan
(``graph_hdc/baselines/minhash_fps.py``). This experiment compares, with the MLP only:

* ``hdf``     HDF, D=2048, L=2, continuous encoding, bidirectional message passing and total hydrogen counts
              (the corrected HDF)
* ``morgan``  Morgan / ECFP4, 2048 bits, radius 2 (the established baseline, as a reference point)
* ``secfp``   SECFP6 = folded MHFP6, 2048 bits, radius 3 (the MHFP default)
* ``map4``    folded MAP4, 2048 bits, radius 2 (the MAP4 default)

on the nine targets of the GNN comparison (ex_14), with 10 seeds and the fixed MLP of ex_14 ((100, 100),
lr 1e-3). KNN is left out: Jaccard KNN in scikit-learn takes about 2 hours per QM9 run.

Within a seed, all four representations use the same split: caching is disabled for every run, so the
dataset order is the per-seed shuffle of ``load_dataset`` and the split is drawn from it in the same way by
all modules (they share the loading, filtering and splitting hooks of ``predict_molecules.py``). The splits
are not those of ex_14, which loaded a seed-independent cached order; compare ex_18 with ex_14 only in
aggregate. Disabling the cache also means that no two jobs write the same cache file.

The runs take about two days, and other work edits this repository in the meantime. The jobs therefore run
on a frozen copy of the code, ``_ex18/code`` (all ``.py`` files of the package plus its templates and
VERSION, made once by ``snapshot()``; ``SNAPSHOT_INFO.txt`` records the git state it was taken from). The
group scripts put the copy first on ``PYTHONPATH`` and run the experiment modules from its fingerprints
folder, whose ``results`` is a symlink to the results folder of this repository, so the archives land in
the usual place. Every group script prints which ``graph_hdc`` it imported.

Each SLURM job runs one group script from ``_ex18/jobs/``: all four representations for the seeds of one
group (one seed per group for the large datasets, five for the small ones), one run after the other, on the
CPU only (``CUDA_VISIBLE_DEVICES=``; the GPU of euler is shared), with NUM_THREADS threads (more thrash on
the shared machine) and its own ``TMPDIR`` (chem_mat_data rewrites its metadata file in the temp folder on
every load). The datasets are downloaded to /tmp once, sequentially, before submitting, because concurrent
first downloads of the same file can corrupt it. Jobs are pinned to euler (config ``euler_3``). Group
scripts are never rewritten while their jobs may run (bash reads them during execution); a resubmission
writes new scripts with a ``__retry_N`` suffix.

    python _slurm_ex_18.py               # snapshot the code (once), pre-load datasets, write group scripts, submit
    python _slurm_ex_18.py --dry-run     # snapshot (once) and write group scripts and job scripts only
    python _slurm_ex_18.py --smoke       # snapshot (once), write one group script (FreeSolv, seed 0, prefix
                                         # ex_18_smoke) and print its path, to run it locally with bash
    python _slurm_ex_18.py --missing     # resubmit only the runs without a completed archive (no ex18 job may
                                         # be queued or running)
    python _slurm_ex_18.py --resnapshot  # replace the code snapshot (no ex18 job may be queued or running)
    python _slurm_ex_18.py --kcist       # write the command files for the KCIST runner (run_ex18_kcist.sbatch):
                                         # _ex18/kcist_commands.txt (all runs, longest first) and
                                         # _ex18/kcist_smoke.txt (FreeSolv, seed 0, prefix ex_18_kcist_smoke)

On KCIST the runs go through ``run_ex18_kcist.sbatch`` instead of the euler group scripts: a pinned checkout
of the code replaces the snapshot, each node pre-loads its datasets one after the other, and 12 workers per
node (3 per GPU) pull commands from the shard; the MLPs train on the GPU, HDF encodes on the CPU.

Results: ``analyze_ex_18.py``.
"""
import os
import sys
import glob
import shutil
import subprocess
from datetime import datetime

from _slurm_ex_14 import DATASETS, VARIANTS, DOWNSTREAM, _command

PATH = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.abspath(os.path.join(PATH, '..', '..', '..', '.venv'))
OUT = os.path.join(PATH, '_ex18')
# frozen copy of the code that the jobs run (see the module docstring)
CODE = os.path.join(OUT, 'code')
RUN_PATH = os.path.join(CODE, 'graph_hdc', 'experiments', 'fingerprints')
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
JOB_NAME = 'ex18_fp'

PREFIX = 'ex_18_fp'
PREFIX_SMOKE = 'ex_18_smoke'
SEEDS = list(range(10))

AUTOSLURM_CONFIG = 'euler_3'
# threads per run (torch, BLAS) and CPUs per job: small jobs side by side instead of a few wide ones
NUM_THREADS = 2
CPUS = 4

# the MLP of ex_14 for every representation; no KNN
MLP = {
    'MODELS': ['neural_net2'],
    'NN_HIDDEN_LAYER_SIZES': DOWNSTREAM['NN_HIDDEN_LAYER_SIZES'],
    'NN_LEARNING_RATE_INIT': DOWNSTREAM['NN_LEARNING_RATE_INIT'],
    '__CACHING__': False,
}
# (variant name, experiment module suffix, parameters)
REPRESENTATIONS = [
    ('hdf', 'hdc', {**VARIANTS['hdf'][1], **MLP, 'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total'}),
    ('morgan', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'morgan', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}),
    ('secfp', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'secfp', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 3}),
    ('map4', 'fp', {**MLP, 'FINGERPRINT_TYPE': 'map4', 'FINGERPRINT_SIZE': 2048, 'FINGERPRINT_RADIUS': 2}),
]

# seeds per group script and SLURM resources: the large datasets (COMPAS-3X, QM9) get one seed per job.
# Peak memory estimates (review, 2026-10-08): QM9 HDF ~12 GB, COMPAS-3X HDF ~10-12 GB, fingerprint runs ~5 GB,
# small datasets < 5 GB. Expected run time of a QM9 group ~4 h, so 16 h leaves a wide margin for MAP4, which
# runs last in every group.
SEEDS_PER_GROUP = {False: 5, True: 1}
TIME = {False: '06:00:00', True: '16:00:00'}
MEM = {False: '8000mb', True: '24000mb'}


def commands(prefix: str, ds: tuple, seeds: list) -> list:
    """All representations for every seed of one dataset, next to each other."""
    return [_command(module, prefix, seed, ds, params) for seed in seeds for _, module, params in REPRESENTATIONS]


def snapshot(force: bool = False):
    """
    Freeze the code that the jobs run: copy the ``.py`` files (plus templates and VERSION) of the package to
    ``_ex18/code`` and link the results folder. Reuses an existing snapshot unless ``force``.
    """
    if os.path.isdir(RUN_PATH) and not force:
        print(f'using the existing code snapshot {CODE}')
        return
    if os.path.isdir(CODE):
        shutil.rmtree(CODE)
    os.makedirs(CODE)
    excludes = ['results/', '.cache/', 'checkpoints/', '_ex*/', '__pycache__/', 'logs/', '_ged_timing_probe/']
    subprocess.run(['rsync', '-a', '--prune-empty-dirs', *[f'--exclude={e}' for e in excludes],
                    '--include=*/', '--include=*.py', '--include=*.j2', '--include=VERSION', '--exclude=*',
                    os.path.join(REPO, 'graph_hdc'), CODE], check=True)
    os.symlink(os.path.join(PATH, 'results'), os.path.join(RUN_PATH, 'results'))
    os.makedirs(os.path.join(RUN_PATH, '.cache'), exist_ok=True)   # caching is off; nothing is written here
    head = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(['git', '-C', REPO, 'status', '--short'], capture_output=True, text=True).stdout
    with open(os.path.join(CODE, 'SNAPSHOT_INFO.txt'), 'w') as f:
        f.write(f'snapshot taken {datetime.now().isoformat(timespec="seconds")} from {REPO}\n')
        f.write(f'git HEAD {head}; uncommitted changes at that time:\n{status}')
    n = sum(len(files) for _, _, files in os.walk(CODE))
    print(f'code snapshot: {n} files in {CODE}')


def write_group(name: str, lines: list) -> str:
    """One bash script per SLURM job: frozen code, CPU only, private TMPDIR, commands one after the other."""
    os.makedirs(os.path.join(OUT, 'jobs'), exist_ok=True)
    tmp = os.path.join(OUT, 'tmp', name)
    path = os.path.join(OUT, 'jobs', f'{name}.sh')
    with open(path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('export CUDA_VISIBLE_DEVICES=\n')
        f.write(f'export OMP_NUM_THREADS={NUM_THREADS} MKL_NUM_THREADS={NUM_THREADS} OPENBLAS_NUM_THREADS={NUM_THREADS}\n')
        f.write(f'export TMPDIR="{tmp}"\n')
        f.write(f'export PYTHONPATH="{CODE}"\n')
        f.write('mkdir -p "$TMPDIR"\n')
        f.write(f'cd "{RUN_PATH}"\n')
        f.write('python -c "import graph_hdc; print(\'graph_hdc imported from\', graph_hdc.__file__)"\n')
        for line in lines:
            f.write(f'echo "=== $(date +%H:%M:%S) {name}"\n{line}\n')
    return path


def groups(prefix: str, datasets: list, seeds: list, done: set = frozenset(), suffix: str = '') -> list:
    """
    (path, big) of the group scripts: one per dataset and chunk of seeds. Runs whose (variant, dataset, seed)
    is in ``done`` are left out, and groups without any remaining run are skipped.
    """
    out = []
    for ds in datasets:
        big = ds[4]
        step = SEEDS_PER_GROUP[big]
        for i in range(0, len(seeds), step):
            chunk = seeds[i:i + step]
            lines = [_command(module, prefix, seed, ds, params)
                     for seed in chunk for variant, module, params in REPRESENTATIONS
                     if (variant, ds[0], seed) not in done]
            if lines:
                name = f'{prefix}__{ds[0]}__seeds_{chunk[0]}-{chunk[-1]}{suffix}'
                out.append((write_group(name, lines), big))
    return out


def expected_cost(variant: str, ds: tuple) -> int:
    """Rough cost class of a run, to order the KCIST command file longest first (sharding is round-robin)."""
    size = 2 if ds[1] == 'qm9_smiles' else (1 if ds[1] == 'compas_3x' else 0)
    return 2 * size + (variant == 'hdf')


def write_kcist_commands():
    """Command files for run_ex18_kcist.sbatch: all runs longest first, and a FreeSolv smoke file."""
    os.makedirs(OUT, exist_ok=True)
    runs = [(expected_cost(variant, ds), _command(module, PREFIX, seed, ds, params))
            for seed in SEEDS for ds in DATASETS for variant, module, params in REPRESENTATIONS]
    # stable sort: within a cost class the seed-major order is kept
    lines = [line for _, line in sorted(runs, key=lambda r: -r[0])]
    smoke = [_command(module, 'ex_18_kcist_smoke', 0, ds, params)
             for ds in DATASETS if ds[0] == 'freesolv_hfe' for _, module, params in REPRESENTATIONS]
    for name, content in (('kcist_commands.txt', lines), ('kcist_smoke.txt', smoke)):
        with open(os.path.join(OUT, name), 'w') as f:
            f.write('\n'.join(content) + '\n')
        print(f'wrote {len(content):4d} commands -> _ex18/{name}')


def jobs_active() -> int:
    """Number of queued or running ex18 jobs of this user."""
    out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', JOB_NAME, '-o', '%i'],
                         capture_output=True, text=True, check=True).stdout
    return len(out.split())


def preload(datasets: list):
    """Download every dataset to /tmp once, one after the other (same call as predict_molecules.py)."""
    from chem_mat_data.main import load_graph_dataset
    for name in sorted({ds[1] for ds in datasets}):
        graphs = load_graph_dataset(name, folder_path='/tmp')
        print(f'pre-loaded {name}: {len(graphs)} graphs')


if __name__ == '__main__':

    if '--kcist' in sys.argv:
        write_kcist_commands()
        sys.exit(0)

    if '--resnapshot' in sys.argv or '--missing' in sys.argv:
        if jobs_active():
            sys.exit(f'{jobs_active()} {JOB_NAME} job(s) are queued or running; wait until they are done.')
    snapshot(force='--resnapshot' in sys.argv)
    if '--resnapshot' in sys.argv:
        sys.exit(0)

    if '--smoke' in sys.argv:
        ds = [d for d in DATASETS if d[0] == 'freesolv_hfe']
        print(groups(PREFIX_SMOKE, ds, [0])[0][0])
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in sys.argv
    if not dry_run:
        preload(DATASETS)

    done, suffix = frozenset(), ''
    if '--missing' in sys.argv:
        from analyze_ex_18 import collect
        records, _ = collect(PREFIX)
        done = {(r['variant'], r['dataset'], r['seed']) for r in records}
        retry = 1 + len({os.path.basename(p).split('__retry_')[1] for p in glob.glob(os.path.join(OUT, 'jobs', '*__retry_*.sh'))})
        suffix = f'__retry_{retry}'
        print(f'{len(done)} runs are complete; writing the remaining ones as {suffix}')

    all_groups = groups(PREFIX, DATASETS, SEEDS, done=done, suffix=suffix)
    total = 0
    # one submitter per resource class (time and memory differ between small and large datasets)
    for big in (True, False):
        paths = [p for p, b in all_groups if b == big]
        submitter = ASlurmSubmitter(
            config_name=AUTOSLURM_CONFIG,
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'time': TIME[big],
                'cpus': str(CPUS),
                'num_threads': str(NUM_THREADS),
                'mem': MEM[big],
                'job_name': JOB_NAME,
                'additional_sbatch_configs': '#SBATCH --nodelist=euler',
            },
            archive_path=PATH,
        )
        if not paths:
            continue
        for path in paths:
            submitter.add_command(f'bash {path}')
        total += submitter.count_jobs()
        submitter.submit()

    n_runs = sum(sum(1 for line in open(p) if line.startswith('python predict_')) for p, _ in all_groups)
    print(f'{n_runs} runs (of {len(REPRESENTATIONS) * len(SEEDS) * len(DATASETS)}: {len(DATASETS)} targets x '
          f'{len(SEEDS)} seeds x {len(REPRESENTATIONS)} representations) in {total} job(s), config {AUTOSLURM_CONFIG}')
