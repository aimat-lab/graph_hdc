"""
Scheduler for Experiment 21 (ex_21): the optional stereo attribute of HDF (``STEREO``).

Every part compares HDF without and with the CIP stereo descriptors of the atoms; nothing else changes between the two
arms (the Morgan fingerprints that molecule_collisions.py computes as well are not changed either).

* ``collisions``: molecule_collisions.py on ZINC250k with stereoisomers counted as distinct molecules (DEDUPLICATION =
  "isomeric"), at D = 256 and 2048. One job per (arm, D).
* ``bace``: BACE pIC50 with the HDF configuration of ex_14 (D = 2048, L = 2, MLP + KNN), seeds 0-9. One job per seed
  runs both arms on the same split, with the same codebooks and the same network seed, and a third run without stereo
  with another network seed (NN_SEED = 1000 + seed, as in ex_20), whose difference to the first run shows how much the
  MLP results vary from the network initialization alone.
* ``chiro``: the enantiomer docking ranking benchmark of Adams et al. (predict_enantiomers__hdc.py on the original
  splits) with the same HDF configuration and models, seeds 0-2 (codebooks, network initialization). One job per
  (seed, arm).
* ``props``: the BACE design on further datasets of the main comparison (FreeSolv, AqSolDB). One job per (dataset,
  seed) with the same three runs as for BACE.
* ``cmrt``: the elution order of enantiomers on the chiral column Chiralpak AD-H (CMRT subset of Baimacheva et al.,
  predict_enantiomers__hdc.py with prepare_cmrt_elution.py) with the same HDF configuration and models, on the
  authors' split (seeds 0-9: codebooks, network initialization) and on random splits by pair with 194 test pairs
  (seeds 0-9). One job per (split, seed) runs both arms.

Job setup as in ex_20: CPU only, 2 threads, 4 CPUs per job, a private TMPDIR, caching disabled (no job writes shared
cache files), this repository first on PYTHONPATH (so that a run from a git worktree uses the worktree's code), jobs
pinned to euler. The datasets are loaded once, one after the other, before submitting.

    python prepare_chiro_docking.py    # once: writes data/chiro_docking/chiro_docking.csv
    python prepare_cmrt_elution.py     # once: writes data/cmrt_elution/cmrt_adh.csv
    python _slurm_ex_21.py [part ...]  # pre-load datasets, write group scripts, submit (default: all parts)
    python _slurm_ex_21.py --dry-run   # write group scripts and job scripts only
    python _slurm_ex_21.py --smoke     # write small group scripts (prefix ex_21_smoke) and print their paths, to run
                                       # them locally with bash
"""
import os
import sys

from _slurm_ex_14 import DATASETS, VARIANTS, _command

PATH = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(PATH, '..', '..', '..'))
OUT = os.path.join(PATH, '_ex21')
# the venv lives in the main checkout only (a worktree has none)
VENV = os.path.join(REPO, '.venv')
if not os.path.isdir(VENV):
    VENV = '/media/ssd2/Programming/graph_hdc/.venv'
CHIRO_CSV = os.path.join(REPO, 'data', 'chiro_docking', 'chiro_docking.csv')
CMRT_CSV = os.path.join(REPO, 'data', 'cmrt_elution', 'cmrt_adh.csv')

PREFIX = 'ex_21'
PREFIX_SMOKE = 'ex_21_smoke'
ARMS = [False, True]

COLLISION_DATASET = 'zinc250k'
COLLISION_DIMS = [256, 2048]
BACE = [ds for ds in DATASETS if ds[0] == 'bace_ic50'][0]
BACE_SEEDS = list(range(10))
NN_SEED_OFFSET = 1000
CHIRO_SEEDS = [0, 1, 2]
PROPS = [ds for ds in DATASETS if ds[0] in ('freesolv_hfe', 'aqsoldb_logs')]
PROPS_SEEDS = list(range(10))
# (SPLIT_MODE, seeds): the authors' split with different codebooks and network seeds, and random splits by pair (with
# 10 repetitions each, so that a paired Wilcoxon test can reach p < 0.05; with 5, the smallest p is 0.0625)
CMRT_SPLITS = [('given', list(range(10))), ('random', list(range(10)))]
# test pairs of a random split: 194, as in the random splits of Baimacheva et al.
CMRT_NUM_TEST_PAIRS = 194
PARTS = ['collisions', 'bace', 'chiro', 'props', 'cmrt']

AUTOSLURM_CONFIG = 'euler_3'
NUM_THREADS = 2
CPUS = 4
# (memory, time limit) per part
# (BACE: the HDF encoding with stereo peaks at about 6.1 GB, ex_20 without stereo at about 4.9 GB)
RESOURCES = {'collisions': ('24000mb', '24:00:00'), 'bace': ('10000mb', '12:00:00'), 'chiro': ('16000mb', '24:00:00'),
             'props': ('10000mb', '12:00:00'), 'cmrt': ('10000mb', '12:00:00')}


def _line(module: str, params: dict) -> str:
    return ' '.join(['python', module] + [f'--{k}="{v!r}"' for k, v in params.items()])


def collision_command(prefix: str, dataset: str, dim: int, stereo: bool) -> str:
    return _line('molecule_collisions.py', {
        '__DEBUG__': False, '__PREFIX__': f'{prefix}_collisions', 'DATASET_NAME': dataset, 'EMBEDDING_SIZES': [dim],
        'HYDROGEN_COUNT': 'total', 'DEDUPLICATION': 'isomeric', 'STEREO': stereo,
    })


def hdf_params() -> dict:
    """The HDF configuration of ex_14 (encoder, MLP and KNN settings), caching disabled."""
    _module, params = VARIANTS['hdf']
    return {**params, '__CACHING__': False}


def prediction_command(prefix: str, part: str, ds: tuple, seed: int, stereo: bool, nn_seed=None) -> str:
    """predict_molecules__hdc.py with the ex_14 HDF configuration (parts "bace" and "props")."""
    return _command('hdc', f'{prefix}_{part}', seed, ds, {**hdf_params(), 'STEREO': stereo, 'NN_SEED': nn_seed})


def prediction_group(prefix: str, part: str, ds: tuple, seed: int) -> list:
    """Both arms on the same split, codebooks and network seed, plus the re-seeded run without stereo."""
    lines = [prediction_command(prefix, part, ds, seed, stereo) for stereo in ARMS]
    lines.append(prediction_command(prefix, part, ds, seed, False, nn_seed=NN_SEED_OFFSET + seed))
    return lines


def chiro_command(prefix: str, seed: int, stereo: bool, num_data=None) -> str:
    params = {
        '__DEBUG__': False, '__PREFIX__': f'{prefix}_chiro', 'SEED': seed, 'CHIRO_CSV': CHIRO_CSV,
        'NUM_DATA': num_data, 'NN_NUM_WORKERS': 0, **hdf_params(),
        'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'STEREO': stereo,
    }
    return _line('predict_enantiomers__hdc.py', params)


def cmrt_command(prefix: str, split_mode: str, seed: int, stereo: bool, num_data=None) -> str:
    """Elution order on Chiralpak AD-H: regression of the first-eluted label (1/0), ranking accuracy over the pairs
    and accuracy of the predictions thresholded at 0.5 (as the classification accuracy of Baimacheva et al.)."""
    params = {
        '__DEBUG__': False, '__PREFIX__': f'{prefix}_cmrt', 'SEED': seed, 'PAIRS_CSV': CMRT_CSV,
        'TARGET_COLUMN': 'first_eluted', 'SPLIT_MODE': split_mode, 'NUM_TEST': CMRT_NUM_TEST_PAIRS,
        'DATASET_NAME': 'cmrt_adh', 'DATASET_NAME_ID': 'cmrt_adh',
        'ACCURACY_THRESHOLD': 0.5, 'NUM_DATA': num_data, 'NN_NUM_WORKERS': 0, **hdf_params(),
        'BIDIRECTIONAL': True, 'HYDROGEN_COUNT': 'total', 'STEREO': stereo,
    }
    return _line('predict_enantiomers__hdc.py', params)


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


def groups(prefix: str, smoke: bool = False, parts: list = PARTS) -> dict:
    """Group script paths of the given parts (only their scripts are written, so that the scripts of running jobs of
    other parts are never rewritten). The smoke variant uses small datasets: BACE (as SMILES table) for the
    collisions, one BACE seed, 2 % of the ChIRo pairs and one seed of FreeSolv."""
    tag = lambda stereo: 'stereo' if stereo else 'plain'
    out = {part: [] for part in parts}
    if 'collisions' in parts:
        dataset, dims = ('bace_cls', [64]) if smoke else (COLLISION_DATASET, COLLISION_DIMS)
        # largest D first: the longest jobs start first
        for dim in sorted(dims, reverse=True):
            for stereo in ARMS:
                name = f'{prefix}_collisions__{dataset}__D{dim}__{tag(stereo)}'
                out['collisions'].append(write_group(name, [collision_command(prefix, dataset, dim, stereo)]))
    if 'chiro' in parts:
        for seed in CHIRO_SEEDS[:1] if smoke else CHIRO_SEEDS:
            for stereo in ARMS:
                name = f'{prefix}_chiro__seed_{seed}__{tag(stereo)}'
                line = chiro_command(prefix, seed, stereo, num_data=0.02 if smoke else None)
                out['chiro'].append(write_group(name, [line]))
    if 'bace' in parts:
        for seed in BACE_SEEDS[:1] if smoke else BACE_SEEDS:
            name = f'{prefix}_bace__seed_{seed}'
            out['bace'].append(write_group(name, prediction_group(prefix, 'bace', BACE, seed)))
    if 'props' in parts:
        for ds in PROPS[:1] if smoke else PROPS:
            for seed in PROPS_SEEDS[:1] if smoke else PROPS_SEEDS:
                name = f'{prefix}_props__{ds[0]}__seed_{seed}'
                out['props'].append(write_group(name, prediction_group(prefix, 'props', ds, seed)))
    if 'cmrt' in parts:
        for split_mode, seeds in CMRT_SPLITS:
            for seed in seeds[:1] if smoke else seeds:
                name = f'{prefix}_cmrt__{split_mode}__seed_{seed}'
                lines = [cmrt_command(prefix, split_mode, seed, stereo) for stereo in ARMS]
                out['cmrt'].append(write_group(name, lines))
    return out


def preload(parts: list):
    """Load the datasets of the given parts once, one after the other (same calls as the experiments)."""
    from chem_mat_data import load_smiles_dataset
    from chem_mat_data.main import load_graph_dataset
    if 'collisions' in parts:
        print(f'pre-loaded {COLLISION_DATASET}: {len(load_smiles_dataset(COLLISION_DATASET))} rows')
    names = ([BACE[1]] if 'bace' in parts else []) + ([ds[1] for ds in PROPS] if 'props' in parts else [])
    for name in names:
        print(f'pre-loaded {name}: {len(load_graph_dataset(name, folder_path="/tmp"))} graphs')


if __name__ == '__main__':

    for path, script in ((CHIRO_CSV, 'prepare_chiro_docking.py'), (CMRT_CSV, 'prepare_cmrt_elution.py')):
        if not os.path.exists(path):
            sys.exit(f'{path} is missing, run {script} first')

    # a misspelled flag must not lead to a real submission
    flags = {arg for arg in sys.argv[1:] if arg.startswith('--')}
    if flags - {'--dry-run', '--smoke'}:
        sys.exit(f'unknown flag(s) {sorted(flags - {"--dry-run", "--smoke"})}, expected --dry-run or --smoke')
    parts = [arg for arg in sys.argv[1:] if not arg.startswith('--')] or PARTS
    unknown = set(parts) - set(PARTS)
    if unknown:
        sys.exit(f'unknown part(s) {sorted(unknown)}, expected some of {PARTS}')

    if '--smoke' in flags:
        for part, paths in groups(PREFIX_SMOKE, smoke=True, parts=parts).items():
            for path in paths:
                print(part, path)
        sys.exit(0)

    from auto_slurm.aslurmx import ASlurmSubmitter

    dry_run = '--dry-run' in flags
    if not dry_run:
        preload(parts)

    for part, paths in groups(PREFIX, parts=parts).items():
        mem, time_limit = RESOURCES[part]
        submitter = ASlurmSubmitter(
            config_name=AUTOSLURM_CONFIG,
            batch_size=1,
            randomize=False,
            dry_run=dry_run,
            overwrite_fillers={
                'venv': VENV,
                'cwd': PATH,
                'time': time_limit,
                'cpus': str(CPUS),
                'num_threads': str(NUM_THREADS),
                'mem': mem,
                'job_name': f'ex21_{part}',
                'additional_sbatch_configs': '#SBATCH --nodelist=euler',
            },
            archive_path=PATH,
        )
        for path in paths:
            submitter.add_command(f'bash {path}')
        print(f'{part}: {len(paths)} group script(s) in {submitter.count_jobs()} job(s), config {AUTOSLURM_CONFIG}')
        submitter.submit()
